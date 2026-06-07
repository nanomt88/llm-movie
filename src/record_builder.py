# -*- coding: utf-8 -*-
"""
记录组装模块 — 将用户文本、情绪标签、电影类型、节假日等信息
组装为分析记录列表，供热图生成模块使用。

数据流：读取 DataFrame + user_seg + idx_to_emotion + cal + movie_info

文件输出：
  intermediate/records_summary.json — 记录统计 + 前 50 条样本
  intermediate/records_full.jsonl   — 完整记录（JSON Lines 格式）
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import (
    INTERMEDIATE_DIR,
    generate_run_id, log, log_error, log_warn,
)
from src.debug_dump import (
    write_json, write_json_lines,
    extract_run_id, load_latest_by_step,
)
from src.text_extractor import extract_movie_text_from_system, extract_imdb_ids
from src.genre_classifier import classify_genre_from_ids, log_not_found_ids


def build_analysis_data(df: pd.DataFrame,
                        user_seg: dict,
                        idx_to_emotion: dict,
                        cal,
                        movie_info: dict) -> list[dict]:
    """
    核心数据组装函数。

    遍历 DataFrame 的每一行，将情绪标签、年龄分段、节假日、电影类型
    组装为扁平化的分析记录列表。

    参数：
        df             : 对话 DataFrame
        user_seg       : {user_id → age_segment}
        idx_to_emotion : {df_index → emotion_label}
        cal            : HolidayCalendar 实例
        movie_info     : {imdb_id → info}

    返回：
        records — [{'age_segment', 'holiday_name', 'emotion', 'genre'}, ...]
    """
    not_found_ids: list[str] = []
    records = []
    skipped_no_next = 0
    skipped_no_genre = 0
    skipped_no_emotion = 0

    total_rows = len(df)
    log('RecordBuilder', f'开始组装记录: 共 {total_rows} 行')

    for idx, row in df.iterrows():
        try:
            # is_seeker == True 的行是系统回复，其后一行包含推荐电影
            if row.get('is_seeker') != True:
                continue
            if idx not in idx_to_emotion:
                skipped_no_emotion += 1
                continue

            # 获取基础信息
            user_id = row.get('user_id')
            age_seg = user_seg.get(user_id, 'unknown')
            emotion = idx_to_emotion[idx]

            # 检查下一行（系统推荐）
            current_pos = df.index.get_loc(idx)
            if current_pos >= len(df) - 1:
                skipped_no_next += 1
                continue

            next_row = df.iloc[current_pos + 1]
            sys_text = extract_movie_text_from_system(next_row.get('processed'))
            imdb_ids = extract_imdb_ids(sys_text)
            genres = classify_genre_from_ids(imdb_ids, movie_info, not_found_ids)

            if not genres:
                skipped_no_genre += 1
                continue

            # 节假日判定
            utc_val = row.get('utc_time')
            holiday_name = '非节假日'
            try:
                dt = datetime.utcfromtimestamp(int(utc_val))
                if cal.is_holiday(dt):
                    names = cal.get_holiday_names(dt)
                    if names:
                        holiday_name = names[0]
            except (ValueError, TypeError, OverflowError):
                pass

            # 每个类型生成一条记录
            for genre in genres:
                records.append({
                    'age_segment': age_seg,
                    'holiday_name': holiday_name,
                    'emotion': emotion,
                    'genre': genre,
                })

        except Exception as e:
            log_error('RecordBuilder', f'行 {idx} 处理失败: {e}')
            continue

    # 日志统计
    log('RecordBuilder', f'组装完成: {len(records)} 条记录')
    log('RecordBuilder', f'  跳过(无下一行): {skipped_no_next}')
    log('RecordBuilder', f'  跳过(无类型): {skipped_no_genre}')
    log('RecordBuilder', f'  跳过(无情绪): {skipped_no_emotion}')

    # 输出未匹配的 IMDB ID
    log_not_found_ids(not_found_ids)

    return records


def save_records(records: list[dict], run_id: str = None) -> dict:
    """
    保存分析记录到 intermediate 文件。

    参数：
        run_id : 流水号，为 None 时自动生成

    返回：
        {'summary': path, 'full': path}
    """
    files = {}

    if not records:
        path = write_json({'total_records': 0}, 'records_summary',
                          run_id=run_id, step='04')
        if path:
            files['summary'] = path
        return files

    # 统计摘要
    df_rec = pd.DataFrame(records)
    summary = {
        'total_records': len(records),
        'total_unique_ages': int(df_rec['age_segment'].nunique()),
        'total_unique_genres': int(df_rec['genre'].nunique()),
        'total_unique_holidays': int(df_rec['holiday_name'].nunique()),
        'emotion_distribution': df_rec['emotion'].value_counts().to_dict(),
        'age_distribution': df_rec['age_segment'].value_counts().to_dict(),
        'holiday_records_count': int((df_rec['holiday_name'] != '非节假日').sum()),
        'samples': records[:50],
    }
    path = write_json(summary, 'records_summary', run_id=run_id, step='04')
    if path:
        files['summary'] = path

    # 完整记录写入 JSON Lines
    path = write_json_lines(records, 'records_full', run_id=run_id, step='04')
    if path:
        files['full'] = path

    return files


def run(df: pd.DataFrame,
        user_seg: dict,
        idx_to_emotion: dict,
        cal,
        movie_info: dict,
        save_files: bool = True,
        run_id: str = None) -> dict:
    """
    统一入口：组装分析记录。

    参数：
        df             : 对话 DataFrame
        user_seg       : 用户年龄分段
        idx_to_emotion : 情绪标签映射
        cal            : 节假日日历
        movie_info     : 电影信息
        save_files     : 是否保存 intermediate 文件
        run_id         : 流水号（同一轮流水线执行共享此 ID）

    返回：
        dict — {
            'run_id': str,
            'records': list[dict],
            'files': {...}  (save_files=True 时)
        }
    """
    if run_id is None:
        run_id = generate_run_id()

    log('RecordBuilder', '=' * 40)
    log('RecordBuilder', f'开始组装分析记录 (run_id={run_id})')

    records = build_analysis_data(df, user_seg, idx_to_emotion, cal, movie_info)

    result = {'run_id': run_id, 'records': records}

    if save_files:
        log('RecordBuilder', '保存 intermediate 文件...')
        files = save_records(records, run_id=run_id)
        result['files'] = files

    log('RecordBuilder', f'记录组装完成 ({len(records)} 条)')
    log('RecordBuilder', '=' * 40)
    return result


def main():
    """
    独立运行入口。

    用法：python -m src.record_builder
    说明：自动发现最新的 S03 idx_to_emotion_full 文件，加载所有上游数据，
          完成记录组装，沿用上游的 run_id 保存 S04 文件。
    """
    # 加载 S03 idx_to_emotion
    data, filepath = load_latest_by_step('03', 'idx_to_emotion_full')
    if not data:
        log_error('RecordBuilder', '未找到 idx_to_emotion_full 文件，无法独立运行')
        sys.exit(1)

    upstream_run_id = extract_run_id(filepath)
    idx_to_emotion = {int(k): v for k, v in data.items()}
    log('RecordBuilder', f'从文件恢复 idx_to_emotion (run_id={upstream_run_id}): '
                         f'{os.path.basename(filepath)}')

    # 加载原始数据
    from src.data_loader import load_csv, load_holiday_calendar, load_movie_info, segment_users
    df = load_csv()
    cal = load_holiday_calendar()
    movie_info = load_movie_info()
    user_seg = segment_users()

    result = run(df, user_seg, idx_to_emotion, cal, movie_info,
                 save_files=True, run_id=upstream_run_id)
    print(f'\n组装完成: {len(result["records"])} 条记录, run_id: {result["run_id"]}')


if __name__ == '__main__':
    main()
