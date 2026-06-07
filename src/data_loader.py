# -*- coding: utf-8 -*-
"""
数据加载模块 — 加载 CSV、节假日日历、电影信息、年龄分段。

文件输出：
  intermediate/user_seg.json   — 用户年龄分段结果
  intermediate/df_info.json    — DataFrame 概要信息
  intermediate/holiday_info.json — 节假日列表
  intermediate/movie_info.json — 电影信息概要
"""

import json
import os
import sys
from collections import Counter

import pandas as pd

from src.config import (
    CSV_PATH, HOLIDAY_CSV, MOVIE_INFO_PATH,
    generate_run_id, log, log_error, log_warn,
)
from src.debug_dump import write_json

# ── 项目依赖 ──────────────────────────────────────────────────────
# holiday_util 和 age_segment 在项目根目录，需要加入 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src import age_segment
from src.holiday_util import HolidayCalendar


def load_csv(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """加载对话 CSV 文件"""
    log('DataLoader', f'加载 CSV: {csv_path}')
    try:
        df = pd.read_csv(csv_path)
        log('DataLoader', f'  行数: {len(df)}, 列数: {len(df.columns)}')
        log('DataLoader', f'  列名: {list(df.columns)}')
        return df
    except Exception as e:
        log_error('DataLoader', f'CSV 加载失败: {e}')
        raise


def load_holiday_calendar(holiday_csv: str = HOLIDAY_CSV) -> HolidayCalendar:
    """加载节假日日历"""
    log('DataLoader', f'加载节假日: {holiday_csv}')
    try:
        cal = HolidayCalendar(holiday_csv)
        log('DataLoader', f'  共 {len(cal.all_holidays)} 个节假日')
        return cal
    except Exception as e:
        log_error('DataLoader', f'节假日加载失败: {e}')
        raise


def load_movie_info(movie_info_path: str = MOVIE_INFO_PATH) -> dict:
    """加载电影信息 JSON"""
    log('DataLoader', f'加载电影信息: {movie_info_path}')
    try:
        with open(movie_info_path, 'r', encoding='utf-8') as f:
            movie_info = json.load(f)
        log('DataLoader', f'  共 {len(movie_info)} 部电影')
        return movie_info
    except Exception as e:
        log_error('DataLoader', f'电影信息加载失败: {e}')
        raise


def segment_users(csv_path: str = CSV_PATH) -> dict:
    """
    用户年龄分段。

    返回：
        dict — {user_id: age_segment}
    """
    log('DataLoader', f'用户年龄分段: {csv_path}')
    try:
        user_seg_df = age_segment.segment_users(csv_path)
        user_seg = dict(zip(user_seg_df['user_id'], user_seg_df['age_segment']))
        known = user_seg_df[user_seg_df['age_segment'] != 'unknown']
        log('DataLoader', f'  总用户: {len(user_seg_df)}, '
                          f'可分段: {len(known)} '
                          f'({len(known) / max(len(user_seg_df), 1) * 100:.1f}%)')
        return user_seg
    except Exception as e:
        log_error('DataLoader', f'年龄分段失败: {e}')
        raise


def save_intermediate(user_seg: dict, df: pd.DataFrame,
                      cal: HolidayCalendar, movie_info: dict,
                      run_id: str = None) -> dict:
    """
    将所有加载的数据保存到 intermediate 目录，返回保存的文件路径字典。

    参数：
        run_id : 流水号，为 None 时自动生成

    返回：
        {key: file_path} 其中 key 为数据标识
    """
    files = {}

    # 1. user_seg
    user_seg_data = {
        'total_users': len(user_seg),
        'age_distribution': {},
        'total': user_seg,
    }
    # 统计年龄段分布（需要从 age_segment 重新统计）
    try:
        user_seg_df = age_segment.segment_users(CSV_PATH)
        user_seg_data['age_distribution'] = user_seg_df['age_segment'].value_counts().to_dict()
    except Exception as e:
        log_warn('DataLoader', f'年龄段分布统计失败: {e}')
    path = write_json(user_seg_data, 'user_seg', run_id=run_id, step='01')
    if path:
        files['user_seg'] = path

    # 2. df_info
    df_info = {
        'rows': len(df),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'is_seeker_distribution': df['is_seeker'].value_counts().to_dict() if 'is_seeker' in df.columns else {},
    }
    path = write_json(df_info, 'df_info', run_id=run_id, step='01')
    if path:
        files['df_info'] = path

    # 3. holiday_info
    holiday_info = {
        'total': len(cal.all_holidays),
        'holidays': [
            {'date': str(h.date), 'description': h.description, 'type': h.type}
            for h in cal.all_holidays
        ],
    }
    path = write_json(holiday_info, 'holiday_info', run_id=run_id, step='01')
    if path:
        files['holiday_info'] = path

    # 4. movie_info summary
    all_genres = set()
    for info in movie_info.values():
        if isinstance(info, dict) and 'genres' in info:
            all_genres.update(info['genres'])
    movie_summary = {
        'total_movies': len(movie_info),
        'unique_genres': sorted(all_genres),
        'genre_count': len(all_genres),
    }
    #path = write_json(movie_summary, 'movie_info', run_id=run_id, step='01')
    #if path:
    files['movie_info'] = "D:\workspaces\python\llm-movie\data\movie_info.json"

    return files


def run(csv_path: str = CSV_PATH,
        holiday_csv: str = HOLIDAY_CSV,
        movie_info_path: str = MOVIE_INFO_PATH,
        save_files: bool = True,
        run_id: str = None) -> dict:
    """
    统一入口：加载所有数据。

    参数：
        csv_path        : 对话 CSV 路径
        holiday_csv     : 节假日 CSV 路径
        movie_info_path : 电影信息 JSON 路径
        save_files      : 是否保存 intermediate 文件
        run_id          : 流水号（同一轮流水线执行共享此 ID）

    返回：
        dict — {
            'run_id': str,
            'df': DataFrame,
            'cal': HolidayCalendar,
            'movie_info': dict,
            'user_seg': dict,
            'files': {...}  (save_files=True 时)
        }
    """
    if run_id is None:
        run_id = generate_run_id()

    result = {'run_id': run_id}

    log('DataLoader', '=' * 40)
    log('DataLoader', f'开始加载所有数据 (run_id={run_id})')

    # 分段加载（独立 try/except 以保证后续模块仍能运行）
    try:
        df = load_csv(csv_path)
        result['df'] = df
    except Exception as e:
        log_error('DataLoader', f'CSV 加载失败，无法继续: {e}')
        raise

    try:
        cal = load_holiday_calendar(holiday_csv)
        result['cal'] = cal
    except Exception as e:
        log_error('DataLoader', f'节假日加载失败: {e}')
        raise

    try:
        movie_info = load_movie_info(movie_info_path)
        result['movie_info'] = movie_info
    except Exception as e:
        log_error('DataLoader', f'电影信息加载失败: {e}')
        raise

    try:
        csv_path = "D:\workspaces\python\llm-movie\data\my-train-data.csv"
        user_seg = segment_users(csv_path)
        result['user_seg'] = user_seg
    except Exception as e:
        log_error('DataLoader', f'年龄分段失败: {e}')
        raise
    # 统计用户占比信息
    get_age_segment_summary(user_seg, df)
    # 保存 intermediate 文件
    if save_files:
        log('DataLoader', '保存 intermediate 文件...')
        files = save_intermediate(user_seg, df, cal, movie_info, run_id=run_id)
        result['files'] = files

    log('DataLoader', '数据加载完成')
    log('DataLoader', '=' * 40)
    return result


def get_age_segment_summary(user_seg_total: dict, df_real: pd.DataFrame) -> dict:
    """
    使用 DataFrame API 统计年龄段分布摘要。

    参数：
        df: 包含 user_id 列的 DataFrame
        user_seg: 用户年龄段字典 {user_id: age_segment}

    返回：
        dict — 年龄段分布统计信息
    """

    real_user =  {uid: '' for uid in  df_real[df_real['is_seeker'] == True]['user_id'].unique()}

    real_user.update({k: user_seg_total[k] for k in user_seg_total.keys() & real_user.keys()})

    counts = Counter(real_user.values())
    summary = {k: {'count': v, 'percentage': round(v / len(real_user) * 100, 2)} for k, v in counts.items()}
    # log('DataLoader', f'用户年龄识别结果：{summary}')
    log('DataLoader', f'用户年龄识别结果：\n{json.dumps(summary, indent=2, ensure_ascii=False)}')
    return summary

def main():
    """
    独立运行入口。

    用法：python -m src.data_loader
    说明：生成新的 run_id，加载所有数据并保存中间文件。
    """
    result = run(save_files=True)
    print(f'\n加载完成，run_id: {result["run_id"]}')
    for key, path in result.get('files', {}).items():
        print(f'  {key}: {path}')


if __name__ == '__main__':
    main()
