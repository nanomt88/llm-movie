# -*- coding: utf-8 -*-
"""
文本提取模块 — 从 CSV 字段中解析用户发言、系统回复、IMDB ID。

文件输出：
  intermediate/text_batch_info.json — 提取的文本批次信息（含样本）
  intermediate/text_batch_full.json — 完整文本批次（用于后续模块恢复运行）
"""

import ast
import re
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import INTERMEDIATE_DIR, generate_run_id, log, log_error, log_warn
from src.debug_dump import write_json


# ══════════════════════════════════════════════════════════════════
#  字段解析器
# ══════════════════════════════════════════════════════════════════

def parse_raw_field(value):
    """将 CSV 中的字符串字段安全地转换为 Python 对象"""
    if pd.isna(value) or not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        log_warn('TextExtractor', f'literal_eval 解析失败: {repr(value[:80])}..., 错误: {e}')
        return None


def extract_user_text_from_context(context_raw) -> str | None:
    """从上下文字段中提取用户的实际说话内容"""
    ctx = parse_raw_field(context_raw)
    if not ctx or not isinstance(ctx, list):
        return None
    if len(ctx) >= 2 and ctx[0] == 'USER':
        return ctx[1]
    return None


def extract_user_text_direct(processed_value) -> str | None:
    """直接从处理后的字段中提取用户发言"""
    parsed = parse_raw_field(processed_value)
    if not parsed or not isinstance(parsed, list) or len(parsed) < 2:
        return None
    if parsed[0] == 'USER':
        return parsed[1]
    return None


def extract_movie_text_from_system(raw_value) -> str | None:
    """从系统推荐字段中提取电影相关的描述文本"""
    parsed = parse_raw_field(raw_value)
    if not parsed or not isinstance(parsed, list) or len(parsed) < 2:
        return None
    if parsed[0] == 'SYSTEM':
        return parsed[1]
    return None


# ══════════════════════════════════════════════════════════════════
#  IMDB ID 提取
# ══════════════════════════════════════════════════════════════════

_IMDB_ID_RE = re.compile(r'tt\d+')


def extract_imdb_ids(text: str) -> list[str]:
    """从文本中提取所有 IMDB ID (ttXXXXXXX) 并去重"""
    if not text:
        return []
    seen = set()
    result = []
    for m in _IMDB_ID_RE.finditer(text):
        tid = m.group()
        if tid not in seen:
            seen.add(tid)
            result.append(tid)
    return result


# ══════════════════════════════════════════════════════════════════
#  批量文本提取
# ══════════════════════════════════════════════════════════════════

def build_text_batch(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    """
    从 DataFrame 中提取所有用户发言文本。

    逻辑（与 v6 完全一致）：
      - is_seeker == False 的行视为用户发言
      - 优先从 processed 字段提取，失败则回退 raw 字段

    返回：
        (text_batch, text_indices)
          text_batch    — 用户文本列表
          text_indices  — 对应 DataFrame 行索引列表
    """
    text_batch: list[str] = []
    text_indices: list[int] = []
    skipped_no_text = 0

    for idx, row in df.iterrows():
        try:
            # 只处理 is_seeker == True 的行（用户发言）
            # 与 v6 保持一致：if row.get('is_seeker') != True: continue
            if row.get('is_seeker') != True:
                continue

            user_text = extract_user_text_from_context(row.get('processed'))
            if not user_text:
                user_text = extract_user_text_from_context(row.get('raw'))

            if user_text:
                text_batch.append(user_text)
                text_indices.append(idx)
            else:
                skipped_no_text += 1

        except Exception as e:
            log_error('TextExtractor', f'行 {idx} 提取失败: {e}')
            continue

    log('TextExtractor', f'提取完成: {len(text_batch)} 条用户文本 '
                         f'(跳过 {skipped_no_text} 条无文本行)')
    return text_batch, text_indices


def save_text_batch(text_batch: list[str],
                    text_indices: list[int],
                    run_id: str = None) -> dict:
    """
    将文本批次保存到 intermediate 文件。

    保存两份：
      1. text_batch_info — 统计信息 + 前 20 条样本
      2. text_batch_full — 完整的 {idx→text} 映射（JSON）

    参数：
        run_id : 流水号，为 None 时自动生成

    返回：
        {'info': path, 'full': path}
    """
    files = {}

    # 统计信息
    if text_batch:
        avg_chars = int(np.mean([len(t) for t in text_batch]))
    else:
        avg_chars = 0

    info = {
        'total_count': len(text_batch),
        'empty_count': sum(1 for t in text_batch if not t.strip()),
        'avg_chars': avg_chars,
        'samples': [
            {'idx': text_indices[i], 'text': text_batch[i][:200]}
            for i in range(min(20, len(text_batch)))
        ],
    }
    path = write_json(info, 'text_batch_info', run_id=run_id, step='02')
    if path:
        files['info'] = path

    # 完整 text_batch（key: str(idx) → text），便于后续模块恢复
    full = {str(idx): text for idx, text in zip(text_indices, text_batch)}
    path = write_json(full, 'text_batch_full', run_id=run_id, step='02')
    if path:
        files['full'] = path

    return files


def run(df: pd.DataFrame, save_files: bool = True,
        run_id: str = None) -> dict:
    """
    统一入口：从 DataFrame 提取所有用户文本。

    参数：
        df         : 对话 DataFrame
        save_files : 是否保存 intermediate 文件
        run_id     : 流水号（同一轮流水线执行共享此 ID）

    返回：
        dict — {
            'run_id': str,
            'text_batch': list[str],
            'text_indices': list[int],
            'files': {...}  (save_files=True 时)
        }
    """
    if run_id is None:
        run_id = generate_run_id()

    log('TextExtractor', '=' * 40)
    log('TextExtractor', f'开始提取用户文本 (run_id={run_id})')

    text_batch, text_indices = build_text_batch(df)

    result = {
        'run_id': run_id,
        'text_batch': text_batch,
        'text_indices': text_indices,
    }

    if save_files:
        log('TextExtractor', '保存 intermediate 文件...')
        files = save_text_batch(text_batch, text_indices, run_id=run_id)
        result['files'] = files

    log('TextExtractor', f'文本提取完成 ({len(text_batch)} 条)')
    log('TextExtractor', '=' * 40)
    return result


def main():
    """
    独立运行入口。

    用法：python -m src.text_extractor
    说明：从 data_loader 加载 CSV，提取文本，使用新的 run_id 保存。
    """
    from src.data_loader import load_csv
    df = load_csv()
    result = run(df, save_files=True)
    print(f'\n提取完成: {len(result["text_batch"])} 条文本, run_id: {result["run_id"]}')
    if 'files' in result:
        for key, path in result['files'].items():
            print(f'  {key}: {path}')


if __name__ == '__main__':
    main()
