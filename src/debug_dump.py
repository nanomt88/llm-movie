# -*- coding: utf-8 -*-
"""
调试导出模块 — 将中间变量序列化为 JSON 文件，便于排查问题。

写入目录：output/intermediate/
文件名格式：{run_id}_S{step:02d}_{name}.json

设计说明：
  - run_id（流水号）：同一轮流水线执行生成的所有文件共享一个 run_id，
    方便识别哪些文件属于同一组。
  - step（步骤编号）：S01~S05 对应流水线的 5 个步骤。
"""

import glob
import json
import os
import re
from datetime import datetime

from src.config import INTERMEDIATE_DIR, generate_run_id, log, log_error


def _sizeof_fmt(num: int) -> str:
    """字节 → 人类可读格式"""
    for unit in ('B', 'KB', 'MB'):
        if num < 1024:
            return f'{num:.1f} {unit}'
        num /= 1024
    return f'{num:.1f} GB'


def make_filename(run_id: str, step: str, name: str, ext: str = '.json') -> str:
    """
    按统一规则生成中间文件名。

    格式：{run_id}_S{step}_{name}.json
    示例：20260606_143022_S01_user_seg.json
    """
    return f'{run_id}_S{step}_{name}{ext}'


def write_json(data, name: str, run_id: str = None, step: str = '00',
               subdir: str = None) -> str | None:
    """
    将 data 序列化为 JSON 写入 output/intermediate 目录。

    参数：
        data   : 可 JSON 序列化的 Python 对象
        name   : 文件名（不含 .json）
        run_id : 流水号，为 None 时自动生成
        step   : 步骤编号（如 '01', '02'），默认 '00'
        subdir : 子目录名（如 'intermediate'），默认写入 INTERMEDIATE_DIR

    返回：
        完整路径，失败返回 None
    """
    if run_id is None:
        run_id = generate_run_id()

    filename = make_filename(run_id, step, name)

    if subdir == 'intermediate' or subdir is None:
        target_dir = INTERMEDIATE_DIR
    elif subdir:
        target_dir = os.path.join(os.path.dirname(INTERMEDIATE_DIR), subdir)
        os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = os.path.dirname(INTERMEDIATE_DIR)

    out_path = os.path.join(target_dir, filename)

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        size_str = _sizeof_fmt(os.path.getsize(out_path))
        log('DebugDump', f'{filename} ({size_str})')
        return out_path
    except Exception as e:
        log_error('DebugDump', f'{filename} 写入失败: {e}')
        return None


def write_json_lines(records: list, name: str, run_id: str = None,
                     step: str = '00') -> str | None:
    """
    将记录列表写出为 JSON Lines（每行一个 JSON 对象），适合大批量数据。

    参数：
        records : 字典列表
        name    : 文件名（不含 .jsonl）
        run_id  : 流水号，为 None 时自动生成
        step    : 步骤编号（如 '01', '02'），默认 '00'

    返回：
        完整路径，失败返回 None
    """
    if run_id is None:
        run_id = generate_run_id()

    filename = make_filename(run_id, step, name, ext='.jsonl')
    out_path = os.path.join(INTERMEDIATE_DIR, filename)

    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        size_str = _sizeof_fmt(os.path.getsize(out_path))
        log('DebugDump', f'{filename} ({size_str}, {len(records)} 行)')
        return out_path
    except Exception as e:
        log_error('DebugDump', f'{filename} 写入失败: {e}')
        return None


def read_json(path: str):
    """读取 JSON 文件，失败返回 None"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log_error('DebugDump', f'读取失败 {path}: {e}')
        return None


def extract_run_id(filepath: str) -> str | None:
    """
    从文件名中提取 run_id。

    文件名格式：{run_id}_S{step}_{name}.json
    例如 20260606_143022_S01_user_seg.json → '20260606_143022'
    """
    basename = os.path.basename(filepath)
    m = re.match(r'^(.+?)_S\d{2}_', basename)
    return m.group(1) if m else None


def find_latest_by_step(step: str, name_suffix: str = None,
                        extension: str = '.json') -> str | None:
    """
    按步骤编号查找最新的中间文件。

    参数：
        step         : 步骤编号（如 '01', '02'）
        name_suffix  : 文件名后缀匹配（如 'text_batch_full'），为 None 时匹配该步骤所有文件
        extension    : 文件扩展名（默认 '.json'）

    返回：
        最新文件的完整路径，未找到返回 None
    """
    if name_suffix:
        pattern = f'*_S{step}_*{name_suffix}*{extension}'
    else:
        pattern = f'*_S{step}_*{extension}'

    abs_pattern = os.path.join(INTERMEDIATE_DIR, pattern)
    files = sorted(glob.glob(abs_pattern))
    if not files:
        log_error('DebugDump', f'未找到匹配文件 (S{step}, {name_suffix}): {pattern}')
        return None
    return files[-1]


def load_latest_by_step(step: str, name_suffix: str = None,
                        extension: str = '.json'):
    """
    按步骤编号加载最新的中间文件内容。

    返回：
        (parsed_data, file_path) 或 (None, None)
    """
    filepath = find_latest_by_step(step, name_suffix, extension)
    if filepath is None:
        return None, None
    data = read_json(filepath)
    if data is None:
        return None, None
    return data, filepath


def load_latest_records():
    """
    找最新的 S04 records_full.jsonl 并加载全部记录。

    返回：
        list[dict] | None
    """
    pattern = os.path.join(INTERMEDIATE_DIR, '*_S04_*records_full.jsonl')
    files = sorted(glob.glob(pattern))
    if not files:
        log_error('DebugDump', '未找到 records_full 文件')
        return None
    records = []
    latest = files[-1]
    try:
        with open(latest, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        log('DebugDump', f'从 {os.path.basename(latest)} 加载 {len(records)} 条记录')
        return records
    except Exception as e:
        log_error('DebugDump', f'加载记录失败: {e}')
        return None


if __name__ == '__main__':
    records = load_latest_records()
    print(records)
