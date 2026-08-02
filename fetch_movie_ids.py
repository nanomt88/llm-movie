"""
fetch_movie_ids.py
==================
Phase 1: 从 data/conv/data_all.csv 提取用户和系统对话中提到的电影 ID，
         去重后与 data/movie_id.csv 对比，将不存在的 ID 写入临时文件，
         再追加到 movie_id.csv。
Phase 2: 逐个通过 IMDb ID 从 TMDB 获取电影信息，写入 data/movie_info.json。
         未找到的记录到 data/movie_not_found.json。
         支持代理、任务中断后继续运行。

Usage:
    python fetch_movie_ids.py              # 运行两个阶段
    python fetch_movie_ids.py --phase 1    # 仅 Phase 1（提取 ID）
    python fetch_movie_ids.py --phase 2    # 仅 Phase 2（下载信息）
    python fetch_movie_ids.py --resume     # 从上次中断处继续 Phase 2
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

# ── 控制台编码（Windows GBK 环境修正）──────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.upper() in ("GBK", "GB2312"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── 路径 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CONV_CSV = os.path.join(DATA_DIR, 'conv', 'data_all.csv')
MOVIE_ID_CSV = os.path.join(DATA_DIR, 'movie_id.csv')
MOVIE_INFO_JSON = os.path.join(DATA_DIR, 'movie_info.json')
NOT_FOUND_JSON = os.path.join(DATA_DIR, 'movie_not_found.json')
NOT_FOUND_TXT = os.path.join(DATA_DIR, 'movie_not_found.txt')
TEMP_NEW_IDS = os.path.join(DATA_DIR, '_new_movie_ids.txt')
CHECKPOINT_FILE = os.path.join(DATA_DIR, '_fetch_checkpoint.txt')

# 每 N 个 ID 保存一次进度
SAVE_EVERY = 10

# ── 导入 TMDB 配置（复用 his/src/movie_info.py）─────────────────────────
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'his', 'src'))
try:
    from movie_info import (
        fetch_movie_info,
        MAX_RETRIES,
        RETRY_DELAY,
        PROXIES,
        TMDB_API_KEY,
    )
except ImportError as e:
    print(f'ERROR: 无法导入 his/src/movie_info.py: {e}', file=sys.stderr)
    print('请确保 requests 库已安装且 his/src/movie_info.py 存在。', file=sys.stderr)
    sys.exit(1)

# 重试次数改为 1 次（只调用一次 API，不重试）
MAX_RETRIES = 1


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def is_valid_imdb_id(tid: str) -> bool:
    """检查是否为有效的 IMDb ID（tt + 7~9 位数字）。"""
    return bool(re.match(r'^tt\d{7,9}$', tid))


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1: 从 data_all.csv 提取电影 ID，对比 movie_id.csv，追加新 ID
# ═══════════════════════════════════════════════════════════════════════
def extract_movie_ids_from_csv(filepath: str) -> tuple[list, list]:
    """
    Parse the processed column of data_all.csv for ttXXXX patterns.
    Returns (user_ids, system_ids) where each is a list of (imdb_id, conv_id).
    从 data_all.csv 的 processed 列提取 ttXXXX 格式的电影 ID。
    返回 (user_ids, system_ids)，每个是 (imdb_id, conv_id) 的列表。
    """
    user_ids = []
    system_ids = []
    total = 0

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            processed = row.get('processed', '')
            if not processed:
                continue
            ids_found = re.findall(r'tt\d{7,9}', processed)
            if not ids_found:
                continue
            is_seeker = row.get('is_seeker', '').strip() == 'True'
            target = user_ids if is_seeker else system_ids
            conv_id = row.get('conv_id', '')
            for tid in ids_found:
                target.append((tid, conv_id))

    log(f'扫描了 {total} 行')
    log(f'  用户提问中的 ID（含重复）: {len(user_ids)}')
    log(f'  系统回复中的 ID（含重复）: {len(system_ids)}')

    unique_user = len(set(tid for tid, _ in user_ids))
    unique_system = len(set(tid for tid, _ in system_ids))
    unique_all = len(set(tid for tid, _ in user_ids) | set(tid for tid, _ in system_ids))
    log(f'  去重后用户 ID:  {unique_user}')
    log(f'  去重后系统 ID:  {unique_system}')
    log(f'  去重后总计:     {unique_all}')

    return user_ids, system_ids


def get_existing_ids_from_movie_csv() -> set[str]:
    """
    快速读取 movie_id.csv 中已有的 imdb_id（去重）。
    使用行分割而非 csv.DictReader，适合 1500 万行大文件。
    """
    if not os.path.exists(MOVIE_ID_CSV):
        return set()
    ids = set()
    with open(MOVIE_ID_CSV, 'r', encoding='utf-8') as f:
        f.readline()  # 跳过表头
        for line in f:
            # imdb_id 是第一列，格式为 ttXXXXXXX，不含逗号
            tid = line.split(',', 1)[0].strip().strip('"')
            if tid:
                ids.add(tid)
    log(f'从 movie_id.csv 加载了 {len(ids)} 个唯一 ID')
    return ids


def get_unique_movie_ids_from_csv() -> set[str]:
    """读取 movie_id.csv 中所有唯一的电影 ID（Phase 2 独立运行时使用）。"""
    if not os.path.exists(MOVIE_ID_CSV):
        log('ERROR: movie_id.csv 不存在，请先运行 Phase 1。')
        sys.exit(1)
    return get_existing_ids_from_movie_csv()


def write_temp_file(new_ids: set[str], path: str):
    """将新 ID 写入临时文件（按字典序排序）。"""
    sorted_ids = sorted(new_ids)
    with open(path, 'w', encoding='utf-8') as f:
        for tid in sorted_ids:
            f.write(tid + '\n')
    log(f'将 {len(sorted_ids)} 个新 ID 写入 {path}')


def append_to_movie_csv(new_entries: list, path: str):
    """追加 (imdb_id, group, conv_id) 行到 movie_id.csv。"""
    file_exists = os.path.exists(path)
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['imdb_id', 'group', 'conv_id'])
        for tid, group, conv_id in new_entries:
            writer.writerow([tid, group, conv_id])
    log(f'追加了 {len(new_entries)} 行到 {path}')


def run_phase1() -> set[str]:
    """
    Phase 1: 提取 ID、对比、写临时文件、追加到 movie_id.csv。
    返回所有唯一 ID（data_all.csv 提取的 ∪ movie_id.csv 已有的），供 Phase 2 使用。
    """
    log('=' * 55)
    log('Phase 1: 从 data_all.csv 提取电影 ID')
    log('=' * 55)
    log(f'源文件: {CONV_CSV}')

    # 1. 从 data_all.csv 提取所有 ID
    user_ids, system_ids = extract_movie_ids_from_csv(CONV_CSV)

    # 2. 去重
    all_unique_ids = set(tid for tid, _ in user_ids) | set(tid for tid, _ in system_ids)

    # 3. 读取 movie_id.csv 中已有的 ID
    existing_ids = get_existing_ids_from_movie_csv()

    # 4. 找出不存在于 movie_id.csv 的新 ID
    new_ids = all_unique_ids - existing_ids
    log(f'  不在 movie_id.csv 中的新 ID: {len(new_ids)}')

    if not new_ids:
        log('没有新 ID，Phase 1 完成。')
        return all_unique_ids | existing_ids

    # 5. 写入临时文件
    write_temp_file(new_ids, TEMP_NEW_IDS)

    # 6. 追加新条目到 movie_id.csv
    new_entries = [(tid, 'user', conv_id) for tid, conv_id in user_ids if tid in new_ids]
    new_entries += [(tid, 'system', conv_id) for tid, conv_id in system_ids if tid in new_ids]
    append_to_movie_csv(new_entries, MOVIE_ID_CSV)

    log('Phase 1 完成!')
    return all_unique_ids | existing_ids


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: 从 TMDB 获取电影信息（通过 IMDb ID 查找）
# ═══════════════════════════════════════════════════════════════════════
def load_existing_json(path: str) -> dict:
    """加载已存在的 JSON 文件，失败时返回空 dict。"""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            log(f'  警告: 无法读取 {path}，从头开始')
    return {}


def save_json_atomic(data: dict, path: str):
    """原子保存 JSON：先写 .tmp 再替换。
    Windows 上可能因文件锁导致 os.replace 失败，自动重试。"""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt < 4:
                time.sleep(0.5)
            else:
                log(f'  警告: os.replace 失败，回退为直接写入 {path}')
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                try:
                    os.remove(tmp)
                except OSError:
                    pass


def save_not_found_txt(not_found: dict, path: str):
    """将未找到的 ID 列表保存为文本文件。"""
    ids = sorted(not_found.keys())
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'{len(ids)} 个 IMDb ID 在 TMDB 中未找到:\n')
        for tid in ids:
            f.write(f'  {tid}\n')


def write_checkpoint(imdb_id: str):
    """保存恢复检查点（最后完成的 imdb_id）。"""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        f.write(imdb_id + '\n')


def read_checkpoint() -> str | None:
    """读取恢复检查点，返回 imdb_id 或 None。"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            return line if line else None
    return None


def download_movie_info(all_ids: set[str], retry_not_found: bool = False):
    """
    逐个通过 IMDb ID 从 TMDB 获取电影信息。
    跳过已存在于 movie_info.json 或 movie_not_found.json 的 ID。
    支持通过检查点文件恢复。
    retry_not_found=True 时，重新获取之前未找到的 ID。
    """
    # ── 1. 加载已有数据 ──
    existing_info = load_existing_json(MOVIE_INFO_JSON)
    existing_not_found = load_existing_json(NOT_FOUND_JSON)

    if retry_not_found:
        # 重试之前未找到的 ID（仅有效的 7-9 位）
        all_retry = set(existing_not_found.keys())
        retry_ids = {tid for tid in all_retry if is_valid_imdb_id(tid)}
        skipped_invalid = len(all_retry) - len(retry_ids)
        # 从 not_found 中移除待重试的 ID，以便重新评估
        for tid in retry_ids:
            del existing_not_found[tid]
        todo = sorted(retry_ids)

        log(f'已有 movie_info.json: {len(existing_info)} 条')
        log(f'已有 not_found:       {len(all_retry)} 条')
        log(f'重试有效 ID:         {len(todo)} 个')
        if skipped_invalid:
            log(f'跳过无效 ID:         {skipped_invalid} 个（非 7-9 位数字）')
    else:
        already_processed = set(existing_info.keys()) | set(existing_not_found.keys())
        todo = sorted(
            tid for tid in (all_ids - already_processed) if is_valid_imdb_id(tid)
        )

        log(f'已有 movie_info.json: {len(existing_info)} 条')
        log(f'已有 not_found:       {len(existing_not_found)} 条')
        log(f'需要获取的新 ID:       {len(todo)}')

    if not todo:
        log('所有电影 ID 已处理完毕，无需操作。')
        return

    # ── 2. 处理恢复 ──
    checkpoint_tid = read_checkpoint()
    start_idx = 0
    if checkpoint_tid:
        for i, tid in enumerate(todo):
            if tid == checkpoint_tid:
                start_idx = i + 1
                log(f'从 {checkpoint_tid} 之后恢复（索引 {start_idx}/{len(todo)}）')
                break
        else:
            log(f'检查点 ID {checkpoint_tid} 不在待处理列表中，从头开始')

    # ── 3. 代理信息 ──
    if PROXIES:
        log(f'已读取系统代理: {list(PROXIES.values())[0]}')
    else:
        log('未检测到系统代理，将直连 TMDB')

    # ── 4. 下载循环 ──
    stats = {'ok': 0, 'fail': 0, 'skip': 0}
    last_save_count = len(existing_info) + len(existing_not_found)

    for idx in range(start_idx, len(todo)):
        imdb_id = todo[idx]
        # 跳过已处理的（恢复时的边缘情况）
        if imdb_id in existing_info or imdb_id in existing_not_found:
            stats['skip'] += 1
            continue

        log(f'[{idx + 1}/{len(todo)}] {imdb_id}')

        info = None
        for attempt in range(1, MAX_RETRIES + 1):
            info = fetch_movie_info(imdb_id)
            if info is not None:
                break
            if attempt < MAX_RETRIES:
                log(f'  重试 {attempt}/{MAX_RETRIES}...')
                time.sleep(RETRY_DELAY * attempt)

        if info:
            existing_info[imdb_id] = info
            stats['ok'] += 1
            genre_str = ' / '.join(info.get('genres', [])) or 'N/A'
            log(f'  OK {info["title"]} [{genre_str}]')
        else:
            placeholder = {'imdb_id': imdb_id, 'title': ''}
            existing_not_found[imdb_id] = placeholder
            stats['fail'] += 1
            log(f'  NOT FOUND')

        # 保存检查点 + 定期保存
        write_checkpoint(imdb_id)
        current_total = len(existing_info) + len(existing_not_found)
        if (idx + 1) % SAVE_EVERY == 0 and last_save_count < current_total:
            save_json_atomic(existing_info, MOVIE_INFO_JSON)
            save_json_atomic(existing_not_found, NOT_FOUND_JSON)
            save_not_found_txt(existing_not_found, NOT_FOUND_TXT)
            last_save_count = current_total
            log(f'  [检查点已保存 @ {imdb_id}]')

    # ── 5. 最终保存 ──
    save_json_atomic(existing_info, MOVIE_INFO_JSON)
    save_json_atomic(existing_not_found, NOT_FOUND_JSON)
    save_not_found_txt(existing_not_found, NOT_FOUND_TXT)

    # 成功完成后清理检查点文件
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    total_attempted = stats['ok'] + stats['fail']
    hit_rate = stats['ok'] / max(total_attempted, 1) * 100
    log('')
    log('=' * 55)
    log('Phase 2 完成!')
    log(f'  成功:  {stats["ok"]}')
    log(f'  失败:  {stats["fail"]}')
    log(f'  跳过:  {stats["skip"]}')
    log(f'  命中率: {hit_rate:.1f}%')
    log(f'  movie_info.json 总计: {len(existing_info)}')
    log(f'  not_found 总计:       {len(existing_not_found)}')


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='提取电影 ID 并从 TMDB 获取电影信息'
    )
    parser.add_argument('--phase', type=int, choices=[1, 2], default=None,
                        help='仅运行 Phase 1（提取）或 Phase 2（下载）')
    parser.add_argument('--resume', action='store_true',
                        help='从上次检查点继续 Phase 2')
    parser.add_argument('--retry-not-found', action='store_true',
                        help='重新获取之前未找到的电影信息')
    args = parser.parse_args()

    phase = args.phase
    if args.resume or args.retry_not_found:
        phase = 2

    # ── Phase 1: 提取 ──
    if phase is None or phase == 1:
        all_ids = run_phase1()
        log('')

    # ── Phase 2: 下载 ──
    if phase is None or phase == 2:
        log('=' * 55)
        if args.retry_not_found:
            log('Phase 2: 重新获取未找到的电影信息')
        else:
            log('Phase 2: 从 TMDB 获取电影信息')
        log('=' * 55)
        log(f'ID 来源: {MOVIE_ID_CSV}')

        if not TMDB_API_KEY:
            log('ERROR: TMDB_API_KEY 未在 his/src/movie_info.py 中设置')
            sys.exit(1)

        # 如果只运行 Phase 2，从 movie_id.csv 读取所有唯一 ID
        # retry-not-found 模式下不需要读取 movie_id.csv
        all_ids = set()
        if (phase == 2 or args.resume) and not args.retry_not_found:
            all_ids = get_unique_movie_ids_from_csv()

        download_movie_info(all_ids, retry_not_found=args.retry_not_found)
        log('Phase 2 完成!')

    log('\n全部完成!')


if __name__ == '__main__':
    main()
