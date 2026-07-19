"""
fetch_movie_ids.py
=============
Phase 1: Extract all movie IDs from data/conv/data_all.csv, split by user vs system,
         and write to data/movie_id.csv.
Phase 2: Download movie info from TMDB for all new movie IDs, append to
         data/movie_info.json, record not-found IDs to data/movie_not_found.*.
         Supports resume if interrupted.

Usage:
    python fetch_movie_ids.py              # Run both phases from start
    python fetch_movie_ids.py --phase 1    # Phase 1 only (extract IDs)
    python fetch_movie_ids.py --phase 2    # Phase 2 only (download)
    python fetch_movie_ids.py --resume     # Resume Phase 2 from last checkpoint
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

# Ensure we can import from src/
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CONV_CSV = os.path.join(DATA_DIR, 'conv', 'data_all.csv')
MOVIE_ID_CSV = os.path.join(DATA_DIR, 'movie_id.csv')
MOVIE_INFO_JSON = os.path.join(DATA_DIR, 'movie_info.json')
NOT_FOUND_JSON = os.path.join(DATA_DIR, 'movie_not_found.json')
NOT_FOUND_TXT = os.path.join(DATA_DIR, 'movie_not_found.txt')
CHECKPOINT_FILE = os.path.join(DATA_DIR, '_fetch_checkpoint.txt')

# Progress save interval
SAVE_EVERY = 10


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1: Extract movie IDs from data_all.csv
# ═══════════════════════════════════════════════════════════════════════
def extract_movie_ids_from_csv(filepath: str) -> tuple[list, list]:
    """
    Parse the processed column of data_all.csv for ttXXXX patterns.
    Returns (user_ids, system_ids) where each is a list of (imdb_id, conv_id).
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
            ids_found = re.findall(r'tt\d+', processed)
            if not ids_found:
                continue
            is_seeker = row.get('is_seeker', '').strip() == 'True'
            target = user_ids if is_seeker else system_ids
            conv_id = row.get('conv_id', '')
            for tid in ids_found:
                target.append((tid, conv_id))

    log(f'Scanned {total} rows')
    log(f'  User questions:   {len(user_ids)} movie IDs (with duplicates)')
    log(f'  System replies:   {len(system_ids)} movie IDs (with duplicates)')

    # Unique count
    unique_user = len(set(tid for tid, _ in user_ids))
    unique_system = len(set(tid for tid, _ in system_ids))
    unique_all = len(set(tid for tid, _ in user_ids) | set(tid for tid, _ in system_ids))
    log(f'  Unique user IDs:  {unique_user}')
    log(f'  Unique system IDs:{unique_system}')
    log(f'  Unique total:     {unique_all}')

    return user_ids, system_ids


def write_movie_id_csv(user_ids: list, system_ids: list, output_path: str):
    """Write movie IDs to CSV with group column (append mode)."""
    file_exists = os.path.exists(output_path)
    mode = 'a' if file_exists else 'w'

    with open(output_path, mode, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['imdb_id', 'group', 'conv_id'])

        for tid, conv_id in user_ids:
            writer.writerow([tid, 'user', conv_id])
        for tid, conv_id in system_ids:
            writer.writerow([tid, 'system', conv_id])

    if file_exists:
        log(f'Appended {len(user_ids) + len(system_ids)} rows to {output_path}')
    else:
        log(f'Wrote {len(user_ids) + len(system_ids)} rows to {output_path}')


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: Download movie info from TMDB API
# ═══════════════════════════════════════════════════════════════════════

def load_existing_json(path: str) -> dict:
    """Load existing JSON file, return empty dict on failure."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            log(f'  Warning: could not read {path}, starting fresh')
    return {}


def save_json_atomic(data: dict, path: str):
    """Atomic JSON save: write to .tmp then replace."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def save_not_found_txt(not_found: dict, path: str):
    """Save the list of not-found IDs as a simple text file."""
    ids = sorted(not_found.keys())
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f' {len(ids)} IMDB IDs not found in TMDB:\n')
        for tid in ids:
            f.write(f'  {tid}\n')


def write_checkpoint(imdb_id: str):
    """Save resume checkpoint (last completed imdb_id)."""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        f.write(imdb_id + '\n')


def read_checkpoint() -> str | None:
    """Read resume checkpoint, return imdb_id or None."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            return line if line else None
    return None


def get_unique_movie_ids_from_csv() -> set[str]:
    """Read all unique movie IDs from movie_id.csv."""
    if not os.path.exists(MOVIE_ID_CSV):
        log('ERROR: movie_id.csv not found. Run Phase 1 first.')
        sys.exit(1)
    ids = set()
    with open(MOVIE_ID_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row['imdb_id'].strip())
    log(f'Loaded {len(ids)} unique movie IDs from {MOVIE_ID_CSV}')
    return ids


def download_movie_info(all_ids: set[str]):
    """
    Download movie info for all new IDs.
    Skips IDs already in movie_info.json or movie_not_found.json.
    Supports resume via checkpoint file.
    """
    # ── 1. Load existing data ──
    existing_info = load_existing_json(MOVIE_INFO_JSON)
    existing_not_found = load_existing_json(NOT_FOUND_JSON)

    already_processed = set(existing_info.keys()) | set(existing_not_found.keys())
    todo = sorted(all_ids - already_processed)

    log(f'Existing movie_info.json: {len(existing_info)} entries')
    log(f'Existing not_found:       {len(existing_not_found)} entries')
    log(f'New IDs to fetch:         {len(todo)}')

    if not todo:
        log('All movie IDs already processed. Nothing to do.')
        return

    # ── 2. Handle resume ──
    checkpoint_tid = read_checkpoint()
    start_idx = 0
    if checkpoint_tid:
        # Find where we left off
        for i, tid in enumerate(todo):
            if tid == checkpoint_tid:
                start_idx = i + 1
                log(f'Resuming from after {checkpoint_tid} (index {start_idx}/{len(todo)})')
                break
        else:
            log(f'Checkpoint ID {checkpoint_tid} not found in todo list, starting from beginning')

    # ── 3. Setup TMDB API from src/movie_info.py ──
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    from movie_info import fetch_movie_info, MAX_RETRIES, RETRY_DELAY, PROXIES

    if PROXIES:
        log(f'已读取系统代理: {list(PROXIES.values())[0]}')
    else:
        log('未检测到系统代理，将直连 TMDB')

    # ── 4. Download loop ──
    stats = {'ok': 0, 'fail': 0, 'skip': 0}
    last_save_count = len(existing_info) + len(existing_not_found)

    for idx in range(start_idx, len(todo)):
        imdb_id = todo[idx]
        # Skip if it got processed in a previous run (edge case from resume)
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
                log(f'  Retry {attempt}/{MAX_RETRIES}...')
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

        # Save checkpoint + periodic full save
        write_checkpoint(imdb_id)
        current_total = len(existing_info) + len(existing_not_found)
        if (idx + 1) % SAVE_EVERY == 0 and last_save_count < current_total:
            save_json_atomic(existing_info, MOVIE_INFO_JSON)
            save_json_atomic(existing_not_found, NOT_FOUND_JSON)
            save_not_found_txt(existing_not_found, NOT_FOUND_TXT)
            last_save_count = current_total
            log(f'  [Checkpoint saved at {imdb_id}]')

    # ── 5. Final save ──
    save_json_atomic(existing_info, MOVIE_INFO_JSON)
    save_json_atomic(existing_not_found, NOT_FOUND_JSON)
    save_not_found_txt(existing_not_found, NOT_FOUND_TXT)

    # Clean up checkpoint file on success
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    total_attempted = stats['ok'] + stats['fail']
    hit_rate = stats['ok'] / max(total_attempted, 1) * 100
    log('')
    log('=' * 55)
    log('Phase 2 complete!')
    log(f'  OK:  {stats["ok"]}')
    log(f'  Fail:{stats["fail"]}')
    log(f'  Skip:{stats["skip"]}')
    log(f'  Hit rate: {hit_rate:.1f}%')
    log(f'  Total movie_info.json: {len(existing_info)}')
    log(f'  Total not_found:       {len(existing_not_found)}')


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Extract movie IDs and fetch info from TMDB')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=None,
                        help='Run only Phase 1 (extract) or Phase 2 (download)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume Phase 2 from last checkpoint')
    args = parser.parse_args()

    phase = args.phase
    if args.resume:
        phase = 2

    # ── Phase 1: Extract ──
    if phase is None or phase == 1:
        log('=' * 55)
        log('Phase 1: Extracting movie IDs from data_all.csv')
        log('=' * 55)
        log(f'Source: {CONV_CSV}')
        user_ids, system_ids = extract_movie_ids_from_csv(CONV_CSV)
        write_movie_id_csv(user_ids, system_ids, MOVIE_ID_CSV)
        log('Phase 1 complete!\n')

    # ── Phase 2: Download ──
    if phase is None or phase == 2:
        log('=' * 55)
        log('Phase 2: Downloading movie info from TMDB')
        log('=' * 55)
        log(f'Source IDs: {MOVIE_ID_CSV}')

        # Validate src/movie_info.py has TMDB_API_KEY set
        try:
            from src.movie_info import TMDB_API_KEY
            if not TMDB_API_KEY:
                log('ERROR: TMDB_API_KEY not found in src/movie_info.py')
                sys.exit(1)
        except ImportError:
            log('ERROR: Cannot import src/movie_info.py')
            sys.exit(1)

        all_ids = get_unique_movie_ids_from_csv()
        download_movie_info(all_ids)
        log('Phase 2 complete!')

    log('\nAll done!')


if __name__ == '__main__':
    main()
