# -*- coding: utf-8 -*-
"""
Module 1: Data Loader
Loads all data sources, cleans text, labels holiday/non-holiday, validates.
Can run standalone to verify data integrity.
"""

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone

from his.data_analyzer.config import (
    HOLIDAY_CSV, HOLIDAY_CONV_CSV, FULL_YEAR_CSV, MOVIE_INFO_PATH,
    log,
)


# ── 1. Holiday definitions ────────────────────────────────────────────
def load_holiday_definitions() -> dict[str, dict]:
    """
    Load holiday.csv, return dict: {date_str: {description, type}}.
    Supports all years present in the file.
    """
    holidays = {}
    with open(HOLIDAY_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row['date'].strip()
            holidays[d] = {
                'description': row['description'].strip(),
                'type': row['type'].strip(),
            }
    # Merge duplicate dates (e.g., 2022-06-19 = Juneteenth & Father's Day)
    merged = {}
    for d, info in holidays.items():
        if d in merged:
            merged[d]['description'] = f"{merged[d]['description']}&{info['description']}"
        else:
            merged[d] = dict(info)

    years = sorted(set(d[:4] for d in merged))
    log(f"Loaded {len(merged)} holiday dates across years: {', '.join(years)}")
    return merged


# ── 2. Conversation data ──────────────────────────────────────────────
def parse_processed_text(row: dict) -> str:
    """
    Extract the user's actual text from the 'raw' field.
    raw is stored as: "['USER', 'text here']"  (Python repr of a list)
    We parse it safely.
    """
    raw = row.get('raw', '')
    if not raw:
        return ''

    # The field looks like "['USER', 'actual message text']"
    # Use regex to extract the message part
    m = re.search(r"\[\s*'USER'\s*,\s*'(.*)'\s*\]", raw, re.DOTALL)
    if m:
        return m.group(1)
    # Fallback: try to remove the ['USER',  prefix and trailing ]
    cleaned = re.sub(r"^\s*\[\s*'USER'\s*,\s*", '', raw)
    cleaned = re.sub(r"\s*\]\s*$", '', cleaned)
    # Remove surrounding quotes
    cleaned = cleaned.strip().strip("'\"")
    return cleaned


def parse_processed_messages(row: dict) -> str:
    """
    Extract text from 'processed' field (same format as raw but with tt IDs).
    """
    processed = row.get('processed', '')
    if not processed:
        return ''
    m = re.search(r"\[\s*'(?:USER|SYSTEM)'\s*,\s*'(.*)'\s*\]", processed, re.DOTALL)
    if m:
        return m.group(1)
    cleaned = re.sub(r"^\s*\[\s*'(?:USER|SYSTEM)'\s*,\s*", '', processed)
    cleaned = re.sub(r"\s*\]\s*$", '', cleaned)
    return cleaned.strip().strip("'\"")


def clean_word_count(text: str) -> int:
    """
    Count English words after stripping non-alpha characters.
    Removes digits, punctuation, extra spaces.
    """
    if not text:
        return 0
    # Keep only letters and spaces
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        return 0
    return len(cleaned.split())


def extract_imdb_ids(text: str) -> list[str]:
    """Extract all tt... IDs from text."""
    if not text:
        return []
    return re.findall(r'tt\d+', text)


def load_conversations(filepath: str) -> list[dict]:
    """
    Load a conversation CSV, parse raw/processed, compute word counts.
    Returns list of dicts with parsed fields.
    """
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'user_id': row.get('user_id', ''),
                'turn_order': int(row.get('turn_order', 0)),
                'is_seeker': row.get('is_seeker', '').strip() == 'True',
                'utc_time': int(row.get('utc_time', 0)),
                'upvotes': float(row.get('upvotes', 0) or 0),
                'raw_text': parse_processed_text(row),
                'proc_text': parse_processed_messages(row),
                'processed_raw': row.get('processed', ''),
            }
            # Word count from raw user text
            parsed['word_count'] = clean_word_count(parsed['raw_text'])
            # IMDB IDs from processed
            parsed['imdb_ids'] = extract_imdb_ids(row.get('processed', ''))
            # Has movie mention
            parsed['has_movie'] = len(parsed['imdb_ids']) > 0
            rows.append(parsed)
    log(f"Loaded {len(rows)} rows from {filepath}")
    return rows


# ── 3. Holiday tagging ────────────────────────────────────────────────
def tag_holiday(rows: list[dict], holiday_map: dict) -> list[dict]:
    """
    Add 'date', 'is_holiday', 'holiday_name', 'holiday_type' fields to each row
    based on the utc_time.
    """
    tagged = []
    holiday_date_set = set(holiday_map.keys())
    stats = {'holiday': 0, 'non_holiday': 0}

    for row in rows:
        dt = datetime.fromtimestamp(row['utc_time'], tz=timezone.utc)
        date_str = dt.strftime('%Y-%m-%d')
        row['date'] = date_str
        row['hour'] = dt.hour
        row['weekday'] = dt.weekday()  # 0=Mon, 6=Sun

        if date_str in holiday_date_set:
            row['is_holiday'] = True
            row['holiday_name'] = holiday_map[date_str]['description']
            row['holiday_type'] = holiday_map[date_str]['type']
            stats['holiday'] += 1
        else:
            row['is_holiday'] = False
            row['holiday_name'] = ''
            row['holiday_type'] = ''
            stats['non_holiday'] += 1

        tagged.append(row)

    log(f"Tagged: {stats['holiday']} holiday rows, {stats['non_holiday']} non-holiday rows")
    return tagged


# ── 4. Movie info ─────────────────────────────────────────────────────
def load_movie_info() -> dict:
    """Load movie_info.json into a dict keyed by IMDB ID."""
    with open(MOVIE_INFO_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    log(f"Loaded {len(data)} movies from movie_info.json")
    return data


def lookup_genres(imdb_ids: list[str], movie_info: dict) -> list[str]:
    """Look up genre names for a list of IMDB IDs. Returns deduplicated list."""
    genres = set()
    for tid in imdb_ids:
        info = movie_info.get(tid)
        if info and 'genres' in info:
            genres.update(info['genres'])
    return sorted(genres)


def lookup_genre_counts(imdb_ids: list[str], movie_info: dict) -> dict[str, int]:
    """Count genre occurrences for a list of IMDB IDs."""
    counts = defaultdict(int)
    for tid in imdb_ids:
        info = movie_info.get(tid)
        if info and 'genres' in info:
            for g in info['genres']:
                counts[g] += 1
    return dict(counts)


# ── 5. Validation utility ─────────────────────────────────────────────
def validate_data(rows: list[dict]):
    """Print data validation summary."""
    total = len(rows)
    seekers = sum(1 for r in rows if r['is_seeker'])
    non_seekers = total - seekers
    unique_conv = len(set(r['conv_id'].rsplit('/', 1)[0] for r in rows))
    with_movie = sum(1 for r in rows if r['is_seeker'] and r['has_movie'])

    log(f"=== Data Validation ===")
    log(f"Total rows: {total}")
    log(f"  User questions: {seekers}")
    log(f"  System replies: {non_seekers}")
    log(f"  Unique conversations: {unique_conv}")
    log(f"  User Q with movie mentions: {with_movie} ({with_movie/max(seekers,1)*100:.1f}%)")

    # Date range
    dates = sorted(set(r['date'] for r in rows))
    log(f"Date range: {dates[0]} ~ {dates[-1]} ({len(dates)} days)")

    # Holidays coverage
    holiday_dates = sorted(set(r['date'] for r in rows if r['is_holiday']))
    log(f"Holiday dates with data: {len(holiday_dates)}")
    for d in holiday_dates:
        cnt = sum(1 for r in rows if r['date'] == d and r['is_seeker'])
        log(f"  {d}: {cnt} user questions")


# ── 6. Main (standalone run) ─────────────────────────────────────────
def main():
    """Standalone: load + tag + validate + export summary."""
    log("=== Step 1: Data Preparation ===", "DataLoader")

    # 6a: Holiday definitions
    holiday_map = load_holiday_definitions()

    # 6b: Full year data
    log("Loading full year data...")
    full_rows = load_conversations(FULL_YEAR_CSV)
    tagged_full = tag_holiday(full_rows, holiday_map)
    validate_data(tagged_full)

    # 6c: Holiday conversation data
    log("\nLoading holiday conversation data...")
    holi_rows = load_conversations(HOLIDAY_CONV_CSV)
    tagged_holi = tag_holiday(holi_rows, holiday_map)
    validate_data(tagged_holi)

    # 6d: Movie info
    movie_info = load_movie_info()
    log(f"Movie info keys (sample): {list(movie_info.keys())[:3]}")

    # 6e: Per-holiday summary (user questions only)
    log("\n=== Per-Holiday Question Count ===")
    seekers = [r for r in tagged_full if r['is_seeker']]
    holiday_counts = defaultdict(lambda: {'total': 0, 'with_movie': 0, 'avg_words': 0})
    word_sums = defaultdict(int)
    for r in seekers:
        if r['is_holiday']:
            name = r['holiday_name']
            holiday_counts[name]['total'] += 1
            word_sums[name] += r['word_count']
            if r['has_movie']:
                holiday_counts[name]['with_movie'] += 1

    for name, info in sorted(holiday_counts.items(), key=lambda x: -x[1]['total']):
        avg_w = word_sums[name] / max(info['total'], 1)
        log(f"  {name}: {info['total']} Q, {info['with_movie']} with movie, avg {avg_w:.1f} words")

    log("\nData preparation complete!")


if __name__ == '__main__':
    main()
