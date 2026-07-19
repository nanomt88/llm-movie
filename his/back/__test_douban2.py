"""Test Douban data format more thoroughly"""
import requests
import json
import re
import sys

# Force UTF-8 for stdout
sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

test_ids = [
    'tt0110148',   # Interview with the Vampire
    'tt0109424',   # Chungking Express
    'tt0119675',   # Mimic
    'tt0082158',   # Chariots of Fire
    'tt0034862',   # Holiday Inn
    'tt1040023',   # From the CSV
]

for imdb_id in test_ids:
    print(f'\n=== {imdb_id} ===')
    r = requests.get(
        f'https://search.douban.com/movie/subject_search?search_text={imdb_id}',
        headers=HEADERS, timeout=15
    )
    match = re.search(r'window\.__DATA__\s*=\s*({.*?});', r.text, re.DOTALL)
    if match:
        data = json.loads(match.group(1))
        items = data.get('items', [])
        if items:
            item = items[0]
            print(f"  ID: {item.get('id')}")
            print(f"  Title: {repr(item.get('title'))}")
            print(f"  Abstract: {repr(item.get('abstract'))}")
            print(f"  Abstract2: {repr(item.get('abstract_2'))}")
            print(f"  Labels: {item.get('labels')}")
            print(f"  Rating: {item.get('rating')}")
        else:
            print('  No items')
    else:
        print('  No data')
