"""Test Douban API scraping"""
import requests
import json
import re
import sys

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://movie.douban.com/',
}

test_ids = [
    'tt0110148',   # Interview with the Vampire
    'tt0119675',   # Mimic
    'tt0109424',   # Chungking Express
    'tt9999999',   # Non-existent
]

for imdb_id in test_ids:
    print(f'\n=== {imdb_id} ===')
    try:
        r = requests.get(
            f'https://search.douban.com/movie/subject_search?search_text={imdb_id}',
            headers=HEADERS,
            timeout=15
        )
        print(f'Status: {r.status_code}')

        match = re.search(r'window\.__DATA__\s*=\s*({.*?});', r.text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            items = data.get('items', [])
            if items:
                item = items[0]
                print(f"  Douban ID: {item.get('id')}")
                print(f"  Title: {item.get('title')}")
                print(f"  Abstract: {item.get('abstract')}")
                print(f"  Abstract2: {item.get('abstract_2')}")
                print(f"  Rating: {item.get('rating')}")
            else:
                print('  No items found')
        else:
            print('  No __DATA__ found')
            if len(r.text) < 500:
                print(f'  Body: {r.text[:300]}')
    except Exception as e:
        print(f'  Error: {e}')
