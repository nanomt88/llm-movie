"""Test Douban one at a time with more debugging"""
import requests
import json
import re
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

test_ids = [
    'tt0109424',   # Chungking Express - single test
]

for imdb_id in test_ids:
    print(f'\n=== {imdb_id} ===')
    time.sleep(2)  # Be polite
    r = requests.get(
        f'https://search.douban.com/movie/subject_search?search_text={imdb_id}',
        headers=HEADERS, timeout=15
    )
    print(f'  Status: {r.status_code}')
    print(f'  URL: {r.url}')
    print(f'  Length: {len(r.text)}')
    
    match = re.search(r'window\.__DATA__\s*=\s*({.*?});', r.text, re.DOTALL)
    if match:
        data = json.loads(match.group(1))
        items = data.get('items', [])
        print(f'  Total items: {data.get("total")}')
        print(f'  Items count: {len(items)}')
        if items:
            item = items[0]
            print(f"  ID: {item.get('id')}")
            print(f"  Title: {repr(item.get('title'))}")
            print(f"  Abstract: {repr(item.get('abstract'))}")
            print(f"  Abstract2: {repr(item.get('abstract_2'))}")
        else:
            print(f'  Text query: {data.get("text")}')
            # Check if we got redirect HTML
            if 'subject' in r.url or 'subject' in r.text[:200]:
                print('  -> Might have redirect to subject page')
    else:
        print('  No __DATA__ found')
        # Check for subject page redirect
        print(f'  URL after redirect: {r.url}')
        if '/subject/' in r.url:
            print(f'  -> Redirected to subject page: {r.url}')
        elif len(r.text) < 200:
            print(f'  Body: {r.text[:200]}')
