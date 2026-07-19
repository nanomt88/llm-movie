"""Test Douban search by movie name vs IMDb ID"""
import requests
import json
import re
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# Test cases: (IMDb ID, movie name)
cases = [
    ('tt0109424', 'Chungking Express'),
    ('tt0110148', 'Interview with the Vampire'),
    ('tt0119675', 'Mimic'),
    ('tt0082158', 'Chariots of Fire'),
    ('tt0034862', 'Holiday Inn'),
    ('tt0113537', 'Kicking and Screaming'),
]

for imdb_id, name in cases:
    print(f'\n=== {name} ({imdb_id}) ===')

    # Method 1: Search by IMDb ID
    r1 = requests.get(
        f'https://search.douban.com/movie/subject_search?search_text={imdb_id}',
        headers=HEADERS, timeout=15
    )
    m1 = re.search(r'window\.__DATA__\s*=\s*({.*?});', r1.text, re.DOTALL)
    r1_items = 0
    if m1:
        d1 = json.loads(m1.group(1))
        r1_items = d1.get('total', 0)
    print(f'  IMDb ID search: {r1_items} items | URL: {r1.url}')

    if r1_items == 0:
        # Check if redirected to subject page
        if '/subject/' in r1.url:
            did = re.search(r'/subject/(\d+)', r1.url)
            if did:
                print(f'  -> Redirect: douban ID {did.group(1)}')

    # Method 2: Search by movie name
    time.sleep(1)
    r2 = requests.get(
        f'https://search.douban.com/movie/subject_search?search_text={name}',
        headers=HEADERS, timeout=15
    )
    m2 = re.search(r'window\.__DATA__\s*=\s*({.*?});', r2.text, re.DOTALL)
    r2_items = 0
    if m2:
        d2 = json.loads(m2.group(1))
        r2_items = d2.get('total', 0)
    print(f'  Name search: {r2_items} items | URL: {r2.url}')
