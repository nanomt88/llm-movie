"""Test various Douban endpoints for movie info"""
import requests
import json
import re
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# Movie that works with IMDb search: tt0110148, tt0109424
# Movie that doesn't: tt0082158

# Test 1: Parse abstract for genre extraction
cases = {
    'tt0110148': '美国 / 剧情 / 奇幻 / 惊悚 / 吸血迷情(港) / 123分钟',
    'tt0119675': None,  # Will fetch
    'tt0109424': None,  # Will fetch
}

# Try direct Douban subject page by known ID
print('=== Direct Douban subject page ===')
resp = requests.get('https://movie.douban.com/subject/1299327/', headers=HEADERS, timeout=15)
print(f'Status: {resp.status_code} | Length: {len(resp.text)}')

# Check for structured data on the page
# Look for JSON-LD
jsonld = re.findall(r'<script type="application/ld\+json">(.*?)</script>', resp.text, re.DOTALL)
if jsonld:
    for j in jsonld:
        data = json.loads(j)
        print(f'JSON-LD keys: {list(data.keys())}')
        print(f'  @type: {data.get("@type")}')
        print(f'  genre: {data.get("genre")}')
else:
    print('No JSON-LD found')

# Check for rating/interest data
interest = re.findall(r'window\.__INTEREST_DATA__\s*=\s*({.*?});', resp.text, re.DOTALL)
print(f'Interest data: {len(interest)} blocks')

time.sleep(1)

# Test 2: Try the /j/suggest endpoint with different formats
print('\n=== Subject suggest (IMDb ID) ===')
resp2 = requests.get(
    'https://movie.douban.com/j/subject_suggest?q=tt0110148',
    headers=HEADERS, timeout=15
)
print(f'Status: {resp2.status_code} | {resp2.text[:300]}')

time.sleep(1)

# Test 3: Check Douban search with full names
print('\n=== Subject suggest (Chinese name) ===')
resp3 = requests.get(
    'https://movie.douban.com/j/subject_suggest?q=%E9%87%8D%E5%BA%86%E6%A3%AE%E6%9E%97',
    headers=HEADERS, timeout=15
)
print(f'Status: {resp3.status_code} | {resp3.text[:300]}')

time.sleep(1)

# Test 4: Check Douban search with English name
print('\n=== Subject suggest (English name) ===')
resp4 = requests.get(
    'https://movie.douban.com/j/subject_suggest?q=Chungking+Express',
    headers=HEADERS, timeout=15
)
print(f'Status: {resp4.status_code} | {resp4.text[:300]}')
