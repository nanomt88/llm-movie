"""Debug Douban search failures - check raw response"""
import requests
import json
import re
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# Movie known to be on Douban but search failed
for imdb_id in ['tt0082158', 'tt1040023']:
    print(f'\n=== {imdb_id} ===')
    
    resp = requests.get(
        f'https://search.douban.com/movie/subject_search?search_text={imdb_id}',
        headers=HEADERS, timeout=15
    )
    print(f'Status: {resp.status_code} | URL: {resp.url}')
    print(f'Content-Length: {len(resp.text)}')
    
    # Try to extract any JSON data
    json_patterns = re.findall(r'window\.__DATA__\s*=\s*({.*?});', resp.text, re.DOTALL)
    print(f'__DATA__ blocks found: {len(json_patterns)}')
    
    for j in json_patterns[:1]:
        try:
            data = json.loads(j)
            print(f'  total: {data.get("total")}, items: {len(data.get("items", []))}')
            print(f'  text: {data.get("text")}')
            if data.get("items"):
                print(f'  first item keys: {list(data["items"][0].keys())}')
        except json.JSONDecodeError as e:
            print(f'  JSON error: {e}')
    
    # Look for any subject IDs in the HTML
    subject_ids = re.findall(r'/subject/(\d+)', resp.text)
    print(f'Subject IDs found: {set(subject_ids)}')
    
    time.sleep(1)
    
    # Try also direct Douban API-like endpoint
    resp2 = requests.get(
        f'https://movie.douban.com/subject_search?search_text={imdb_id}',
        headers=HEADERS, timeout=15
    )
    print(f'\nmovie.douban.com search: Status={resp2.status_code} URL={resp2.url}')
    subject_ids2 = re.findall(r'/subject/(\d+)', resp2.text)
    print(f'Subject IDs: {set(subject_ids2)}')
    
    time.sleep(1)
    
    # Try suggest API
    resp3 = requests.get(
        f'https://movie.douban.com/j/subject_suggest?q={imdb_id}',
        headers=HEADERS, timeout=15
    )
    print(f'\nSuggest API: Status={resp3.status_code}')
    if resp3.text.strip():
        try:
            suggest_data = resp3.json()
            print(f'Results: {len(suggest_data)}')
            for s in suggest_data[:1]:
                print(f'  {s}')
        except:
            print(f'  Not JSON: {resp3.text[:200]}')
    else:
        print(f'  Empty response')
