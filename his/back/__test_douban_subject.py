"""Check Douban subject page for inline data"""
import requests
import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

s = requests.Session()
s.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
})

# First visit homepage to get cookies
s.get('https://movie.douban.com/', timeout=15)

# Then visit subject page
resp = s.get('https://movie.douban.com/subject/1299327/', timeout=15)
print(f'Status: {resp.status_code} | Length: {len(resp.text)}')

# Check various data patterns in the page
patterns = [
    r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
    r'window\.__DATA__\s*=\s*({.*?});',
    r'application/ld\+json',
    r'<script[^>]*>var\s+_INITIAL_',
]
for p in patterns:
    matches = re.findall(p, resp.text)
    print(f'Pattern {p[:30]}: {len(matches)} matches')

# Look at the body content
body_start = resp.text.find('<body')
body_end = resp.text.find('</body>')
body = resp.text[body_start:body_end+7] if body_start >= 0 else ''
print(f'\nBody content length: {len(body)}')

# Check for any JSON in script tags
scripts = re.findall(r'<script[^>]*>([^<]+)</script>', resp.text)
for s in scripts:
    s = s.strip()
    if s.startswith('{') or s.startswith('['):
        print(f'  JSON-like script: {s[:100]}...')
