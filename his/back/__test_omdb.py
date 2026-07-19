"""Test OMDb API availability"""
import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {'User-Agent': 'Mozilla/5.0'}

# OMDb test with a free tier key (rate-limited to 1000/day)
# Try the demo endpoint
url = 'http://www.omdbapi.com/?i=tt0110148&apikey=8776f253'
print(f'Testing OMDb API...')
resp = requests.get(url, headers=HEADERS, timeout=15)
print(f'Status: {resp.status_code}')
if resp.status_code == 200:
    data = resp.json()
    if data.get('Response') == 'True':
        print(f'Title: {data.get("Title")}')
        print(f'Year: {data.get("Year")}')
        print(f'Genre: {data.get("Genre")}')
        print(f'Director: {data.get("Director")}')
        print(f'Actors: {data.get("Actors")}')
        print(f'Country: {data.get("Country")}')
        print(f'Runtime: {data.get("Runtime")}')
        print(f'Language: {data.get("Language")}')
        print(f'imdbRating: {data.get("imdbRating")}')
    else:
        print(f'Error: {data.get("Error")}')
else:
    print(f'Body: {resp.text[:300]}')
