"""Check and download a newer wordcloud version."""
import urllib.request, re, os, subprocess, sys

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_downloads')

resp = urllib.request.urlopen('https://pypi.org/simple/wordcloud/', timeout=15)
html = resp.read().decode('utf-8')
links = re.findall(r'href="([^"]+)"', html)

import re as regex
candidates = []
for l in links:
    if 'cp310' in l and 'win_amd64' in l:
        m = regex.search(r'wordcloud-([\d.]+[a-z\d.]*)-cp310', l)
        if m:
            from packaging.version import Version
            candidates.append((Version(m.group(1)), l))

candidates.sort(key=lambda x: x[0], reverse=True)
print('Available cp310 win_amd64 versions:')
for v, url in candidates:
    print(f'  v{v}')

if candidates:
    best = candidates[0]
    url = best[1]
    if not url.startswith('http'):
        url = 'https://pypi.org' + url
    dl_url = url.split('#')[0]
    filename = dl_url.rsplit('/')[-1]
    filepath = os.path.join(dest, filename)
    print(f'\nDownloading wordcloud v{best[0]}...')
    urllib.request.urlretrieve(dl_url, filepath)
    print(f'Uninstalling old wordcloud...')
    subprocess.run(['pip', 'uninstall', 'wordcloud', '-y'], capture_output=True)
    print(f'Installing {filename}...')
    result = subprocess.run(['pip', 'install', filepath, '--no-index'], capture_output=True, text=True)
    print(result.stdout.strip()[-300:])
    if result.returncode != 0:
        print(f'Error: {result.stderr.strip()[-300:]}')
    
    # Test
    result = subprocess.run([sys.executable, '-c', 'from wordcloud import WordCloud; wc = WordCloud(); print("wordcloud import OK")'],
                          capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f'Error: {result.stderr.strip()[-500:]}')
