"""Quick test for movie_info.py"""
import sys
sys.path.insert(0, '../..')
from his.src.movie_info import *

test_ids = ['tt0110148', 'tt0119675', 'tt0109424']
id_to_name = {
    'tt0110148': 'Interview with the Vampire',
    'tt0119675': 'Mimic',
    'tt0109424': 'Chungking Express',
}

for imdb_id in test_ids:
    name = id_to_name.get(imdb_id)
    label = name or imdb_id
    print(f'\n=== {label} ({imdb_id}) ===')
    info = fetch_movie_info(imdb_id, name)
    if info:
        for k, v in info.items():
            print(f'  {k}: {v}')
    else:
        print('  Not found')
