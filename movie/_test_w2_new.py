"""Test W2 with new elevated words chart."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from movie.data_loader import load_conversations, tag_period, load_holiday_definitions, load_holiday_workday_adjustments

holiday_map = load_holiday_definitions()
adjustments = load_holiday_workday_adjustments()
rows = load_conversations('..\\data\\conv\\data_all.csv', max_rows=50000)
rows = tag_period(rows, holiday_map, adjustments)
seekers = [r for r in rows if r['is_seeker']]

from movie.step7_wordcloud import dim_w2_holiday_vs_nonholiday_words

# Default threshold 1.5
print("=== threshold=1.5 ===")
dim_w2_holiday_vs_nonholiday_words(seekers, ratio_threshold=1.5)

# Custom threshold 3.0
print("\n=== threshold=3.0 ===")
dim_w2_holiday_vs_nonholiday_words(seekers, ratio_threshold=3.0)
