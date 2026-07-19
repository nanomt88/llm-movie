# -*- coding: utf-8 -*-
"""
Shared configuration for the 2018-2022 holiday movie analysis pipeline.
Paths, constants, font setup, logging.
"""

import os
import sys
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# Input files
HOLIDAY_CSV = os.path.join(DATA_DIR, 'holiday.csv')
HOLIDAY_CONV_CSV = os.path.join(DATA_DIR, 'conv/all_holiday_records_v3.csv')  # holiday-focused
FULL_YEAR_CSV = os.path.join(DATA_DIR, 'conv/data_all.csv')           # full year
MOVIE_INFO_PATH = os.path.join(DATA_DIR, 'movie_info.json')

# ── Thresholds (all configurable) ─────────────────────────────────────
LONG_QUESTION_WORD_THRESHOLD = 25   # >= this many words = "long question"
LONG_DETAILED_CHAR_THRESHOLD = 100 # deprecated in v2, kept for compat
MIN_DATA_ROWS = 10                  # skip holiday with fewer rows than this

# Sentiment intensity thresholds (based on VADER compound absolute value)
INTENSITY_MILD = 0.2
INTENSITY_MODERATE = 0.5

# ── Output directories ────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
STEP_DIRS = {}
for step in range(1, 6):
    d = os.path.join(OUTPUT_DIR, f'step{step}')
    os.makedirs(d, exist_ok=True)
    STEP_DIRS[step] = d


# ── Matplotlib font setup ─────────────────────────────────────────────
def setup_matplotlib():
    """Configure matplotlib: Agg backend + font."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Try Chinese fonts (even though data is English, labels may have Chinese)
    cn_fonts = [
        'Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'Source Han Sans SC',
    ]
    for f in cn_fonts:
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            matplotlib.rcParams['font.sans-serif'] = [f] + matplotlib.rcParams['font.sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            continue

    # English-friendly fallback
    plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150})


# ── Logging ───────────────────────────────────────────────────────────
def log(msg: str, module: str = "Analyzer"):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [{module}] {msg}", flush=True)
