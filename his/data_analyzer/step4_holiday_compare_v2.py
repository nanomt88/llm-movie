# -*- coding: utf-8 -*-
"""
Step 4 v2: Holiday Comparison Analysis (Dimensions 6 + 7)
  - Dimension 6: Holiday vs baseline (same weekday before) comparison
      * Sentiment type (positive/neutral/negative)
      * Sentiment intensity (mild/moderate/strong)
      * Movie genre preference
  - Dimension 7: Cross-holiday comparison
      * Question volume
      * Sentiment & intensity distribution
      * Movie genre preference
      * Holiday type aggregation

v2 Changes from v1:
  1. Holidays are grouped by name across years and averaged for charts.
  2. d6_sentiment_comparison adds a 3rd subplot: baseline = same day one month before.
  3. d8_sentiment_vs_nonholiday_avg / d8_intensity_vs_nonholiday_avg:
       Compare each holiday's sentiment/intensity distribution against the
       global non-holiday average across the full 2018-2022 dataset.

Output: output/step4_v2/*.png + CSV

Usage:
    python -m data_analyzer.step4_holiday_compare_v2

Performance note: sentiment analysis on 142k+ questions takes ~2-3 min.
"""

import os
import csv
import calendar
from collections import defaultdict, Counter
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from his.data_analyzer.config import (
    OUTPUT_DIR, MIN_DATA_ROWS, setup_matplotlib, log,
)
from his.data_analyzer.data_loader import (
    extract_imdb_ids, lookup_genre_counts,
)

# Reuse shared logic from v1
from his.data_analyzer.step4_holiday_compare import (
    load_and_prepare, get_sentiment_for_texts, get_baseline_date,
)

setup_matplotlib()
STEP_OUT = os.path.join(OUTPUT_DIR, 'step4_v2')
os.makedirs(STEP_OUT, exist_ok=True)


# ── v2: Month-before baseline date helper ──────────────────────────────
def get_baseline_date_month_before(holiday_date: str, holiday_map: dict) -> str:
    """
    Get the same calendar day one month before a holiday.
    Handles month-end edge cases (e.g., Mar 31 -> Feb 28/29).
    If candidate day is also a holiday, go back 7 days.
    Returns date string or empty string if not found.
    """
    dt = datetime.strptime(holiday_date, '%Y-%m-%d')
    year, month, day = dt.year, dt.month, dt.day

    # Go back one month
    if month == 1:
        prev_month = 12
        prev_year = year - 1
    else:
        prev_month = month - 1
        prev_year = year

    # Handle month-end: if day > days in prev month, use last day
    last_day = calendar.monthrange(prev_year, prev_month)[1]
    candidate_day = min(day, last_day)

    candidate = f"{prev_year:04d}-{prev_month:02d}-{candidate_day:02d}"

    # If candidate is also a holiday, go back up to 5 weeks
    for _ in range(6):
        if candidate not in holiday_map:
            return candidate
        dt_candidate = datetime.strptime(candidate, '%Y-%m-%d')
        candidate = (dt_candidate - timedelta(days=7)).strftime('%Y-%m-%d')

    return ''


# ── v2: Aggregate results by holiday name across years ────────────────
def aggregate_by_name(all_results: list[dict]) -> list[dict]:
    """
    Group all_results by holiday name across different years.
    Merge sentiments, intensities, scores, genre_counts per name.
    Returns a new list with one entry per unique holiday name.
    """
    groups = defaultdict(lambda: {
        'name': None,
        'type': set(),
        'n': 0,
        'sentiments': [],
        'intensities': [],
        'scores': [],
        'genre_counts': Counter(),
        'baseline_week': {
            'n': 0, 'sentiments': [], 'intensities': [], 'scores': [],
            'genre_counts': Counter(),
        },
        'baseline_month': {
            'n': 0, 'sentiments': [], 'intensities': [], 'scores': [],
            'genre_counts': Counter(),
        },
    })

    for r in all_results:
        name = r['name']
        g = groups[name]
        g['name'] = name
        g['type'].add(r['type'])

        g['n'] += r['n']
        g['sentiments'].extend(r['sentiments'])
        g['intensities'].extend(r['intensities'])
        g['scores'].extend(r['scores'])
        g['genre_counts'].update(r['genre_counts'])

        # Week-before baseline
        bw = r['baseline']
        g['baseline_week']['n'] += bw['n']
        g['baseline_week']['sentiments'].extend(bw['sentiments'])
        g['baseline_week']['intensities'].extend(bw['intensities'])
        g['baseline_week']['scores'].extend(bw['scores'])
        g['baseline_week']['genre_counts'].update(bw['genre_counts'])

        # Month-before baseline
        if 'baseline_month' in r:
            bm = r['baseline_month']
            g['baseline_month']['n'] += bm['n']
            g['baseline_month']['sentiments'].extend(bm['sentiments'])
            g['baseline_month']['intensities'].extend(bm['intensities'])
            g['baseline_month']['scores'].extend(bm['scores'])
            g['baseline_month']['genre_counts'].update(bm['genre_counts'])

    # Convert sets to comma-joined type string
    result = []
    for v in groups.values():
        v['type'] = '|'.join(sorted(v['type']))
        result.append(v)

    # Sort by volume descending
    result.sort(key=lambda r: r['n'], reverse=True)
    return result


# ── v2: Compute non-holiday average baseline (10% sample) ────────────
def compute_nonholiday_baseline(seekers: list[dict]) -> dict:
    """
    Run sentiment analysis on a 10% random sample of non-holiday user questions
    and return aggregated sentiment/intensity statistics.
    This provides the global baseline for the d8 comparison charts.
    """
    import random
    random.seed(42)  # reproducible sample
    non_h = [r for r in seekers
             if not r['is_holiday']
             and r['raw_text'].strip()]
    sampled = random.sample(non_h, max(1, len(non_h) // 10))
    log(f"Computing non-holiday baseline: sampling 10% = {len(sampled)} from {len(non_h)} total texts...")

    # Sentiment analysis (uses SENTIMENT_CACHE shared with dim6)
    texts = [r['raw_text'] for r in sampled]
    sentiments = get_sentiment_for_texts(texts)

    baseline = {
        'n': len(sentiments),
        'sentiments': [s['sentiment'] for s in sentiments],
        'intensities': [s['intensity'] for s in sentiments],
        'scores': [s['score'] for s in sentiments],
    }

    # Log summary
    n = baseline['n']
    pos = sum(1 for s in baseline['sentiments'] if s == 'positive') / max(n, 1) * 100
    neu = sum(1 for s in baseline['sentiments'] if s == 'neutral') / max(n, 1) * 100
    neg = sum(1 for s in baseline['sentiments'] if s == 'negative') / max(n, 1) * 100
    mild = sum(1 for s in baseline['intensities'] if s == 'mild') / max(n, 1) * 100
    mod = sum(1 for s in baseline['intensities'] if s == 'moderate') / max(n, 1) * 100
    strong = sum(1 for s in baseline['intensities'] if s == 'strong') / max(n, 1) * 100
    log(f"Non-holiday baseline: Pos={pos:.1f}% Neu={neu:.1f}% Neg={neg:.1f}%  "
        f"Mild={mild:.1f}% Mod={mod:.1f}% Strong={strong:.1f}%  (n={n})")

    return baseline


# ═══════════════════════════════════════════════════════════════════════
#  Dimension 6 v2: Holiday vs Baseline (with month-before + name grouping)
# ═══════════════════════════════════════════════════════════════════════
def dim6_holiday_vs_baseline_v2(seekers: list[dict], holiday_map: dict,
                                 movie_info: dict) -> tuple[list[dict], list[dict]]:
    """
    Compare sentiment, intensity, and genre for each holiday vs two baselines
    (week-before and month-before). Returns (raw_results, grouped_results).
    """
    log("=" * 50)
    log("Dimension 6 v2: Holiday vs Baseline Comparison")
    log("Running sentiment analysis (this may take a moment)...")

    # Build date -> questions lookup
    questions_by_date = defaultdict(list)
    for r in seekers:
        questions_by_date[r['date']].append(r)

    # Process each holiday with sufficient data
    holiday_dates = sorted(set(r['date'] for r in seekers if r['is_holiday']))
    holiday_dates = [d for d in holiday_dates
                     if len(questions_by_date.get(d, [])) >= MIN_DATA_ROWS]

    all_results = []

    for h_date in holiday_dates:
        h_name = holiday_map[h_date]['description']
        h_type = holiday_map[h_date]['type']
        h_rows = questions_by_date[h_date]

        # Baseline week-before (same as v1)
        b_date = get_baseline_date(h_date, holiday_map)  # imported from step4
        # We import get_baseline_date via step4 module, but we need to call it.
        # It's imported from step4_holiday_compare, so we reference the function.
        # Actually, get_baseline_date is a function defined in step4_holiday_compare,
        # but it's not imported in the __init__. Let me import it directly.
        # Wait - I imported get_questions_for_date from step4 but not get_baseline_date.
        # Let me use the one from step4.

        if not b_date or b_date not in questions_by_date:
            log(f"  {h_name}: no week-baseline found, skipping")
            continue
        b_rows = questions_by_date[b_date]

        # Baseline month-before
        bm_date = get_baseline_date_month_before(h_date, holiday_map)
        bm_rows = []
        if bm_date and bm_date in questions_by_date:
            bm_rows = questions_by_date[bm_date]
        else:
            log(f"  {h_name}: no month-baseline found ({bm_date}), skipping month baseline")

        # Sentiment on holiday questions
        h_texts = [r['raw_text'] for r in h_rows]
        h_sentiments = get_sentiment_for_texts(h_texts)
        for r, s in zip(h_rows, h_sentiments):
            r['_sentiment'] = s['sentiment']
            r['_intensity'] = s['intensity']
            r['_sentiment_score'] = s['score']

        # Sentiment on week-before baseline
        b_texts = [r['raw_text'] for r in b_rows]
        b_sentiments = get_sentiment_for_texts(b_texts)
        for r, s in zip(b_rows, b_sentiments):
            r['_sentiment'] = s['sentiment']
            r['_intensity'] = s['intensity']
            r['_sentiment_score'] = s['score']

        # Sentiment on month-before baseline
        bm_sentiments = []
        if bm_rows:
            bm_texts = [r['raw_text'] for r in bm_rows]
            bm_sentiments = get_sentiment_for_texts(bm_texts)
            for r, s in zip(bm_rows, bm_sentiments):
                r['_sentiment'] = s['sentiment']
                r['_intensity'] = s['intensity']
                r['_sentiment_score'] = s['score']

        # Genres for holiday
        h_all_ids = []
        for r in h_rows:
            h_all_ids.extend(extract_imdb_ids(r.get('processed_raw', '')))
        h_genre_counts = lookup_genre_counts(h_all_ids, movie_info)

        # Genres for week-before baseline
        b_all_ids = []
        for r in b_rows:
            b_all_ids.extend(extract_imdb_ids(r.get('processed_raw', '')))
        b_genre_counts = lookup_genre_counts(b_all_ids, movie_info)

        # Genres for month-before baseline
        bm_genre_counts = {}
        if bm_rows:
            bm_all_ids = []
            for r in bm_rows:
                bm_all_ids.extend(extract_imdb_ids(r.get('processed_raw', '')))
            bm_genre_counts = lookup_genre_counts(bm_all_ids, movie_info)

        all_results.append({
            'date': h_date, 'name': h_name, 'type': h_type,
            'rows': h_rows, 'n': len(h_rows),
            'sentiments': [s['sentiment'] for s in h_sentiments],
            'intensities': [s['intensity'] for s in h_sentiments],
            'scores': [s['score'] for s in h_sentiments],
            'genre_counts': h_genre_counts,
            'baseline': {
                'date': b_date,
                'rows': b_rows, 'n': len(b_rows),
                'sentiments': [s['sentiment'] for s in b_sentiments],
                'intensities': [s['intensity'] for s in b_sentiments],
                'scores': [s['score'] for s in b_sentiments],
                'genre_counts': b_genre_counts,
            },
            'baseline_month': {
                'date': bm_date,
                'rows': bm_rows, 'n': len(bm_rows) if bm_rows else 0,
                'sentiments': [s['sentiment'] for s in bm_sentiments],
                'intensities': [s['intensity'] for s in bm_sentiments],
                'scores': [s['score'] for s in bm_sentiments],
                'genre_counts': bm_genre_counts,
            },
        })

    log(f"Analyzed {len(all_results)} holiday-date entries")

    # Group by name across years
    grouped_results = aggregate_by_name(all_results)
    log(f"Aggregated into {len(grouped_results)} unique holiday names")

    # ── 6a-c: plot using grouped data ──
    _plot_sentiment_comparison_v2(grouped_results)
    _plot_intensity_comparison_v2(grouped_results)
    _plot_genre_comparison_v2(grouped_results)

    # ── Save detailed CSV (raw per-date) ──
    csv_path = os.path.join(STEP_OUT, 'd6_holiday_vs_baseline.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'date', 'type', 'n_questions',
                     'sentiment_positive_pct', 'sentiment_neutral_pct',
                     'sentiment_negative_pct',
                     'intensity_mild_pct', 'intensity_moderate_pct',
                     'intensity_strong_pct',
                     'baseline_week_date', 'baseline_week_n',
                     'baseline_week_positive_pct', 'baseline_week_neutral_pct',
                     'baseline_week_negative_pct',
                     'baseline_week_mild_pct', 'baseline_week_moderate_pct',
                     'baseline_week_strong_pct',
                     'baseline_month_date', 'baseline_month_n',
                     'baseline_month_positive_pct', 'baseline_month_neutral_pct',
                     'baseline_month_negative_pct',
                     'baseline_month_mild_pct', 'baseline_month_moderate_pct',
                     'baseline_month_strong_pct'])
        for r in all_results:
            b = r['baseline']
            bm = r['baseline_month']
            def pct(arr, label):
                return f"{sum(1 for x in arr if x == label) / max(len(arr), 1) * 100:.1f}"
            w.writerow([
                r['name'], r['date'], r['type'], r['n'],
                pct(r['sentiments'], 'positive'),
                pct(r['sentiments'], 'neutral'),
                pct(r['sentiments'], 'negative'),
                pct(r['intensities'], 'mild'),
                pct(r['intensities'], 'moderate'),
                pct(r['intensities'], 'strong'),
                b['date'], b['n'],
                pct(b['sentiments'], 'positive'),
                pct(b['sentiments'], 'neutral'),
                pct(b['sentiments'], 'negative'),
                pct(b['intensities'], 'mild'),
                pct(b['intensities'], 'moderate'),
                pct(b['intensities'], 'strong'),
                bm['date'], bm['n'],
                pct(bm['sentiments'], 'positive'),
                pct(bm['sentiments'], 'neutral'),
                pct(bm['sentiments'], 'negative'),
                pct(bm['intensities'], 'mild'),
                pct(bm['intensities'], 'moderate'),
                pct(bm['intensities'], 'strong'),
            ])
    log(f"Saved: {csv_path}")

    return all_results, grouped_results


# ═══════════════════════════════════════════════════════════════════════
#  Plot helpers (v2: grouped by name)
# ═══════════════════════════════════════════════════════════════════════

def _plot_sentiment_comparison_v2(grouped: list[dict]):
    """
    6a v2: Grouped bar chart - sentiment type holiday vs week-baseline vs month-baseline.
    Three subplots: Holiday, Week-Before, Month-Before.
    """
    names = [r['name'][:10] for r in grouped]

    # Holiday sentiment
    h_pos = [sum(1 for s in r['sentiments'] if s == 'positive') / max(r['n'], 1) * 100
             for r in grouped]
    h_neu = [sum(1 for s in r['sentiments'] if s == 'neutral') / max(r['n'], 1) * 100
             for r in grouped]
    h_neg = [sum(1 for s in r['sentiments'] if s == 'negative') / max(r['n'], 1) * 100
             for r in grouped]

    # Week-before baseline sentiment
    b_pos = [sum(1 for s in r['baseline_week']['sentiments'] if s == 'positive')
             / max(r['baseline_week']['n'], 1) * 100 for r in grouped]
    b_neu = [sum(1 for s in r['baseline_week']['sentiments'] if s == 'neutral')
             / max(r['baseline_week']['n'], 1) * 100 for r in grouped]
    b_neg = [sum(1 for s in r['baseline_week']['sentiments'] if s == 'negative')
             / max(r['baseline_week']['n'], 1) * 100 for r in grouped]

    # Month-before baseline sentiment
    m_pos = [sum(1 for s in r['baseline_month']['sentiments'] if s == 'positive')
             / max(r['baseline_month']['n'], 1) * 100 for r in grouped]
    m_neu = [sum(1 for s in r['baseline_month']['sentiments'] if s == 'neutral')
             / max(r['baseline_month']['n'], 1) * 100 for r in grouped]
    m_neg = [sum(1 for s in r['baseline_month']['sentiments'] if s == 'negative')
             / max(r['baseline_month']['n'], 1) * 100 for r in grouped]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
    x = np.arange(len(names))
    width = 0.25

    # Holiday chart
    ax1.bar(x - width, h_pos, width, label='Positive', color='#2ecc71', alpha=0.8)
    ax1.bar(x, h_neu, width, label='Neutral', color='#95a5a6', alpha=0.8)
    ax1.bar(x + width, h_neg, width, label='Negative', color='#e74c3c', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('Proportion (%)')
    ax1.set_title('Holiday Sentiment Distribution (averaged across years)')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Week-before baseline chart
    ax2.bar(x - width, b_pos, width, label='Positive', color='#2ecc71', alpha=0.8)
    ax2.bar(x, b_neu, width, label='Neutral', color='#95a5a6', alpha=0.8)
    ax2.bar(x + width, b_neg, width, label='Negative', color='#e74c3c', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Proportion (%)')
    ax2.set_title('Baseline (Week Before) Sentiment Distribution')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Month-before baseline chart
    ax3.bar(x - width, m_pos, width, label='Positive', color='#2ecc71', alpha=0.8)
    ax3.bar(x, m_neu, width, label='Neutral', color='#95a5a6', alpha=0.8)
    ax3.bar(x + width, m_neg, width, label='Negative', color='#e74c3c', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax3.set_ylabel('Proportion (%)')
    ax3.set_title('Baseline (Month Before) Sentiment Distribution')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd6_sentiment_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_intensity_comparison_v2(grouped: list[dict]):
    """6b v2: Scatter - intensity comparison, grouped by name."""
    names = [r['name'][:12] for r in grouped]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    width = 0.35

    for i, level in enumerate(['mild', 'moderate', 'strong']):
        # Holiday
        h_vals = [sum(1 for s in r['intensities'] if s == level) / max(r['n'], 1) * 100
                  for r in grouped]
        # Baseline week
        b_vals = [sum(1 for s in r['baseline_week']['intensities'] if s == level)
                  / max(r['baseline_week']['n'], 1) * 100 for r in grouped]

        offset = (i - 1) * 0.08
        colors = {'mild': '#3498db', 'moderate': '#f39c12', 'strong': '#e74c3c'}
        ax.scatter(x - width / 2 + offset, h_vals, marker='o', s=60,
                   color=colors[level], alpha=0.7,
                   label=f'{level.capitalize()} (Holiday)' if i == 0 else '')
        ax.scatter(x + width / 2 + offset, b_vals, marker='x', s=60,
                   color=colors[level], alpha=0.7,
                   label=f'{level.capitalize()} (Baseline)' if i == 0 else '')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Intensity: Holiday vs Baseline (o=holiday, x=baseline, averaged across years)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd6_intensity_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_genre_comparison_v2(grouped: list[dict]):
    """6c v2: Heatmap - top genres for holidays vs baselines, grouped by name."""
    # Aggregate genre counts across all groups
    all_genres = Counter()
    for r in grouped:
        all_genres.update(r['genre_counts'])
        all_genres.update(r['baseline_week']['genre_counts'])

    top_genres = [g for g, _ in all_genres.most_common(12)]

    if not top_genres:
        log("WARN: No genre data for heatmap")
        return

    # Build matrix: rows=holidays (grouped), cols=genres, val=ratio diff (holiday - week baseline)
    matrix = np.zeros((len(grouped), len(top_genres)))
    for i, r in enumerate(grouped):
        h_total = max(sum(r['genre_counts'].values()), 1)
        b_total = max(sum(r['baseline_week']['genre_counts'].values()), 1)
        for j, g in enumerate(top_genres):
            h_ratio = r['genre_counts'].get(g, 0) / h_total
            b_ratio = r['baseline_week']['genre_counts'].get(g, 0) / b_total
            matrix[i, j] = h_ratio - b_ratio

    fig, ax = plt.subplots(figsize=(14, max(6, len(grouped) * 0.5 + 2)))
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)

    ax.set_xticks(range(len(top_genres)))
    ax.set_xticklabels(top_genres, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels([r['name'][:12] for r in grouped], fontsize=8)

    for i in range(len(grouped)):
        for j in range(len(top_genres)):
            val = matrix[i, j]
            if abs(val) > 0.01:
                color = 'white' if abs(val) > 0.08 else 'black'
                ax.text(j, i, f'{val:+.0%}', ha='center', va='center',
                        fontsize=7, color=color)

    ax.set_title('Genre Proportion Difference: Holiday - Week Baseline (red=more on holiday, averaged across years)')
    fig.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd6_genre_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Dimension 8 v2: Holiday vs Global Non-Holiday Average
# ═══════════════════════════════════════════════════════════════════════
def _plot_sentiment_vs_nonholiday_avg(grouped: list[dict], nh_baseline: dict):
    """
    d8a: Grouped bar chart comparing each holiday's sentiment distribution
    against the global non-holiday average.
    Each holiday gets 3 grouped bars (positive/neutral/negative);
    dashed horizontal lines show the non-holiday reference values.
    """
    names = [r['name'][:6] for r in grouped]

    # Holiday percentages per sentiment type
    h_pos = [sum(1 for s in r['sentiments'] if s == 'positive') / max(r['n'], 1) * 100
             for r in grouped]
    h_neu = [sum(1 for s in r['sentiments'] if s == 'neutral') / max(r['n'], 1) * 100
             for r in grouped]
    h_neg = [sum(1 for s in r['sentiments'] if s == 'negative') / max(r['n'], 1) * 100
             for r in grouped]

    # Non-holiday global average percentages
    nh_n = max(nh_baseline['n'], 1)
    nh_pos = sum(1 for s in nh_baseline['sentiments'] if s == 'positive') / nh_n * 100
    nh_neu = sum(1 for s in nh_baseline['sentiments'] if s == 'neutral') / nh_n * 100
    nh_neg = sum(1 for s in nh_baseline['sentiments'] if s == 'negative') / nh_n * 100

    fig, ax = plt.subplots(figsize=(24, 6))
    x = np.arange(len(names))
    width = 0.22

    # Grouped bars: positive / neutral / negative for each holiday
    bars_pos = ax.bar(x - width, h_pos, width, label='Positive',
                      color='#2ecc71', alpha=0.8)
    bars_neu = ax.bar(x, h_neu, width, label='Neutral',
                      color='#95a5a6', alpha=0.8)
    bars_neg = ax.bar(x + width, h_neg, width, label='Negative',
                      color='#e74c3c', alpha=0.8)

    # Horizontal reference lines from non-holiday average
    ax.axhline(y=nh_pos, color='#2ecc71', linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Pos ({nh_pos:.1f}%)')
    ax.axhline(y=nh_neu, color='#95a5a6', linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Neu ({nh_neu:.1f}%)')
    ax.axhline(y=nh_neg, color='#e74c3c', linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Neg ({nh_neg:.1f}%)')

    # Highlight holidays that deviate noticeably
    for i in range(len(names)):
        for bar, val, ref in [(bars_pos[i], h_pos[i], nh_pos),
                               (bars_neu[i], h_neu[i], nh_neu),
                               (bars_neg[i], h_neg[i], nh_neg)]:
            diff = val - ref
            if abs(diff) > 3.0:  # >3pp deviation
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Distribution: Each Holiday vs Non-Holiday Average')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd8_sentiment_vs_nonholiday_avg.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV: deviation per holiday ──
    csv_path = os.path.join(STEP_OUT, 'd8_sentiment_vs_nonholiday_avg.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'n_questions',
                     'pos_pct', 'neu_pct', 'neg_pct',
                     'nh_pos_pct', 'nh_neu_pct', 'nh_neg_pct',
                     'pos_diff', 'neu_diff', 'neg_diff'])
        for i, r in enumerate(grouped):
            w.writerow([
                r['name'], r['n'],
                f'{h_pos[i]:.1f}', f'{h_neu[i]:.1f}', f'{h_neg[i]:.1f}',
                f'{nh_pos:.1f}', f'{nh_neu:.1f}', f'{nh_neg:.1f}',
                f'{h_pos[i] - nh_pos:+.1f}',
                f'{h_neu[i] - nh_neu:+.1f}',
                f'{h_neg[i] - nh_neg:+.1f}',
            ])
    log(f"Saved: {csv_path}")


def _plot_intensity_vs_nonholiday_avg(grouped: list[dict], nh_baseline: dict):
    """
    d8b: Grouped bar chart comparing each holiday's intensity distribution
    against the global non-holiday average.
    """
    names = [r['name'][:10] for r in grouped]

    # Holiday percentages per intensity level
    h_mild = [sum(1 for s in r['intensities'] if s == 'mild') / max(r['n'], 1) * 100
              for r in grouped]
    h_mod = [sum(1 for s in r['intensities'] if s == 'moderate') / max(r['n'], 1) * 100
             for r in grouped]
    h_strong = [sum(1 for s in r['intensities'] if s == 'strong') / max(r['n'], 1) * 100
                for r in grouped]

    # Non-holiday global average percentages
    nh_n = max(nh_baseline['n'], 1)
    nh_mild = sum(1 for s in nh_baseline['intensities'] if s == 'mild') / nh_n * 100
    nh_mod = sum(1 for s in nh_baseline['intensities'] if s == 'moderate') / nh_n * 100
    nh_strong = sum(1 for s in nh_baseline['intensities'] if s == 'strong') / nh_n * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    width = 0.22

    bars_mild = ax.bar(x - width, h_mild, width, label='Mild',
                       color='#3498db', alpha=0.8)
    bars_mod = ax.bar(x, h_mod, width, label='Moderate',
                      color='#f39c12', alpha=0.8)
    bars_strong = ax.bar(x + width, h_strong, width, label='Strong',
                         color='#e74c3c', alpha=0.8)

    # Horizontal reference lines
    ax.axhline(y=nh_mild, color='#3498db', linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Mild ({nh_mild:.1f}%)')
    ax.axhline(y=nh_mod, color='#f39c12', linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Mod ({nh_mod:.1f}%)')
    ax.axhline(y=nh_strong, color='#e74c3c', linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Strong ({nh_strong:.1f}%)')

    # Highlight deviations >3pp
    for i in range(len(names)):
        for bar, val, ref in [(bars_mild[i], h_mild[i], nh_mild),
                               (bars_mod[i], h_mod[i], nh_mod),
                               (bars_strong[i], h_strong[i], nh_strong)]:
            diff = val - ref
            if abs(diff) > 3.0:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Intensity: Each Holiday vs Non-Holiday Average')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd8_intensity_vs_nonholiday_avg.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd8_intensity_vs_nonholiday_avg.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'n_questions',
                     'mild_pct', 'mod_pct', 'strong_pct',
                     'nh_mild_pct', 'nh_mod_pct', 'nh_strong_pct',
                     'mild_diff', 'mod_diff', 'strong_diff'])
        for i, r in enumerate(grouped):
            w.writerow([
                r['name'], r['n'],
                f'{h_mild[i]:.1f}', f'{h_mod[i]:.1f}', f'{h_strong[i]:.1f}',
                f'{nh_mild:.1f}', f'{nh_mod:.1f}', f'{nh_strong:.1f}',
                f'{h_mild[i] - nh_mild:+.1f}',
                f'{h_mod[i] - nh_mod:+.1f}',
                f'{h_strong[i] - nh_strong:+.1f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Dimension 7 v2: Cross-Holiday Comparison (grouped by name)
# ═══════════════════════════════════════════════════════════════════════
def dim7_cross_holiday_v2(grouped: list[dict]):
    """Compare question volume, sentiment, and genres across holidays (grouped by name)."""
    log("-" * 50)
    log("Dimension 7 v2: Cross-Holiday Comparison (averaged across years)")

    if not grouped:
        log("WARN: No holiday data")
        return

    # ── 7a: Question volume comparison (bar, sorted) ──
    sorted_by_vol = sorted(grouped, key=lambda r: r['n'], reverse=True)
    names = [r['name'][:12] for r in sorted_by_vol]
    volumes = [r['n'] for r in sorted_by_vol]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    ax.bar(range(len(names)), volumes, color=colors_bar, alpha=0.8, edgecolor='grey')
    for i, v in enumerate(volumes):
        ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Number of Questions (all years combined)')
    ax.set_title('Cross-Holiday Question Volume (sorted, summed across years)')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd7_question_volume.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── 7b: Sentiment stacked bar ──
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(grouped))
    pos_vals = [sum(1 for s in r['sentiments'] if s == 'positive') / max(r['n'], 1) * 100
                for r in grouped]
    neu_vals = [sum(1 for s in r['sentiments'] if s == 'neutral') / max(r['n'], 1) * 100
                for r in grouped]
    neg_vals = [sum(1 for s in r['sentiments'] if s == 'negative') / max(r['n'], 1) * 100
                for r in grouped]

    ax.bar(x, pos_vals, label='Positive', color='#2ecc71', alpha=0.8)
    ax.bar(x, neu_vals, bottom=pos_vals, label='Neutral', color='#95a5a6', alpha=0.8)
    bottom2 = [p + n for p, n in zip(pos_vals, neu_vals)]
    ax.bar(x, neg_vals, bottom=bottom2, label='Negative', color='#e74c3c', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([r['name'][:10] for r in grouped], rotation=30, ha='right')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Distribution Across Holidays (averaged across years)')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd7_sentiment_by_holiday.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── 7c: Intensity grouped bars ──
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(grouped))
    width = 0.25

    for i, level in enumerate(['mild', 'moderate', 'strong']):
        vals = [sum(1 for s in r['intensities'] if s == level) / max(r['n'], 1) * 100
                for r in grouped]
        colors = {'mild': '#3498db', 'moderate': '#f39c12', 'strong': '#e74c3c'}
        ax.bar(x + (i - 1) * width, vals, width, label=level.capitalize(),
               color=colors[level], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([r['name'][:10] for r in grouped], rotation=30, ha='right')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Intensity Across Holidays (averaged across years)')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd7_intensity_by_holiday.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── 7d: Holiday type aggregation ──
    _plot_holiday_type_aggregation_v2(grouped)

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd7_cross_holiday_stats.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'type', 'n_questions_all_years',
                     'positive_pct', 'neutral_pct', 'negative_pct',
                     'mild_pct', 'moderate_pct', 'strong_pct',
                     'avg_sentiment_score'])
        for r in grouped:
            avg_score = np.mean(r['scores']) if r['scores'] else 0
            def pct(arr, label):
                return f"{sum(1 for x in arr if x == label) / max(len(arr), 1) * 100:.1f}"
            w.writerow([
                r['name'], r['type'], r['n'],
                pct(r['sentiments'], 'positive'),
                pct(r['sentiments'], 'neutral'),
                pct(r['sentiments'], 'negative'),
                pct(r['intensities'], 'mild'),
                pct(r['intensities'], 'moderate'),
                pct(r['intensities'], 'strong'),
                f"{avg_score:.3f}",
            ])
    log(f"Saved: {csv_path}")


def _plot_holiday_type_aggregation_v2(grouped: list[dict]):
    """7d v2: Aggregate by holiday type using name-grouped data."""
    type_groups = defaultdict(list)
    for r in grouped:
        type_groups[r['type']].append(r)

    type_names = sorted(type_groups.keys())
    if len(type_names) < 2:
        log("  Skipping holiday type aggregation: only 1 type")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1) Avg sentiment per type
    ax = axes[0]
    x = np.arange(len(type_names))
    for i, level in enumerate(['positive', 'neutral', 'negative']):
        vals = []
        for tn in type_names:
            all_s = []
            for r in type_groups[tn]:
                all_s.extend(r['sentiments'])
            vals.append(sum(1 for s in all_s if s == level) / max(len(all_s), 1) * 100)
        colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
        offset = (i - 1) * 0.25
        ax.bar(x + offset, vals, 0.25, label=level.capitalize(),
               color=colors[level], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(type_names, rotation=15, ha='right')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment by Holiday Type')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 2) Avg intensity per type
    ax = axes[1]
    for i, level in enumerate(['mild', 'moderate', 'strong']):
        vals = []
        for tn in type_names:
            all_i = []
            for r in type_groups[tn]:
                all_i.extend(r['intensities'])
            vals.append(sum(1 for s in all_i if s == level) / max(len(all_i), 1) * 100)
        colors_i = {'mild': '#3498db', 'moderate': '#f39c12', 'strong': '#e74c3c'}
        offset = (i - 1) * 0.25
        ax.bar(x + offset, vals, 0.25, label=level.capitalize(),
               color=colors_i[level], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(type_names, rotation=15, ha='right')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Intensity by Holiday Type')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 3) Radar: genre preference by type
    ax = axes[2]
    all_genre_counts = Counter()
    for r in grouped:
        all_genre_counts.update(r['genre_counts'])
    top_g = [g for g, _ in all_genre_counts.most_common(8)]

    if top_g:
        angles = np.linspace(0, 2 * np.pi, len(top_g), endpoint=False).tolist()
        angles += angles[:1]

        for tn in type_names:
            type_gc = Counter()
            total_g = 0
            for r in type_groups[tn]:
                type_gc.update(r['genre_counts'])
                total_g += sum(r['genre_counts'].values())

            vals = [type_gc.get(g, 0) / max(total_g, 1) * 100 for g in top_g]
            vals += vals[:1]
            ax.plot(angles, vals, 'o-', label=tn, alpha=0.7)
            ax.fill(angles, vals, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_g, fontsize=7)
        ax.set_title('Genre by Holiday Type')
        ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd7_holiday_type_aggregation.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── Type aggregation CSV ──
    csv_path = os.path.join(STEP_OUT, 'd7_holiday_type_aggregation.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_type', 'n_holidays', 'n_questions',
                     'positive_pct', 'neutral_pct', 'negative_pct',
                     'mild_pct', 'moderate_pct', 'strong_pct'])
        for tn in type_names:
            all_s = []
            all_i = []
            for r in type_groups[tn]:
                all_s.extend(r['sentiments'])
                all_i.extend(r['intensities'])
            n_q = len(all_s)
            def pct(arr, label):
                return f"{sum(1 for x in arr if x == label)/max(len(arr),1)*100:.1f}"
            w.writerow([
                tn, len(type_groups[tn]), n_q,
                pct(all_s, 'positive'), pct(all_s, 'neutral'), pct(all_s, 'negative'),
                pct(all_i, 'mild'), pct(all_i, 'moderate'), pct(all_i, 'strong'),
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 60)
    log("Step 4 v2: Holiday Comparison Analysis (Dimensions 6 + 7)")
    log("  Features: name-grouped across years, month-before baseline")
    log("=" * 60)

    seekers, holiday_map, movie_info = load_and_prepare()

    # Dimension 6 (returns both raw and grouped)
    log("")
    all_results, grouped_results = dim6_holiday_vs_baseline_v2(
        seekers, holiday_map, movie_info)

    # Dimension 8: Holiday vs global non-holiday average
    log("")
    nh_baseline = compute_nonholiday_baseline(seekers)
    log("")
    _plot_sentiment_vs_nonholiday_avg(grouped_results, nh_baseline)
    _plot_intensity_vs_nonholiday_avg(grouped_results, nh_baseline)

    # Dimension 7 (uses grouped data)
    log("")
    dim7_cross_holiday_v2(grouped_results)

    log("")
    log("=" * 60)
    log(f"Step 4 v2 complete! Results saved to {STEP_OUT}")
    log("=" * 60)

    # Print summary (using grouped data)
    if grouped_results:
        print(f"\n{'Holiday':<20} {'n':>6} {'Pos':>7} {'Neu':>7} {'Neg':>7} "
              f"{'Mild':>7} {'Mod':>7} {'Strong':>7} {'Score':>7}")
        print("-" * 75)
        for r in grouped_results:
            n = r['n']
            pos = sum(1 for s in r['sentiments'] if s == 'positive') / n * 100
            neu = sum(1 for s in r['sentiments'] if s == 'neutral') / n * 100
            neg = sum(1 for s in r['sentiments'] if s == 'negative') / n * 100
            mild = sum(1 for s in r['intensities'] if s == 'mild') / n * 100
            mod = sum(1 for s in r['intensities'] if s == 'moderate') / n * 100
            strong = sum(1 for s in r['intensities'] if s == 'strong') / n * 100
            score = np.mean(r['scores'])
            name_display = r['name'][:18]
            print(f"{name_display:<20} {n:>6} {pos:>5.1f}% {neu:>5.1f}% {neg:>5.1f}% "
                  f"{mild:>5.1f}% {mod:>5.1f}% {strong:>5.1f}% {score:>+6.3f}")


if __name__ == '__main__':
    main()
