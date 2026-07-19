# -*- coding: utf-8 -*-
"""
Step 4: Holiday Comparison Analysis (Dimensions 6 + 7)
  - Dimension 6: Holiday vs baseline (same weekday before) comparison
      * Sentiment type (positive/neutral/negative)
      * Sentiment intensity (mild/moderate/strong)
      * Movie genre preference
  - Dimension 7: Cross-holiday comparison
      * Question volume
      * Sentiment & intensity distribution
      * Movie genre preference
      * Holiday type aggregation

Output: output/step4/*.png + CSV

Performance note: sentiment analysis on 142k+ questions takes ~2-3 min.
"""

import os
import csv
from collections import defaultdict, Counter
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from his.data_analyzer.config import (
    OUTPUT_DIR, MIN_DATA_ROWS, FULL_YEAR_CSV, setup_matplotlib, log,
)
from his.data_analyzer.data_loader import (
    load_conversations, load_holiday_definitions, tag_holiday,
    load_movie_info, extract_imdb_ids, lookup_genre_counts,
)
from his.data_analyzer.sentiment import analyze_batch

setup_matplotlib()
STEP_OUT = os.path.join(OUTPUT_DIR, 'step4')
os.makedirs(STEP_OUT, exist_ok=True)


# ── 1. Data preparation ──────────────────────────────────────────────
def load_and_prepare() -> tuple[list[dict], dict, dict]:
    """Load full year data, tag holidays, load movie info."""
    log("Loading full year data...")
    holiday_map = load_holiday_definitions()
    rows = load_conversations(FULL_YEAR_CSV)
    rows = tag_holiday(rows, holiday_map)
    seekers = [r for r in rows if r['is_seeker'] and r['raw_text'].strip()]
    log(f"Extracted {len(seekers)} user questions with text")

    log("Loading movie info...")
    movie_info = load_movie_info()
    return seekers, holiday_map, movie_info


# ── 2. Run sentiment analysis (with caching) ──────────────────────────
SENTIMENT_CACHE = {}  # text_hash -> result


def get_sentiment_for_texts(texts: list[str]) -> list[dict]:
    """Run sentiment analysis with simple text caching."""
    uncached = []
    uncached_idx = []
    results = [None] * len(texts)

    for i, t in enumerate(texts):
        h = hash(t)
        if h in SENTIMENT_CACHE:
            results[i] = SENTIMENT_CACHE[h]
        else:
            uncached.append(t)
            uncached_idx.append(i)

    if uncached:
        log(f"Running sentiment analysis on {len(uncached)} uncached texts...")
        batch_results = analyze_batch(uncached)
        for idx, res in zip(uncached_idx, batch_results):
            h = hash(texts[idx])
            SENTIMENT_CACHE[h] = res
            results[idx] = res

    return results


# ── 3. Helper: get baseline date (same weekday before) ────────────────
def get_baseline_date(holiday_date: str, holiday_map: dict) -> str:
    """
    Get the same weekday 7 days before a holiday.
    If that day is also a holiday, go back another 7 days.
    Returns date string or empty string if not found.
    """
    dt = datetime.strptime(holiday_date, '%Y-%m-%d')
    for weeks_back in range(1, 6):  # up to 5 weeks back
        candidate = (dt - timedelta(weeks=weeks_back)).strftime('%Y-%m-%d')
        if candidate not in holiday_map:
            return candidate
    return ''


# ── 4. Get baseline text set from full data ───────────────────────────
def get_questions_for_date(seekers: list[dict], date_str: str) -> list[dict]:
    """Get all user questions for a specific date."""
    return [r for r in seekers if r['date'] == date_str]


# ═══════════════════════════════════════════════════════════════════════
#  Dimension 6: Holiday vs Baseline Comparison
# ═══════════════════════════════════════════════════════════════════════
def dim6_holiday_vs_baseline(seekers: list[dict], holiday_map: dict, movie_info: dict):
    """Compare sentiment, intensity, and genre for each holiday vs its baseline."""
    log("=" * 50)
    log("Dimension 6: Holiday vs Baseline Comparison")
    log("Running sentiment analysis (this may take a moment)...")

    # Build date -> questions lookup (for fast retrieval)
    questions_by_date = defaultdict(list)
    for r in seekers:
        questions_by_date[r['date']].append(r)

    # Process each holiday with sufficient data
    holiday_dates = sorted(set(r['date'] for r in seekers if r['is_holiday']))
    holiday_dates = [d for d in holiday_dates
                     if len(questions_by_date.get(d, [])) >= MIN_DATA_ROWS]

    all_results = []  # for aggregation in dimension 7
    baseline_results_list = []

    for h_date in holiday_dates:
        h_name = holiday_map[h_date]['description']
        h_type = holiday_map[h_date]['type']
        h_rows = questions_by_date[h_date]

        # Baseline
        b_date = get_baseline_date(h_date, holiday_map)
        if not b_date or b_date not in questions_by_date:
            log(f"  {h_name}: no baseline found, skipping")
            continue
        b_rows = questions_by_date[b_date]

        # Sentiment on holiday questions
        h_texts = [r['raw_text'] for r in h_rows]
        h_sentiments = get_sentiment_for_texts(h_texts)
        for r, s in zip(h_rows, h_sentiments):
            r['_sentiment'] = s['sentiment']
            r['_intensity'] = s['intensity']
            r['_sentiment_score'] = s['score']

        # Sentiment on baseline questions
        b_texts = [r['raw_text'] for r in b_rows]
        b_sentiments = get_sentiment_for_texts(b_texts)
        for r, s in zip(b_rows, b_sentiments):
            r['_sentiment'] = s['sentiment']
            r['_intensity'] = s['intensity']
            r['_sentiment_score'] = s['score']

        # Genres for holiday
        h_all_ids = []
        for r in h_rows:
            h_all_ids.extend(extract_imdb_ids(r.get('processed_raw', '')))
        h_genre_counts = lookup_genre_counts(h_all_ids, movie_info)

        # Genres for baseline
        b_all_ids = []
        for r in b_rows:
            b_all_ids.extend(extract_imdb_ids(r.get('processed_raw', '')))
        b_genre_counts = lookup_genre_counts(b_all_ids, movie_info)

        # Collect
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
        })

    log(f"Analyzed {len(all_results)} holidays with baselines")

    # ── 6a: Sentiment type comparison ──
    _plot_sentiment_comparison(all_results)
    # ── 6b: Sentiment intensity comparison ──
    _plot_intensity_comparison(all_results)
    # ── 6c: Genre comparison ──
    _plot_genre_comparison(all_results)

    # ── Save detailed CSV ──
    csv_path = os.path.join(STEP_OUT, 'd6_holiday_vs_baseline.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'date', 'type', 'n_questions',
                     'sentiment_positive_pct', 'sentiment_neutral_pct',
                     'sentiment_negative_pct',
                     'intensity_mild_pct', 'intensity_moderate_pct',
                     'intensity_strong_pct',
                     'baseline_date', 'baseline_n',
                     'baseline_positive_pct', 'baseline_neutral_pct',
                     'baseline_negative_pct',
                     'baseline_mild_pct', 'baseline_moderate_pct',
                     'baseline_strong_pct'])
        for r in all_results:
            b = r['baseline']
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
            ])
    log(f"Saved: {csv_path}")

    return all_results


def _plot_sentiment_comparison(all_results: list[dict]):
    """6a: Grouped bar chart - sentiment type holiday vs baseline."""
    names = [r['name'][:6] for r in all_results]

    # Holiday sentiment
    h_pos = [sum(1 for s in r['sentiments'] if s == 'positive') / max(r['n'], 1) * 100
             for r in all_results]
    h_neu = [sum(1 for s in r['sentiments'] if s == 'neutral') / max(r['n'], 1) * 100
             for r in all_results]
    h_neg = [sum(1 for s in r['sentiments'] if s == 'negative') / max(r['n'], 1) * 100
             for r in all_results]

    # Baseline sentiment
    b_pos = [sum(1 for s in r['baseline']['sentiments'] if s == 'positive')
             / max(r['baseline']['n'], 1) * 100 for r in all_results]
    b_neu = [sum(1 for s in r['baseline']['sentiments'] if s == 'neutral')
             / max(r['baseline']['n'], 1) * 100 for r in all_results]
    b_neg = [sum(1 for s in r['baseline']['sentiments'] if s == 'negative')
             / max(r['baseline']['n'], 1) * 100 for r in all_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10))
    x = np.arange(len(names))
    width = 0.25

    # Holiday chart
    ax1.bar(x - width, h_pos, width, label='Positive', color='#2ecc71', alpha=0.8)
    ax1.bar(x, h_neu, width, label='Neutral', color='#95a5a6', alpha=0.8)
    ax1.bar(x + width, h_neg, width, label='Negative', color='#e74c3c', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Proportion (%)')
    ax1.set_title('Holiday Sentiment Distribution')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Baseline chart
    ax2.bar(x - width, b_pos, width, label='Positive', color='#2ecc71', alpha=0.8)
    ax2.bar(x, b_neu, width, label='Neutral', color='#95a5a6', alpha=0.8)
    ax2.bar(x + width, b_neg, width, label='Negative', color='#e74c3c', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Proportion (%)')
    ax2.set_title('Baseline (Week Before) Sentiment Distribution')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd6_sentiment_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_intensity_comparison(all_results: list[dict]):
    """6b: Stacked bar - intensity holiday vs baseline side by side."""
    names = [r['name'][:12] for r in all_results]

    fig, ax = plt.subplots(figsize=(24, 6))
    x = np.arange(len(names))
    width = 0.35

    for i, level in enumerate(['mild', 'moderate', 'strong']):
        # Holiday
        h_vals = [sum(1 for s in r['intensities'] if s == level) / max(r['n'], 1) * 100
                  for r in all_results]
        # Baseline
        b_vals = [sum(1 for s in r['baseline']['intensities'] if s == level)
                  / max(r['baseline']['n'], 1) * 100 for r in all_results]

        offset = (i - 1) * 0.08
        colors = {'mild': '#3498db', 'moderate': '#f39c12', 'strong': '#e74c3c'}
        ax.scatter(x - width / 2 + offset, h_vals, marker='o', s=60,
                   color=colors[level], alpha=0.7,
                   label=f'{level.capitalize()} (Holiday)' if i == 0 else '')
        ax.scatter(x + width / 2 + offset, b_vals, marker='x', s=60,
                   color=colors[level], alpha=0.7,
                   label=f'{level.capitalize()} (Baseline)' if i == 0 else '')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Intensity: Holiday vs Baseline (o=holiday, x=baseline)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd6_intensity_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_genre_comparison(all_results: list[dict]):
    """6c: Heatmap - top genres for holidays vs baselines."""
    # Aggregate genre counts
    all_genres = Counter()
    for r in all_results:
        all_genres.update(r['genre_counts'])
        all_genres.update(r['baseline']['genre_counts'])

    top_genres = [g for g, _ in all_genres.most_common(12)]

    if not top_genres:
        log("WARN: No genre data for heatmap")
        return

    # Build matrix: rows=holidays, cols=genres, val=ratio diff (holiday - baseline)
    matrix = np.zeros((len(all_results), len(top_genres)))
    for i, r in enumerate(all_results):
        h_total = max(sum(r['genre_counts'].values()), 1)
        b_total = max(sum(r['baseline']['genre_counts'].values()), 1)
        for j, g in enumerate(top_genres):
            h_ratio = r['genre_counts'].get(g, 0) / h_total
            b_ratio = r['baseline']['genre_counts'].get(g, 0) / b_total
            matrix[i, j] = h_ratio - b_ratio  # positive = more on holiday

    fig, ax = plt.subplots(figsize=(14, max(6, len(all_results) * 0.5 + 2)))
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)

    ax.set_xticks(range(len(top_genres)))
    ax.set_xticklabels(top_genres, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(all_results)))
    ax.set_yticklabels([r['name'][:12] for r in all_results], fontsize=8)

    # Add text in cells
    for i in range(len(all_results)):
        for j in range(len(top_genres)):
            val = matrix[i, j]
            if abs(val) > 0.01:
                color = 'white' if abs(val) > 0.08 else 'black'
                ax.text(j, i, f'{val:+.0%}', ha='center', va='center',
                        fontsize=7, color=color)

    ax.set_title('Genre Proportion Difference: Holiday - Baseline (red=more on holiday)')
    fig.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd6_genre_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Dimension 7: Cross-Holiday Comparison
# ═══════════════════════════════════════════════════════════════════════
def dim7_cross_holiday(all_results: list[dict]):
    """Compare question volume, sentiment, and genres across holidays."""
    log("-" * 50)
    log("Dimension 7: Cross-Holiday Comparison")

    if not all_results:
        log("WARN: No holiday data from dimension 6")
        return

    # ── 7a: Question volume comparison (bar, sorted) ──
    sorted_by_vol = sorted(all_results, key=lambda r: r['n'], reverse=True)
    names = [r['name'][:6] for r in sorted_by_vol]
    volumes = [r['n'] for r in sorted_by_vol]

    fig, ax = plt.subplots(figsize=(24, 5))
    colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    ax.bar(range(len(names)), volumes, color=colors_bar, alpha=0.8, edgecolor='grey')
    for i, v in enumerate(volumes):
        ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Number of Questions')
    ax.set_title('Cross-Holiday Question Volume (sorted)')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd7_question_volume.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── 7b: Sentiment stacked bar ──
    fig, ax = plt.subplots(figsize=(24, 6))
    x = np.arange(len(all_results))
    pos_vals = [sum(1 for s in r['sentiments'] if s == 'positive') / max(r['n'], 1) * 100
                for r in all_results]
    neu_vals = [sum(1 for s in r['sentiments'] if s == 'neutral') / max(r['n'], 1) * 100
                for r in all_results]
    neg_vals = [sum(1 for s in r['sentiments'] if s == 'negative') / max(r['n'], 1) * 100
                for r in all_results]

    ax.bar(x, pos_vals, label='Positive', color='#2ecc71', alpha=0.8)
    ax.bar(x, neu_vals, bottom=pos_vals, label='Neutral', color='#95a5a6', alpha=0.8)
    bottom2 = [p + n for p, n in zip(pos_vals, neu_vals)]
    ax.bar(x, neg_vals, bottom=bottom2, label='Negative', color='#e74c3c', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([r['name'][:6] for r in all_results], rotation=45, ha='right')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Distribution Across Holidays')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd7_sentiment_by_holiday.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── 7c: Intensity grouped bars ──
    fig, ax = plt.subplots(figsize=(24, 5))
    x = np.arange(len(all_results))
    width = 0.25

    for i, level in enumerate(['mild', 'moderate', 'strong']):
        vals = [sum(1 for s in r['intensities'] if s == level) / max(r['n'], 1) * 100
                for r in all_results]
        colors = {'mild': '#3498db', 'moderate': '#f39c12', 'strong': '#e74c3c'}
        ax.bar(x + (i - 1) * width, vals, width, label=level.capitalize(),
               color=colors[level], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([r['name'][:6] for r in all_results], rotation=45, ha='right')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Intensity Across Holidays')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd7_intensity_by_holiday.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── 7d: Holiday type aggregation ──
    _plot_holiday_type_aggregation(all_results)

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd7_cross_holiday_stats.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'date', 'type', 'n_questions',
                     'positive_pct', 'neutral_pct', 'negative_pct',
                     'mild_pct', 'moderate_pct', 'strong_pct',
                     'avg_sentiment_score'])
        for r in all_results:
            avg_score = np.mean(r['scores']) if r['scores'] else 0
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
                f"{avg_score:.3f}",
            ])
    log(f"Saved: {csv_path}")


def _plot_holiday_type_aggregation(all_results: list[dict]):
    """7d: Aggregate by holiday type (federal, traditional, religious, family)."""
    type_groups = defaultdict(list)
    for r in all_results:
        type_groups[r['type']].append(r)

    type_names = sorted(type_groups.keys())
    if len(type_names) < 2:
        log("  Skipping holiday type aggregation: only 1 type")
        return

    # Aggregate sentiment
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
    # Get top genres across all
    all_genre_counts = Counter()
    for r in all_results:
        all_genre_counts.update(r['genre_counts'])
    top_g = [g for g, _ in all_genre_counts.most_common(8)]

    if top_g:
        angles = np.linspace(0, 2 * np.pi, len(top_g), endpoint=False).tolist()
        angles += angles[:1]

        for tn in type_names:
            # Aggregate genre counts for this type
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
    log("Step 4: Holiday Comparison Analysis (Dimensions 6 + 7)")
    log("=" * 60)

    seekers, holiday_map, movie_info = load_and_prepare()

    # Dimension 6
    log("")
    all_results = dim6_holiday_vs_baseline(seekers, holiday_map, movie_info)

    # Dimension 7 (requires dim6 results)
    log("")
    dim7_cross_holiday(all_results)

    log("")
    log("=" * 60)
    log(f"Step 4 complete! Results saved to {STEP_OUT}")
    log("=" * 60)

    # Print summary
    if all_results:
        print(f"\n{'Holiday':<20} {'n':>6} {'Pos':>7} {'Neu':>7} {'Neg':>7} "
              f"{'Mild':>7} {'Mod':>7} {'Strong':>7} {'Score':>7}")
        print("-" * 75)
        for r in all_results:
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
