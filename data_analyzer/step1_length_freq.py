# -*- coding: utf-8 -*-
"""
Step 1: Question Length & Access Frequency Analysis
  - Dimension 1a: Holiday vs Non-holiday question word length
  - Dimension 1b: Per-holiday vs yearly average question length
  - Dimension 2:  Access frequency comparison

Output: output/step1/*.png + stats CSV
"""

import os
import csv
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data_analyzer.config import (
    OUTPUT_DIR, MIN_DATA_ROWS, LONG_QUESTION_WORD_THRESHOLD,
    FULL_YEAR_CSV, HOLIDAY_CSV, setup_matplotlib, log,
)
from data_analyzer.data_loader import (
    load_conversations, load_holiday_definitions, tag_holiday,
    clean_word_count, parse_processed_text,
)

setup_matplotlib()
STEP_OUT = os.path.join(OUTPUT_DIR, 'step1')
os.makedirs(STEP_OUT, exist_ok=True)


# ── Load & prep data ─────────────────────────────────────────────────
def load_and_prep() -> tuple[list[dict], dict]:
    """Load full year data, tag holidays, return (rows, holiday_map)."""
    log("Loading full year data...")
    holiday_map = load_holiday_definitions()
    rows = load_conversations(FULL_YEAR_CSV)
    rows = tag_holiday(rows, holiday_map)
    # Keep only user questions
    seekers = [r for r in rows if r['is_seeker']]
    log(f"Extracted {len(seekers)} user questions")
    return seekers, holiday_map


# ═══════════════════════════════════════════════════════════════════════
#  Dimension 1a: Holiday vs Non-holiday Question Length
# ═══════════════════════════════════════════════════════════════════════
def dim1a_holiday_vs_nonholiday(seekers: list[dict]):
    """Boxplot + long-ratio bar chart comparing word counts."""
    log("=" * 50)
    log("Dimension 1a: Holiday vs Non-holiday Question Length")

    holiday_words = [r['word_count'] for r in seekers if r['is_holiday']]
    nonholiday_words = [r['word_count'] for r in seekers if not r['is_holiday']]

    if not holiday_words or not nonholiday_words:
        log("WARN: Not enough data for D1a")
        return

    # Stats
    def stats(arr):
        return {
            'count': len(arr), 'mean': np.mean(arr), 'median': np.median(arr),
            'q1': np.percentile(arr, 25), 'q3': np.percentile(arr, 75),
            'std': np.std(arr),
        }

    s_h = stats(holiday_words)
    s_n = stats(nonholiday_words)
    log(f"Holiday: mean={s_h['mean']:.1f}, median={s_h['median']:.1f}, "
        f"Q1={s_h['q1']:.1f}, Q3={s_h['q3']:.1f} (n={s_h['count']})")
    log(f"Non-holiday: mean={s_n['mean']:.1f}, median={s_n['median']:.1f}, "
        f"Q1={s_n['q1']:.1f}, Q3={s_n['q3']:.1f} (n={s_n['count']})")

    # ── Figure 1a-1: Boxplot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([holiday_words, nonholiday_words],
                    patch_artist=True, widths=0.5,
                    medianprops={'color': 'white', 'linewidth': 2})
    colors = ['#ff9999', '#66b3ff']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xticklabels([f'Holiday\n(n={s_h["count"]})',
                        f'Non-holiday\n(n={s_n["count"]})'])
    ax.set_ylabel('Word Count')
    ax.set_title('Question Length: Holiday vs Non-holiday')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Add mean markers
    for i, mean_val in enumerate([s_h['mean'], s_n['mean']], 1):
        ax.plot(i, mean_val, 'D', color='darkred', markersize=6, zorder=5)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd1a_word_count_boxplot.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── Figure 1a-2: Long question ratio ──
    h_long = sum(1 for w in holiday_words if w >= LONG_QUESTION_WORD_THRESHOLD)
    n_long = sum(1 for w in nonholiday_words if w >= LONG_QUESTION_WORD_THRESHOLD)

    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ['Holiday', 'Non-holiday']
    long_ratios = [h_long / len(holiday_words) * 100,
                   n_long / len(nonholiday_words) * 100]
    short_ratios = [100 - r for r in long_ratios]

    ax.bar(categories, short_ratios, label=f'Short (<{LONG_QUESTION_WORD_THRESHOLD} words)',
           color='#66b3ff', alpha=0.8)
    ax.bar(categories, long_ratios, bottom=short_ratios,
           label=f'Long (≥{LONG_QUESTION_WORD_THRESHOLD} words)',
           color='#ff9999', alpha=0.8)

    for i, (l, s) in enumerate(zip(long_ratios, short_ratios)):
        ax.text(i, s / 2, f'{s:.1f}%', ha='center', va='center', fontsize=11)
        ax.text(i, s + l / 2, f'{l:.1f}%', ha='center', va='center', fontsize=11)

    ax.set_ylabel('Proportion (%)')
    ax.set_title(f'Long Question Ratio (threshold={LONG_QUESTION_WORD_THRESHOLD} words)')
    ax.legend()

    fig.tight_layout()
    path2 = os.path.join(STEP_OUT, 'd1a_long_question_ratio.png')
    fig.savefig(path2)
    plt.close(fig)
    log(f"Saved: {path2}")

    # ── Save stats CSV ──
    csv_path = os.path.join(STEP_OUT, 'd1a_word_length_stats.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'count', 'mean', 'median', 'q1', 'q3', 'std',
                     f'long_ratio(≥{LONG_QUESTION_WORD_THRESHOLD}words)'])
        w.writerow(['holiday', s_h['count'], f"{s_h['mean']:.2f}",
                     f"{s_h['median']:.1f}", f"{s_h['q1']:.1f}", f"{s_h['q3']:.1f}",
                     f"{s_h['std']:.2f}", f"{h_long/len(holiday_words)*100:.2f}%"])
        w.writerow(['non_holiday', s_n['count'], f"{s_n['mean']:.2f}",
                     f"{s_n['median']:.1f}", f"{s_n['q1']:.1f}", f"{s_n['q3']:.1f}",
                     f"{s_n['std']:.2f}", f"{n_long/len(nonholiday_words)*100:.2f}%"])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Dimension 1b: Per-holiday vs Yearly Average Length
# ═══════════════════════════════════════════════════════════════════════
def dim1b_per_holiday_length(seekers: list[dict]):
    """Bar chart: each holiday's avg word count vs yearly non-holiday baseline."""
    log("-" * 50)
    log("Dimension 1b: Per-holiday Question Length")

    # Yearly non-holiday average
    non_h_words = [r['word_count'] for r in seekers if not r['is_holiday']]
    yearly_avg = np.mean(non_h_words) if non_h_words else 0

    # Per holiday averages
    holiday_data = defaultdict(list)
    for r in seekers:
        if r['is_holiday']:
            holiday_data[r['holiday_name']].append(r['word_count'])

    # Sort by question count descending, filter low-data
    sorted_holidays = sorted(holiday_data.items(),
                             key=lambda x: len(x[1]), reverse=True)
    sorted_holidays = [(name, words) for name, words in sorted_holidays
                       if len(words) >= MIN_DATA_ROWS]

    if not sorted_holidays:
        log("WARN: No holidays with sufficient data")
        return

    names = [h[0] for h in sorted_holidays]
    avgs = [np.mean(h[1]) for h in sorted_holidays]
    counts = [len(h[1]) for h in sorted_holidays]

    log(f"Yearly non-holiday average: {yearly_avg:.1f} words")
    for n, a, c in zip(names, avgs, counts):
        diff = a - yearly_avg
        arrow = "▲" if diff > 0 else "▼"
        log(f"  {n}: {a:.1f} (n={c}) {arrow}{abs(diff):.1f} vs baseline")

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    colors_bar = ['#ff6b6b' if v > yearly_avg else '#74b9ff' for v in avgs]
    bars = ax.bar(x, avgs, color=colors_bar, alpha=0.8, edgecolor='grey', linewidth=0.5)

    # Baseline
    ax.axhline(y=yearly_avg, color='red', linestyle='--', linewidth=1.5,
               label=f'Non-holiday avg ({yearly_avg:.1f})')

    # Labels
    for i, (v, c) in enumerate(zip(avgs, counts)):
        ax.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i, 0.5, f'n={c}', ha='center', va='bottom', fontsize=7, rotation=90,
                color='grey', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Avg Word Count')
    ax.set_title('Per-Holiday Average Question Length vs Non-Holiday Baseline')
    ax.legend()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd1b_per_holiday_length.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd1b_per_holiday_length.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'question_count', 'avg_word_count',
                     'yearly_non_holiday_avg', 'diff_from_baseline'])
        for n, a, c in zip(names, avgs, counts):
            w.writerow([n, c, f"{a:.2f}", f"{yearly_avg:.2f}",
                        f"{a - yearly_avg:+.2f}"])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Dimension 2: Access Frequency Comparison
# ═══════════════════════════════════════════════════════════════════════
def dim2_access_frequency(seekers: list[dict], holiday_map: dict):
    """
    For each holiday: compare daily question count against:
      - Yearly non-holiday daily average
      - Same weekday one week before
      - Same weekday one month before (4 weeks before)
    """
    log("-" * 50)
    log("Dimension 2: Access Frequency Comparison")

    # Daily question counts
    daily_counts = Counter()
    for r in seekers:
        daily_counts[r['date']] += 1

    # Non-holiday daily average (include zero-count days for fair baseline)
    # Compute the full date range from the actual data instead of hardcoding a year
    from datetime import datetime, timedelta
    all_dates_data = sorted(set(r['date'] for r in seekers))
    if all_dates_data:
        data_start = datetime.strptime(all_dates_data[0], '%Y-%m-%d')
        data_end = datetime.strptime(all_dates_data[-1], '%Y-%m-%d')
    else:
        log("WARN: No date data available")
        return

    all_possible_dates = set()
    cursor = data_start
    while cursor <= data_end:
        all_possible_dates.add(cursor.strftime('%Y-%m-%d'))
        cursor += timedelta(days=1)

    non_h_counts_filled = []
    for d in sorted(all_possible_dates):
        if d not in holiday_map:
            non_h_counts_filled.append(daily_counts.get(d, 0))
    baseline_avg = np.mean(non_h_counts_filled) if non_h_counts_filled else 0
    log(f"Non-holiday daily avg: {baseline_avg:.1f} questions "
        f"(across {len(non_h_counts_filled)} non-holiday days, "
        f"span {data_start.date()} ~ {data_end.date()})")

    # Per-holiday frequency data
    holiday_dates = sorted(set(r['date'] for r in seekers if r['is_holiday']))
    freq_data = []
    for d in holiday_dates:
        cnt = daily_counts.get(d, 0)
        if cnt < MIN_DATA_ROWS:
            continue

        # Find same weekday 7 days before (non-holiday preferred)
        dt = datetime.strptime(d, '%Y-%m-%d')
        week_before = (dt - timedelta(days=7)).strftime('%Y-%m-%d')
        # If week_before is also a holiday, go another week back
        while week_before in holiday_map:
            week_before = (datetime.strptime(week_before, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
        wb_cnt = daily_counts.get(week_before, 0)

        # Same weekday ~4 weeks before (28 days)
        month_before = (dt - timedelta(days=28)).strftime('%Y-%m-%d')
        while month_before in holiday_map:
            month_before = (datetime.strptime(month_before, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
        mb_cnt = daily_counts.get(month_before, 0)

        holiday_name = holiday_map[d]['description']
        freq_data.append({
            'date': d, 'name': holiday_name,
            'holiday_cnt': cnt,
            'week_before_cnt': wb_cnt,
            'month_before_cnt': mb_cnt,
            'yearly_avg': baseline_avg,
        })

    if not freq_data:
        log("WARN: No frequency data")
        return

    log(f"Total holidays analyzed: {len(freq_data)}")

    # ── Figure: Grouped bar chart ──
    fig, ax = plt.subplots(figsize=(20, 6))
    x = np.arange(len(freq_data))
    width = 0.2

    holiday_cnts = [d['holiday_cnt'] for d in freq_data]
    wb_cnts = [d['week_before_cnt'] for d in freq_data]
    mb_cnts = [d['month_before_cnt'] for d in freq_data]
    yearly_avgs = [d['yearly_avg'] for d in freq_data]

    ax.bar(x - 1.5 * width, holiday_cnts, width, label='Holiday', color='#ff6b6b', alpha=0.85)
    ax.bar(x - 0.5 * width, wb_cnts, width, label='Week Before', color='#feca57', alpha=0.85)
    ax.bar(x + 0.5 * width, mb_cnts, width, label='Month Before', color='#48dbfb', alpha=0.85)
    ax.bar(x + 1.5 * width, yearly_avgs, width, label='Non-Holiday Avg', color='#636e72', alpha=0.7)

    ax.set_xticks(x)
    names_short = [d['name'][:6] for d in freq_data]
    ax.set_xticklabels(names_short, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Number of Questions')
    ax.set_title('Access Frequency: Holiday vs Baseline Periods')
    ax.legend(fontsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd2_access_frequency.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd2_access_frequency.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'date', 'holiday_count', 'week_before_count',
                     'month_before_count', 'non_holiday_daily_avg'])
        for d in freq_data:
            w.writerow([d['name'], d['date'], d['holiday_cnt'],
                        d['week_before_cnt'], d['month_before_cnt'],
                        f"{d['yearly_avg']:.1f}"])
    log(f"Saved: {csv_path}")

    # ── Print table ──
    print(f"\n{'Holiday':<20} {'Holiday':>8} {'Week-7':>8} {'Month-28':>8} {'YearAvg':>8}")
    print("-" * 55)
    for d in freq_data:
        print(f"{d['name']:<20} {d['holiday_cnt']:>8} {d['week_before_cnt']:>8} "
              f"{d['month_before_cnt']:>8} {d['yearly_avg']:>8.1f}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 60)
    log("Step 1: Question Length & Access Frequency Analysis")
    log("=" * 60)

    seekers, holiday_map = load_and_prep()

    # Dimension 1a
    log("")
    dim1a_holiday_vs_nonholiday(seekers)

    # Dimension 1b
    log("")
    dim1b_per_holiday_length(seekers)

    # Dimension 2
    log("")
    dim2_access_frequency(seekers, holiday_map)

    log("")
    log("=" * 60)
    log(f"Step 1 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
