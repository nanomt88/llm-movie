# -*- coding: utf-8 -*-
"""
Step 2: Time Distribution & Conversation Turn Analysis
  - Dimension 3: Intra-day time distribution (holiday vs non-holiday)
  - Dimension 4: Conversation turn count distribution

Output: output/step2/*.png + CSV
"""

import os
import csv
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from his.data_analyzer.config import (
    OUTPUT_DIR, FULL_YEAR_CSV,
    setup_matplotlib, log,
)
from his.data_analyzer.data_loader import (
    load_conversations, load_holiday_definitions, tag_holiday,
)

setup_matplotlib()
STEP_OUT = os.path.join(OUTPUT_DIR, 'step2')
os.makedirs(STEP_OUT, exist_ok=True)


# ── Load & prep ───────────────────────────────────────────────────────
def load_and_prep() -> tuple[list[dict], list[dict]]:
    """Load full year data, split into holiday/non-holiday seekers + all_rows."""
    log("Loading full year data...")
    holiday_map = load_holiday_definitions()
    rows = load_conversations(FULL_YEAR_CSV)
    rows = tag_holiday(rows, holiday_map)
    seekers = [r for r in rows if r['is_seeker']]
    log(f"Extracted {len(seekers)} user questions")
    return rows, seekers


# ═══════════════════════════════════════════════════════════════════════
#  Dimension 3: Intra-day Time Distribution
# ═══════════════════════════════════════════════════════════════════════
def dim3_time_distribution(seekers: list[dict]):
    """Overlapping density plot of question hour distribution."""
    log("=" * 50)
    log("Dimension 3: Intra-day Time Distribution")

    holiday_hours = [r['hour'] for r in seekers if r['is_holiday']]
    nonholiday_hours = [r['hour'] for r in seekers if not r['is_holiday']]

    if not holiday_hours or not nonholiday_hours:
        log("WARN: Not enough data")
        return

    log(f"Holiday questions: {len(holiday_hours)}, Non-holiday: {len(nonholiday_hours)}")

    # ── Figure: Overlapping histogram/density ──
    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram with density normalization
    bins = np.arange(0, 25, 1)  # 0..24
    ax.hist(holiday_hours, bins=bins, density=True, alpha=0.6,
            color='#ff6b6b', label=f'Holiday (n={len(holiday_hours)})',
            edgecolor='white', linewidth=0.5)
    ax.hist(nonholiday_hours, bins=bins, density=True, alpha=0.5,
            color='#74b9ff', label=f'Non-holiday (n={len(nonholiday_hours)})',
            edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Hour of Day (UTC)')
    ax.set_ylabel('Density')
    ax.set_title('Question Time Distribution: Holiday vs Non-holiday')
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd3_time_distribution.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── Figure 2: Stacked comparison (side-by-side) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    h_counts = Counter(holiday_hours)
    n_counts = Counter(nonholiday_hours)

    h_pct = [h_counts.get(h, 0) / len(holiday_hours) * 100 for h in range(24)]
    n_pct = [n_counts.get(h, 0) / len(nonholiday_hours) * 100 for h in range(24)]

    hours = list(range(24))
    ax1.bar(hours, h_pct, color='#ff6b6b', alpha=0.8, edgecolor='white')
    ax1.set_title('Holiday')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Proportion (%)')
    ax1.set_xticks(range(0, 24, 3))

    ax2.bar(hours, n_pct, color='#74b9ff', alpha=0.8, edgecolor='white')
    ax2.set_title('Non-holiday')
    ax2.set_xlabel('Hour')
    ax2.set_xticks(range(0, 24, 3))

    fig.suptitle('Question Hour Distribution (%)', fontsize=14)
    fig.tight_layout()
    path2 = os.path.join(STEP_OUT, 'd3_time_side_by_side.png')
    fig.savefig(path2)
    plt.close(fig)
    log(f"Saved: {path2}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd3_hourly_distribution.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['hour', 'holiday_count', 'holiday_pct',
                     'non_holiday_count', 'non_holiday_pct'])
        for h in range(24):
            w.writerow([h, h_counts.get(h, 0),
                        f"{h_pct[h]:.2f}%",
                        n_counts.get(h, 0),
                        f"{n_pct[h]:.2f}%"])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Dimension 4: Conversation Turn Analysis
# ═══════════════════════════════════════════════════════════════════════
def dim4_turn_analysis(rows: list[dict]):
    """
    Analyze conversation depth distribution.
    Turn count = number of rows per conv_id base.
    Buckets based on observed distribution:
      - 1 round (1-2 rows)
      - 2 rounds (3-4 rows)
      - 3-4 rounds (5-8 rows)
      - 5+ rounds (>=9 rows)
    """
    log("-" * 50)
    log("Dimension 4: Conversation Turn Analysis")

    # Group by conversation base ID
    conv_turns = Counter()
    conv_holiday = {}  # conv_base -> is_holiday (if any user Q is on holiday)
    for r in rows:
        base = r['conv_id'].rsplit('/', 1)[0]
        conv_turns[base] += 1
        if r['is_seeker'] and r['is_holiday']:
            conv_holiday[base] = True
        elif r['is_seeker'] and base not in conv_holiday:
            conv_holiday[base] = False

    total_conv = len(conv_turns)
    log(f"Total conversations: {total_conv}")

    # Bucket definition (based on rows count: 1 user + 1 system = 2 rows for 1 round)
    bucket_defs = [
        ('1 round (1-2 rows)', lambda t: t <= 2),
        ('2 rounds (3-4 rows)', lambda t: 3 <= t <= 4),
        ('3-4 rounds (5-8 rows)', lambda t: 5 <= t <= 8),
        ('5+ rounds (>=9 rows)', lambda t: t >= 9),
    ]

    # Overall distribution
    dist = np.zeros(len(bucket_defs), dtype=int)
    for t in conv_turns.values():
        for i, (_, fn) in enumerate(bucket_defs):
            if fn(t):
                dist[i] += 1
                break

    log("Turn distribution (all):")
    for (name, _), cnt in zip(bucket_defs, dist):
        log(f"  {name}: {cnt} ({cnt / total_conv * 100:.1f}%)")

    # Holiday vs non-holiday split
    holiday_turns = {}
    non_holiday_turns = {}
    for base, t in conv_turns.items():
        is_h = conv_holiday.get(base, False)
        target = holiday_turns if is_h else non_holiday_turns
        target[base] = t

    h_dist = np.zeros(len(bucket_defs), dtype=int)
    n_dist = np.zeros(len(bucket_defs), dtype=int)
    for t in holiday_turns.values():
        for i, (_, fn) in enumerate(bucket_defs):
            if fn(t):
                h_dist[i] += 1
                break
    for t in non_holiday_turns.values():
        for i, (_, fn) in enumerate(bucket_defs):
            if fn(t):
                n_dist[i] += 1
                break

    h_total = len(holiday_turns)
    n_total = len(non_holiday_turns)

    # ── Figure 1: Stacked bar ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bucket_defs))
    width = 0.3

    h_pcts = [h_dist[i] / max(h_total, 1) * 100 for i in range(len(bucket_defs))]
    n_pcts = [n_dist[i] / max(n_total, 1) * 100 for i in range(len(bucket_defs))]

    bars1 = ax.bar(x - width / 2, h_pcts, width, label=f'Holiday (n={h_total})',
                   color='#ff6b6b', alpha=0.8)
    bars2 = ax.bar(x + width / 2, n_pcts, width, label=f'Non-holiday (n={n_total})',
                   color='#74b9ff', alpha=0.8)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

    bucket_labels = [name for name, _ in bucket_defs]
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=9)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Conversation Depth: Holiday vs Non-holiday')
    ax.legend()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd4_turn_distribution.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── Figure 2: Pie charts side by side ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    ax1.pie(h_dist, labels=[n for n, _ in bucket_defs],
            autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax1.set_title(f'Holiday Conversations (n={h_total})')

    ax2.pie(n_dist, labels=[n for n, _ in bucket_defs],
            autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax2.set_title(f'Non-holiday Conversations (n={n_total})')

    fig.tight_layout()
    path2 = os.path.join(STEP_OUT, 'd4_turn_pie.png')
    fig.savefig(path2)
    plt.close(fig)
    log(f"Saved: {path2}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd4_turn_distribution.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket', 'holiday_count', 'holiday_pct',
                     'non_holiday_count', 'non_holiday_pct'])
        for i, (name, _) in enumerate(bucket_defs):
            w.writerow([
                name, int(h_dist[i]), f"{h_pcts[i]:.2f}%",
                int(n_dist[i]), f"{n_pcts[i]:.2f}%",
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 60)
    log("Step 2: Time Distribution & Conversation Turn Analysis")
    log("=" * 60)

    rows, seekers = load_and_prep()

    # Dimension 3
    log("")
    dim3_time_distribution(seekers)

    # Dimension 4
    log("")
    dim4_turn_analysis(rows)

    log("")
    log("=" * 60)
    log(f"Step 2 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
