# -*- coding: utf-8 -*-
"""
Step 3: Special Scenario Demand Analysis (Dimension 5)
Identify and compare viewing context scenarios (family, couple, alone, friends)
in holiday vs non-holiday conversations.

Keyword-based classification on English text from Reddit movie conversations.

Output: output/step3/*.png + CSV
"""

import os
import csv
import re
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_analyzer.config import (
    OUTPUT_DIR, MIN_DATA_ROWS, FULL_YEAR_CSV,
    setup_matplotlib, log,
)
from data_analyzer.data_loader import (
    load_conversations, load_holiday_definitions, tag_holiday,
    parse_processed_text,
)

setup_matplotlib()
STEP_OUT = os.path.join(OUTPUT_DIR, 'step3')
os.makedirs(STEP_OUT, exist_ok=True)


# ── Scenario keyword definitions ──────────────────────────────────────
# Each scenario: (name, regex_pattern)
# All patterns are case-insensitive, whole-word or phrase matching.
SCENARIOS = [
    ('Family Viewing', re.compile(
        r'(?i)\b(family|kids|children|parents|grandma|grandpa|grandparent|'
        r'whole\s*family|family\s*movie|with\s*my\s*kids|'
        r'for\s*the\s*kids|child|teenager|family-friendly|pg|everyone)\b'
    )),
    ('Couple Date', re.compile(
        r'(?i)\b(date\s*nite|date\s*night|romantic|romance|'
        r'couple|girlfriend|boyfriend|valentine|'
        r'with\s*my\s*girlfriend|with\s*my\s*boyfriend|'
        r'with\s*my\s*partner|anniversary|wedding|honeymoon)\b'
    )),
    ('Alone / Solo', re.compile(
        r'(?i)\b(alone|by\s*myself|solo|just\s*me|on\s*my\s*own|'
        r'lonely|single|nobody\s*to\s*watch|by\s*myself)\b'
    )),
    ('Friend Gathering', re.compile(
        r'(?i)\b(friends|hang\s*out|party|group|sleepover|'
        r'get\s*together|watch\s*party|with\s*friends|'
        r'for\s*a\s*group|group\s*of\s*friends|mates|buddies)\b'
    )),
    ('Weekend / Holiday Plan', re.compile(
        r'(?i)\b(this\s*weekend|long\s*weekend|for\s*the\s*holiday|'
        r'weekend\s*plan|on\s*my\s*day\s*off|staycation|'
        r'for\s*break|during\s*break|christmas\s*break)\b'
    )),
]


def classify_scenarios(text: str) -> list[str]:
    """
    Classify which scenarios a question text belongs to.
    Returns list of scenario names (may be empty, may have multiple).
    """
    if not text:
        return []
    matched = []
    for name, pattern in SCENARIOS:
        if pattern.search(text):
            matched.append(name)
    return matched


# ── Load & prep ───────────────────────────────────────────────────────
def load_and_prep() -> list[dict]:
    """Load full year data, tag holidays, keep only seekers with non-empty text."""
    log("Loading full year data...")
    holiday_map = load_holiday_definitions()
    rows = load_conversations(FULL_YEAR_CSV)
    rows = tag_holiday(rows, holiday_map)
    seekers = [r for r in rows if r['is_seeker'] and r['raw_text'].strip()]
    log(f"Extracted {len(seekers)} user questions with text")
    return seekers


# ═══════════════════════════════════════════════════════════════════════
#  Dimension 5: Special Scenario Analysis
# ═══════════════════════════════════════════════════════════════════════
def dim5_scenario_analysis(seekers: list[dict]):
    """Classify scenarios, compare holiday vs non-holiday proportions."""
    log("=" * 50)
    log("Dimension 5: Special Scenario Demand Analysis")

    # Classify all questions
    scenario_counts = defaultdict(lambda: {'holiday': 0, 'non_holiday': 0})
    scenario_users = defaultdict(lambda: {'holiday': set(), 'non_holiday': set()})
    total_h = 0
    total_n = 0

    for r in seekers:
        scenarios = classify_scenarios(r['raw_text'])
        if r['is_holiday']:
            total_h += 1
            for s in scenarios:
                scenario_counts[s]['holiday'] += 1
                scenario_users[s]['holiday'].add(r['user_id'])
        else:
            total_n += 1
            for s in scenarios:
                scenario_counts[s]['non_holiday'] += 1
                scenario_users[s]['non_holiday'].add(r['user_id'])

    log(f"Holiday questions: {total_h}, Non-holiday: {total_n}")

    # Print results
    scenario_names = [s[0] for s in SCENARIOS]
    print(f"\n{'Scenario':<25} {'Holiday':>10} {'H%':>8} {'Non-Holiday':>12} {'NH%':>8}")
    print("-" * 68)
    for name in scenario_names:
        h_cnt = scenario_counts[name]['holiday']
        n_cnt = scenario_counts[name]['non_holiday']
        h_pct = h_cnt / max(total_h, 1) * 100
        n_pct = n_cnt / max(total_n, 1) * 100
        h_users = len(scenario_users[name]['holiday'])
        n_users = len(scenario_users[name]['non_holiday'])
        print(f"{name:<25} {h_cnt:>6} ({h_pct:>5.1f}%) {n_cnt:>8} ({n_pct:>5.1f}%)  "
              f"[users: {h_users}/{n_users}]")
        log(f"  {name} - Holiday: {h_cnt} ({h_pct:.1f}%), "
            f"Non-holiday: {n_cnt} ({n_pct:.1f}%)")

    # Any match vs no match
    any_match_h = sum(1 for r in seekers if r['is_holiday'] and classify_scenarios(r['raw_text']))
    any_match_n = sum(1 for r in seekers if not r['is_holiday'] and classify_scenarios(r['raw_text']))
    log(f"\nAny scenario matched - Holiday: {any_match_h}/{total_h} "
        f"({any_match_h/max(total_h,1)*100:.1f}%)")
    log(f"Any scenario matched - Non-holiday: {any_match_n}/{total_n} "
        f"({any_match_n/max(total_n,1)*100:.1f}%)")

    # ── Figure 1: Grouped bar chart ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scenario_names))
    width = 0.3

    h_rates = [scenario_counts[n]['holiday'] / max(total_h, 1) * 100
               for n in scenario_names]
    n_rates = [scenario_counts[n]['non_holiday'] / max(total_n, 1) * 100
               for n in scenario_names]

    bars1 = ax.bar(x - width / 2, h_rates, width, label='Holiday',
                   color='#ff6b6b', alpha=0.8)
    bars2 = ax.bar(x + width / 2, n_rates, width, label='Non-holiday',
                   color='#74b9ff', alpha=0.8)

    # Labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=8, rotation=0)
    for bar in bars2:
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=9)
    ax.set_ylabel('Proportion of Questions (%)')
    ax.set_title('Special Viewing Scenarios: Holiday vs Non-holiday')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd5_scenario_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── Figure 2: Per-holiday scenario breakdown ──
    # (Only for top holidays with sufficient data)
    holiday_scenario = defaultdict(lambda: Counter())
    for r in seekers:
        if r['is_holiday']:
            scenarios = classify_scenarios(r['raw_text'])
            for s in scenarios:
                holiday_scenario[r['holiday_name']][s] += 1

    # Filter holidays with enough questions
    holiday_q_counts = Counter()
    for r in seekers:
        if r['is_holiday']:
            holiday_q_counts[r['holiday_name']] += 1

    top_holidays = [h for h, c in holiday_q_counts.most_common(8)
                    if c >= MIN_DATA_ROWS]

    if top_holidays:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(top_holidays))
        width = 0.15

        for i, s_name in enumerate(scenario_names):
            rates = [
                holiday_scenario[h].get(s_name, 0) / max(holiday_q_counts[h], 1) * 100
                for h in top_holidays
            ]
            pos = x + (i - len(scenario_names) / 2 + 0.5) * width
            ax.bar(pos, rates, width, label=s_name, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(top_holidays, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Proportion (%)')
        ax.set_title('Scenario Distribution Across Holidays')
        ax.legend(fontsize=8, loc='upper right')

        fig.tight_layout()
        path2 = os.path.join(STEP_OUT, 'd5_scenario_per_holiday.png')
        fig.savefig(path2)
        plt.close(fig)
        log(f"Saved: {path2}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd5_scenario_comparison.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'holiday_count', 'holiday_pct',
                     'non_holiday_count', 'non_holiday_pct',
                     'holiday_unique_users', 'non_holiday_unique_users'])
        for name in scenario_names:
            w.writerow([
                name,
                scenario_counts[name]['holiday'],
                f"{scenario_counts[name]['holiday']/max(total_h,1)*100:.2f}%",
                scenario_counts[name]['non_holiday'],
                f"{scenario_counts[name]['non_holiday']/max(total_n,1)*100:.2f}%",
                len(scenario_users[name]['holiday']),
                len(scenario_users[name]['non_holiday']),
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    log("=" * 60)
    log("Step 3: Special Scenario Demand Analysis")
    log("=" * 60)

    seekers = load_and_prep()

    log("")
    dim5_scenario_analysis(seekers)

    log("")
    log("=" * 60)
    log(f"Step 3 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
