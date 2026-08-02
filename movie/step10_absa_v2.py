# -*- coding: utf-8 -*-
"""
Step 10 ABSA Phase 1 (v2): 语料切换 + 去噪 + 会话级聚合.
B: 分析系统回复（非用户提问）；③.1: detect_aspects_v2 去噪；⑤: 会话级聚合.

输出到 output/movie/step10/v2/，与旧 step10_absa.py（seekers 路径）对比。
"""

import os
import csv
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log
from movie.utils.absa_aspects import detect_aspects_v2, ASPECTS_V2, ASPECT_NAMES
from movie.utils.conv_pairs import build_pairs_from_rows
from movie.step10_absa import SentimentAnalyzer, _sentiment_to_numeric

# ── 初始化 ──────────────────────────────────────────────────────────
setup_matplotlib()
STEP_OUT = os.path.join(STEP_DIRS[10], 'v2')       # 输出子目录
os.makedirs(STEP_OUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  提取 + 会话级聚合
# ═══════════════════════════════════════════════════════════════════════

def extract_v2(rows: list[dict], analyzer: SentimentAnalyzer):
    """Phase 1 提取：系统回复语料 + 去噪 + 会话级聚合。
    Returns: (pair_records, conv_records)"""
    pairs = build_pairs_from_rows(rows)
    pair_records = []
    stats = Counter()
    for p in pairs:
        text = p['system_text']
        if not text:
            continue
        for a in detect_aspects_v2(text):
            lab, sc = analyzer.predict(a['snippet'])
            pair_records.append({
                'session_id': p['session_id'],
                'pair_id': p['pair_id'],
                'date': p.get('date', ''),
                'period': p.get('period', ''),
                'is_holiday': p.get('is_holiday', False),
                'holiday_name': p.get('holiday_name', ''),
                'cross_day': p.get('cross_day', False),
                'aspect': a['aspect'],
                'aspect_label': ASPECTS_V2[a['aspect']]['label_en'],
                'sentiment': lab,
                'score': sc,
                'keyword': a['keyword'],
                'snippet': a['snippet'][:200],
                'model_used': getattr(analyzer, 'last_model', 'unknown'),
            })
            stats[a['aspect']] += 1
    log(f"  v2 Pairs: {len(pairs)} | records: {len(pair_records)}", "v2")
    log(f"  v2 Aspects: {dict(stats)}", "v2")
    conv_records = _aggregate_to_conv_level(pair_records)
    log(f"  v2 Conv-level: {len(conv_records)}", "v2")
    return pair_records, conv_records


def _aggregate_to_conv_level(pair_records: list[dict]) -> list[dict]:
    """会话级聚合：1 session × 1 aspect = 1 row。"""
    grouped = defaultdict(list)
    for r in pair_records:
        grouped[(r['session_id'], r['aspect'])].append(r)
    conv = []
    for (sid, aspect), recs in grouped.items():
        scores = [_sentiment_to_numeric(r['sentiment']) for r in recs]
        arr = np.array(scores)
        base = recs[0]
        conv.append({
            'session_id': sid,
            'date': base['date'],
            'period': base['period'],
            'is_holiday': base['is_holiday'],
            'holiday_name': base['holiday_name'],
            'cross_day': base['cross_day'],
            'aspect': aspect,
            'aspect_label': base['aspect_label'],
            'sentiment': base['sentiment'],
            'mean_sentiment': float(arr.mean()),
            'std_sentiment': float(arr.std()) if len(arr) > 1 else 0.0,
            'n_pairs': len(arr),
            'pos_ratio': float((arr > 0).sum() / max(len(arr), 1)),
        })
    return conv


# ═══════════════════════════════════════════════════════════════════════
#  聚合统计
# ═══════════════════════════════════════════════════════════════════════

def _agg_conv(conv_records, filter_fn=None, group_by_period=True):
    filtered = conv_records if filter_fn is None else [r for r in conv_records if filter_fn(r)]
    grouped = defaultdict(lambda: defaultdict(list))
    for r in filtered:
        gk = r.get('period', 'unknown') if group_by_period else 'overall'
        grouped[r['aspect']][gk].append(r['mean_sentiment'])
    result = {}
    for a, gd in grouped.items():
        result[a] = {}
        for g, v in gd.items():
            arr = np.array(v)
            result[a][g] = {
                'mean': float(arr.mean()), 'std': float(arr.std()) if len(arr) > 1 else 0.0,
                'count': len(arr),
                'pos_ratio': float((arr > 0).sum() / max(len(arr), 1)),
            }
    return result


def _overall(conv_records, filter_fn=None):
    """合并所有 period 为 overall 组。"""
    filtered = conv_records if filter_fn is None else [r for r in conv_records if filter_fn(r)]
    grouped = defaultdict(list)
    for r in filtered:
        grouped[r['aspect']].append(r['mean_sentiment'])
    return {a: {'overall': {
        'mean': float(np.mean(v)) if v else 0.0,
        'std': float(np.std(v)) if len(v) > 1 else 0.0,
        'count': len(v),
        'pos_ratio': float((np.array(v) > 0).sum() / max(len(v), 1)),
    }} for a, v in grouped.items()}


# ═══════════════════════════════════════════════════════════════════════
#  可视化（复用 step10_absa 的 _plot_aspect_sentiment_bars 思路）
# ═══════════════════════════════════════════════════════════════════════

def _plot_bars(aspect_data, title, filename):
    aspects = sorted(aspect_data.keys(), key=lambda a: sum(
        aspect_data[a].get(g, {}).get('mean', 0) for g in aspect_data[a]
    ), reverse=True)
    groups = sorted(set(g for a in aspects for g in aspect_data[a]))
    if not aspects or not groups:
        return
    fig, ax = plt.subplots(figsize=(12, max(5, len(aspects) * 0.4 + 1)))
    x = np.arange(len(aspects))
    n = len(groups)
    w = 0.8 / max(n, 1)
    colors = ['#ff6b6b', '#74b9ff', '#feca57', '#48dbfb']
    for i, g in enumerate(groups):
        means = [aspect_data[a].get(g, {}).get('mean', 0) for a in aspects]
        errs = [aspect_data[a].get(g, {}).get('std', 0) for a in aspects]
        ax.bar(x + (i - (n - 1) / 2) * w, means, w, yerr=errs, capsize=3,
               label=g, color=colors[i % len(colors)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ASPECTS_V2[a]['label_en'] for a in aspects],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean Sentiment (-1 to +1)')
    ax.set_title(title, fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


def _save_csv(conv_records, filename):
    p = os.path.join(STEP_OUT, filename)
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['session_id', 'date', 'period', 'holiday_name', 'cross_day',
                    'aspect', 'aspect_label', 'mean_sentiment', 'std_sentiment',
                    'n_pairs', 'pos_ratio'])
        for r in conv_records:
            w.writerow([r['session_id'], r['date'], r['period'], r['holiday_name'],
                        r['cross_day'], r['aspect'], r['aspect_label'],
                        f'{r["mean_sentiment"]:.4f}', f'{r["std_sentiment"]:.4f}',
                        r['n_pairs'], f'{r["pos_ratio"]:.3f}'])
    log(f"Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════
#  A1-A5 会话级
# ═══════════════════════════════════════════════════════════════════════

def dim_a1(conv_records):
    log("A1 (v2): Aspect Distribution")
    c = Counter(r['aspect_label'] for r in conv_records)
    total = sum(c.values())
    labels = [l for l, _ in c.most_common()]
    values = [v for _, v in c.most_common()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, _, _ = ax.pie(values, labels=None, autopct='%1.1f%%',
                          colors=colors, startangle=90)
    ax.legend(wedges, labels, title="Aspect", loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    ax.set_title(f'v2 Aspect Distribution (Conv-level, n={total})', fontsize=13)
    fig.tight_layout()
    p = os.path.join(STEP_OUT, 'a1_aspect_distribution_v2.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {p}")


def dim_a2(conv_records):
    log("A2 (v2): Overall Aspect Sentiment")
    agg = _overall(conv_records)
    if not agg:
        return
    _plot_bars(agg, 'v2 Overall Sentiment by Aspect (Conv)',
               'a2_overall_aspect_sentiment_v2.png')


def dim_a3(conv_records):
    log("A3 (v2): Holiday vs Non-Holiday")
    h = _agg_conv(conv_records, lambda r: r['period'] == 'holiday')
    nh = _agg_conv(conv_records, lambda r: r['period'] != 'holiday')
    merged = {}
    for a in set(list(h.keys()) + list(nh.keys())):
        merged[a] = {}
        if a in h:
            merged[a]['Holiday'] = list(h[a].values())[0] if h[a] else {}
        if a in nh:
            merged[a]['Non-holiday'] = list(nh[a].values())[0] if nh[a] else {}
    _plot_bars(merged, 'v2 Holiday vs Non-Holiday (Conv)',
               'a3_holiday_vs_nonholiday_v2.png')


def dim_a4(conv_records):
    log("A4 (v2): Holiday vs Workday vs Weekend")
    pmap = {'holiday': 'Holiday', 'workday': 'Workday', 'weekend': 'Weekend'}
    merged = {}
    for pc, pl in pmap.items():
        agg = _agg_conv(conv_records, lambda r, p=pc: r['period'] == p)
        for a, gd in agg.items():
            if a not in merged:
                merged[a] = {}
            merged[a][pl] = list(gd.values())[0] if gd else {}
    _plot_bars(merged, 'v2 Holiday vs Workday vs Weekend (Conv)',
               'a4_holiday_workday_weekend_v2.png')


def dim_a5(conv_records):
    log("A5 (v2): Per-Holiday Heatmap")
    ha = defaultdict(lambda: defaultdict(list))
    for r in conv_records:
        if r['is_holiday'] and r['holiday_name']:
            ha[r['holiday_name'][:8]][r['aspect']].append(r['mean_sentiment'])
    ha = {k: v for k, v in ha.items()
          if sum(len(s) for s in v.values()) >= MIN_DATA_ROWS}
    if not ha:
        log("  No holiday groups with sufficient data")
        return
    names = sorted(ha.keys())
    matrix = np.zeros((len(names), len(ASPECT_NAMES)))
    for i, n in enumerate(names):
        for j, a in enumerate(ASPECT_NAMES):
            v = ha[n].get(a, [])
            matrix[i, j] = np.mean(v) if v else np.nan
    fig, ax = plt.subplots(figsize=(max(12, len(ASPECT_NAMES) * 0.8),
                                     max(5, len(names) * 0.5 + 1)))
    mask = np.isnan(matrix)
    cmap = plt.cm.RdBu_r
    cmap.set_bad('lightgray')
    sns.heatmap(np.ma.masked_invalid(matrix), annot=True, fmt='.2f', cmap=cmap,
                xticklabels=[ASPECTS_V2[a]['label_en'] for a in ASPECT_NAMES],
                yticklabels=names, ax=ax, center=0, linewidths=0.5, mask=mask,
                cbar_kws={'label': 'Mean Sentiment'})
    ax.set_title('v2 Per-Holiday Aspect Sentiment (Conv)', fontsize=13)
    ax.set_xlabel('Aspect'); ax.set_ylabel('Holiday')
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    p = os.path.join(STEP_OUT, 'a5_per_holiday_aspect_v2.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None, analyzer: SentimentAnalyzer = None):
    log("=" * 60)
    log("Step 10 ABSA v2 (Phase 1: corpus switch + de-noise + conv aggregate)")
    log("=" * 60)
    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    if analyzer is None:
        analyzer = SentimentAnalyzer()
    pair_records, conv_records = extract_v2(data['rows'], analyzer)
    if not conv_records:
        log("  No conv data. Skip.")
        return pair_records, conv_records
    _save_csv(conv_records, 'a0_conv_records_v2.csv')
    dim_a1(conv_records)
    dim_a2(conv_records)
    dim_a3(conv_records)
    dim_a4(conv_records)
    dim_a5(conv_records)
    log(f"v2 done. Output: {STEP_OUT}")
    return pair_records, conv_records


if __name__ == '__main__':
    main()
