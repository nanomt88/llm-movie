# -*- coding: utf-8 -*-
"""
Step 10 ABSA v2 vs v3 对比：同样本跑 Phase 1 与 Phase 2，量化对比哪个更好。

为加速，用 VADER-only 分析器（避免 transformer 推理过慢）。
子采样固定数量会话，保证 v2/v3 跑相同数据。
"""

import os
import json
import random
from collections import Counter, defaultdict

import numpy as np

from movie.config import STEP_DIRS, setup_matplotlib, log
from movie.data_loader import load_all
from movie.utils.conv_pairs import regroup_sessions
from movie.utils.absa_aspects import ASPECTS_V2, ASPECT_NAMES
from movie.step10_absa_v2 import extract_v2, _aggregate_to_conv_level
from movie.step10_absa_v3 import extract_v3

setup_matplotlib()
STEP_OUT = os.path.join(STEP_DIRS[10], 'compare')
os.makedirs(STEP_OUT, exist_ok=True)


class VAnalyzer:
    """VADER-only 快速分析器（含 last_model 字段，接口与 SentimentAnalyzer 一致）。"""
    last_model = 'vader'

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._v = SentimentIntensityAnalyzer()

    def predict(self, text):
        if not text or len(text.strip()) < 3:
            self.last_model = 'none'
            return ('NEUTRAL', 0.0)
        c = self._v.polarity_scores(text)
        comp = c['compound']
        self.last_model = 'vader'
        if comp >= 0.05:
            return ('POSITIVE', comp)
        if comp <= -0.05:
            return ('NEGATIVE', -abs(comp))
        return ('NEUTRAL', 0.0)


def subsample_rows_by_session(rows, n_sessions, seed=42):
    """按 session 子采样 rows，保证 v2/v3 跑相同数据。"""
    random.seed(seed)
    sessions = regroup_sessions(rows)
    sids = list(sessions.keys())
    if len(sids) > n_sessions:
        sids = random.sample(sids, n_sessions)
    sub_rows = [r for s in sids for r in sessions[s]]
    log(f"  Subsampled {len(sids)} sessions -> {len(sub_rows)} rows")
    return sub_rows


def conv_aspect_means(conv_records, filter_fn=None):
    """各 aspect 的会话级均值（跨会话平均）。"""
    filtered = conv_records if filter_fn is None else [r for r in conv_records if filter_fn(r)]
    grouped = defaultdict(list)
    for r in filtered:
        grouped[r['aspect']].append(r['mean_sentiment'])
    return {a: {'mean': float(np.mean(v)) if v else 0.0,
                'std': float(np.std(v)) if len(v) > 1 else 0.0,
                'n': len(v)}
            for a, v in grouped.items()}


def main(n_sessions=8000, seed=42):
    log("=" * 60)
    log(f"Step 10 ABSA v2 vs v3 Comparison (n_sessions={n_sessions})")
    log("=" * 60)

    data = load_all()
    sub_rows = subsample_rows_by_session(data['rows'], n_sessions, seed)
    sub_data = {'rows': sub_rows, 'seekers': [r for r in sub_rows if r['is_seeker']]}

    analyzer = VAnalyzer()
    log(f"Analyzer: VADER (fast mode)")

    # ── v2 (Phase 1) ──
    log("\n[1/2] Running v2 (Phase 1: corpus switch + de-noise + conv aggregate)...")
    v2_pairs, v2_conv = extract_v2(sub_rows, analyzer)
    log(f"  v2 done: {len(v2_pairs)} pair-records, {len(v2_conv)} conv-records")

    # ── v3 (Phase 2) ──
    log("\n[2/2] Running v3 (Phase 2: v2 + NLI + dual-track)...")
    v3_pairs, v3_conv, v3_stats = extract_v3(sub_rows, analyzer)
    log(f"  v3 done: {len(v3_pairs)} pair-records, {len(v3_conv)} conv-records")

    # ── 对比指标 ──
    log("\n" + "=" * 60)
    log("COMPARISON RESULTS")
    log("=" * 60)

    # 1. 计数对比
    v2_cnt = Counter(r['aspect'] for r in v2_pairs)
    v3_cnt = Counter(r['aspect'] for r in v3_pairs)
    log("\n[1] Aspect mention counts (pair-level):")
    log(f"  {'aspect':14} {'v2':>6} {'v3':>6} {'v3 drop':>10}")
    for a in ASPECT_NAMES:
        log(f"  {a:14} {v2_cnt.get(a,0):6} {v3_cnt.get(a,0):6} "
            f"{(v2_cnt.get(a,0)-v3_cnt.get(a,0))/max(v2_cnt.get(a,0),1)*100:9.1f}%")

    # 2. NLI 过滤率
    nli_rate = v3_stats['n_nli_filtered'] / max(
        v3_stats['n_nli_filtered'] + v3_stats['n_records'], 1) * 100
    log(f"\n[2] NLI filter rate (v3): {v3_stats['n_nli_filtered']} filtered "
        f"= {nli_rate:.1f}% of candidates")

    # 3. 双轨覆盖率
    dual_rate = v3_stats['n_dual_track'] / max(v3_stats['n_pairs'], 1) * 100
    log(f"[3] Dual-track coverage (v3): {v3_stats['n_dual_track']} pairs "
        f"= {dual_rate:.1f}% of total pairs")

    # 4. 各 aspect 情感均值对比
    v2_means = conv_aspect_means(v2_conv)
    v3_means = conv_aspect_means(v3_conv)
    log(f"\n[4] Sentiment means (conv-level, across sessions):")
    log(f"  {'aspect':14} {'v2 mean':>9} {'v3 mean':>9} {'delta':>8} {'v2 std':>8} {'v3 std':>8}")
    for a in ASPECT_NAMES:
        v2m = v2_means.get(a, {})
        v3m = v3_means.get(a, {})
        log(f"  {a:14} {v2m.get('mean',0):9.3f} {v3m.get('mean',0):9.3f} "
            f"{v3m.get('mean',0)-v2m.get('mean',0):8.3f} "
            f"{v2m.get('std',0):8.3f} {v3m.get('std',0):8.3f}")

    # 5. 节假日 vs 非节假日 差异（哪个区分度更大）
    def holiday_delta(conv):
        h = conv_aspect_means(conv, lambda r: r['period'] == 'holiday')
        nh = conv_aspect_means(conv, lambda r: r['period'] != 'holiday')
        deltas = {}
        for a in ASPECT_NAMES:
            hm = h.get(a, {}).get('mean', 0)
            nm = nh.get(a, {}).get('mean', 0)
            deltas[a] = hm - nm
        return deltas, h, nh

    v2_hd, v2_h, v2_nh = holiday_delta(v2_conv)
    v3_hd, v3_h, v3_nh = holiday_delta(v3_conv)
    log(f"\n[5] Holiday - Non-holiday sentiment delta (differentiation signal):")
    log(f"  {'aspect':14} {'v2 delta':>9} {'v3 delta':>9}")
    for a in ASPECT_NAMES:
        log(f"  {a:14} {v2_hd.get(a,0):9.3f} {v3_hd.get(a,0):9.3f}")
    v2_signal = np.mean([abs(v) for v in v2_hd.values()])
    v3_signal = np.mean([abs(v) for v in v3_hd.values()])
    log(f"  mean |delta|: v2={v2_signal:.3f}  v3={v3_signal:.3f}  "
        f"(larger = stronger holiday differentiation)")

    # 6. 稳定性（会话级 std 中位）
    v2_stds = [v2_means[a]['std'] for a in ASPECT_NAMES if a in v2_means]
    v3_stds = [v3_means[a]['std'] for a in ASPECT_NAMES if a in v3_means]
    v2_med_std = float(np.median(v2_stds)) if v2_stds else 0.0
    v3_med_std = float(np.median(v3_stds)) if v3_stds else 0.0

    # 保存 JSON
    report = {
        'config': {'n_sessions': n_sessions, 'seed': seed, 'analyzer': 'vader'},
        'counts': {'v2': dict(v2_cnt), 'v3': dict(v3_cnt)},
        'nli_filter': {'filtered': v3_stats['n_nli_filtered'], 'rate_pct': nli_rate},
        'dual_track': {'pairs': v3_stats['n_dual_track'], 'rate_pct': dual_rate},
        'sentiment_means': {
            'v2': {a: v2_means.get(a, {}) for a in ASPECT_NAMES},
            'v3': {a: v3_means.get(a, {}) for a in ASPECT_NAMES},
        },
        'holiday_delta': {'v2': v2_hd, 'v3': v3_hd},
        'stability': {'v2_median_std': v2_med_std, 'v3_median_std': v3_med_std},
    }
    path = os.path.join(STEP_OUT, 'v2_vs_v3_report.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    log(f"\nReport saved: {path}")

    # 判定
    log("\n" + "=" * 60)
    log("VERDICT")
    log("=" * 60)
    log(f"  v2 = Phase 1 (corpus switch + de-noise + conv aggregate)")
    log(f"  v3 = Phase 2 (v2 + NLI filter + dual-track pairing)")
    log(f"  NLI filtered {nli_rate:.1f}% non-evaluative candidates (precision up)")
    log(f"  Dual-track covered {dual_rate:.1f}% pairs (richer sentiment signal)")
    log(f"  Stability: v2 median std={v2_med_std:.4f} vs v3={v3_med_std:.4f}")
    log(f"  Holiday signal: v2 |delta|={v2_signal:.3f} vs v3={v3_signal:.3f}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sessions', type=int, default=8000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    main(n_sessions=args.sessions, seed=args.seed)
