# -*- coding: utf-8 -*-
"""
Step 10 ABSA Phase 1 evaluation harness (standalone, 不改动 step10_absa.py).
评估：
  1. 旧 detect_aspects vs 新 detect_aspects_v2 检测对比（200 条系统回复）
  2. Bootstrap 聚合稳定性：会话级 vs 消息级 各 aspect 均值方差

为避免 transformer 推理过慢，bootstrap 用 VADER（稳定性比较只需聚合方法一致，模型不影响结论）。
"""

import os
import csv
import json
import random
from collections import defaultdict

import numpy as np

from movie.config import STEP_DIRS, log, setup_matplotlib
from movie.data_loader import load_all
from movie.utils.absa_aspects import detect_aspects_v2
from movie.utils.conv_pairs import build_pairs_from_rows
from movie.step10_absa import (
    detect_aspects,           # 旧：纯关键词子串匹配
    _sentiment_to_numeric,
)

setup_matplotlib()
STEP_OUT = STEP_DIRS[10]
EVAL_DIR = os.path.join(STEP_OUT, 'eval')
os.makedirs(EVAL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  [1] 旧 vs 新 方面检测对比
# ═══════════════════════════════════════════════════════════════════════

def run_old_vs_new(rows: list[dict], sample_size: int = 200, seed: int = 42) -> str:
    """旧 detect_aspects vs 新 detect_aspects_v2 对比，输出 CSV + 统计。
       仅做检测对比，不跑情感模型（快）。"""
    random.seed(seed)
    system_rows = [r for r in rows if not r['is_seeker'] and r.get('proc_text')]
    sample = random.sample(system_rows, min(sample_size, len(system_rows)))

    path = os.path.join(EVAL_DIR, 'old_vs_new_detection.csv')
    old_total = defaultdict(int)
    new_total = defaultdict(int)
    n_old_empty = 0
    n_new_empty = 0

    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'text', 'old_aspects', 'new_aspects',
                    'old_count', 'new_count', 'delta'])
        for i, r in enumerate(sample):
            text = r['proc_text']
            old = detect_aspects(text)               # 旧：纯关键词
            new = detect_aspects_v2(text)           # 新：去噪
            old_names = sorted(old.keys())
            new_names = sorted({a['aspect'] for a in new})
            for a in old_names:
                old_total[a] += 1
            for a in new_names:
                new_total[a] += 1
            if not old_names:
                n_old_empty += 1
            if not new_names:
                n_new_empty += 1
            w.writerow([i, text[:300], '|'.join(old_names), '|'.join(new_names),
                        len(old_names), len(new_names),
                        len(new_names) - len(old_names)])

    log(f"  Sample: {len(sample)} system replies")
    log(f"  Old: empty detections = {n_old_empty} | New: empty = {n_new_empty}")
    log(f"  Wrote {path}")
    log("  Aspect totals (old -> new, change%):")
    all_aspects = sorted(set(list(old_total.keys()) + list(new_total.keys())))
    for a in all_aspects:
        o = old_total.get(a, 0)
        n = new_total.get(a, 0)
        chg = f"{(n - o) / o * 100:+.1f}%" if o > 0 else "(new only)"
        log(f"    {a}: {o} -> {n}  {chg}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  [2] Bootstrap 聚合稳定性（会话级 vs 消息级）
# ═══════════════════════════════════════════════════════════════════════

_VADER = None


def _vader_sentiment(text: str) -> str:
    """Fast VADER sentiment for bootstrap (避免 transformer 推理过慢)。
       返回 'POSITIVE'/'NEGATIVE'/'NEUTRAL'。"""
    global _VADER
    if _VADER is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _VADER = SentimentIntensityAnalyzer()
        except Exception:
            _VADER = False
    if _VADER is False:
        return 'NEUTRAL'
    try:
        c = _VADER.polarity_scores(text)
        comp = c['compound']
        if comp >= 0.05:
            return 'POSITIVE'
        elif comp <= -0.05:
            return 'NEGATIVE'
    except Exception:
        pass
    return 'NEUTRAL'


def _aggregate_to_conv_level_inline(pair_records: list[dict]) -> list[dict]:
    """会话级聚合：1 session × 1 aspect = 1 row（与计划 Task 3 一致）。"""
    grouped = defaultdict(list)
    for r in pair_records:
        grouped[(r['session_id'], r['aspect'])].append(r)
    conv_records = []
    for (sid, aspect), recs in grouped.items():
        scores = [_sentiment_to_numeric(r['sentiment']) for r in recs]
        arr = np.array(scores)
        conv_records.append({
            'session_id': sid,
            'aspect': aspect,
            'mean_sentiment': float(arr.mean()),
            'n_pairs': len(arr),
        })
    return conv_records


def run_bootstrap_stability(
    rows: list[dict],
    n_runs: int = 5,
    boot_sessions: int = 2000,
    sample_frac: float = 0.8,
    seed: int = 42,
) -> str:
    """Bootstrap 稳定性：会话级 vs 消息级 各 aspect 均值方差。
       每次随机抽 boot_sessions 个会话的 80%，跑 detect+sentiment，
       分别按消息级和会话级聚合，比较跨 n_runs 次的 std。"""
    random.seed(seed)
    pairs = build_pairs_from_rows(rows)
    pairs_by_sid = defaultdict(list)
    for p in pairs:
        pairs_by_sid[p['session_id']].append(p)
    sids = list(pairs_by_sid.keys())
    log(f"  Total sessions: {len(sids)} | total pairs: {len(pairs)}")
    log(f"  Bootstrap: {n_runs} runs × {boot_sessions} sessions × {sample_frac:.0%} subsample")

    seeds = [seed + i * 10 for i in range(n_runs)]
    conv_runs = []
    msg_runs = []
    for s in seeds:
        random.seed(s)
        # 先抽 boot_sessions 个会话，再从中抽 80%
        if len(sids) > boot_sessions:
            boot_pool = random.sample(sids, boot_sessions)
        else:
            boot_pool = sids
        sub = random.sample(boot_pool, max(1, int(len(boot_pool) * sample_frac)))
        sub_pairs = [p for sid in sub for p in pairs_by_sid[sid]]

        recs = []
        for p in sub_pairs:
            text = p['system_text']
            if not text:
                continue
            for a in detect_aspects_v2(text):
                lab = _vader_sentiment(a['snippet'])
                recs.append({
                    'session_id': p['session_id'],
                    'aspect': a['aspect'],
                    'sentiment': lab,
                })
        # 消息级均值（所有 pair_records 直接平均）
        msg_grouped = defaultdict(list)
        for r in recs:
            msg_grouped[r['aspect']].append(_sentiment_to_numeric(r['sentiment']))
        msg_means = {a: float(np.mean(v)) if v else 0.0 for a, v in msg_grouped.items()}
        # 会话级均值（先 session×aspect 聚合，再跨 session 平均）
        conv = _aggregate_to_conv_level_inline(recs)
        conv_grouped = defaultdict(list)
        for r in conv:
            conv_grouped[r['aspect']].append(r['mean_sentiment'])
        conv_means = {a: float(np.mean(v)) if v else 0.0 for a, v in conv_grouped.items()}
        conv_runs.append(conv_means)
        msg_runs.append(msg_means)
        log(f"    seed={s}: subsessions={len(sub)} pairs={len(sub_pairs)} "
            f"records={len(recs)} conv_records={len(conv)}")

    all_aspects = set()
    for run in conv_runs + msg_runs:
        all_aspects |= set(run.keys())
    stability = {}
    for a in sorted(all_aspects):
        conv_vals = [r.get(a, 0.0) for r in conv_runs]
        msg_vals = [r.get(a, 0.0) for r in msg_runs]
        msg_std = float(np.std(msg_vals))
        conv_std = float(np.std(conv_vals))
        stability[a] = {
            'conv_mean': float(np.mean(conv_vals)),
            'conv_std': conv_std,
            'msg_mean': float(np.mean(msg_vals)),
            'msg_std': msg_std,
            'std_reduction_pct': float((msg_std - conv_std) / max(msg_std, 1e-9) * 100),
            'n_runs': n_runs,
        }
    path = os.path.join(EVAL_DIR, 'stability_bootstrap.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stability, f, indent=2, ensure_ascii=False)
    log(f"  Wrote {path}")
    log("  Stability (conv_std vs msg_std, reduction%):")
    for a, s in stability.items():
        log(f"    {a}: conv_std={s['conv_std']:.4f} msg_std={s['msg_std']:.4f} "
            f"reduction={s['std_reduction_pct']:+.1f}%  "
            f"(conv_mean={s['conv_mean']:+.3f}, msg_mean={s['msg_mean']:+.3f})")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════

def main(sample_size: int = 200, n_runs: int = 5, boot_sessions: int = 2000):
    log("=" * 60)
    log("Step 10 ABSA Phase 1 Evaluation")
    log("=" * 60)
    data = load_all()
    rows = data['rows']
    log(f"Loaded {len(rows)} rows")

    log("\n[1/2] Old vs New detection comparison...")
    run_old_vs_new(rows, sample_size=sample_size)

    log("\n[2/2] Bootstrap stability (conv-level vs message-level)...")
    run_bootstrap_stability(rows, n_runs=n_runs, boot_sessions=boot_sessions)

    log("\nEvaluation complete. See output/movie/step10/eval/")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sample', type=int, default=200, help='old-vs-new sample size')
    p.add_argument('--runs', type=int, default=5, help='bootstrap runs')
    p.add_argument('--boot-sessions', type=int, default=2000,
                   help='sessions per bootstrap run (speed cap)')
    args = p.parse_args()
    main(sample_size=args.sample, n_runs=args.runs, boot_sessions=args.boot_sessions)
