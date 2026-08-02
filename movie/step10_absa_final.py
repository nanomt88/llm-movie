# -*- coding: utf-8 -*-
"""
Step 10 ABSA Final (v3-lite): v2 + NLI 过滤，无双轨配对.
最终推荐版本：B 语料切换 + ③.1 去噪 + ③.2 NLI + ⑤ 会话级聚合。
（去掉 v3 的双轨配对 —— 实测覆盖率仅 0.8%，性价比低）

输出到 output/movie/step10/final/，复用 v2 的 dim/plot 函数。
分析器用归一化 VADER（last_model + 有符号 [-1,1] 分数），全量数据约 8 分钟。
"""

import os
from collections import Counter

from movie.config import STEP_DIRS, setup_matplotlib, log
from movie.utils.absa_aspects import detect_aspects_v2, ASPECTS_V2
from movie.utils.conv_pairs import build_pairs_from_rows
from movie.utils.absa_nli import has_evaluative_context
from movie.step10_absa import _sentiment_to_numeric
import movie.step10_absa_v2 as v2

# ── 初始化：复用 v2 的全部 dim/plot，仅重定向输出目录 ──────────────
setup_matplotlib()
v2.STEP_OUT = os.path.join(STEP_DIRS[10], 'final')   # 重定向到 final/
os.makedirs(v2.STEP_OUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  归一化 VADER 分析器（有符号 [-1,1]，含 last_model）
# ═══════════════════════════════════════════════════════════════════════

class NormalizedVaderAnalyzer:
    """VADER-only，分数归一化到 [-1,1]（NEGATIVE 返回负值），含 last_model。
       接口与 step10_absa.SentimentAnalyzer 一致（predict/last_model）。"""
    last_model = 'vader'

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._v = SentimentIntensityAnalyzer()

    def predict(self, text):
        if not text or len(text.strip()) < 3:
            self.last_model = 'none'
            return ('NEUTRAL', 0.0)
        c = self._v.polarity_scores(text)
        comp = c['compound']                      # [-1, 1]
        self.last_model = 'vader'
        if comp >= 0.05:
            return ('POSITIVE', comp)              # 正值
        if comp <= -0.05:
            return ('NEGATIVE', comp)              # 负值（有符号）
        return ('NEUTRAL', 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  v3-lite 提取：v2 + NLI 过滤，无双轨
# ═══════════════════════════════════════════════════════════════════════

def extract_final(rows, analyzer):
    """v3-lite 提取：系统回复语料 + 去噪 + NLI 过滤 + 会话级聚合（无双轨）。
    Returns: (pair_records, conv_records, stats)"""
    pairs = build_pairs_from_rows(rows)
    pair_records = []
    stats = Counter()
    n_nli_filtered = 0

    for p in pairs:
        text = p['system_text']
        if not text:
            continue
        candidates = detect_aspects_v2(text)
        # ③.2 NLI 过滤：丢弃纯提及（无评价词）候选
        for c in candidates:
            if not has_evaluative_context(c['snippet']):
                n_nli_filtered += 1
                continue
            lab, sc = analyzer.predict(c['snippet'])
            pair_records.append({
                'session_id': p['session_id'],
                'pair_id': p['pair_id'],
                'date': p.get('date', ''),
                'period': p.get('period', ''),
                'is_holiday': p.get('is_holiday', False),
                'holiday_name': p.get('holiday_name', ''),
                'cross_day': p.get('cross_day', False),
                'aspect': c['aspect'],
                'aspect_label': ASPECTS_V2[c['aspect']]['label_en'],
                'sentiment': lab,
                'score': sc,
                'keyword': c['keyword'],
                'snippet': c['snippet'][:200],
                'model_used': getattr(analyzer, 'last_model', 'vader'),
            })
            stats[c['aspect']] += 1

    log(f"  final Pairs: {len(pairs)} | records: {len(pair_records)}", "final")
    log(f"  final Aspects: {dict(stats)}", "final")
    log(f"  final NLI filtered (non-evaluative): {n_nli_filtered} "
        f"= {n_nli_filtered / max(n_nli_filtered + len(pair_records), 1) * 100:.1f}%", "final")
    conv_records = v2._aggregate_to_conv_level(pair_records)
    log(f"  final Conv-level: {len(conv_records)}", "final")
    stats_out = {'n_pairs': len(pairs), 'n_records': len(pair_records),
                 'n_nli_filtered': n_nli_filtered, 'n_conv': len(conv_records),
                 'aspect_counts': dict(stats)}
    return pair_records, conv_records, stats_out


# ═══════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════

def main(data=None, analyzer=None):
    log("=" * 60)
    log("Step 10 ABSA Final (v3-lite: v2 + NLI, no dual-track)")
    log("=" * 60)
    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    if analyzer is None:
        analyzer = NormalizedVaderAnalyzer()
        log("Analyzer: VADER (normalized, signed [-1,1])")
    pair_records, conv_records, stats = extract_final(data['rows'], analyzer)
    if not conv_records:
        log("  No conv data. Skip.")
        return pair_records, conv_records, stats
    # 复用 v2 的 A1-A5 + CSV（输出到 v2.STEP_OUT = final/）
    v2._save_csv(conv_records, 'a0_conv_records_final.csv')
    v2.dim_a1(conv_records)
    v2.dim_a2(conv_records)
    v2.dim_a3(conv_records)
    v2.dim_a4(conv_records)
    v2.dim_a5(conv_records)
    log(f"Final done. Output: {v2.STEP_OUT}")
    return pair_records, conv_records, stats


if __name__ == '__main__':
    main()
