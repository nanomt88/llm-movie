# -*- coding: utf-8 -*-
"""
Step 12: Recommendation Insight Analysis
步骤 12：推荐洞察分析

Dimensions / 分析维度:
  N1 - Query Intent Classification & Holiday Patterns
      查询意图分类与节假日模式：分析用户在节假日/工作日/周末的提问意图分布，
      以及不同意图下的会话深度（平均轮次），帮助理解节假日场景下用户的真实需求。
  N2 - Sentiment-Genre Affinity by Holiday Period
      情感-类型关联分析：结合情感分析结果与电影类型数据，发现节假日期间
      哪些类型与正向/负向情感高度相关，为情感感知推荐提供依据。

Output: output/movie/step12/*.png
"""

import os
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from movie.config import STEP_DIRS, setup_matplotlib, log
from movie.utils.text import parse_conv_turn


setup_matplotlib()
STEP_OUT = STEP_DIRS[12]
os.makedirs(STEP_OUT, exist_ok=True)

# ── Module-level cache for full-row scan ───────────────────────────────
_CONV_SYSTEM_CACHE: dict[tuple[str, str], list[str]] | None = None


def _build_conv_system(all_rows: list[dict]) -> dict[tuple[str, str], list[str]]:
    """Build (session_id, turn_num) -> list of system reply processed_raw texts.

    Uses turn-level keys so each seeker only gets the system reply from the
    same turn, avoiding cross-turn genre contamination.
    Cached at module level so N1 and N2 don't each scan 1.6M rows.
    """
    global _CONV_SYSTEM_CACHE
    if _CONV_SYSTEM_CACHE is not None:
        return _CONV_SYSTEM_CACHE

    conv_system: dict[tuple[str, str], list[str]] = {}
    for row in all_rows:
        is_seeker = row.get('is_seeker', False)
        if is_seeker:
            continue
        processed = row.get('processed_raw', row.get('processed', ''))
        if not processed:
            continue
        conv_id = row.get('conv_id', '')
        key = parse_conv_turn(conv_id)  # (session_id, turn_num)
        if key not in conv_system:
            conv_system[key] = []
        conv_system[key].append(processed)

    _CONV_SYSTEM_CACHE = conv_system
    return conv_system


# ── Color scheme (consistent with previous steps) ──────────────────────
PERIOD_COLORS = {
    'holiday':  '#ff6b6b',
    'workday':  '#74b9ff',
    'weekend':  '#feca57',
}
SENTIMENT_COLORS = {
    'positive': '#2ecc71',
    'neutral':  '#95a5a6',
    'negative': '#e74c3c',
}
INTENT_COLORS = {
    'recommendation': '#3498db',
    'identification': '#e67e22',
    'opinion':        '#9b59b6',
    'factual':        '#1abc9c',
    'comparison':     '#e74c3c',
    'general':        '#95a5a6',
}

# ── Intent classification patterns ──────────────────────────────────────
# Order matters: more specific patterns checked first.
_INTENT_PATTERNS = [
    ('recommendation', [
        r'\brecommend\b', r'\bsuggest\b', r'\bsuggestion\b',
        r'\blooking for\b', r'\bwant to watch\b', r'\bneed a movie\b',
        r'\bany good\b', r'\bsimilar to\b', r'\blike\s+\w+\s+movie\b',
        r'\bwhat should I watch\b', r"\bcan'?t decide\b",
        r'\bmust watch\b', r'\bunderrated\b', r'\boverrated\b',
    ]),
    ('identification', [
        r'\bwhat movie\b', r'\bwhich movie\b', r'\bwhat film\b',
        r"\bwhat's the\b", r"\bwhat is the\b", r'\bname of\b',
        r'\bcalled\b', r'\btitle\b', r'\bwho (?:plays|stars|acted)\b',
        r'\bwhere can I\b', r'\bhow to\b',
    ]),
    ('opinion', [
        r'\bis\s+\w+\s+good\b', r'\bis\s+\w+\s+bad\b', r'\bis\s+\w+\s+worth\b',
        r'\bworth watching\b', r'\bworth it\b', r'\bis it good\b',
        r'\bis it bad\b', r'\bhow is\b', r'\bwhat do you think\b',
        r'\bopinion\b', r'\bthoughts?\b', r'\bimpression\b',
        r'\bterrible\b', r'\bamazing\b', r'\bincredible\b',
        r'\bfavorite\b', r'\bbest\b', r'\bworst\b',
    ]),
    ('comparison', [
        r'\bvs\b', r'\bversus\b', r'\bbetter than\b',
        r'\bcompare\b', r'\balternative\b', r'\bor\s+.+\s+or\b',
        r'\bwhich is better\b', r'\bdifference between\b',
    ]),
    ('factual', [
        r'\brelease\b', r'\bcast\b', r'\bdirector\b', r'\bplot\b',
        r'\bgenre\b', r'\byear\b', r'\brating\b', r'\bimdb\b',
        r'\brotten tomatoes\b', r'\bruntime\b', r'\bbudget\b',
        r'\bbox office\b', r'\bseries\b', r'\bsequel\b', r'\bprequel\b',
        r'\bremake\b', r'\badaptation\b', r'\bbased on\b',
        r'\bnetflix\b', r'\bstreaming\b', r'\btheater\b',
    ]),
]


def _classify_intent(text: str) -> str:
    """Classify a query text into an intent category.

    Args:
        text: Raw or processed query text from a seeker record.

    Returns:
        One of: 'recommendation', 'identification', 'opinion',
                'comparison', 'factual', 'general'.
    """
    if not text:
        return 'general'
    text_lower = text.lower().strip()
    for intent, patterns in _INTENT_PATTERNS:
        for pat in patterns:
            if re.search(pat, text_lower):
                return intent
    return 'general'


def _session_turn_counts(seekers: list[dict], rows: list[dict]) -> dict[str, dict[str, float]]:
    """Compute average turns per session, grouped by intent and period.

    Pre-computes unique system turn counts per session in one pass,
    then aggregates by period and intent.

    Args:
        seekers: List of seeker records (with intent classification added).
        rows: Full conversation rows.

    Returns:
        dict: {period: {intent: avg_turns}}
    """
    # Pre-compute unique system turn count per session in one pass
    session_unique_turns = {}
    session_turns: dict[str, set] = {}
    for r in rows:
        if not r['is_seeker']:
            sid = r['session_id']
            if sid not in session_turns:
                session_turns[sid] = set()
            session_turns[sid].add(r['turn_order'])
    for sid, turns in session_turns.items():
        session_unique_turns[sid] = len(turns)

    # Group seekers by period and intent
    period_intent_turns = defaultdict(lambda: defaultdict(list))
    for s in seekers:
        period = s.get('period', 'workday')
        intent = s.get('_intent', 'general')
        sid = s.get('session_id', '')
        period_intent_turns[period][intent].append(session_unique_turns.get(sid, 0))

    # Compute averages
    result: dict[str, dict[str, float]] = {}
    for period, intent_data in period_intent_turns.items():
        result[period] = {}
        for intent, turn_list in intent_data.items():
            if turn_list:
                result[period][intent] = sum(turn_list) / len(turn_list)
            else:
                result[period][intent] = 0.0
    return result


def _build_seeker_genres(
    seekers: list[dict],
    all_rows: list[dict],
    movie_info: dict,
) -> list[dict]:
    """Augment seekers with genre info from system replies.

    Same approach as step5_genre._build_seeker_genres.
    Caches turn-level genre data for performance.

    Args:
        seekers: List of seeker records.
        all_rows: Full conversation rows.
        movie_info: Dict keyed by IMDB tt-id with 'genres' key.

    Returns:
        List of seeker dicts with 'genres' key added.
    """
    # Phase 1: Get system replies from cache (avoids re-scanning 1.6M rows)
    conv_system = _build_conv_system(all_rows)

    # Phase 2: Pre-compute genres per turn (cache, not per-seeker)
    tt_pattern = re.compile(r'\b(tt\d+)\b')
    turn_genres_cache: dict[tuple[str, str], set[str]] = {}

    # Collect all unique (session_id, turn_num) keys from seekers first
    need_turns = set()
    for r in seekers:
        conv_id = r.get('conv_id', '')
        need_turns.add(parse_conv_turn(conv_id))

    for key in need_turns:
        system_msgs = conv_system.get(key, [])
        if not system_msgs:
            turn_genres_cache[key] = {'unknown'}
            continue

        movie_ids = set()
        for msg in system_msgs:
            movie_ids.update(tt_pattern.findall(str(msg)))

        genres_found = set()
        for mid in movie_ids:
            info = movie_info.get(mid, {})
            if isinstance(info, dict):
                genre_list = info.get('genres', []) or []
                if genre_list:
                    genres_found.update(g.strip() for g in genre_list if g.strip())

        turn_genres_cache[key] = genres_found if genres_found else {'unknown'}

    # Phase 3: Assign cached genres to each seeker
    result = []
    for r in seekers:
        conv_id = r.get('conv_id', '')
        key = parse_conv_turn(conv_id)
        rec = dict(r)
        rec['genres'] = turn_genres_cache.get(key, {'unknown'})
        result.append(rec)
    return result


# ═══════════════════════════════════════════════════════════════════════
#  N1: Query Intent Classification & Holiday Patterns
# ═══════════════════════════════════════════════════════════════════════

def _plot_intent_distribution(
    intent_stats: dict[str, dict[str, int]],
    avg_turns: dict[str, dict[str, float]],
    filename: str,
):
    """Plot intent distribution and engagement depth side by side."""
    periods = ['holiday', 'workday', 'weekend']
    period_labels = {'holiday': 'Holiday', 'workday': 'Workday', 'weekend': 'Weekend'}
    intents = ['recommendation', 'identification', 'opinion', 'factual', 'comparison', 'general']
    intent_labels = {
        'recommendation': 'Recommendation',
        'identification': 'Identification',
        'opinion': 'Opinion',
        'factual': 'Factual',
        'comparison': 'Comparison',
        'general': 'General',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Intent distribution (stacked bar) ──
    x = np.arange(len(periods))
    width = 0.55
    bottoms = np.zeros(len(periods))
    for intent in intents:
        vals = [intent_stats.get(p, {}).get(intent, 0) for p in periods]
        total = sum(vals)
        pcts = [v / total * 100 if total > 0 else 0 for v in vals]
        ax1.bar(x, pcts, width, bottom=bottoms,
                label=intent_labels[intent],
                color=INTENT_COLORS[intent])
        bottoms += pcts

    ax1.set_xticks(x)
    ax1.set_xticklabels([period_labels[p] for p in periods], fontsize=10)
    ax1.set_ylabel('Proportion of Queries (%)', fontsize=10)
    ax1.set_title('Query Intent Distribution by Period', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_ylim(0, 105)

    # ── Right: Engagement depth (avg system turns) ──
    x2 = np.arange(len(intents))
    w = 0.25
    for i, period in enumerate(periods):
        vals = [avg_turns.get(period, {}).get(intent, 0) for intent in intents]
        ax2.bar(x2 + i * w, vals, w,
                label=period_labels[period],
                color=PERIOD_COLORS[period])

    ax2.set_xticks(x2 + w)
    ax2.set_xticklabels([intent_labels[i] for i in intents], fontsize=8, rotation=30, ha='right')
    ax2.set_ylabel('Avg System Replies per Session', fontsize=10)
    ax2.set_title('Engagement Depth by Intent & Period', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_intent_period_heatmap(
    intent_stats: dict[str, dict[str, int]],
    filename: str,
):
    """Plot normalized heatmap: intent x period showing over/under representation."""
    periods = ['holiday', 'workday', 'weekend']
    period_labels = ['Holiday', 'Workday', 'Weekend']
    intents = ['recommendation', 'identification', 'opinion', 'factual', 'comparison', 'general']
    intent_labels = ['Recommendation', 'Identification', 'Opinion', 'Factual', 'Comparison', 'General']

    matrix = np.zeros((len(periods), len(intents)))
    for i, p in enumerate(periods):
        total = sum(intent_stats.get(p, {}).values())
        for j, intent in enumerate(intents):
            cnt = intent_stats.get(p, {}).get(intent, 0)
            matrix[i, j] = cnt / total * 100 if total > 0 else 0

    if matrix.size == 0:
        log("No data for intent-period heatmap", "Step12")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')

    for i in range(len(periods)):
        for j in range(len(intents)):
            val = matrix[i, j]
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=9, color='white' if val > matrix.max() * 0.6 else 'black')

    ax.set_xticks(range(len(intents)))
    ax.set_xticklabels(intent_labels, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(len(periods)))
    ax.set_yticklabels(period_labels, fontsize=10)
    ax.set_title('Query Intent × Period (% of queries)', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, label='% of queries in period')

    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  N1: 查询意图分类与节假日模式
#  N1: Query Intent Classification & Holiday Patterns
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】双面板: 左=意图分布柱状图(holiday/workday/weekend), 
#            右=意图×period热力图
# 
# 【统计口径】
#   意图分类: 使用规则/正则匹配对 proc_text 进行分类
#     - _classify_intent(text) → 'movie_search'/'recommendation'/'info_query'/...
#   统计各 period(holiday/workday/weekend) 下各意图的数量和占比
#   会话深度: _session_turn_counts() 统计各意图下的平均对话轮次
# 
# 【坐标轴】
#   柱状图: X轴=意图类别, Y轴=占比(%), 分三组对比
#   热力图: 行=意图, 列=period, 值=占比
# 
# 【输出文件】
#   PNG: n1_intent_distribution.png (意图分布柱状图)
#   PNG: n1_intent_heatmap.png (意图×时期热力图)
# 
# 【特殊说明】
#   - 使用 _build_conv_system() 缓存的会话数据(模块级缓存避免重复扫描)
#   - 意图分类规则定义在 _classify_intent() 中
#   - 会话轮次统计用于评估不同意图下的参与深度
# 
# 【代码中处理逻辑】
#   1. 遍历 seekers, 调用 _classify_intent(text) 给每条记录打上意图标签 s['_intent']
#   2. 按 period 分组统计各意图计数 → intent_stats[period][intent] += 1
#   3. 计算各组的占比并输出日志
#   4. 调用 _session_turn_counts() 计算各意图的平均会话轮次
#   5. 绘制双面板图: _plot_intent_distribution() + _plot_intent_period_heatmap()
# ═══════════════════════════════════════════════════════════════════════

def n1_intent_holiday_patterns(seekers: list[dict], rows: list[dict]):
    """N1: Classify query intents and compare across holiday/workday/weekend.

    Args:
        seekers: List of seeker records (must have 'period', 'date', 'proc_text').
        rows: Full conversation rows for turn count analysis.
    """
    log("=" * 50, "Step12")
    log("N1: Query Intent Classification & Holiday Patterns", "Step12")
    log("=" * 50, "Step12")

    if not seekers:
        log("No seeker data available, skipping N1", "Step12")
        return

    # Classify each seeker
    for s in seekers:
        text = s.get('proc_text', '') or s.get('raw_text', '')
        s['_intent'] = _classify_intent(text)

    # Count intents per period
    intent_stats = defaultdict(lambda: defaultdict(int))
    for s in seekers:
        period = s.get('period', 'workday')
        intent = s.get('_intent', 'general')
        intent_stats[period][intent] += 1

    # Log summary
    log("Intent distribution by period:", "Step12")
    for period in ['holiday', 'workday', 'weekend']:
        stats = intent_stats.get(period, {})
        total = sum(stats.values())
        if total > 0:
            parts = ', '.join(f'{k}: {v} ({v/total*100:.1f}%)'
                             for k, v in sorted(stats.items(), key=lambda x: -x[1]))
            log(f"  {period}: {total} total — {parts}", "Step12")

    # Compute average turns per session (engagement depth)
    avg_turns = _session_turn_counts(seekers, rows)

    # Visualize
    _plot_intent_distribution(dict(intent_stats), avg_turns, 'n1_intent_distribution.png')
    _plot_intent_period_heatmap(dict(intent_stats), 'n1_intent_heatmap.png')

    # Key findings
    log("N1 complete. Charts saved.", "Step12")


# ═══════════════════════════════════════════════════════════════════════
#  N2: 情感-类型关联分析 (节假日对比)
#  N2: Sentiment-Genre Affinity by Holiday Period
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】双面板: 左=情感×类型热力图(holiday vs baseline),
#            右=情感×类型分组柱状图对比
# 
# 【统计口径】
#   数据准备: _build_seeker_genres() 将情感数据与类型提及关联
#     - 从 conversation rows 中提取 system reply 中的电影类型
#     - 关联到每条 seeker 记录的情感标签(sentiment)
#   分组: holiday(节假日) vs baseline(工作日+周末, 等量采样)
#   情感分类: positive / neutral / negative
#   统计各(情感, 类型)组合下的提及次数或占比
# 
# 【坐标轴】
#   热力图: X轴=情感类别(positive/neutral/negative), Y轴=电影类型
#   柱状图: 分组对比 holiday 与 baseline 下各类型的正向情感占比
# 
# 【输出文件】
#   PNG: n2_sentiment_genre_heatmap.png (情感×类型热力图)
#   PNG: n2_sentiment_genre_comparison.png (节假日VS基线对比柱状图)
# 
# 【特殊说明】
#   - 使用模块级缓存 _CONV_SYSTEM_CACHE 避免重复扫描完整数据
#   - 节假日与基线等量采样保证对比公平性
#   - 仅保留 sentiment 为 positive/neutral/negative 的记录
#   - 过滤 'unknown' 类型
# 
# 【代码中处理逻辑】
#   1. 调用 _build_seeker_genres() 获取带有情感和类型信息的记录
#   2. 过滤 sentiment 为 positive/neutral/negative 的记录
#   3. 收集所有有效类型(排除 unknown)
#   4. 构建情感×类型的计数矩阵 (holiday 和 baseline 分别)
#   5. 绘制左面板热力图 _plot_sentiment_genre_heatmap()
#   6. 绘制右面板柱状图 _plot_sentiment_genre_comparison()
# ═══════════════════════════════════════════════════════════════════════

def _plot_sentiment_genre_heatmap(
    matrix: np.ndarray,
    genre_names: list[str],
    sentiment_order: list[str],
    period: str,
    filename: str,
):
    """Heatmap: sentiment x genre for a given period."""
    if matrix.size == 0:
        return

    sentiment_labels = {
        'positive': 'Positive',
        'neutral':  'Neutral',
        'negative': 'Negative',
    }

    fig_h = max(4, len(genre_names) * 0.3 + 2)
    fig_w = max(6, len(sentiment_order) * 1.5 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix.T, aspect='auto', cmap='YlOrRd')
    # Normalize each row (sentiment) to % for fair comparison
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    pct_matrix = matrix / row_sums * 100

    for i in range(len(genre_names)):
        for j in range(len(sentiment_order)):
            val = pct_matrix[j, i]
            if val > 0:
                ax.text(i, j, f'{val:.1f}%', ha='center', va='center',
                        fontsize=8,
                        color='white' if val > pct_matrix.max() * 0.6 else 'black')

    ax.set_xticks(range(len(genre_names)))
    ax.set_xticklabels(genre_names, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(len(sentiment_order)))
    ax.set_yticklabels([sentiment_labels[s] for s in sentiment_order], fontsize=9)
    ax.set_title(f'Sentiment × Genre Affinity – {period.capitalize()}', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, label='Raw mention count')

    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_sentiment_genre_comparison(
    holiday_matrix: np.ndarray,
    baseline_matrix: np.ndarray,
    genre_names: list[str],
    sentiment_order: list[str],
    filename: str,
):
    """Side-by-side comparison: holiday vs baseline sentiment-genre affinity."""
    sentiment_labels = {
        'positive': 'Positive',
        'neutral':  'Neutral',
        'negative': 'Negative',
    }

    n_genres = len(genre_names)
    n_sentiments = len(sentiment_order)
    if n_genres == 0 or n_sentiments == 0:
        return

    # Normalize to % within each sentiment
    def _normalize(mat):
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return mat / row_sums * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n_genres * 0.3 + 2)))

    for idx, (mat, title, ax) in enumerate([
        (holiday_matrix, 'Holiday', axes[0]),
        (baseline_matrix, 'Baseline (Non-Holiday)', axes[1]),
    ]):
        pct = _normalize(mat)
        im = ax.imshow(pct, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
        for i in range(n_genres):
            for j in range(n_sentiments):
                val = pct[j, i]
                if val > 0:
                    ax.text(i, j, f'{val:.0f}%', ha='center', va='center',
                            fontsize=7,
                            color='white' if val > 60 else 'black')

        ax.set_xticks(range(n_genres))
        ax.set_xticklabels(genre_names, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(n_sentiments))
        ax.set_yticklabels([sentiment_labels[s] for s in sentiment_order], fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')

    fig.colorbar(im, ax=axes, shrink=0.6, label='% within sentiment')
    fig.suptitle('Genre Preference by Sentiment: Holiday vs Baseline', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


def n2_sentiment_genre_affinity(
    seekers: list[dict],
    all_rows: list[dict],
    movie_info: dict,
):
    """N2: Analyze sentiment-genre affinity by holiday period.

    Cross-references seeker sentiment data with genre mentions from
    system replies to discover which genres are associated with each
    sentiment state during holidays vs baseline periods.

    Args:
        seekers: List of seeker records (must have 'sentiment', 'period').
        all_rows: Full conversation rows (for system reply genre lookup).
        movie_info: Dict of movie info keyed by IMDB tt-id.
    """
    log("=" * 50, "Step12")
    log("N2: Sentiment-Genre Affinity by Holiday Period", "Step12")
    log("=" * 50, "Step12")

    if not seekers:
        log("No seeker data available, skipping N2", "Step12")
        return

    # Get seekers with genre info and sentiment
    seeker_genres = _build_seeker_genres(seekers, all_rows, movie_info)

    # Filter to records with valid sentiment
    valid = [s for s in seeker_genres if s.get('sentiment') in ('positive', 'neutral', 'negative')]
    log(f"N2: {len(valid)} seekers with sentiment + genre data", "Step12")

    if len(valid) < 5:
        log("Too few records for N2 analysis, skipping", "Step12")
        return

    # Collect all known genres across the data (filter 'unknown')
    all_genres = set()
    for s in valid:
        for g in s.get('genres', set()):
            if g != 'unknown':
                all_genres.add(g)
    genre_list = sorted(all_genres)
    sentiment_order = ['positive', 'neutral', 'negative']
    log(f"N2: {len(genre_list)} genres found across {len(valid)} records", "Step12")

    if not genre_list:
        log("No valid genres found, skipping N2", "Step12")
        return

    # Build period × sentiment × genre count matrix
    period_genre_sent = {
        'holiday':  np.zeros((len(sentiment_order), len(genre_list)), dtype=int),
        'baseline': np.zeros((len(sentiment_order), len(genre_list)), dtype=int),
    }

    for s in valid:
        period = s.get('period', 'workday')
        sent = s.get('sentiment', 'neutral')
        if sent not in ('positive', 'neutral', 'negative'):
            continue
        sent_idx = sentiment_order.index(sent)
        bucket = 'holiday' if period == 'holiday' else 'baseline'
        for g in s.get('genres', set()):
            if g in all_genres:
                g_idx = genre_list.index(g)
                period_genre_sent[bucket][sent_idx, g_idx] += 1

    # Filter to top genres by total mentions
    total_mentions = period_genre_sent['holiday'] + period_genre_sent['baseline']
    genre_totals = total_mentions.sum(axis=0)
    top_n = 15
    if len(genre_list) > top_n:
        top_indices = np.argsort(genre_totals)[-top_n:]
        top_indices = sorted(top_indices)
        genre_list_top = [genre_list[i] for i in top_indices]
        for bucket in period_genre_sent:
            period_genre_sent[bucket] = period_genre_sent[bucket][:, top_indices]
    else:
        genre_list_top = genre_list

    # Log findings
    for bucket in ['holiday', 'baseline']:
        mat = period_genre_sent[bucket]
        log(f"  {bucket}: total sentiment-genre pairs = {mat.sum()}", "Step12")
        top_genres = sorted(
            [(genre_list_top[i], mat[:, i].sum()) for i in range(len(genre_list_top))],
            key=lambda x: -x[1]
        )[:5]
        log(f"    Top genres: {', '.join(f'{g}({c})' for g, c in top_genres)}", "Step12")

    # Visualize per-period heatmaps
    for bucket in ['holiday', 'baseline']:
        _plot_sentiment_genre_heatmap(
            period_genre_sent[bucket],
            genre_list_top,
            sentiment_order,
            bucket,
            f'n2_{bucket}_sentiment_genre.png',
        )

    # Side-by-side comparison
    _plot_sentiment_genre_comparison(
        period_genre_sent['holiday'],
        period_genre_sent['baseline'],
        genre_list_top,
        sentiment_order,
        'n2_sentiment_genre_comparison.png',
    )

    log("N2 complete. Charts saved.", "Step12")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Run N1 and N2 analyses."""
    from movie.data_loader import load_all
    data = load_all()
    seekers = data['seekers']
    rows = data['rows']
    movie_info = data.get('movie_info', {})

    n1_intent_holiday_patterns(seekers, rows)

    # N2: sample holiday + balanced baseline, then run sentiment inline
    holiday_s = [s for s in seekers if s.get('period') == 'holiday']
    non_holiday_s = [s for s in seekers if s.get('period') != 'holiday']
    # Sample baseline to match holiday count for balanced comparison
    import random
    random.seed(42)
    if len(non_holiday_s) > len(holiday_s):
        non_holiday_s = random.sample(non_holiday_s, len(holiday_s))
    sample_seekers = holiday_s + non_holiday_s
    log(f"N2: sampled {len(holiday_s)} holiday + {len(non_holiday_s)} baseline "
        f"= {len(sample_seekers)} total for genre-sentiment analysis", "Step12")

    # Check if sentiment already present (from step11 pipeline)
    has_sentiment = any(s.get('sentiment') in ('positive', 'neutral', 'negative')
                        for s in sample_seekers[:100])
    if not has_sentiment:
        log("N2: running sentiment analysis on sample...", "Step12")
        try:
            from data_analyzer.sentiment import analyze_batch
            texts = [s.get('proc_text', '') or s.get('raw_text', '') for s in sample_seekers]
            results = analyze_batch(texts)
            for s, res in zip(sample_seekers, results):
                s['sentiment'] = res['sentiment']
                s['intensity'] = res['intensity']
                s['sentiment_score'] = res['score']
            log("N2: sentiment analysis complete", "Step12")
        except ImportError:
            log("N2: data_analyzer.sentiment not available, skipping", "Step12")
            return
        except Exception as e:
            log(f"N2: sentiment analysis failed: {e}, skipping", "Step12")
            return

    n2_sentiment_genre_affinity(sample_seekers, rows, movie_info)


if __name__ == '__main__':
    main()
