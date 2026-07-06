# -*- coding: utf-8 -*-
"""
Step 11: Sentiment Analysis of Movie-Related Queries
步骤 11：电影相关查询的情感分析

Analysis:
  S1 - Holiday vs Non-Holiday sentiment & intensity comparison (dual-panel)
  S2 - Sentiment by period (holiday / workday / weekend) with intensity (dual-panel)
  S3 - Per-holiday sentiment profile
  S4 - Per-holiday genre sentiment heatmap vs non-holiday baseline
  S5 - Sentiment-associated keywords (positive / negative markers)
  S6 - Sentiment comparison: holiday vs week-before / month-before baseline
  S7 - Intensity comparison: holiday vs baseline
  S8 - Intensity distribution across holidays
  S9 - Sentiment vs global non-holiday average
  S10 - Intensity vs global non-holiday average

Dependencies: vaderSentiment, afinn (via data_analyzer.sentiment)
Output: output/movie/step11/*.png + CSV
"""

import os               # 文件路径操作
import csv              # CSV 读写
import re               # 正则表达式，用于分词
import calendar          # 日期计算（月前基线）
from collections import Counter, defaultdict  # 计数器与默认字典
from datetime import datetime, timedelta      # 基线日期计算

import numpy as np      # 数值计算

import matplotlib
matplotlib.use('Agg')   # 非交互式后端（服务器环境）
import matplotlib.pyplot as plt

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log
from movie.utils.text import tokenize as _shared_tokenize, deduplicate_seekers
from data_analyzer.sentiment import analyze_batch

# ── 初始化 ──────────────────────────────────────────────────────────
setup_matplotlib()
STEP_OUT = STEP_DIRS[11]                        # 输出目录：output/movie/step11/
os.makedirs(STEP_OUT, exist_ok=True)

# ── 颜色方案（与步骤 7/8 风格一致）─────────────────────────────────
SENT_COLORS = {
    'positive': '#2ecc71',
    'neutral':  '#95a5a6',
    'negative': '#e74c3c',
}
INTEN_COLORS = {
    'mild':     '#74b9ff',
    'moderate': '#feca57',
    'strong':   '#ff6b6b',
}
PERIOD_COLORS = {
    '节假日':  '#ff6b6b',
    '工作日':  '#74b9ff',
    '周末':    '#feca57',
}

# ── S5 关键词分析用的轻量停用词表（不含情感词汇）──────────────────
_S5_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'i', 'you', 'he', 'she',
    'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'its', 'our', 'their', 'mine', 'yours', 'this', 'that',
    'these', 'those', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
    'for', 'with', 'on', 'in', 'at', 'to', 'from', 'by', 'about', 'into',
    'through', 'during', 'before', 'after', 'of', 'up', 'down', 'out',
    'off', 'over', 'under', 'again', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
    'than', 'too', 'very', 'just', 'because', 'as', 'until', 'while',
    'if', 'else', 'like', 'also', 'any', 'many', 'much', 'who', 'what',
    'which', 'im', 'ive', 'id', 'youre', 'youve', 'dont', 'doesnt',
    'didnt', 'wont', 'wouldnt', 'couldnt', 'shouldnt', 'isnt', 'arent',
    'wasnt', 'werent', 'hasnt', 'havent', 'hadnt',
    'movie', 'movies', 'film', 'films', 'show', 'shows', 'watch',
    'watched', 'watching', 'looking', 'look', 'know', 'seen', 'seeing',
    'go', 'going', 'want', 'wants', 'need', 'needs', 'get', 'got',
    'make', 'made', 'say', 'says', 'said', 'think', 'thinks', 'tell',
    'tells', 'told', 'find', 'finds', 'found', 'give', 'gives', 'gave',
    'try', 'tries', 'tried', 'thanks', 'please', 'help', 'really',
    'actually', 'well', 'even', 'still', 'though', 'thing', 'things',
    'something', 'lot', 'lots', 'bit', 'way', 'ways', 'time', 'times',
    'day', 'days', 'year', 'years', 'new', 'old', 'first', 'last',
    'next', 'ever', 'never', 'always', 'also', 'much', 'many', 'back',
    'around', 'away', 'maybe', 'perhaps', 'probably', 'definitely',
    'pretty', 'quite', 'rather', 'guess', 'wonder', 'suppose',
    'yeah', 'yea', 'ok', 'okay', 'oh', 'hmm', 'haha', 'lol',
    'actually', 'honestly', 'seriously', 'hopefully', 'unfortunately',
    'supposed', 'gonna', 'gotta', 'wanna', 'tryna', 'yall',
    'guys', 'guy', 'people', 'person', 'someone', 'anyone', 'everyone',
    'something', 'anything', 'everything', 'nothing',
    'let', 'took', 'take', 'takes', 'taken', 'using', 'use', 'used',
    'reddit', 'post', 'sub', 'title', 'amp', 'x200b', 'gt', 'br',
    'https', 'http', 'www', 'com', 'org', 'edit', 'update',
    'nt', 've', 'll', 're', 'one', 'two', 'list', 'end', 'long',
    'big', 'top', 'done', 'favorite', 'anyone', 'any', 'somebody',
    'everybody', 'nobody', 'anybody', 'everywhere',
    'also', 'else', 'though', 'although', 'however', 'therefore',
    'thus', 'hence', 'furthermore', 'meanwhile',
    'nevertheless', 'nonetheless', 'moreover', 'besides', 'indeed',
    'instead', 'regarding', 'concerning', 'including', 'except',
    'without', 'within', 'upon', 'across', 'along', 'among',
    'amongst', 'throughout', 'outside', 'inside', 'around',
    'behind', 'beneath', 'beside', 'beyond', 'via',
    'versus', 'vs', 'per',
    'ntt', 'utm', 'nedit', 'nthanks', 'nthe', 'nany', 'nso',
    'nthank', 'nsome', 'nwhat', 'nalso', 'nif', 'nmovies',
    'nmy', 'nand', 'nfor', 'nit', 'nwe', 'ni', 'ctt', 'cxld',
    'thett', 'andtt', 'mitt', 'mett', 'nm',
}


# ── 数据准备 ────────────────────────────────────────────────────────

def tokenize_simple(text: str) -> list[str]:
    """Tokenize for keyword analysis using shared utility."""
    return _shared_tokenize(text, stopwords=_S5_STOPWORDS)


# deduplicate_seekers is imported from movie.utils.text


def annotate_sentiment(seekers: list[dict]) -> list[dict]:
    """Run sentiment analysis on all seekers and attach results in-place."""
    texts = [r.get('proc_text', '') or r.get('raw_text', '') for r in seekers]
    results = analyze_batch(texts)
    for r, res in zip(seekers, results):
        r['sentiment'] = res['sentiment']
        r['intensity'] = res['intensity']
        r['sentiment_score'] = res['score']
    return seekers


def _prepare_text(seekers: list[dict], idx: int) -> str:
    """Get text for a seeker record (proc_text fallback to raw_text)."""
    return seekers[idx].get('proc_text', '') or seekers[idx].get('raw_text', '')


# ── 辅助绘图 ────────────────────────────────────────────────────────

def _plot_sentiment_pie(counts: dict[str, int], title: str, filename: str,
                        colors: dict[str, str] = None):
    """Pie chart of sentiment or intensity distribution."""
    labels = list(counts.keys())
    values = list(counts.values())
    if not values or sum(values) == 0:
        return
    clrs = [colors.get(k, '#95a5a6') for k in labels] if colors else None
    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.1f%%',
        colors=clrs, startangle=90, pctdistance=0.78,
        textprops={'fontsize': 10},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_grouped_bar(
    group_data: dict[str, dict[str, float]],
    title: str, filename: str,
    ylabel: str = 'Proportion',
    colors: dict[str, str] = None,
    ylim_top: float = None,
):
    """Grouped bar chart of distributions across groups."""
    categories = list(next(iter(group_data.values())).keys())
    groups = list(group_data.keys())

    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.2), 5))
    x = np.arange(len(categories))
    n_groups = len(groups)
    width = 0.7 / max(n_groups, 1)

    default_c = ['#ff6b6b', '#74b9ff', '#feca57', '#48dbfb', '#a29bfe', '#fd79a8']
    for i, group in enumerate(groups):
        vals = [group_data[group].get(c, 0) for c in categories]
        offset = (i - (n_groups - 1) / 2) * width
        c = colors.get(group, default_c[i]) if colors else default_c[i]
        ax.bar(x + offset, vals, width, label=group, color=c, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    if ylim_top:
        ax.set_ylim(0, ylim_top)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_dual_grouped_bar(
    left_data: dict[str, dict[str, float]],
    left_title: str,
    right_data: dict[str, dict[str, float]],
    right_title: str,
    filename: str,
    colors: dict[str, str] = None,
    ylabel: str = 'Proportion',
    suptitle: str = '',
):
    """Two grouped bar charts side by side in one figure (left + right panels)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    def _plot_on_ax(ax, data, title_text):
        categories = list(next(iter(data.values())).keys())
        groups = list(data.keys())
        x = np.arange(len(categories))
        n_groups = len(groups)
        width = 0.7 / max(n_groups, 1)
        default_c = ['#ff6b6b', '#74b9ff', '#feca57', '#48dbfb', '#a29bfe', '#fd79a8']
        for i, group in enumerate(groups):
            vals = [data[group].get(c, 0) for c in categories]
            offset = (i - (n_groups - 1) / 2) * width
            c = colors.get(group, default_c[i]) if colors else default_c[i]
            ax.bar(x + offset, vals, width, label=group, color=c, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title_text, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    _plot_on_ax(ax1, left_data, left_title)
    _plot_on_ax(ax2, right_data, right_title)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_horizontal_bar(
    items: list[tuple[str, float]],
    title: str, filename: str,
    xlabel: str = 'Value',
    top_n: int = 20,
    color: str = '#74b9ff',
):
    """Horizontal bar chart for ranked items."""
    top = items[:top_n]
    if not top:
        return
    labels = [t[0] for t in top[::-1]]
    values = [t[1] for t in top[::-1]]

    fig, ax = plt.subplots(figsize=(max(7, top_n * 0.35), max(4, top_n * 0.35)))
    ax.barh(range(len(labels)), values, color=color, alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str, filename: str,
    fmt: str = '.2f',
    cmap: str = 'RdBu_r',
    center: float = None,
    cbar_label: str = '',
):
    """Heatmap with labeled rows and columns."""
    if matrix.size == 0:
        return
    fig_h = max(5, len(row_labels) * 0.35 + 2)
    fig_w = max(7, len(col_labels) * 0.8 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect='auto', cmap=cmap,
                   **({} if center is None else {'vmin': -abs(matrix).max(),
                                                   'vmax': abs(matrix).max()}))
    # 在热力图上标注数值
    _mat = np.ma.getdata(matrix) if isinstance(matrix, np.ma.MaskedArray) else np.asarray(matrix)
    for _i in range(_mat.shape[0]):
        for _j in range(_mat.shape[1]):
            _v = _mat[_i, _j]
            if not np.isnan(_v) and abs(_v) > 1e-6:
                ax.text(_j, _i, f'{_v:.1f}', ha='center', va='center',
                        fontsize=5, color='black')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.6, label=cbar_label)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  分析维度
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  S1: 节假日 VS 非节假日 情感与强度 (Dual Panel)
#  S1: Holiday vs Non-Holiday Sentiment & Intensity
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】双面板图: 上=情感得分分布(柱状/提琴), 下=情感强度分布
# 【统计口径】
#   情感得分(sentiment): -1~1, 基于文本的 VADER/TextBlob 情感分析
#   情感强度(intensity): 0~1, |sentiment| 绝对值
#   按 period=holiday/non_holiday 分组
# 【输出文件】PNG: s1_holiday_vs_nonholiday.png, CSV: s1_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_s1_holiday_vs_nonholiday(seekers: list[dict]):
    """节假日 vs 非节假日情感与强度对比（双面板）。"""
    log("=" * 50)
    log("S1: 节假日 vs 非节假日情感与强度对比")

    holiday = [r for r in seekers if r['period'] == 'holiday']
    non_holiday = [r for r in seekers if r['period'] != 'holiday']

    h_n = len(holiday)
    nh_n = len(non_holiday)
    log(f"  节假日: {h_n} 条记录, 非节假日: {nh_n} 条记录")

    if h_n == 0 or nh_n == 0:
        log("  其中一组为空，无法对比")
        return

    # ── Sentiment proportions ──
    h_sent = Counter(r['sentiment'] for r in holiday)
    nh_sent = Counter(r['sentiment'] for r in non_holiday)

    sent_data = {
        '节假日':     {k: h_sent.get(k, 0) / h_n for k in ['positive', 'neutral', 'negative']},
        '非节假日':   {k: nh_sent.get(k, 0) / nh_n for k in ['positive', 'neutral', 'negative']},
    }

    # ── Intensity proportions ──
    h_int = Counter(r['intensity'] for r in holiday)
    nh_int = Counter(r['intensity'] for r in non_holiday)

    inten_data = {
        '节假日':     {k: h_int.get(k, 0) / h_n for k in ['mild', 'moderate', 'strong']},
        '非节假日':   {k: nh_int.get(k, 0) / nh_n for k in ['mild', 'moderate', 'strong']},
    }

    # Log summary
    log(f"  节假日    — 正面={sent_data['节假日']['positive']*100:.1f}%, "
        f"负面={sent_data['节假日']['negative']*100:.1f}%, "
        f"强烈={inten_data['节假日']['strong']*100:.1f}%")
    log(f"  非节假日  — 正面={sent_data['非节假日']['positive']*100:.1f}%, "
        f"负面={sent_data['非节假日']['negative']*100:.1f}%, "
        f"强烈={inten_data['非节假日']['strong']*100:.1f}%")

    group_colors = {'节假日': '#ff6b6b', '非节假日': '#74b9ff'}
    _plot_dual_grouped_bar(
        sent_data, '情感倾向分布',
        inten_data, '强度分布',
        's1_holiday_vs_nonholiday.png',
        colors=group_colors,
        suptitle='节假日 vs 非节假日情感与强度对比',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 's1_holiday_vs_nonholiday.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'metric', 'class', 'count', 'proportion_pct'])
        for group, n, sent_c, int_c in [
            ('holiday', h_n, h_sent, h_int),
            ('non_holiday', nh_n, nh_sent, nh_int),
        ]:
            for cls, cnt in sorted(sent_c.items()):
                w.writerow([group, 'sentiment', cls, cnt, f'{cnt / n * 100:.1f}'])
            for cls, cnt in sorted(int_c.items()):
                w.writerow([group, 'intensity', cls, cnt, f'{cnt / n * 100:.1f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S2: 按时期的情感与强度分布 (Dual Panel)
#  S2: Sentiment & Intensity by Period
# ═══════════════════════════════════════════════════════════════════════
# 【统计口径】3组(holiday/workday/weekend) 的情感与强度分布
# 【输出文件】PNG: s2_sentiment_by_period.png, CSV: s2_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_s2_by_period(seekers: list[dict]):
    """按时期的情感与强度分布（双面板）。"""
    log("=" * 50)
    log("S2: 按时期的情感与强度分布")

    periods = ['holiday', 'workday', 'weekend']
    period_cn = {'holiday': '节假日', 'workday': '工作日', 'weekend': '周末'}
    sent_data = {}
    inten_data = {}
    period_summaries = []

    for p in periods:
        subset = [r for r in seekers if r['period'] == p]
        if not subset:
            continue
        n = len(subset)
        label = period_cn[p]

        # Sentiment
        s_counts = Counter(r['sentiment'] for r in subset)
        sent_data[label] = {
            k: s_counts.get(k, 0) / n for k in ['positive', 'neutral', 'negative']
        }

        # Intensity
        i_counts = Counter(r['intensity'] for r in subset)
        inten_data[label] = {
            k: i_counts.get(k, 0) / n for k in ['mild', 'moderate', 'strong']
        }

        avg = np.mean([r['sentiment_score'] for r in subset])
        period_summaries.append(
            f"  {label}: 情感={dict(s_counts)}, 强度={dict(i_counts)}, "
            f"平均分={avg:.3f} ({n} 条记录)"
        )

    for line in period_summaries:
        log(line)

    if not sent_data or not inten_data:
        log("  无可用时期数据")
        return

    _plot_dual_grouped_bar(
        sent_data, '情感倾向分布',
        inten_data, '强度分布',
        's2_sentiment_intensity_by_period.png',
        colors=PERIOD_COLORS,
        suptitle='按时期的情感与强度分布（节假日 / 工作日 / 周末）',
    )

    # Save CSV
    csv_path = os.path.join(STEP_OUT, 's2_sentiment_intensity_by_period.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['period', 'total',
                     'pos', 'pos_pct', 'neu', 'neu_pct', 'neg', 'neg_pct',
                     'mild', 'mild_pct', 'moderate', 'moderate_pct',
                     'strong', 'strong_pct', 'avg_score'])
        for p in periods:
            subset = [r for r in seekers if r['period'] == p]
            if not subset:
                continue
            n = len(subset)
            s_counts = Counter(r['sentiment'] for r in subset)
            i_counts = Counter(r['intensity'] for r in subset)
            avg = np.mean([r['sentiment_score'] for r in subset])
            w.writerow([
                p, n,
                s_counts.get('positive', 0),
                f'{s_counts.get("positive", 0) / n * 100:.1f}',
                s_counts.get('neutral', 0),
                f'{s_counts.get("neutral", 0) / n * 100:.1f}',
                s_counts.get('negative', 0),
                f'{s_counts.get("negative", 0) / n * 100:.1f}',
                i_counts.get('mild', 0),
                f'{i_counts.get("mild", 0) / n * 100:.1f}',
                i_counts.get('moderate', 0),
                f'{i_counts.get("moderate", 0) / n * 100:.1f}',
                i_counts.get('strong', 0),
                f'{i_counts.get("strong", 0) / n * 100:.1f}',
                f'{avg:.3f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S3: 各节假日情感画像 (Heatmap)
#  S3: Per-Holiday Sentiment Profile
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】热力图: 行=节假日, 列=情感得分区间, 值=占比差值 vs 全局基线
# 【输出文件】PNG: s3_per_holiday_sentiment.png, CSV: s3_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_s3_per_holiday(seekers: list[dict]):
    """各节假日情感画像，热力图 vs 全局基线。"""
    log("=" * 50)
    log("S3: 各节假日情感画像")

    # Global baseline
    global_counts = Counter(r['sentiment'] for r in seekers)
    global_n = len(seekers)
    global_pct = {k: global_counts.get(k, 0) / max(global_n, 1)
                  for k in ['positive', 'neutral', 'negative']}

    # Group by holiday
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['is_holiday']:
            name = r.get('holiday_name', '')[:8]
            holiday_groups[name].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items()
                      if len(v) >= MIN_DATA_ROWS}

    if not holiday_groups:
        log("  无数据足够的节假日分组")
        return

    holiday_names = sorted(holiday_groups.keys())

    # Build heatmap matrix: holidays × (positive, neutral, negative) as % diff from global
    matrix = np.zeros((len(holiday_names), 3))
    csv_rows = []
    for i, hn in enumerate(holiday_names):
        group = holiday_groups[hn]
        n = len(group)
        counts = Counter(r['sentiment'] for r in group)
        avg_score = np.mean([r['sentiment_score'] for r in group])
        row = {'holiday': hn, 'count': n}
        for j, sent in enumerate(['positive', 'neutral', 'negative']):
            pct = counts.get(sent, 0) / max(n, 1)
            diff = (pct - global_pct[sent]) * 100  # percentage points diff
            matrix[i, j] = diff
            row[f'{sent}_pct'] = f'{pct * 100:.1f}'
            row[f'{sent}_diff'] = f'{diff:+.1f}'
        row['avg_score'] = f'{avg_score:.3f}'
        csv_rows.append(row)
        log(f"  {hn} (n={n}): 正面={counts.get('positive', 0)}, "
            f"中性={counts.get('neutral', 0)}, 负面={counts.get('negative', 0)}, "
            f"平均分={avg_score:.3f}")

    # Heatmap
    _plot_heatmap(matrix, holiday_names,
                  ['正面', '中性', '负面'],
                  '各节假日情感差异 vs 全局基线（百分点）',
                  's3_per_holiday_sentiment.png',
                  fmt='+.1f', cmap='RdBu_r', cbar_label='vs 全局基线差异（pp）')

    # CSV
    csv_path = os.path.join(STEP_OUT, 's3_per_holiday_sentiment.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['holiday', 'count',
                                          'positive_pct', 'positive_diff',
                                          'neutral_pct', 'neutral_diff',
                                          'negative_pct', 'negative_diff',
                                          'avg_score'])
        w.writeheader()
        w.writerows(csv_rows)
        # Global baseline row
        global_row = {
            'holiday': '__global_baseline__',
            'count': global_n,
            'positive_pct': f'{global_pct["positive"] * 100:.1f}',
            'positive_diff': '0.0',
            'neutral_pct': f'{global_pct["neutral"] * 100:.1f}',
            'neutral_diff': '0.0',
            'negative_pct': f'{global_pct["negative"] * 100:.1f}',
            'negative_diff': '0.0',
            'avg_score': f'{np.mean([r["sentiment_score"] for r in seekers]):.3f}',
        }
        w.writerow(global_row)
    log(f"Saved: {csv_path}")


# ── S4: Per-Holiday Genre Sentiment vs Baseline ───────────────────────

def _compute_genre_sentiment(
    seekers_subset: list[dict], movie_info: dict,
) -> dict[str, dict]:
    """Compute average sentiment score per genre for a subset of seekers.

    Uses raw Chinese genre strings from movie_info (same as step5_genre.py).

    Returns:
        dict[genre] -> {'score_sum': float, 'count': int, 'avg': float}
    """
    stats = defaultdict(lambda: {'score_sum': 0.0, 'count': 0})
    for r in seekers_subset:
        score = r.get('sentiment_score', 0.0)
        for tid in r.get('imdb_ids', []):
            info = movie_info.get(tid)
            if info and 'genres' in info:
                for g in info['genres']:
                    s = stats[g]
                    s['score_sum'] += score
                    s['count'] += 1
    # Compute averages
    for g, s in stats.items():
        s['avg'] = s['score_sum'] / max(s['count'], 1)
    return dict(stats)


# ═══════════════════════════════════════════════════════════════════════
#  S4: 各节假日题材情感差值 (Heatmap)
#  S4: Per-Holiday Genre Sentiment Difference
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】热力图: 行=节假日, 列=电影类型, 值=情感差值 vs 非节假日基线
# 【统计口径】按节假日×类型聚合情感得分，计算差值
#   过滤: 仅保留数据充足的节假日×类型组合
# 【输出文件】PNG: s4_genre_sentiment_heatmap.png, CSV: s4_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_s4_genre_by_holiday(seekers: list[dict], movie_info: dict):
    """Per-holiday genre sentiment difference vs non-holiday baseline.

    For each holiday (with sufficient data) and each genre (with sufficient
    mentions in baseline), compute the average sentiment score.  Display as
    a heatmap: rows = holidays, columns = genres, colored by the difference
    from the non-holiday baseline (pp = percentage-point-like offset in
    VADER score space).
    """
    log("=" * 50)
    log("S4: Per-Holiday Genre Sentiment vs Baseline")

    if not movie_info:
        log("  No movie_info available, skipping")
        return

    # ── 1. Non-holiday baseline ──
    non_holiday = [r for r in seekers if r['period'] != 'holiday']
    if not non_holiday:
        log("  No non-holiday data for baseline")
        return
    base_stats = _compute_genre_sentiment(non_holiday, movie_info)
    # Filter: genres with enough baseline mentions
    base_genres = {g for g, s in base_stats.items() if s['count'] >= MIN_DATA_ROWS}
    if not base_genres:
        log("  No genres in baseline meet the minimum mention threshold")
        return

    # ── 2. Per-holiday groups ──
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['is_holiday']:
            holiday_groups[r.get('holiday_name', '')].append(r)
    # Filter: holidays with enough data
    holiday_groups = {k: v for k, v in holiday_groups.items()
                      if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  No holiday groups with sufficient data")
        return

    holiday_names = sorted(holiday_groups.keys())
    log(f"  Baseline: {len(base_genres)} genres from {len(non_holiday)} non-holiday records")
    log(f"  Holidays: {len(holiday_names)} groups")

    # ── 3. Build matrix: holidays × genres ──
    # Also collect which holiday-genre cell passes the mention threshold
    shared_genres = sorted(base_genres)  # use baseline genre list
    n_holidays = len(holiday_names)
    n_genres = len(shared_genres)
    matrix = np.full((n_holidays, n_genres), np.nan)  # NaN = insufficient data
    csv_rows = []

    for i, hn in enumerate(holiday_names):
        h_stats = _compute_genre_sentiment(holiday_groups[hn], movie_info)
        row = {'holiday': hn, 'count': len(holiday_groups[hn])}
        for j, genre in enumerate(shared_genres):
            base_avg = base_stats[genre]['avg']
            h_entry = h_stats.get(genre)
            if h_entry and h_entry['count'] >= MIN_DATA_ROWS:
                diff = h_entry['avg'] - base_avg
                matrix[i, j] = diff
                row[f'{genre}_holiday_avg'] = f'{h_entry["avg"]:.3f}'
                row[f'{genre}_baseline_avg'] = f'{base_avg:.3f}'
                row[f'{genre}_diff'] = f'{diff:+.3f}'
            else:
                row[f'{genre}_holiday_avg'] = ''
                row[f'{genre}_baseline_avg'] = f'{base_avg:.3f}'
                row[f'{genre}_diff'] = ''
        csv_rows.append(row)

        # Log top-5 most positive/negative deviating genres for this holiday
        valid = [(genre, matrix[i, j]) for j, genre in enumerate(shared_genres)
                 if not np.isnan(matrix[i, j])]
        valid.sort(key=lambda x: x[1], reverse=True)
        if valid:
            top_pos = valid[:3]
            top_neg = valid[-3:]
            log(f"  {hn}: 偏差最大正面={[(g, f'{d:+.3f}') for g, d in top_pos]}, "
                f"偏差最大负面={[(g, f'{d:+.3f}') for g, d in top_neg]}")

    # ── 4. Filter: drop genres with no data for ANY holiday ──
    col_has_data = ~np.all(np.isnan(matrix), axis=0)
    if col_has_data.sum() == 0:
        log("  No genre-holiday cells meet the minimum mention threshold")
        return
    matrix = matrix[:, col_has_data]
    active_genres = [g for g, keep in zip(shared_genres, col_has_data) if keep]

    log(f"  Heatmap: {n_holidays} holidays × {len(active_genres)} genres")

    # ── 5. Heatmap ──
    fig_h = max(5, n_holidays * 0.35 + 2)
    fig_w = max(10, len(active_genres) * 0.7 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Use masked array for NaN -> white in RdBu_r
    masked = np.ma.masked_invalid(matrix)
    vmax = max(np.nanmax(np.abs(matrix)), 0.01)
    im = ax.imshow(masked, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    # 在热力图上标注数值
    _mat = np.ma.getdata(masked) if isinstance(masked, np.ma.MaskedArray) else np.asarray(masked)
    for _i in range(_mat.shape[0]):
        for _j in range(_mat.shape[1]):
            _v = _mat[_i, _j]
            if not np.isnan(_v) and abs(_v) > 1e-6:
                _mask = np.ma.is_masked(masked) and masked.mask[_i, _j]
                if not _mask:
                    ax.text(_j, _i, f'{_v:.2f}', ha='center', va='center',
                            fontsize=5, color='black')

    ax.set_xticks(range(len(active_genres)))
    ax.set_xticklabels(active_genres, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_holidays))
    ax.set_yticklabels(holiday_names, fontsize=8)
    ax.set_xlabel('Movie Genre', fontsize=10)
    ax.set_ylabel('Holiday', fontsize=10)
    ax.set_title('Genre Sentiment Difference: Holiday vs Baseline\n'
                 '(Red = more positive on holiday, Blue = more negative)',
                 fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Sentiment Score Difference (Holiday − Baseline)', fontsize=9)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 's4_genre_sentiment_by_holiday.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")

    # ── 6. CSV ──
    csv_path = os.path.join(STEP_OUT, 's4_genre_sentiment_by_holiday.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        header = ['holiday', 'seeker_count']
        for genre in active_genres:
            header.extend([f'{genre}_avg', f'{genre}_diff'])
        w.writerow(header)
        for i, hn in enumerate(holiday_names):
            row = [hn, len(holiday_groups[hn])]
            for j, genre in enumerate(active_genres):
                val = matrix[i, j]
                if np.isnan(val):
                    row.extend(['', ''])
                else:
                    stats_h = _compute_genre_sentiment(holiday_groups[hn], movie_info)
                    h_avg = stats_h[genre]['avg']
                    row.extend([f'{h_avg:.4f}', f'{val:+.4f}'])
            w.writerow(row)
        # Baseline row
        base_row = ['__baseline__', len(non_holiday)]
        for genre in active_genres:
            base_avg = base_stats[genre]['avg']
            base_row.extend([f'{base_avg:.4f}', '0.0000'])
        w.writerow(base_row)
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S5: 情感关联关键词 (Table)
#  S5: Sentiment-Associated Keywords
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】表格/柱状图: 显示与积极/消极情感最相关的词
# 【统计口径】使用 log-odds ratio + additive smoothing 评分
#   积极词: 在正面情感文本中出现频率显著更高的词
#   消极词: 在负面情感文本中出现频率显著更高的词
# 【输出文件】PNG: s5_sentiment_keywords.png, CSV: s5_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_s5_keywords(seekers: list[dict]):
    """Find words most strongly associated with positive vs negative sentiment.

    Uses log-odds ratio with additive smoothing to score each word's
    sentiment polarity. Only words appearing in >= 10 documents are scored.
    """
    log("=" * 50)
    log("S5: 情感关联关键词分析")

    # Split seekers by sentiment
    pos_texts = [r for r in seekers if r['sentiment'] == 'positive']
    neg_texts = [r for r in seekers if r['sentiment'] == 'negative']
    # Neutral included as a baseline reference but not used for keyword
    # association extremes

    log(f"  正面文本: {len(pos_texts)} 条, 负面文本: {len(neg_texts)} 条")

    if not pos_texts or not neg_texts:
        log("  需要同时包含正面和负面文本才能进行关键词对比")
        return

    def _tokenize_and_count(texts: list[dict]) -> Counter:
        """Tokenize texts and count document-level presence of each word."""
        freq = Counter()
        for r in texts:
            text = r.get('proc_text', '') or r.get('raw_text', '')
            tokens = set(tokenize_simple(text))
            freq.update(tokens)
        return freq

    pos_freq = _tokenize_and_count(pos_texts)
    neg_freq = _tokenize_and_count(neg_texts)

    pos_n = len(pos_texts)
    neg_n = len(neg_texts)

    # Compute log-odds ratio with additive smoothing
    MIN_TOTAL = 10
    results = []
    all_words = set(pos_freq.keys()) | set(neg_freq.keys())
    for w in all_words:
        total = pos_freq.get(w, 0) + neg_freq.get(w, 0)
        if total < MIN_TOTAL:
            continue
        # Additive smoothing: (count + 0.5) / (n + 1)
        pos_rate = (pos_freq.get(w, 0) + 0.5) / (pos_n + 1)
        neg_rate = (neg_freq.get(w, 0) + 0.5) / (neg_n + 1)
        log_or = np.log2(pos_rate / neg_rate)
        results.append((w, log_or, pos_freq.get(w, 0), neg_freq.get(w, 0), total))

    results.sort(key=lambda x: x[1], reverse=True)

    # Top positive-associated words
    pos_top = [r for r in results if r[1] > 0.5][:20]
    # Top negative-associated words
    neg_top = [r for r in reversed(results) if r[1] < -0.5][:20]

    log("  正面关联关键词（log-odds ratio > 0.5）：")
    for w, lor, pos_c, neg_c, tot in pos_top[:10]:
        log(f"    {w}: log2 比率={lor:+.2f} (正面={pos_c}, 负面={neg_c})")

    log("  负面关联关键词（log-odds ratio < -0.5）：")
    for w, lor, pos_c, neg_c, tot in neg_top[:10]:
        log(f"    {w}: log2 比率={lor:+.2f} (正面={pos_c}, 负面={neg_c})")

    # Bar charts
    if pos_top:
        _plot_horizontal_bar(
            [(w, lor) for w, lor, _, _, _ in pos_top],
            '正面关联关键词 (log2 正面/负面比率)',
            's5_positive_keywords.png',
            xlabel='log2(正面/负面比率)',
            color='#2ecc71',
        )
    if neg_top:
        _plot_horizontal_bar(
            [(w, abs(lor)) for w, lor, _, _, _ in neg_top],
            '负面关联关键词 (log2 正面/负面比率)',
            's5_negative_keywords.png',
            xlabel='|log2(正面/负面比率)|',
            color='#e74c3c',
        )

    # CSV
    csv_path = os.path.join(STEP_OUT, 's5_sentiment_keywords.csv')
    all_scored = sorted(results, key=lambda x: x[1], reverse=True)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['word', 'log2_pos_neg_ratio', 'pos_doc_count',
                     'neg_doc_count', 'total_doc_count'])
        for word, lor, pos_c, neg_c, tot in all_scored:
            w.writerow([word, f'{lor:.3f}', pos_c, neg_c, tot])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S6-S10: 从 step4_holiday_compare_v2 移植（使用 step11 数据处理方式）
# ═══════════════════════════════════════════════════════════════════════

# ── 基线日期工具函数 ──────────────────────────────────────────────

def _get_week_before(date_str: str, holiday_dates: set[str]) -> str:
    """获取周前基线日期（同一星期 7 天前，若也是节假日则再退 7 天）。"""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    for _ in range(6):
        dt -= timedelta(days=7)
        candidate = dt.strftime('%Y-%m-%d')
        if candidate not in holiday_dates:
            return candidate
    return ''


def _get_month_before(date_str: str, holiday_dates: set[str]) -> str:
    """获取月前基线日期（同一日历日一个月前，处理月末边界）。"""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    year, month, day = dt.year, dt.month, dt.day
    if month == 1:
        prev_month = 12
        prev_year = year - 1
    else:
        prev_month = month - 1
        prev_year = year
    last_day = calendar.monthrange(prev_year, prev_month)[1]
    candidate_day = min(day, last_day)
    candidate = f"{prev_year:04d}-{prev_month:02d}-{candidate_day:02d}"
    for _ in range(6):
        if candidate not in holiday_dates:
            return candidate
        dt_candidate = datetime.strptime(candidate, '%Y-%m-%d')
        candidate = (dt_candidate - timedelta(days=7)).strftime('%Y-%m-%d')
    return ''


# ═══════════════════════════════════════════════════════════════════════
#  S6: 节假日 VS 周前/月前基线 情感分布 (Grouped Bar)
#  S6: Holiday vs Week/Month Baseline Sentiment
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】三面板分组柱状图: 各节假日, 各情感类别占比, 对比周前/月前基线
# 【统计口径】
#   - 周前基线: 节假日前7天的情感分布
#   - 月前基线: 节假日前30天的情感分布
# 【输出文件】PNG: s6_sentiment_comparison.png, CSV: s6_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_s6_sentiment_comparison(seekers: list[dict]):
    """比较各节假日与周前/月前基线的情感分布（三面板分组柱状图）。"""
    log("=" * 50)
    log("S6: 节假日与周前/月前基线的情感分布对比")

    # ── Build date → seekers lookup ──
    seekers_by_date = defaultdict(list)
    holiday_dates = set()
    for r in seekers:
        seekers_by_date[r['date']].append(r)
        if r['is_holiday']:
            holiday_dates.add(r['date'])

    if not holiday_dates:
        log("  无节假日数据，跳过")
        return

    # ── For each holiday date, find week-before & month-before ──
    name_groups = defaultdict(lambda: {'holiday': [], 'week_before': [], 'month_before': []})
    for h_date in sorted(holiday_dates):
        rows = seekers_by_date[h_date]
        if not rows:
            continue
        h_name = rows[0].get('holiday_name', '')
        if not h_name:
            continue
        name_groups[h_name]['holiday'].extend(rows)

        wb = _get_week_before(h_date, holiday_dates)
        if wb and wb in seekers_by_date:
            name_groups[h_name]['week_before'].extend(seekers_by_date[wb])

        mb = _get_month_before(h_date, holiday_dates)
        if mb and mb in seekers_by_date:
            name_groups[h_name]['month_before'].extend(seekers_by_date[mb])

    # Filter: holidays meeting minimum data threshold
    groups = {k: v for k, v in name_groups.items() if len(v['holiday']) >= MIN_DATA_ROWS}
    if not groups:
        log("  无数据足够的节假日")
        return

    names = sorted(groups.keys())
    log(f"  节假日: {len(names)} 个")

    def _pct(records: list[dict], sent_type: str) -> float:
        return sum(1 for r in records if r['sentiment'] == sent_type) / max(len(records), 1) * 100

    h_data = {n: {k: _pct(groups[n]['holiday'], k) for k in ['positive', 'neutral', 'negative']} for n in names}
    w_data = {n: {k: _pct(groups[n]['week_before'], k) if groups[n]['week_before'] else 0 for k in ['positive', 'neutral', 'negative']} for n in names}
    m_data = {n: {k: _pct(groups[n]['month_before'], k) if groups[n]['month_before'] else 0 for k in ['positive', 'neutral', 'negative']} for n in names}

    # ── 3-panel grouped bar chart ──
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(10, len(names) * 1.8), 14))
    x = np.arange(len(names))
    width = 0.25
    sent_cfg = [('positive', SENT_COLORS['positive']), ('neutral', SENT_COLORS['neutral']), ('negative', SENT_COLORS['negative'])]

    for ax, data, title in [
        (ax1, h_data, 'Holiday Sentiment Distribution (averaged across years)'),
        (ax2, w_data, 'Baseline (Week Before) Sentiment Distribution'),
        (ax3, m_data, 'Baseline (Month Before) Sentiment Distribution'),
    ]:
        for i, (s_type, color) in enumerate(sent_cfg):
            vals = [data[n].get(s_type, 0) for n in names]
            ax.bar(x + (i - 1) * width, vals, width, label=s_type.capitalize(), color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Proportion (%)')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 's6_sentiment_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 's6_sentiment_comparison.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'n_holiday', 'n_week_before', 'n_month_before',
                     'holiday_pos_pct', 'holiday_neu_pct', 'holiday_neg_pct',
                     'week_before_pos_pct', 'week_before_neu_pct', 'week_before_neg_pct',
                     'month_before_pos_pct', 'month_before_neu_pct', 'month_before_neg_pct'])
        for n in names:
            g = groups[n]
            def _v(d, k): return f'{d[n][k]:.1f}'
            w.writerow([
                n, len(g['holiday']), len(g['week_before']), len(g['month_before']),
                _v(h_data, 'positive'), _v(h_data, 'neutral'), _v(h_data, 'negative'),
                _v(w_data, 'positive'), _v(w_data, 'neutral'), _v(w_data, 'negative'),
                _v(m_data, 'positive'), _v(m_data, 'neutral'), _v(m_data, 'negative'),
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S7: 节假日 VS 基线 情感强度 (Scatter)
#  S7: Holiday vs Baseline Intensity Comparison
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】散点图: 各节假日的情感强度 vs 周前/月前基线
# 【统计口径】同 S6 的基线窗口定义
# 【输出文件】PNG: s7_intensity_comparison.png, CSV: s7_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_s7_intensity_comparison(seekers: list[dict]):
    """比较各节假日与周前/月前基线的情感强度分布（散点图）。"""
    log("=" * 50)
    log("S7: 节假日与基线的情感强度对比")

    seekers_by_date = defaultdict(list)
    holiday_dates = set()
    for r in seekers:
        seekers_by_date[r['date']].append(r)
        if r['is_holiday']:
            holiday_dates.add(r['date'])

    if not holiday_dates:
        log("  无节假日数据，跳过")
        return

    name_groups = defaultdict(lambda: {'holiday': [], 'week_before': [], 'month_before': []})
    for h_date in sorted(holiday_dates):
        rows = seekers_by_date[h_date]
        if not rows:
            continue
        h_name = rows[0].get('holiday_name', '')
        if not h_name:
            continue
        name_groups[h_name]['holiday'].extend(rows)

        wb = _get_week_before(h_date, holiday_dates)
        if wb and wb in seekers_by_date:
            name_groups[h_name]['week_before'].extend(seekers_by_date[wb])

        mb = _get_month_before(h_date, holiday_dates)
        if mb and mb in seekers_by_date:
            name_groups[h_name]['month_before'].extend(seekers_by_date[mb])

    groups = {k: v for k, v in name_groups.items() if len(v['holiday']) >= MIN_DATA_ROWS}
    if not groups:
        log("  无数据足够的节假日")
        return

    names = sorted(groups.keys())

    def _pct_int(records, level):
        return sum(1 for r in records if r['intensity'] == level) / max(len(records), 1) * 100

    # ── Scatter plot: holiday (o) vs week-before (x) ──
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.2), 6))
    x = np.arange(len(names))
    width = 0.3
    inten_markers = {
        'mild':    {'color': INTEN_COLORS['mild'],    'marker': 'o', 'label': 'Mild'},
        'moderate': {'color': INTEN_COLORS['moderate'], 'marker': 's', 'label': 'Moderate'},
        'strong':  {'color': INTEN_COLORS['strong'],   'marker': '^', 'label': 'Strong'},
    }

    for i, level in enumerate(['mild', 'moderate', 'strong']):
        h_vals = [_pct_int(groups[n]['holiday'], level) for n in names]
        b_vals = [_pct_int(groups[n]['week_before'], level) if groups[n]['week_before'] else 0 for n in names]
        offset = (i - 1) * 0.08
        cfg = inten_markers[level]
        ax.scatter(x - width/2 + offset, h_vals, marker='o', s=70,
                   color=cfg['color'], alpha=0.7,
                   label=f'{cfg["label"]} (Holiday)' if i == 0 else '')
        ax.scatter(x + width/2 + offset, b_vals, marker='x', s=70,
                   color=cfg['color'], alpha=0.7,
                   label=f'{cfg["label"]} (Baseline)' if i == 0 else '')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Intensity: Holiday vs Baseline (o=holiday, x=baseline, averaged across years)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 's7_intensity_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 's7_intensity_comparison.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'n_holiday', 'n_baseline',
                     'holiday_mild_pct', 'holiday_moderate_pct', 'holiday_strong_pct',
                     'baseline_mild_pct', 'baseline_moderate_pct', 'baseline_strong_pct'])
        for n in names:
            g = groups[n]
            w.writerow([
                n, len(g['holiday']), len(g['week_before']),
                f'{_pct_int(g["holiday"], "mild"):.1f}',
                f'{_pct_int(g["holiday"], "moderate"):.1f}',
                f'{_pct_int(g["holiday"], "strong"):.1f}',
                f'{_pct_int(g["week_before"], "mild") if g["week_before"] else 0:.1f}',
                f'{_pct_int(g["week_before"], "moderate") if g["week_before"] else 0:.1f}',
                f'{_pct_int(g["week_before"], "strong") if g["week_before"] else 0:.1f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S8: 各节假日情感强度分布 (Grouped Bar)
#  S8: Per-Holiday Sentiment Intensity Distribution
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】分组柱状图: 各节假日的情感强度桶(低/中/高)对比
# 【输出文件】PNG: s8_intensity_by_holiday.png, CSV: s8_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_s8_intensity_by_holiday(seekers: list[dict]):
    """展示各节假日的情感强度分布（分组柱状图）。"""
    log("=" * 50)
    log("S8: 各节假日的情感强度分布")

    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['is_holiday']:
            holiday_groups[r['holiday_name']].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items() if len(v) >= MIN_DATA_ROWS}

    if not holiday_groups:
        log("  无数据足够的节假日")
        return

    names = sorted(holiday_groups.keys())
    log(f"  节假日: {len(names)} 个")

    # ── Grouped bar chart ──
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))
    x = np.arange(len(names))
    width = 0.25

    for i, level in enumerate(['mild', 'moderate', 'strong']):
        vals = [sum(1 for r in holiday_groups[n] if r['intensity'] == level) / max(len(holiday_groups[n]), 1) * 100
                for n in names]
        ax.bar(x + (i - 1) * width, vals, width, label=level.capitalize(),
               color=INTEN_COLORS[level], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Intensity Across Holidays (averaged across years)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 's8_intensity_by_holiday.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 's8_intensity_by_holiday.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'n_questions',
                     'mild_pct', 'moderate_pct', 'strong_pct',
                     'mild_count', 'moderate_count', 'strong_count'])
        for n in names:
            recs = holiday_groups[n]
            n_total = len(recs)
            mild_c = sum(1 for r in recs if r['intensity'] == 'mild')
            mod_c = sum(1 for r in recs if r['intensity'] == 'moderate')
            strong_c = sum(1 for r in recs if r['intensity'] == 'strong')
            w.writerow([
                n, n_total,
                f'{mild_c / max(n_total, 1) * 100:.1f}',
                f'{mod_c / max(n_total, 1) * 100:.1f}',
                f'{strong_c / max(n_total, 1) * 100:.1f}',
                mild_c, mod_c, strong_c,
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S9: 各节假日情感 VS 非节假日均值 (Bar with Baseline)
#  S9: Per-Holiday Sentiment vs Non-Holiday Avg
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】带参考线的柱状图: 各节假日平均情感, 水平线=全局非节假日均值
# 【输出文件】PNG: s9_sentiment_vs_nonholiday_avg.png, CSV: s9_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_s9_sentiment_vs_nonholiday_avg(seekers: list[dict]):
    """每个节假日的情感分布 vs 全局非节假日均值（带参考线的柱状图）。"""
    log("=" * 50)
    log("S9: 各节假日情感 vs 全局非节假日均值")

    non_holiday = [r for r in seekers if r['period'] != 'holiday']
    if not non_holiday:
        log("  无非节假日数据")
        return
    nh_n = len(non_holiday)
    nh_pos = sum(1 for r in non_holiday if r['sentiment'] == 'positive') / nh_n * 100
    nh_neu = sum(1 for r in non_holiday if r['sentiment'] == 'neutral') / nh_n * 100
    nh_neg = sum(1 for r in non_holiday if r['sentiment'] == 'negative') / nh_n * 100

    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['is_holiday']:
            holiday_groups[r['holiday_name']].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items() if len(v) >= MIN_DATA_ROWS}

    if not holiday_groups:
        log("  无数据足够的节假日")
        return

    names = sorted(holiday_groups.keys())
    log(f"  非节假日基线: n={nh_n}, Pos={nh_pos:.1f}%, Neu={nh_neu:.1f}%, Neg={nh_neg:.1f}%")
    log(f"  节假日: {len(names)} 个")

    h_pos = [sum(1 for r in holiday_groups[n] if r['sentiment'] == 'positive') / max(len(holiday_groups[n]), 1) * 100
             for n in names]
    h_neu = [sum(1 for r in holiday_groups[n] if r['sentiment'] == 'neutral') / max(len(holiday_groups[n]), 1) * 100
             for n in names]
    h_neg = [sum(1 for r in holiday_groups[n] if r['sentiment'] == 'negative') / max(len(holiday_groups[n]), 1) * 100
             for n in names]

    # ── Bar chart with reference lines ──
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 1.5), 6))
    x = np.arange(len(names))
    width = 0.22

    bars_pos = ax.bar(x - width, h_pos, width, label='Positive', color=SENT_COLORS['positive'], alpha=0.8)
    bars_neu = ax.bar(x, h_neu, width, label='Neutral', color=SENT_COLORS['neutral'], alpha=0.8)
    bars_neg = ax.bar(x + width, h_neg, width, label='Negative', color=SENT_COLORS['negative'], alpha=0.8)

    ax.axhline(y=nh_pos, color=SENT_COLORS['positive'], linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Pos ({nh_pos:.1f}%)')
    ax.axhline(y=nh_neu, color=SENT_COLORS['neutral'], linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Neu ({nh_neu:.1f}%)')
    ax.axhline(y=nh_neg, color=SENT_COLORS['negative'], linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Neg ({nh_neg:.1f}%)')

    # Highlight bars with >5pp deviation
    for i, (bar, val, ref) in enumerate(
        [(bars_pos[i], h_pos[i], nh_pos) for i in range(len(names))]
        + [(bars_neu[i], h_neu[i], nh_neu) for i in range(len(names))]
        + [(bars_neg[i], h_neg[i], nh_neg) for i in range(len(names))]
    ):
        if abs(val - ref) > 5.0:
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Distribution: Each Holiday vs Non-Holiday Average')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 's9_sentiment_vs_nonholiday_avg.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 's9_sentiment_vs_nonholiday_avg.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'n_questions',
                     'pos_pct', 'neu_pct', 'neg_pct',
                     'nh_pos_pct', 'nh_neu_pct', 'nh_neg_pct',
                     'pos_diff', 'neu_diff', 'neg_diff'])
        for i, n in enumerate(names):
            w.writerow([
                n, len(holiday_groups[n]),
                f'{h_pos[i]:.1f}', f'{h_neu[i]:.1f}', f'{h_neg[i]:.1f}',
                f'{nh_pos:.1f}', f'{nh_neu:.1f}', f'{nh_neg:.1f}',
                f'{h_pos[i] - nh_pos:+.1f}', f'{h_neu[i] - nh_neu:+.1f}', f'{h_neg[i] - nh_neg:+.1f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S10: 各节假日情感强度 VS 非节假日均值 (Bar with Baseline)
#  S10: Per-Holiday Intensity vs Non-Holiday Avg
# ═══════════════════════════════════════════════════════════════════════
# 【统计口径】同 S9 但使用情感强度(|sentiment|) 而非得分
# 【输出文件】PNG: s10_intensity_vs_nonholiday_avg.png, CSV: s10_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_s10_intensity_vs_nonholiday_avg(seekers: list[dict]):
    """每个节假日的情感强度 vs 全局非节假日均值（带参考线的柱状图）。"""
    log("=" * 50)
    log("S10: 各节假日情感强度 vs 全局非节假日均值")

    non_holiday = [r for r in seekers if r['period'] != 'holiday']
    if not non_holiday:
        log("  无非节假日数据")
        return
    nh_n = len(non_holiday)
    nh_mild = sum(1 for r in non_holiday if r['intensity'] == 'mild') / nh_n * 100
    nh_mod = sum(1 for r in non_holiday if r['intensity'] == 'moderate') / nh_n * 100
    nh_strong = sum(1 for r in non_holiday if r['intensity'] == 'strong') / nh_n * 100

    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['is_holiday']:
            holiday_groups[r['holiday_name']].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items() if len(v) >= MIN_DATA_ROWS}

    if not holiday_groups:
        log("  无数据足够的节假日")
        return

    names = sorted(holiday_groups.keys())
    log(f"  非节假日基线: n={nh_n}, Mild={nh_mild:.1f}%, Mod={nh_mod:.1f}%, Strong={nh_strong:.1f}%")
    log(f"  节假日: {len(names)} 个")

    h_mild = [sum(1 for r in holiday_groups[n] if r['intensity'] == 'mild') / max(len(holiday_groups[n]), 1) * 100
              for n in names]
    h_mod = [sum(1 for r in holiday_groups[n] if r['intensity'] == 'moderate') / max(len(holiday_groups[n]), 1) * 100
             for n in names]
    h_strong = [sum(1 for r in holiday_groups[n] if r['intensity'] == 'strong') / max(len(holiday_groups[n]), 1) * 100
                for n in names]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 1.5), 6))
    x = np.arange(len(names))
    width = 0.22

    bars_mild = ax.bar(x - width, h_mild, width, label='Mild', color=INTEN_COLORS['mild'], alpha=0.8)
    bars_mod = ax.bar(x, h_mod, width, label='Moderate', color=INTEN_COLORS['moderate'], alpha=0.8)
    bars_strong = ax.bar(x + width, h_strong, width, label='Strong', color=INTEN_COLORS['strong'], alpha=0.8)

    ax.axhline(y=nh_mild, color=INTEN_COLORS['mild'], linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Mild ({nh_mild:.1f}%)')
    ax.axhline(y=nh_mod, color=INTEN_COLORS['moderate'], linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Mod ({nh_mod:.1f}%)')
    ax.axhline(y=nh_strong, color=INTEN_COLORS['strong'], linestyle='--', linewidth=1.8,
               alpha=0.9, label=f'Non-holiday Strong ({nh_strong:.1f}%)')

    for i, bar, val, ref in (
        [(i, bars_mild[i], h_mild[i], nh_mild) for i in range(len(names))]
        + [(i, bars_mod[i], h_mod[i], nh_mod) for i in range(len(names))]
        + [(i, bars_strong[i], h_strong[i], nh_strong) for i in range(len(names))]
    ):
        if abs(val - ref) > 5.0:
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Sentiment Intensity: Each Holiday vs Non-Holiday Average')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 's10_intensity_vs_nonholiday_avg.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 's10_intensity_vs_nonholiday_avg.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'n_questions',
                     'mild_pct', 'moderate_pct', 'strong_pct',
                     'nh_mild_pct', 'nh_moderate_pct', 'nh_strong_pct',
                     'mild_diff', 'moderate_diff', 'strong_diff'])
        for i, n in enumerate(names):
            w.writerow([
                n, len(holiday_groups[n]),
                f'{h_mild[i]:.1f}', f'{h_mod[i]:.1f}', f'{h_strong[i]:.1f}',
                f'{nh_mild:.1f}', f'{nh_mod:.1f}', f'{nh_strong:.1f}',
                f'{h_mild[i] - nh_mild:+.1f}', f'{h_mod[i] - nh_mod:+.1f}', f'{h_strong[i] - nh_strong:+.1f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None):
    log("=" * 60)
    log("步骤 11：电影相关查询的情感分析")
    log("=" * 60)

    # 加载数据
    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']
    movie_info = data.get('movie_info', {})
    log(f"已加载 {len(seekers)} 条查询记录")

    # 去重
    seekers = deduplicate_seekers(seekers)

    # 执行情感分析并标注到每条记录
    log("正在对查询文本进行情感分析...")
    seekers = annotate_sentiment(seekers)
    log(f"已完成 {len(seekers)} 条记录的情感标注")

    # S1: 节假日 vs 非节假日情感对比
    dim_s1_holiday_vs_nonholiday(seekers)
    log("")

    # S2: 按时期对比（含强度）
    dim_s2_by_period(seekers)
    log("")

    # S3: 各节假日情感画像
    dim_s3_per_holiday(seekers)
    log("")

    # S4: 各节假日电影类型情感差异
    dim_s4_genre_by_holiday(seekers, movie_info)
    log("")

    # S5: 情感关联关键词
    dim_s5_keywords(seekers)
    log("")

    # S6: 情感对比（周前/月前基线）
    dim_s6_sentiment_comparison(seekers)
    log("")

    # S7: 强度对比（基线）
    dim_s7_intensity_comparison(seekers)
    log("")

    # S8: 各节假日强度分布
    dim_s8_intensity_by_holiday(seekers)
    log("")

    # S9: 情感 vs 非节假日均值
    dim_s9_sentiment_vs_nonholiday_avg(seekers)
    log("")

    # S10: 强度 vs 非节假日均值
    dim_s10_intensity_vs_nonholiday_avg(seekers)

    log("")
    log("=" * 60)
    log(f"步骤 11 完成！结果已保存到 {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
