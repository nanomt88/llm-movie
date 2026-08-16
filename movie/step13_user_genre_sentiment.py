# -*- coding: utf-8 -*-
"""
Step 13: User-Mentioned Movie Genre Sentiment Analysis
步骤 13：用户提问提及电影类型的情感分析

Analysis:
  U1 - 整体类型情感分布 (Genre Sentiment Distribution)
        各电影类型在 positive/neutral/negative 上的占比（分组柱状图）
  U2 - 类型平均情感得分排名 (Genre Avg Sentiment Score Ranking)
        按平均情感得分排序的类型排名（水平柱状图，正面/负面分开）
  U3 - 类型情感强度分布 (Genre Intensity Distribution)
        各电影类型在 mild/moderate/strong 上的占比（分组柱状图）
  U4 - 节假日 VS 非节假日 类型情感 (Holiday vs Non-Holiday Genre Sentiment)
        每个类型在节假日/非节假日下的平均情感得分差值（水平柱状图）
  U5 - 类型 × 情感占比热力图 (Genre × Sentiment Proportion Heatmap)
        类型为行、情感三分为列、占比为值的热力图
  U6 - 类型 × 强度占比热力图 (Genre × Intensity Proportion Heatmap)
        类型为行、强度三分为列、占比为值的热力图
  U7 - 各类型提及次数与平均得分 (Genre Mentions vs Avg Score, dual-axis)
        柱状图(提及次数) + 折线(平均情感得分) 双轴图

与 step11 S4 的区别:
  - step11 S4 优先使用 get_system_movie_ids()（从系统回复提取电影 ID，规则8）
  - step13 直接使用 seeker.imdb_ids（用户提问自身提及的电影 ID）
  目的: 分析用户主动提及的电影所对应类型的情感偏向(positive/neutral/
       negative)与强度(mild/moderate/strong)。

Dependencies: vaderSentiment, afinn (via movie.utils.sentiment)
Output: output/movie/step13/*.png + CSV
"""

import os               # 文件路径操作
import csv              # CSV 读写
from collections import defaultdict  # 默认字典

import numpy as np      # 数值计算

import matplotlib
matplotlib.use('Agg')   # 非交互式后端（服务器环境）
import matplotlib.pyplot as plt

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log
from movie.utils.text import deduplicate_seekers
from movie.utils.sentiment import analyze_batch
from movie.utils.genre_map import to_en

# ── 初始化 ──────────────────────────────────────────────────────────
setup_matplotlib()
STEP_OUT = STEP_DIRS[13]                        # 输出目录：output/movie/step13/
os.makedirs(STEP_OUT, exist_ok=True)

# ── 颜色方案（与步骤 11 风格一致）─────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════
#  数据准备
# ═══════════════════════════════════════════════════════════════════════

def annotate_sentiment(seekers: list[dict]) -> list[dict]:
    """Run sentiment analysis on all seekers and attach results in-place.
       对所有用户提问运行情感分析，将 sentiment/intensity/score 附加到每条记录。"""
    texts = [r.get('proc_text', '') or r.get('raw_text', '') for r in seekers]
    results = analyze_batch(texts)
    for r, res in zip(seekers, results):
        r['sentiment'] = res['sentiment']
        r['intensity'] = res['intensity']
        r['sentiment_score'] = res['score']
    return seekers


# ═══════════════════════════════════════════════════════════════════════
#  核心聚合: 按用户提及电影类型聚合情感与强度
# ═══════════════════════════════════════════════════════════════════════

def _compute_user_genre_stats(seekers: list[dict], movie_info: dict) -> dict[str, dict]:
    """Aggregate sentiment + intensity per genre based on USER-mentioned movies.

    与 step11._compute_genre_sentiment 的关键区别:
      - step11 优先使用 get_system_movie_ids()（从系统回复提取电影 ID，规则8）
      - 本函数直接使用 seeker.imdb_ids（用户提问中提及的电影 ID）
        该字段在 data_loader.load_conversations() 中已从 processed 字段提取。

    每条 seeker 记录的 sentiment_score / sentiment / intensity 会被归到
    该提问中提到的每部电影的所有类型上。

    Args:
        seekers:     用户提问记录列表（已标注 sentiment/intensity/score）
        movie_info:  电影信息字典 {imdb_id: {genres: [...], ...}}
    Returns:
        dict[genre_cn] -> {
            'count', 'positive', 'neutral', 'negative',
            'mild', 'moderate', 'strong',
            'score_sum', 'avg_score', 'genre_en'
        }
    """
    stats = defaultdict(lambda: {
        'count': 0,
        'positive': 0, 'neutral': 0, 'negative': 0,
        'mild': 0, 'moderate': 0, 'strong': 0,
        'score_sum': 0.0,
        'weight_sum': 0.0,  # 加权总权重（多电影提问稀释）
    })
    for r in seekers:
        score = r.get('sentiment_score', 0.0)
        sentiment = r.get('sentiment', 'neutral')
        intensity = r.get('intensity', 'mild')
        # 用户提问中提及的电影 ID（data_loader 已从 processed 字段提取）
        tids = r.get('imdb_ids', []) or []
        # 权重: 一条提问提及 N 部电影时，每部电影权重 = 1/N
        # 避免多电影提问的情感得分被重复分配给所有电影
        weight = 1.0 / len(tids) if tids else 1.0
        for tid in tids:
            info = movie_info.get(tid)
            if not info or 'genres' not in info:
                continue
            for g in info['genres']:
                s = stats[g]
                s['count'] += 1
                s[sentiment] += 1
                s[intensity] += 1
                # 加权得分: 多电影提问的得分被稀释
                s['score_sum'] += score * weight
                s['weight_sum'] += weight
    # 计算加权平均得分与英文标签
    for g, s in stats.items():
        s['avg_score'] = s['score_sum'] / max(s['weight_sum'], 1.0)
        s['genre_en'] = to_en(g)
    return dict(stats)


def _filter_genres(stats: dict[str, dict], min_count: int = MIN_DATA_ROWS) -> list[str]:
    """Return genres with sufficient mentions, sorted by count descending.
       返回提及次数 >= min_count 的类型列表，按次数降序。"""
    return sorted(
        [g for g, s in stats.items() if s['count'] >= min_count],
        key=lambda g: stats[g]['count'],
        reverse=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  辅助绘图函数
# ═══════════════════════════════════════════════════════════════════════

def _plot_grouped_bar(
    group_data: dict[str, dict[str, float]],
    title: str, filename: str,
    ylabel: str = 'Proportion (%)',
    colors: dict[str, str] = None,
    ylim_top: float = None,
):
    """Grouped bar chart of distributions across groups.
       分组柱状图：x 轴=categories，每组并排显示 N 个 group 的柱。
       groups（外层 dict 键）应对应 colors 的键，每组用一种颜色。"""
    if not group_data:
        return
    categories = list(next(iter(group_data.values())).keys())
    groups = list(group_data.keys())
    # 图宽基于 x 轴类别数（类别多时需要更宽），并防止颜色列表越界
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.0), 6))
    x = np.arange(len(categories))
    n_groups = len(groups)
    width = 0.7 / max(n_groups, 1)
    default_c = ['#ff6b6b', '#74b9ff', '#feca57', '#48dbfb', '#a29bfe', '#fd79a8']
    for i, group in enumerate(groups):
        vals = [group_data[group].get(c, 0) for c in categories]
        offset = (i - (n_groups - 1) / 2) * width
        # 用 modulo 防止 default_c 越界（当 group 数超过 6 时循环复用）
        c = colors.get(group, default_c[i % len(default_c)]) if colors else default_c[i % len(default_c)]
        ax.bar(x + offset, vals, width, label=group, color=c, alpha=0.8)
    ax.set_xticks(x)
    # 类别较多时旋转标签防止重叠
    if len(categories) > 8:
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    else:
        ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    if ylim_top:
        ax.set_ylim(0, ylim_top)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_horizontal_bar(
    items: list[tuple[str, float]],
    title: str, filename: str,
    xlabel: str = 'Value',
    top_n: int = 20,
    color: str = '#74b9ff',
):
    """Horizontal bar chart for ranked items. 水平柱状图：显示排名项。"""
    top = items[:top_n]
    if not top:
        return
    labels = [t[0] for t in top[::-1]]
    values = [t[1] for t in top[::-1]]

    fig, ax = plt.subplots(figsize=(max(7, top_n * 0.4), max(4, top_n * 0.4)))
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
    fmt: str = '.1f',
    cmap: str = 'RdBu_r',
    cbar_label: str = '',
):
    """Heatmap with labeled rows and columns. 热力图。"""
    if matrix.size == 0:
        return
    fig_h = max(5, len(row_labels) * 0.35 + 2)
    fig_w = max(7, len(col_labels) * 0.8 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = max(np.nanmax(np.abs(matrix)), 0.01)
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
    # 在热力图上标注数值
    _mat = np.ma.getdata(matrix) if isinstance(matrix, np.ma.MaskedArray) else np.asarray(matrix)
    for _i in range(_mat.shape[0]):
        for _j in range(_mat.shape[1]):
            _v = _mat[_i, _j]
            if not np.isnan(_v) and abs(_v) > 1e-6:
                ax.text(_j, _i, f'{_v:{fmt}}', ha='center', va='center',
                        fontsize=6, color='black')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.6, label=cbar_label)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_dual_axis(
    bar_items: list[tuple[str, int]],
    line_items: list[tuple[str, float]],
    title: str, filename: str,
    bar_label: str = 'Mentions',
    line_label: str = 'Avg Sentiment Score',
    bar_color: str = '#74b9ff',
    line_color: str = '#e74c3c',
    top_n: int = 20,
):
    """Dual-axis chart: bar (counts) + line (scores). 双轴图：柱+线。
       bar_items 与 line_items 顺序需保持一致（同一组类型）。"""
    top_bar = bar_items[:top_n]
    if not top_bar:
        return
    labels = [t[0] for t in top_bar]
    bar_vals = [t[1] for t in top_bar]
    # 与 bar 顺序对齐的 line 值（按标签查找）
    line_map = dict(line_items)
    line_vals = [line_map.get(lbl, 0.0) for lbl in labels]

    fig, ax1 = plt.subplots(figsize=(max(10, top_n * 0.5), 6))
    x = np.arange(len(labels))
    ax1.bar(x, bar_vals, color=bar_color, alpha=0.7, label=bar_label)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel(bar_label, fontsize=10, color=bar_color)
    ax1.tick_params(axis='y', labelcolor=bar_color)
    ax1.grid(axis='y', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, line_vals, color=line_color, marker='o', linewidth=2,
             markersize=6, label=line_label)
    ax2.set_ylabel(line_label, fontsize=10, color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    ax1.set_title(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  分析维度
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  U1: 整体类型情感分布 (Genre Sentiment Distribution)
#  U1: Genre Sentiment Distribution
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】分组柱状图: 各类型在 positive/neutral/negative 上的占比
# 【统计口径】基于用户提问中提及的电影 ID (seeker.imdb_ids) 关联类型
#   每个类型至少 MIN_DATA_ROWS 次提及才纳入
# 【输出文件】PNG: u1_genre_sentiment_distribution.png, CSV: u1_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_u1_genre_sentiment_distribution(seekers: list[dict], movie_info: dict):
    """各类型电影的用户情感偏向分布（分组柱状图）。"""
    log("=" * 50)
    log("U1: 各类型电影的用户情感偏向分布")

    stats = _compute_user_genre_stats(seekers, movie_info)
    genres = _filter_genres(stats)
    if not genres:
        log(f"  无数据足够的类型（每个至少 {MIN_DATA_ROWS} 次提及）")
        return

    log(f"  共 {len(genres)} 个类型符合数据阈值")
    for g in genres[:5]:
        s = stats[g]
        n = s['count']
        log(f"    {s['genre_en']} (n={n}): 正面={s['positive']/n*100:.1f}%, "
            f"中性={s['neutral']/n*100:.1f}%, 负面={s['negative']/n*100:.1f}%")

    # ── 构建分组柱状图数据 ──
    # groups = 情感类型（3个，匹配 SENT_COLORS 键），categories = 类型名（x 轴）
    sent_keys = ['positive', 'neutral', 'negative']
    sent_data = {}
    for sk in sent_keys:
        sent_data[sk] = {}
        for g in genres:
            s = stats[g]
            n = s['count']
            sent_data[sk][s['genre_en']] = s[sk] / n * 100

    _plot_grouped_bar(
        sent_data,
        'Genre Sentiment Distribution (User-Mentioned Movies)',
        'u1_genre_sentiment_distribution.png',
        ylabel='Proportion (%)',
        colors=SENT_COLORS,
    )

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'u1_genre_sentiment_distribution.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['genre_cn', 'genre_en', 'count',
                    'positive', 'positive_pct',
                    'neutral', 'neutral_pct',
                    'negative', 'negative_pct',
                    'avg_score'])
        for g in genres:
            s = stats[g]
            n = s['count']
            w.writerow([
                g, s['genre_en'], n,
                s['positive'], f'{s["positive"]/n*100:.1f}',
                s['neutral'], f'{s["neutral"]/n*100:.1f}',
                s['negative'], f'{s["negative"]/n*100:.1f}',
                f'{s["avg_score"]:.3f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  U2: 类型平均情感得分排名 (Genre Avg Sentiment Score Ranking)
#  U2: Genre Avg Sentiment Score Ranking
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】水平柱状图: 按平均情感得分排序的类型排名（正面/负面分开）
# 【统计口径】avg_score = sum(sentiment_score) / count
#   得分范围 [-1, 1]，正值=正面偏向，负值=负面偏向
# 【输出文件】PNG: u2_genre_avg_score_positive.png, u2_*.png, CSV: u2_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_u2_genre_avg_score_ranking(seekers: list[dict], movie_info: dict):
    """按平均情感得分排序的类型排名（水平柱状图，正/负分开）。"""
    log("=" * 50)
    log("U2: 类型平均情感得分排名")

    stats = _compute_user_genre_stats(seekers, movie_info)
    genres = _filter_genres(stats)
    if not genres:
        log(f"  无数据足够的类型（每个至少 {MIN_DATA_ROWS} 次提及）")
        return

    # 构建 (genre_en, avg_score) 列表
    scored = [(stats[g]['genre_en'], stats[g]['avg_score']) for g in genres]
    # 按得分降序：最正面 -> 最负面
    scored.sort(key=lambda x: x[1], reverse=True)

    pos_top = [(g, v) for g, v in scored if v > 0][:20]
    neg_top = [(g, v) for g, v in reversed(scored) if v < 0][:20]

    log(f"  最正面类型 Top 5: {pos_top[:5]}")
    log(f"  最负面类型 Top 5: {neg_top[:5]}")

    if pos_top:
        _plot_horizontal_bar(
            pos_top,
            'Top Genres by Avg Sentiment Score (Positive)',
            'u2_genre_avg_score_positive.png',
            xlabel='Avg Sentiment Score [-1, 1]',
            top_n=len(pos_top),
            color=SENT_COLORS['positive'],
        )
    if neg_top:
        _plot_horizontal_bar(
            [(g, abs(v)) for g, v in neg_top],
            'Top Genres by Avg Sentiment Score (Negative)',
            'u2_genre_avg_score_negative.png',
            xlabel='|Avg Sentiment Score| (negative)',
            top_n=len(neg_top),
            color=SENT_COLORS['negative'],
        )

    # ── CSV（全部类型按得分降序排序）──
    csv_path = os.path.join(STEP_OUT, 'u2_genre_avg_score_ranking.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'genre_cn', 'genre_en', 'count',
                    'avg_score', 'positive_pct', 'negative_pct'])
        # 按 avg_score 降序排列（与注释一致）
        genres_by_score = sorted(genres, key=lambda g: stats[g]['avg_score'], reverse=True)
        for i, g_cn in enumerate(genres_by_score, start=1):
            s = stats[g_cn]
            n = s['count']
            w.writerow([
                i, g_cn, s['genre_en'], n, f'{s["avg_score"]:.3f}',
                f'{s["positive"]/n*100:.1f}',
                f'{s["negative"]/n*100:.1f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  U3: 类型情感强度分布 (Genre Intensity Distribution)
#  U3: Genre Intensity Distribution
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】分组柱状图: 各类型在 mild/moderate/strong 上的占比
# 【统计口径】基于 VADER |compound| 的强度分级
# 【输出文件】PNG: u3_genre_intensity_distribution.png, CSV: u3_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_u3_genre_intensity_distribution(seekers: list[dict], movie_info: dict):
    """各类型电影的用户情感强度分布（分组柱状图）。"""
    log("=" * 50)
    log("U3: 各类型电影的用户情感强度分布")

    stats = _compute_user_genre_stats(seekers, movie_info)
    genres = _filter_genres(stats)
    if not genres:
        log(f"  无数据足够的类型（每个至少 {MIN_DATA_ROWS} 次提及）")
        return

    log(f"  共 {len(genres)} 个类型符合数据阈值")
    for g in genres[:5]:
        s = stats[g]
        n = s['count']
        log(f"    {s['genre_en']} (n={n}): 强烈={s['strong']/n*100:.1f}%, "
            f"中等={s['moderate']/n*100:.1f}%, 轻微={s['mild']/n*100:.1f}%")

    # ── 构建分组柱状图数据 ──
    # groups = 强度类型（3个，匹配 INTEN_COLORS 键），categories = 类型名（x 轴）
    inten_keys = ['mild', 'moderate', 'strong']
    inten_data = {}
    for ik in inten_keys:
        inten_data[ik] = {}
        for g in genres:
            s = stats[g]
            n = s['count']
            inten_data[ik][s['genre_en']] = s[ik] / n * 100

    _plot_grouped_bar(
        inten_data,
        'Genre Sentiment Intensity Distribution (User-Mentioned Movies)',
        'u3_genre_intensity_distribution.png',
        ylabel='Proportion (%)',
        colors=INTEN_COLORS,
    )

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'u3_genre_intensity_distribution.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['genre_cn', 'genre_en', 'count',
                    'mild', 'mild_pct',
                    'moderate', 'moderate_pct',
                    'strong', 'strong_pct'])
        for g in genres:
            s = stats[g]
            n = s['count']
            w.writerow([
                g, s['genre_en'], n,
                s['mild'], f'{s["mild"]/n*100:.1f}',
                s['moderate'], f'{s["moderate"]/n*100:.1f}',
                s['strong'], f'{s["strong"]/n*100:.1f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  U4: 节假日 VS 非节假日 类型情感 (Holiday vs Non-Holiday Genre Sentiment)
#  U4: Holiday vs Non-Holiday Genre Sentiment
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】水平柱状图: 各类型 (节假日avg - 非节假日avg) 的差值
# 【统计口径】分别计算节假日/非节假日子集的类型平均情感得分，差值>0表示
#   节假日更正面，差值<0表示节假日更负面
# 【输出文件】PNG: u4_holiday_vs_nonholiday.png, CSV: u4_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_u4_holiday_vs_nonholiday(seekers: list[dict], movie_info: dict):
    """节假日 vs 非节假日各类型平均情感得分差值（水平柱状图）。"""
    log("=" * 50)
    log("U4: 节假日 vs 非节假日各类型情感得分")

    holiday_seekers = [r for r in seekers if r.get('period') == 'holiday']
    non_holiday_seekers = [r for r in seekers if r.get('period') != 'holiday']

    if not holiday_seekers or not non_holiday_seekers:
        log("  节假日或非节假日子集为空，跳过")
        return

    h_stats = _compute_user_genre_stats(holiday_seekers, movie_info)
    nh_stats = _compute_user_genre_stats(non_holiday_seekers, movie_info)

    # 取两侧均达阈值的类型交集
    h_genres = {g for g, s in h_stats.items() if s['count'] >= MIN_DATA_ROWS}
    nh_genres = {g for g, s in nh_stats.items() if s['count'] >= MIN_DATA_ROWS}
    common = h_genres & nh_genres
    if not common:
        log("  无两侧均达数据阈值的类型")
        return

    # 按差值排序
    diffs = []
    for g in common:
        diff = h_stats[g]['avg_score'] - nh_stats[g]['avg_score']
        diffs.append((g, h_stats[g]['genre_en'], diff,
                      h_stats[g]['avg_score'], nh_stats[g]['avg_score'],
                      h_stats[g]['count'], nh_stats[g]['count']))
    diffs.sort(key=lambda x: x[2], reverse=True)

    log(f"  共 {len(diffs)} 个类型双侧达标")
    log(f"  节假日更正面 Top 3: {[(d[1], f'{d[2]:+.3f}') for d in diffs[:3]]}")
    log(f"  节假日更负面 Top 3: {[(d[1], f'{d[2]:+.3f}') for d in diffs[-3:]]}")

    # ── 水平柱状图（差值，正值红/负值蓝）──
    items_pos = [(d[1], d[2]) for d in diffs if d[2] > 0][:20]
    items_neg = [(d[1], d[2]) for d in reversed(diffs) if d[2] < 0][:20]

    if items_pos:
        _plot_horizontal_bar(
            items_pos,
            'Holiday vs Non-Holiday: Genres More Positive on Holidays',
            'u4_holiday_more_positive.png',
            xlabel='Avg Score Diff (Holiday - Non-Holiday)',
            top_n=len(items_pos),
            color=SENT_COLORS['positive'],
        )
    if items_neg:
        # 负值用绝对值展示
        _plot_horizontal_bar(
            [(g, abs(v)) for g, v in items_neg],
            'Holiday vs Non-Holiday: Genres More Negative on Holidays',
            'u4_holiday_more_negative.png',
            xlabel='|Avg Score Diff| (Holiday - Non-Holiday, negative)',
            top_n=len(items_neg),
            color=SENT_COLORS['negative'],
        )

    # ── CSV（全部类型按差值排序）──
    csv_path = os.path.join(STEP_OUT, 'u4_holiday_vs_nonholiday.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['genre_cn', 'genre_en',
                    'holiday_count', 'nonholiday_count',
                    'holiday_avg_score', 'nonholiday_avg_score',
                    'diff'])
        for g_cn, g_en, diff, h_avg, nh_avg, h_n, nh_n in diffs:
            w.writerow([
                g_cn, g_en, h_n, nh_n,
                f'{h_avg:.3f}', f'{nh_avg:.3f}', f'{diff:+.3f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  U5: 类型情感占比分布 (Genre Sentiment Proportion Distribution)
#  U5: Genre Sentiment Proportion Distribution
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】分组柱状图: x 轴=类型(含 Overall 参照组), 每组 3 根柱(正面/中性/负面)
# 【统计口径】各类型 positive/neutral/negative 占比%，Overall 为全局均值
# 【输出文件】PNG: u5_genre_sentiment_heatmap.png, CSV: u5_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_u5_genre_sentiment_heatmap(seekers: list[dict], movie_info: dict):
    """各类型情感占比分布（分组柱状图，含 Overall 参照组）。"""
    log("=" * 50)
    log("U5: 类型情感占比分布（柱状图 + Overall 参照）")

    stats = _compute_user_genre_stats(seekers, movie_info)
    genres = _filter_genres(stats)
    if not genres:
        log(f"  无数据足够的类型（每个至少 {MIN_DATA_ROWS} 次提及）")
        return

    # 按正面占比降序排列
    genres = sorted(genres, key=lambda g: stats[g]['positive'] / stats[g]['count'], reverse=True)

    sent_keys = ['positive', 'neutral', 'negative']

    # ── 计算全局 Overall 参照组 ──
    total_count = sum(stats[g]['count'] for g in genres)
    overall = {}
    for k in sent_keys:
        overall[k] = sum(stats[g][k] for g in genres) / max(total_count, 1) * 100
    overall_avg = sum(stats[g]['avg_score'] * stats[g]['weight_sum'] for g in genres) / \
                  max(sum(stats[g]['weight_sum'] for g in genres), 1.0)
    log(f"  Overall: 正面={overall['positive']:.1f}%, "
        f"中性={overall['neutral']:.1f}%, 负面={overall['negative']:.1f}%")

    # ── 构建分组柱状图数据 ──
    # groups = 情感类型（3个），categories = 类型名 + Overall（x 轴）
    sent_data = {}
    for sk in sent_keys:
        sent_data[sk] = {}
        for g in genres:
            s = stats[g]
            n = s['count']
            sent_data[sk][s['genre_en']] = s[sk] / n * 100
        # 添加 Overall 参照组
        sent_data[sk]['Overall'] = overall[sk]

    _plot_grouped_bar(
        sent_data,
        'Genre Sentiment Proportion (User-Mentioned, with Overall Baseline)',
        'u5_genre_sentiment_heatmap.png',
        ylabel='Proportion (%)',
        colors=SENT_COLORS,
    )

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'u5_genre_sentiment_heatmap.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['genre_cn', 'genre_en', 'count',
                    'positive_pct', 'neutral_pct', 'negative_pct', 'avg_score'])
        for g in genres:
            s = stats[g]
            n = s['count']
            w.writerow([
                g, s['genre_en'], n,
                f'{s["positive"]/n*100:.1f}',
                f'{s["neutral"]/n*100:.1f}',
                f'{s["negative"]/n*100:.1f}',
                f'{s["avg_score"]:.3f}',
            ])
        # Overall 参照行
        w.writerow(['__overall__', 'Overall', total_count,
                    f'{overall["positive"]:.1f}', f'{overall["neutral"]:.1f}',
                    f'{overall["negative"]:.1f}', f'{overall_avg:.3f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  U6: 类型强度占比分布 (Genre Intensity Proportion Distribution)
#  U6: Genre Intensity Proportion Distribution
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】分组柱状图: x 轴=类型(含 Overall 参照组), 每组 3 根柱(轻微/中等/强烈)
# 【统计口径】各类型 mild/moderate/strong 占比%，Overall 为全局均值
# 【输出文件】PNG: u6_genre_intensity_heatmap.png, CSV: u6_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_u6_genre_intensity_heatmap(seekers: list[dict], movie_info: dict):
    """各类型情感强度占比分布（分组柱状图，含 Overall 参照组）。"""
    log("=" * 50)
    log("U6: 类型强度占比分布（柱状图 + Overall 参照）")

    stats = _compute_user_genre_stats(seekers, movie_info)
    genres = _filter_genres(stats)
    if not genres:
        log(f"  无数据足够的类型（每个至少 {MIN_DATA_ROWS} 次提及）")
        return

    # 按强烈占比降序排列
    genres = sorted(genres, key=lambda g: stats[g]['strong'] / stats[g]['count'], reverse=True)

    inten_keys = ['mild', 'moderate', 'strong']

    # ── 计算全局 Overall 参照组 ──
    total_count = sum(stats[g]['count'] for g in genres)
    overall = {}
    for k in inten_keys:
        overall[k] = sum(stats[g][k] for g in genres) / max(total_count, 1) * 100
    log(f"  Overall: 轻微={overall['mild']:.1f}%, "
        f"中等={overall['moderate']:.1f}%, 强烈={overall['strong']:.1f}%")

    # ── 构建分组柱状图数据 ──
    # groups = 强度类型（3个），categories = 类型名 + Overall（x 轴）
    inten_data = {}
    for ik in inten_keys:
        inten_data[ik] = {}
        for g in genres:
            s = stats[g]
            n = s['count']
            inten_data[ik][s['genre_en']] = s[ik] / n * 100
        # 添加 Overall 参照组
        inten_data[ik]['Overall'] = overall[ik]

    _plot_grouped_bar(
        inten_data,
        'Genre Intensity Proportion (User-Mentioned, with Overall Baseline)',
        'u6_genre_intensity_heatmap.png',
        ylabel='Proportion (%)',
        colors=INTEN_COLORS,
    )

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'u6_genre_intensity_heatmap.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['genre_cn', 'genre_en', 'count',
                    'mild_pct', 'moderate_pct', 'strong_pct'])
        for g in genres:
            s = stats[g]
            n = s['count']
            w.writerow([
                g, s['genre_en'], n,
                f'{s["mild"]/n*100:.1f}',
                f'{s["moderate"]/n*100:.1f}',
                f'{s["strong"]/n*100:.1f}',
            ])
        # Overall 参照行
        w.writerow(['__overall__', 'Overall', total_count,
                    f'{overall["mild"]:.1f}', f'{overall["moderate"]:.1f}',
                    f'{overall["strong"]:.1f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  U7: 各类型提及次数与平均得分 (Genre Mentions vs Avg Score, dual-axis)
#  U7: Genre Mentions vs Avg Score
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】双轴图: 柱状图(提及次数) + 折线(平均情感得分)
# 【统计口径】同时展示各类型的提及量和情感偏向，便于判断样本量与倾向
# 【输出文件】PNG: u7_genre_mentions_vs_score.png, CSV: u7_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_u7_mentions_vs_score(seekers: list[dict], movie_info: dict):
    """各类型提及次数（柱）与平均情感得分（线）双轴图。"""
    log("=" * 50)
    log("U7: 类型提及次数与平均情感得分双轴图")

    stats = _compute_user_genre_stats(seekers, movie_info)
    genres = _filter_genres(stats)
    if not genres:
        log(f"  无数据足够的类型（每个至少 {MIN_DATA_ROWS} 次提及）")
        return

    # 按提及次数降序排列
    genres = sorted(genres, key=lambda g: stats[g]['count'], reverse=True)

    bar_items = [(stats[g]['genre_en'], stats[g]['count']) for g in genres]
    line_items = [(stats[g]['genre_en'], stats[g]['avg_score']) for g in genres]

    log(f"  共 {len(genres)} 个类型，最高提及: {bar_items[0] if bar_items else 'N/A'}")

    top_n = min(20, len(genres))
    _plot_dual_axis(
        bar_items, line_items,
        'Genre Mentions (bar) vs Avg Sentiment Score (line)',
        'u7_genre_mentions_vs_score.png',
        bar_label='Mention Count',
        line_label='Avg Sentiment Score',
        bar_color='#74b9ff',
        line_color='#e74c3c',
        top_n=top_n,
    )

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'u7_genre_mentions_vs_score.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['genre_cn', 'genre_en', 'mention_count', 'avg_score',
                    'positive_pct', 'negative_pct'])
        for g in genres:
            s = stats[g]
            n = s['count']
            w.writerow([
                g, s['genre_en'], n, f'{s["avg_score"]:.3f}',
                f'{s["positive"]/n*100:.1f}',
                f'{s["negative"]/n*100:.1f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None):
    """Run all user-mentioned genre sentiment dimensions.
       执行全部 7 个分析维度。"""
    log("=" * 60)
    log("步骤 13：用户提问提及电影类型的情感分析")
    log("=" * 60)

    # 加载数据
    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']
    movie_info = data.get('movie_info', {})
    log(f"已加载 {len(seekers)} 条用户提问记录")

    if not movie_info:
        log("  movie_info 为空，无法分析类型情感，跳过本步骤")
        return

    # 去重（规则9：同一会话中相同用户提问排重）
    seekers = deduplicate_seekers(seekers)
    log(f"去重后剩余 {len(seekers)} 条记录")

    # 统计提及电影的用户提问占比
    has_movie = sum(1 for r in seekers if r.get('imdb_ids'))
    log(f"  其中提及电影ID的提问: {has_movie} 条 ({has_movie/len(seekers)*100:.1f}%)")

    # 执行情感分析并标注到每条记录
    log("正在对用户提问文本进行情感分析...")
    seekers = annotate_sentiment(seekers)
    log(f"已完成 {len(seekers)} 条记录的情感标注")

    # U1: 整体类型情感分布
    dim_u1_genre_sentiment_distribution(seekers, movie_info)
    log("")

    # U2: 类型平均情感得分排名
    dim_u2_genre_avg_score_ranking(seekers, movie_info)
    log("")

    # U3: 类型情感强度分布
    dim_u3_genre_intensity_distribution(seekers, movie_info)
    log("")

    # U4: 节假日 vs 非节假日类型情感
    dim_u4_holiday_vs_nonholiday(seekers, movie_info)
    log("")

    # U5: 类型 × 情感占比热力图
    dim_u5_genre_sentiment_heatmap(seekers, movie_info)
    log("")

    # U6: 类型 × 强度占比热力图
    dim_u6_genre_intensity_heatmap(seekers, movie_info)
    log("")

    # U7: 类型提及次数与平均得分
    dim_u7_mentions_vs_score(seekers, movie_info)

    log("")
    log("=" * 60)
    log(f"步骤 13 完成！结果已保存到 {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
