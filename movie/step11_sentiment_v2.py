# -*- coding: utf-8 -*-
"""
Step 11 Sentiment V2: 用户对系统推荐电影的回复情感分析
==========================================================
与 step11_sentiment.py（分析用户提问文本的情感）不同，V2 分析的是：
  **用户对系统推荐电影的回复内容**（user_text from conv_pairs）
即：系统推荐了一部电影后，用户的反馈文本（如"谢谢推荐！"、"这部片很差"等）。

分析流程：
  1. 通过 conv_pairs.build_pairs_from_rows() 提取 (系统回复→用户回应) 配对
  2. 对 user_text（用户回复内容）运行 VADER+AFINN 情感分析
  3. 复用 step11 的 S1-S10 分析维度（节假日/工作日/周末对比）

输出：output/movie/step11/v2/
"""

import os
import csv
import re
import calendar
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log
from movie.utils.text import deduplicate_seekers
from movie.utils.sentiment import analyze_batch
from movie.utils.conv_pairs import build_pairs_from_rows

# IMDB ID 正则：匹配 tt + 7~9 位数字（用于从系统回复中提取推荐电影 ID）
_TT_PATTERN = re.compile(r'tt\d{7,9}')

# ── 初始化 ──────────────────────────────────────────────────────────
setup_matplotlib()
STEP_OUT = os.path.join(STEP_DIRS[11], 'v2')       # 输出到 step11/v2/
os.makedirs(STEP_OUT, exist_ok=True)

# ── 颜色方案（与 step11 一致）────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════
#  数据准备：提取用户对系统推荐的回复
# ═══════════════════════════════════════════════════════════════════════

def _extract_user_reply_records(rows: list[dict]) -> list[dict]:
    """
    从原始数据行中提取 (系统回复 → 用户回应) 配对，并对用户回复文本进行情感分析。

    规则：
      - 使用 conv_pairs.build_pairs_from_rows() 提取会话配对
      - 只保留有用户回复的配对（is_solo_system=False 且 user_text 非空）
      - 继承会话的 period / is_holiday / holiday_name 属性

    Returns:
        每条记录包含: session_id, pair_id, date, period, is_holiday, holiday_name,
                      user_text, sentiment, intensity, sentiment_score
    """
    pairs = build_pairs_from_rows(rows)
    log(f"  共提取 {len(pairs)} 条会话配对")

    # 过滤：只保留有用户回复的配对
    valid_pairs = [p for p in pairs
                   if not p.get('is_solo_system', True)
                   and p.get('user_text', '').strip()]
    log(f"  有效用户回复配对: {len(valid_pairs)} 条（过滤 {len(pairs) - len(valid_pairs)} 条）")

    if not valid_pairs:
        return []

    # 对用户回复文本进行情感分析
    user_texts = [p['user_text'] for p in valid_pairs]
    sentiment_results = analyze_batch(user_texts)

    records = []
    for p, sr in zip(valid_pairs, sentiment_results):
        # 从系统回复文本中提取推荐的电影 ID（规则8）
        system_text = p.get('system_text', '')
        imdb_ids = list(set(_TT_PATTERN.findall(system_text)))
        records.append({
            'session_id':    p.get('session_id', ''),
            'pair_id':       p.get('pair_id', ''),
            'date':          p.get('date', ''),
            'period':        p.get('period', ''),
            'is_holiday':    p.get('is_holiday', False),
            'holiday_name':  p.get('holiday_name', ''),
            'holiday_type':  p.get('holiday_type', ''),
            'cross_day':     p.get('cross_day', False),
            'user_text':     p['user_text'],
            'system_text':   system_text,
            'imdb_ids':      imdb_ids,
            'sentiment':     sr['sentiment'],
            'intensity':     sr['intensity'],
            'sentiment_score': sr['score'],
        })

    log(f"  情感分析完成: {len(records)} 条用户回复记录")
    return records


def _aggregate_to_session(records: list[dict]) -> list[dict]:
    """
    会话级聚合：1 session = 1 row（取该会话所有回复的平均情感）。

    Returns:
        每条记录包含: session_id, date, period, is_holiday, holiday_name, cross_day,
                      mean_sentiment, std_sentiment, n_replies,
                      pos_ratio, neg_ratio, dominant_sentiment, dominant_intensity
    """
    grouped = defaultdict(list)
    for r in records:
        grouped[r['session_id']].append(r)

    sessions = []
    for sid, recs in grouped.items():
        scores = [r['sentiment_score'] for r in recs]
        arr = np.array(scores)
        base = recs[0]

        # 主要情感类型（出现最多的）
        sent_counts = Counter(r['sentiment'] for r in recs)
        dominant_sent = sent_counts.most_common(1)[0][0]
        inten_counts = Counter(r['intensity'] for r in recs)
        dominant_inten = inten_counts.most_common(1)[0][0]

        sessions.append({
            'session_id':      sid,
            'date':            base['date'],
            'period':          base['period'],
            'is_holiday':      base['is_holiday'],
            'holiday_name':    base['holiday_name'],
            'holiday_type':    base['holiday_type'],
            'cross_day':       base['cross_day'],
            'mean_sentiment':  float(arr.mean()),
            'std_sentiment':   float(arr.std()) if len(arr) > 1 else 0.0,
            'n_replies':       len(arr),
            'pos_ratio':       float((arr > 0.05).sum() / max(len(arr), 1)),
            'neg_ratio':       float((arr < -0.05).sum() / max(len(arr), 1)),
            'dominant_sentiment': dominant_sent,
            'dominant_intensity': dominant_inten,
        })
    log(f"  会话级聚合: {len(sessions)} 个会话（共 {len(records)} 条回复）")
    return sessions


# ═══════════════════════════════════════════════════════════════════════
#  辅助绘图（复用 step11_sentiment 的逻辑，重定向输出目录）
# ═══════════════════════════════════════════════════════════════════════

def _plot_dual_grouped_bar(
    left_data, left_title, right_data, right_title,
    filename, colors=None, ylabel='Proportion', suptitle='',
):
    """双面板分组柱状图（左+右）。"""
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


def _plot_heatmap(matrix, row_labels, col_labels, title, filename,
                  fmt='.2f', cmap='RdBu_r', center=None, cbar_label=''):
    """热力图（复用 step11_sentiment 的逻辑）。"""
    if matrix.size == 0:
        return
    fig_h = max(5, len(row_labels) * 0.35 + 2)
    fig_w = max(7, len(col_labels) * 0.8 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect='auto', cmap=cmap,
                   **({} if center is None else {'vmin': -abs(matrix).max(),
                                                   'vmax': abs(matrix).max()}))
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


def _plot_horizontal_bar(items, title, filename, xlabel='Value',
                         top_n=20, color='#74b9ff'):
    """水平柱状图（排名）。"""
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


# ═══════════════════════════════════════════════════════════════════════
#  基线日期工具（同 step11_sentiment）
# ═══════════════════════════════════════════════════════════════════════

def _get_week_before(date_str: str, holiday_dates: set) -> str:
    """获取周前基线日期（同一星期 7 天前，若也是节假日则再退 7 天）。"""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    for _ in range(6):
        dt -= timedelta(days=7)
        candidate = dt.strftime('%Y-%m-%d')
        if candidate not in holiday_dates:
            return candidate
    return ''


def _get_month_before(date_str: str, holiday_dates: set) -> str:
    """获取月前基线日期（同一日历日一个月前，若也是节假日则退 7 天重试）。"""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    year, month, day = dt.year, dt.month, dt.day
    if month == 1:
        prev_month, prev_year = 12, year - 1
    else:
        prev_month, prev_year = month - 1, year
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
#  S1: 节假日 VS 非节假日 用户回复情感（Dual Panel）
# ═══════════════════════════════════════════════════════════════════════
# 【分析对象】用户对系统推荐的回复文本（非用户原始提问）
# 【统计口径】情感得分 VADER compound [-1,1]；强度 = |compound|
# ═══════════════════════════════════════════════════════════════════════

def dim_s1_holiday_vs_nonholiday(records: list[dict]):
    """节假日 vs 非节假日：用户回复情感对比（双面板）。"""
    log("=" * 50)
    log("S1: 节假日 vs 非节假日用户回复情感对比")

    holiday = [r for r in records if r['period'] == 'holiday']
    non_holiday = [r for r in records if r['period'] != 'holiday']

    h_n = len(holiday)
    nh_n = len(non_holiday)
    log(f"  节假日: {h_n} 条回复, 非节假日: {nh_n} 条回复")

    if h_n == 0 or nh_n == 0:
        log("  其中一组为空，无法对比")
        return

    # Sentiment proportions
    h_sent = Counter(r['sentiment'] for r in holiday)
    nh_sent = Counter(r['sentiment'] for r in non_holiday)
    sent_data = {
        '节假日':   {k: h_sent.get(k, 0) / h_n for k in ['positive', 'neutral', 'negative']},
        '非节假日': {k: nh_sent.get(k, 0) / nh_n for k in ['positive', 'neutral', 'negative']},
    }

    # Intensity proportions
    h_int = Counter(r['intensity'] for r in holiday)
    nh_int = Counter(r['intensity'] for r in non_holiday)
    inten_data = {
        '节假日':   {k: h_int.get(k, 0) / h_n for k in ['mild', 'moderate', 'strong']},
        '非节假日': {k: nh_int.get(k, 0) / nh_n for k in ['mild', 'moderate', 'strong']},
    }

    log(f"  节假日    — 正面={sent_data['节假日']['positive']*100:.1f}%, "
        f"负面={sent_data['节假日']['negative']*100:.1f}%, "
        f"强烈={inten_data['节假日']['strong']*100:.1f}%")
    log(f"  非节假日  — 正面={sent_data['非节假日']['positive']*100:.1f}%, "
        f"负面={sent_data['非节假日']['negative']*100:.1f}%, "
        f"强烈={inten_data['非节假日']['strong']*100:.1f}%")

    group_colors = {'节假日': '#ff6b6b', '非节假日': '#74b9ff'}
    _plot_dual_grouped_bar(
        sent_data, '用户回复情感倾向分布',
        inten_data, '用户回复强度分布',
        's1_holiday_vs_nonholiday_v2.png',
        colors=group_colors,
        suptitle='用户对系统推荐的回复情感：节假日 vs 非节假日',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 's1_holiday_vs_nonholiday_v2.csv')
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
#  S2: 按时期的用户回复情感（Dual Panel）
# ═══════════════════════════════════════════════════════════════════════

def dim_s2_by_period(records: list[dict]):
    """按时期的用户回复情感与强度分布（双面板）。"""
    log("=" * 50)
    log("S2: 按时期的用户回复情感分布")

    periods = ['holiday', 'workday', 'weekend']
    period_cn = {'holiday': '节假日', 'workday': '工作日', 'weekend': '周末'}
    sent_data = {}
    inten_data = {}

    for p in periods:
        subset = [r for r in records if r['period'] == p]
        if not subset:
            continue
        n = len(subset)
        label = period_cn[p]

        s_counts = Counter(r['sentiment'] for r in subset)
        sent_data[label] = {k: s_counts.get(k, 0) / n for k in ['positive', 'neutral', 'negative']}

        i_counts = Counter(r['intensity'] for r in subset)
        inten_data[label] = {k: i_counts.get(k, 0) / n for k in ['mild', 'moderate', 'strong']}

        avg = np.mean([r['sentiment_score'] for r in subset])
        log(f"  {label}: 情感={dict(s_counts)}, 强度={dict(i_counts)}, "
            f"平均分={avg:.3f} ({n} 条回复)")

    if not sent_data or not inten_data:
        log("  无可用时期数据")
        return

    _plot_dual_grouped_bar(
        sent_data, '用户回复情感倾向分布',
        inten_data, '用户回复强度分布',
        's2_sentiment_by_period_v2.png',
        colors=PERIOD_COLORS,
        suptitle='用户对系统推荐的回复情感：节假日 / 工作日 / 周末',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 's2_sentiment_by_period_v2.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['period', 'total',
                    'pos', 'pos_pct', 'neu', 'neu_pct', 'neg', 'neg_pct',
                    'mild', 'mild_pct', 'moderate', 'moderate_pct',
                    'strong', 'strong_pct', 'avg_score'])
        for p in periods:
            subset = [r for r in records if r['period'] == p]
            if not subset:
                continue
            n = len(subset)
            s_counts = Counter(r['sentiment'] for r in subset)
            i_counts = Counter(r['intensity'] for r in subset)
            avg = np.mean([r['sentiment_score'] for r in subset])
            w.writerow([
                p, n,
                s_counts.get('positive', 0), f'{s_counts.get("positive", 0) / n * 100:.1f}',
                s_counts.get('neutral', 0),  f'{s_counts.get("neutral", 0) / n * 100:.1f}',
                s_counts.get('negative', 0), f'{s_counts.get("negative", 0) / n * 100:.1f}',
                i_counts.get('mild', 0),     f'{i_counts.get("mild", 0) / n * 100:.1f}',
                i_counts.get('moderate', 0), f'{i_counts.get("moderate", 0) / n * 100:.1f}',
                i_counts.get('strong', 0),   f'{i_counts.get("strong", 0) / n * 100:.1f}',
                f'{avg:.3f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S3: 各节假日用户回复情感画像（Heatmap vs 全局基线）
# ═══════════════════════════════════════════════════════════════════════

def dim_s3_per_holiday(records: list[dict]):
    """各节假日用户回复情感画像，柱状图 + 非节假日基线虚线。"""
    log("=" * 50)
    log("S3: 各节假日用户回复情感画像（柱状图 + 基线虚线）")

    # 非节假日基线
    non_holiday = [r for r in records if r['period'] != 'holiday']
    if not non_holiday:
        log("  无非节假日数据")
        return
    nh_n = len(non_holiday)
    nh_pos = sum(1 for r in non_holiday if r['sentiment'] == 'positive') / nh_n * 100
    nh_neu = sum(1 for r in non_holiday if r['sentiment'] == 'neutral') / nh_n * 100
    nh_neg = sum(1 for r in non_holiday if r['sentiment'] == 'negative') / nh_n * 100
    nh_avg = float(np.mean([r['sentiment_score'] for r in non_holiday]))

    holiday_groups = defaultdict(list)
    for r in records:
        if r['is_holiday']:
            name = r.get('holiday_name', '')[:8]
            holiday_groups[name].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items() if len(v) >= MIN_DATA_ROWS}

    if not holiday_groups:
        log("  无数据足够的节假日分组")
        return

    holiday_names = sorted(holiday_groups.keys())
    csv_rows = []

    # 计算每个节假日的正面/中性/负面占比
    h_pos, h_neu, h_neg, h_avg = [], [], [], []
    for hn in holiday_names:
        group = holiday_groups[hn]
        n = len(group)
        counts = Counter(r['sentiment'] for r in group)
        avg_score = float(np.mean([r['sentiment_score'] for r in group]))
        pos_pct = counts.get('positive', 0) / max(n, 1) * 100
        neu_pct = counts.get('neutral', 0) / max(n, 1) * 100
        neg_pct = counts.get('negative', 0) / max(n, 1) * 100
        h_pos.append(pos_pct)
        h_neu.append(neu_pct)
        h_neg.append(neg_pct)
        h_avg.append(avg_score)

        row = {'holiday': hn, 'count': n}
        row['positive_pct'] = f'{pos_pct:.1f}'
        row['neutral_pct'] = f'{neu_pct:.1f}'
        row['negative_pct'] = f'{neg_pct:.1f}'
        row['positive_diff'] = f'{pos_pct - nh_pos:+.1f}'
        row['neutral_diff'] = f'{neu_pct - nh_neu:+.1f}'
        row['negative_diff'] = f'{neg_pct - nh_neg:+.1f}'
        row['avg_score'] = f'{avg_score:.3f}'
        row['nh_avg_score'] = f'{nh_avg:.3f}'
        csv_rows.append(row)
        log(f"  {hn} (n={n}): 正面={counts.get('positive', 0)}, "
            f"中性={counts.get('neutral', 0)}, 负面={counts.get('negative', 0)}, "
            f"平均分={avg_score:.3f}")

    # ── 柱状图：每个节假日 3 根柱（正面/中性/负面），基线虚线 ──
    fig, ax = plt.subplots(figsize=(max(14, len(holiday_names) * 1.2), 6))
    x = np.arange(len(holiday_names))
    width = 0.22

    bars_pos = ax.bar(x - width, h_pos, width, label='Positive',
                      color=SENT_COLORS['positive'], alpha=0.8)
    bars_neu = ax.bar(x, h_neu, width, label='Neutral',
                      color=SENT_COLORS['neutral'], alpha=0.8)
    bars_neg = ax.bar(x + width, h_neg, width, label='Negative',
                      color=SENT_COLORS['negative'], alpha=0.8)

    # 非节假日基线虚线
    ax.axhline(y=nh_pos, color=SENT_COLORS['positive'], linestyle='--',
               linewidth=1.8, alpha=0.9, label=f'Baseline Pos ({nh_pos:.1f}%)')
    ax.axhline(y=nh_neu, color=SENT_COLORS['neutral'], linestyle='--',
               linewidth=1.8, alpha=0.9, label=f'Baseline Neu ({nh_neu:.1f}%)')
    ax.axhline(y=nh_neg, color=SENT_COLORS['negative'], linestyle='--',
               linewidth=1.8, alpha=0.9, label=f'Baseline Neg ({nh_neg:.1f}%)')

    # 偏差 >5pp 的柱加黑边框
    for bar, val, ref in (
        [(bars_pos[i], h_pos[i], nh_pos) for i in range(len(holiday_names))]
        + [(bars_neu[i], h_neu[i], nh_neu) for i in range(len(holiday_names))]
        + [(bars_neg[i], h_neg[i], nh_neg) for i in range(len(holiday_names))]
    ):
        if abs(val - ref) > 5.0:
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(holiday_names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('各节假日用户回复情感分布 vs 非节假日基线', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 's3_per_holiday_sentiment_v2.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 's3_per_holiday_sentiment_v2.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['holiday', 'count',
                                          'positive_pct', 'positive_diff',
                                          'neutral_pct', 'neutral_diff',
                                          'negative_pct', 'negative_diff',
                                          'avg_score', 'nh_avg_score'])
        w.writeheader()
        w.writerows(csv_rows)
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S4: 各节假日电影类型情感差异 (Heatmap)
#  S4: Per-Holiday Genre Sentiment Difference
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】热力图: 行=节假日, 列=电影类型, 值=用户回复情感均值差值 vs 非节假日基线
# 【统计口径】从系统回复中提取推荐电影ID，关联 movie_info 中的类型，
#   计算各节假日×类型的用户回复平均情感得分与非节假日基线的差值
#   过滤: 仅保留数据充足的节假日×类型组合
# 【输出文件】PNG: s4_genre_sentiment_by_holiday_v2.png, CSV: s4_*.csv
# ═══════════════════════════════════════════════════════════════════════

def _compute_genre_sentiment_v2(
    records_subset: list[dict], movie_info: dict,
) -> dict[str, dict]:
    """计算一组用户回复记录中各电影类型的平均情感得分。

    V2 版本：直接使用 records 中的 imdb_ids（从系统回复中提取），
    关联 movie_info 获取类型，计算用户回复的平均情感。

    Returns:
        dict[genre] -> {'score_sum': float, 'count': int, 'avg': float}
    """
    stats = defaultdict(lambda: {'score_sum': 0.0, 'count': 0})
    for r in records_subset:
        score = r.get('sentiment_score', 0.0)
        for tid in r.get('imdb_ids', []):
            info = movie_info.get(tid)
            if info and 'genres' in info:
                for g in info['genres']:
                    s = stats[g]
                    s['score_sum'] += score
                    s['count'] += 1
    for g, s in stats.items():
        s['avg'] = s['score_sum'] / max(s['count'], 1)
    return dict(stats)


def dim_s4_genre_by_holiday(records: list[dict], movie_info: dict):
    """各节假日电影类型情感差异热力图 vs 非节假日基线。

    对每个节假日（数据充足）和每个类型（基线数据充足），计算用户回复的
    平均情感得分。热力图：行=节假日，列=类型，颜色=与非节假日基线的差值。
    """
    log("=" * 50)
    log("S4: 各节假日电影类型情感差异 vs 非节假日基线")

    if not movie_info:
        log("  无 movie_info 数据，跳过")
        return

    # ── 1. 非节假日基线 ──
    non_holiday = [r for r in records if r['period'] != 'holiday']
    if not non_holiday:
        log("  无非节假日基线数据")
        return
    base_stats = _compute_genre_sentiment_v2(non_holiday, movie_info)
    base_genres = {g for g, s in base_stats.items() if s['count'] >= MIN_DATA_ROWS}
    if not base_genres:
        log("  基线中无类型达到最小提及阈值")
        return

    # ── 2. 各节假日分组 ──
    holiday_groups = defaultdict(list)
    for r in records:
        if r['is_holiday']:
            holiday_groups[r.get('holiday_name', '')[:8]].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items()
                      if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  无数据充足的节假日分组")
        return

    holiday_names = sorted(holiday_groups.keys())
    log(f"  基线: {len(base_genres)} 个类型, 来自 {len(non_holiday)} 条非节假日记录")
    log(f"  节假日: {len(holiday_names)} 个分组")

    # ── 3. 构建矩阵: 节假日 × 类型 ──
    shared_genres = sorted(base_genres)
    n_holidays = len(holiday_names)
    n_genres = len(shared_genres)
    matrix = np.full((n_holidays, n_genres), np.nan)
    csv_rows = []

    for i, hn in enumerate(holiday_names):
        h_stats = _compute_genre_sentiment_v2(holiday_groups[hn], movie_info)
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

        # 记录偏差最大的正面/负面类型
        valid = [(genre, matrix[i, j]) for j, genre in enumerate(shared_genres)
                 if not np.isnan(matrix[i, j])]
        valid.sort(key=lambda x: x[1], reverse=True)
        if valid:
            top_pos = valid[:3]
            top_neg = valid[-3:]
            log(f"  {hn}: 偏差最大正面={[(g, f'{d:+.3f}') for g, d in top_pos]}, "
                f"偏差最大负面={[(g, f'{d:+.3f}') for g, d in top_neg]}")

    # ── 4. 过滤: 去掉所有节假日都无数据的类型 ──
    col_has_data = ~np.all(np.isnan(matrix), axis=0)
    if col_has_data.sum() == 0:
        log("  无类型-节假日组合达到最小提及阈值")
        return
    matrix = matrix[:, col_has_data]
    active_genres = [g for g, keep in zip(shared_genres, col_has_data) if keep]

    log(f"  热力图: {n_holidays} 节假日 × {len(active_genres)} 类型")

    # ── 5. 热力图 ──
    fig_h = max(5, n_holidays * 0.35 + 2)
    fig_w = max(10, len(active_genres) * 0.7 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    masked = np.ma.masked_invalid(matrix)
    vmax = max(np.nanmax(np.abs(matrix)), 0.01)
    im = ax.imshow(masked, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
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
    ax.set_title('用户回复情感差异: 各节假日 vs 非节假日基线\n'
                 '(红=节假日更正面, 蓝=节假日更负面)', fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Sentiment Score Difference (Holiday − Baseline)', fontsize=9)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 's4_genre_sentiment_by_holiday_v2.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")

    # ── 6. CSV ──
    csv_path = os.path.join(STEP_OUT, 's4_genre_sentiment_by_holiday_v2.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        header = ['holiday', 'reply_count']
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
                    stats_h = _compute_genre_sentiment_v2(holiday_groups[hn], movie_info)
                    h_avg = stats_h[genre]['avg']
                    row.extend([f'{h_avg:.4f}', f'{val:+.4f}'])
            w.writerow(row)
        # 基线行
        base_row = ['__baseline__', len(non_holiday)]
        for genre in active_genres:
            base_avg = base_stats[genre]['avg']
            base_row.extend([f'{base_avg:.4f}', '0.0000'])
        w.writerow(base_row)
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S5: 会话级情感均值对比（节假日/工作日/周末，带95%CI）
# ═══════════════════════════════════════════════════════════════════════

def dim_s4_session_sentiment_by_period(sessions: list[dict]):
    """
    会话级情感均值对比（按时期），带95%CI误差线。
    每个会话的 mean_sentiment 作为观测值，按 period 分组比较。
    """
    log("=" * 50)
    log("S4: 会话级用户回复情感均值对比（带95%CI）")

    periods = ['holiday', 'workday', 'weekend']
    period_cn = {'holiday': '节假日', 'workday': '工作日', 'weekend': '周末'}
    period_colors = ['#ff6b6b', '#74b9ff', '#feca57']

    groups = {}
    for p in periods:
        subset = [s for s in sessions if s['period'] == p]
        if not subset:
            continue
        scores = np.array([s['mean_sentiment'] for s in subset])
        n = len(scores)
        mean = float(scores.mean())
        se = float(scores.std()) / np.sqrt(n) if n > 1 else 0.0
        ci95 = 1.96 * se
        groups[p] = {'n': n, 'mean': mean, 'se': se, 'ci95': ci95}
        log(f"  {period_cn[p]}: n={n}, mean={mean:.3f}, ±95%CI={ci95:.3f}")

    if len(groups) < 2:
        log("  时期分组不足，跳过")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [period_cn[p] for p in periods if p in groups]
    means = [groups[p]['mean'] for p in periods if p in groups]
    cis = [groups[p]['ci95'] for p in periods if p in groups]
    colors = [period_colors[periods.index(p)] for p in periods if p in groups]
    x = np.arange(len(labels))

    ax.bar(x, means, 0.5, yerr=cis, capsize=6, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Mean Sentiment Score (±95% CI)', fontsize=10)
    ax.set_title('会话级用户回复情感均值：节假日 / 工作日 / 周末', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-0.2, 0.6)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 's4_session_sentiment_by_period_v2.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 's4_session_sentiment_by_period_v2.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['period', 'n_sessions', 'mean_sentiment', 'se', 'ci95'])
        for p in periods:
            if p in groups:
                g = groups[p]
                w.writerow([p, g['n'], f'{g["mean"]:.4f}',
                            f'{g["se"]:.4f}', f'{g["ci95"]:.4f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S5: 各节假日情感 VS 非节假日均值（带参考线的柱状图）
# ═══════════════════════════════════════════════════════════════════════

def dim_s5_per_holiday_vs_baseline(records: list[dict]):
    """每个节假日的用户回复情感 vs 全局非节假日均值（带参考线柱状图）。"""
    log("=" * 50)
    log("S5: 各节假日用户回复情感 vs 非节假日均值")

    non_holiday = [r for r in records if r['period'] != 'holiday']
    if not non_holiday:
        log("  无非节假日数据")
        return
    nh_n = len(non_holiday)
    nh_pos = sum(1 for r in non_holiday if r['sentiment'] == 'positive') / nh_n * 100
    nh_neu = sum(1 for r in non_holiday if r['sentiment'] == 'neutral') / nh_n * 100
    nh_neg = sum(1 for r in non_holiday if r['sentiment'] == 'negative') / nh_n * 100

    holiday_groups = defaultdict(list)
    for r in records:
        if r['is_holiday']:
            holiday_groups[r.get('holiday_name', '')[:8]].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items() if len(v) >= MIN_DATA_ROWS}

    if not holiday_groups:
        log("  无数据足够的节假日")
        return

    names = sorted(holiday_groups.keys())
    log(f"  非节假日基线: n={nh_n}, Pos={nh_pos:.1f}%, Neu={nh_neu:.1f}%, Neg={nh_neg:.1f}%")

    h_pos = [sum(1 for r in holiday_groups[n] if r['sentiment'] == 'positive') / max(len(holiday_groups[n]), 1) * 100
             for n in names]
    h_neu = [sum(1 for r in holiday_groups[n] if r['sentiment'] == 'neutral') / max(len(holiday_groups[n]), 1) * 100
             for n in names]
    h_neg = [sum(1 for r in holiday_groups[n] if r['sentiment'] == 'negative') / max(len(holiday_groups[n]), 1) * 100
             for n in names]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 1.5), 6))
    x = np.arange(len(names))
    width = 0.22

    bars_pos = ax.bar(x - width, h_pos, width, label='Positive',
                      color=SENT_COLORS['positive'], alpha=0.8)
    bars_neu = ax.bar(x, h_neu, width, label='Neutral',
                      color=SENT_COLORS['neutral'], alpha=0.8)
    bars_neg = ax.bar(x + width, h_neg, width, label='Negative',
                      color=SENT_COLORS['negative'], alpha=0.8)

    ax.axhline(y=nh_pos, color=SENT_COLORS['positive'], linestyle='--',
               linewidth=1.8, alpha=0.9, label=f'Non-holiday Pos ({nh_pos:.1f}%)')
    ax.axhline(y=nh_neu, color=SENT_COLORS['neutral'], linestyle='--',
               linewidth=1.8, alpha=0.9, label=f'Non-holiday Neu ({nh_neu:.1f}%)')
    ax.axhline(y=nh_neg, color=SENT_COLORS['negative'], linestyle='--',
               linewidth=1.8, alpha=0.9, label=f'Non-holiday Neg ({nh_neg:.1f}%)')

    for bar, val, ref in (
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
    ax.set_title('用户回复情感：各节假日 vs 非节假日均值', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 's5_per_holiday_vs_baseline_v2.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 's5_per_holiday_vs_baseline_v2.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'n_replies', 'pos_pct', 'neu_pct', 'neg_pct',
                    'nh_pos_pct', 'nh_neu_pct', 'nh_neg_pct',
                    'pos_diff', 'neu_diff', 'neg_diff'])
        for i, n in enumerate(names):
            w.writerow([n, len(holiday_groups[n]),
                        f'{h_pos[i]:.1f}', f'{h_neu[i]:.1f}', f'{h_neg[i]:.1f}',
                        f'{nh_pos:.1f}', f'{nh_neu:.1f}', f'{nh_neg:.1f}',
                        f'{h_pos[i] - nh_pos:+.1f}',
                        f'{h_neu[i] - nh_neu:+.1f}',
                        f'{h_neg[i] - nh_neg:+.1f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S6: 情感关联关键词（用户回复中的正面/负面标志词）
# ═══════════════════════════════════════════════════════════════════════

# 轻量停用词（同 step11_sentiment）
_S6_STOPWORDS = {
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
}


def _tokenize_reply(text: str) -> list[str]:
    """对用户回复文本进行分词（复用共享工具）。"""
    from movie.utils.text import tokenize as _shared_tokenize
    return _shared_tokenize(text, stopwords=_S6_STOPWORDS)


def dim_s6_keywords(records: list[dict]):
    """情感关联关键词：在正面/负面回复中出现频率差异最大的词。"""
    log("=" * 50)
    log("S6: 用户回复情感关联关键词")

    pos_texts = [r for r in records if r['sentiment'] == 'positive']
    neg_texts = [r for r in records if r['sentiment'] == 'negative']
    log(f"  正面回复: {len(pos_texts)} 条, 负面回复: {len(neg_texts)} 条")

    if not pos_texts or not neg_texts:
        log("  需要同时包含正面和负面回复才能进行关键词对比")
        return

    def _count(texts):
        freq = Counter()
        for r in texts:
            tokens = set(_tokenize_reply(r['user_text']))
            freq.update(tokens)
        return freq

    pos_freq = _count(pos_texts)
    neg_freq = _count(neg_texts)
    pos_n, neg_n = len(pos_texts), len(neg_texts)

    MIN_TOTAL = 5  # 回复文本较短，降低阈值
    results = []
    all_words = set(pos_freq.keys()) | set(neg_freq.keys())
    for w in all_words:
        total = pos_freq.get(w, 0) + neg_freq.get(w, 0)
        if total < MIN_TOTAL:
            continue
        pos_rate = (pos_freq.get(w, 0) + 0.5) / (pos_n + 1)
        neg_rate = (neg_freq.get(w, 0) + 0.5) / (neg_n + 1)
        log_or = np.log2(pos_rate / neg_rate)
        results.append((w, log_or, pos_freq.get(w, 0), neg_freq.get(w, 0), total))

    results.sort(key=lambda x: x[1], reverse=True)
    pos_top = [r for r in results if r[1] > 0.5][:20]
    neg_top = [r for r in reversed(results) if r[1] < -0.5][:20]

    log("  正面关联关键词（log-odds ratio > 0.5）：")
    for w, lor, pos_c, neg_c, tot in pos_top[:10]:
        log(f"    {w}: log2 比率={lor:+.2f} (正面={pos_c}, 负面={neg_c})")
    log("  负面关联关键词（log-odds ratio < -0.5）：")
    for w, lor, pos_c, neg_c, tot in neg_top[:10]:
        log(f"    {w}: log2 比率={lor:+.2f} (正面={pos_c}, 负面={neg_c})")

    if pos_top:
        _plot_horizontal_bar(
            [(w, lor) for w, lor, _, _, _ in pos_top],
            '正面关联关键词 (log2 正面/负面比率)',
            's6_positive_keywords_v2.png',
            xlabel='log2(正面/负面比率)', color='#2ecc71',
        )
    if neg_top:
        _plot_horizontal_bar(
            [(w, abs(lor)) for w, lor, _, _, _ in neg_top],
            '负面关联关键词 (log2 正面/负面比率)',
            's6_negative_keywords_v2.png',
            xlabel='|log2(正面/负面比率)|', color='#e74c3c',
        )

    csv_path = os.path.join(STEP_OUT, 's6_sentiment_keywords_v2.csv')
    all_scored = sorted(results, key=lambda x: x[1], reverse=True)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['word', 'log2_pos_neg_ratio', 'pos_doc_count',
                    'neg_doc_count', 'total_doc_count'])
        for word, lor, pos_c, neg_c, tot in all_scored:
            w.writerow([word, f'{lor:.3f}', pos_c, neg_c, tot])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S7: 节假日 vs 基线（周前/月前）情感对比
# ═══════════════════════════════════════════════════════════════════════

def dim_s7_holiday_vs_baseline(records: list[dict]):
    """节假日 vs 周前/月前基线的用户回复情感对比（三面板柱状图）。"""
    log("=" * 50)
    log("S7: 节假日 vs 周前/月前基线用户回复情感对比")

    by_date = defaultdict(list)
    holiday_dates = set()
    for r in records:
        by_date[r['date']].append(r)
        if r['is_holiday']:
            holiday_dates.add(r['date'])

    if not holiday_dates:
        log("  无节假日数据")
        return

    name_groups = defaultdict(lambda: {'holiday': [], 'week_before': [], 'month_before': []})
    for h_date in sorted(holiday_dates):
        rows = by_date[h_date]
        if not rows:
            continue
        h_name = rows[0].get('holiday_name', '')[:8]
        if not h_name:
            continue
        name_groups[h_name]['holiday'].extend(rows)
        wb = _get_week_before(h_date, holiday_dates)
        if wb and wb in by_date:
            name_groups[h_name]['week_before'].extend(by_date[wb])
        mb = _get_month_before(h_date, holiday_dates)
        if mb and mb in by_date:
            name_groups[h_name]['month_before'].extend(by_date[mb])

    groups = {k: v for k, v in name_groups.items() if len(v['holiday']) >= MIN_DATA_ROWS}
    if not groups:
        log("  无数据足够的节假日")
        return

    names = sorted(groups.keys())

    def _pct(recs, sent_type):
        return sum(1 for r in recs if r['sentiment'] == sent_type) / max(len(recs), 1) * 100

    h_data = {n: {k: _pct(groups[n]['holiday'], k) for k in ['positive', 'neutral', 'negative']} for n in names}
    w_data = {n: {k: _pct(groups[n]['week_before'], k) if groups[n]['week_before'] else 0
                  for k in ['positive', 'neutral', 'negative']} for n in names}
    m_data = {n: {k: _pct(groups[n]['month_before'], k) if groups[n]['month_before'] else 0
                  for k in ['positive', 'neutral', 'negative']} for n in names}

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(10, len(names) * 1.8), 14))
    x = np.arange(len(names))
    width = 0.25
    sent_cfg = [('positive', SENT_COLORS['positive']),
                ('neutral', SENT_COLORS['neutral']),
                ('negative', SENT_COLORS['negative'])]

    for ax, data, title in [
        (ax1, h_data, 'Holiday User-Reply Sentiment (averaged across years)'),
        (ax2, w_data, 'Baseline (Week Before) User-Reply Sentiment'),
        (ax3, m_data, 'Baseline (Month Before) User-Reply Sentiment'),
    ]:
        for i, (s_type, color) in enumerate(sent_cfg):
            vals = [data[n].get(s_type, 0) for n in names]
            ax.bar(x + (i - 1) * width, vals, width, label=s_type.capitalize(),
                   color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Proportion (%)')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 's7_holiday_vs_baseline_v2.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved: {path}")

    csv_path = os.path.join(STEP_OUT, 's7_holiday_vs_baseline_v2.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday', 'n_holiday', 'n_week_before', 'n_month_before',
                    'holiday_pos_pct', 'holiday_neu_pct', 'holiday_neg_pct',
                    'week_before_pos_pct', 'week_before_neu_pct', 'week_before_neg_pct',
                    'month_before_pos_pct', 'month_before_neu_pct', 'month_before_neg_pct'])
        for n in names:
            g = groups[n]
            def _v(d, k): return f'{d[n][k]:.1f}'
            w.writerow([n, len(g['holiday']), len(g['week_before']), len(g['month_before']),
                        _v(h_data, 'positive'), _v(h_data, 'neutral'), _v(h_data, 'negative'),
                        _v(w_data, 'positive'), _v(w_data, 'neutral'), _v(w_data, 'negative'),
                        _v(m_data, 'positive'), _v(m_data, 'neutral'), _v(m_data, 'negative')])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  S8: 回复样本展示（正面/负面典型回复文本）
# ═══════════════════════════════════════════════════════════════════════

def dim_s8_sample_replies(records: list[dict]):
    """保存典型正面/负面回复文本样本到 CSV（供分析参考）。"""
    log("=" * 50)
    log("S8: 典型用户回复样本")

    pos_recs = [r for r in records if r['sentiment'] == 'positive']
    neg_recs = [r for r in records if r['sentiment'] == 'negative']

    # 按情感得分绝对值排序（最强烈的排前面）
    pos_recs.sort(key=lambda r: abs(r['sentiment_score']), reverse=True)
    neg_recs.sort(key=lambda r: abs(r['sentiment_score']), reverse=True)

    csv_path = os.path.join(STEP_OUT, 's8_sample_replies_v2.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sentiment', 'score', 'period', 'holiday_name', 'user_text'])
        for r in pos_recs[:30]:
            w.writerow(['positive', f'{r["sentiment_score"]:.3f}',
                        r['period'], r.get('holiday_name', ''),
                        r['user_text'][:200]])
        for r in neg_recs[:30]:
            w.writerow(['negative', f'{r["sentiment_score"]:.3f}',
                        r['period'], r.get('holiday_name', ''),
                        r['user_text'][:200]])
    log(f"  保存正面样本 {min(30, len(pos_recs))} 条, 负面样本 {min(30, len(neg_recs))} 条")
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None):
    log("=" * 60)
    log("Step 11 V2：用户对系统推荐电影的回复情感分析")
    log("=" * 60)

    # 加载数据
    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    rows = data.get('rows', [])
    movie_info = data.get('movie_info', {})
    log(f"已加载 {len(rows)} 条原始数据行, {len(movie_info)} 部电影信息")

    # ── 1. 提取用户对系统推荐的回复并进行情感分析 ──
    log("正在提取 (系统回复→用户回应) 配对...")
    records = _extract_user_reply_records(rows)

    if not records:
        log("  无有效用户回复数据，跳过所有分析")
        return records, []

    # ── 2. 会话级聚合 ──
    sessions = _aggregate_to_session(records)

    # ── 3. 保存全量回复级 CSV ──
    csv_path = os.path.join(STEP_OUT, 's0_all_user_replies_v2.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['session_id', 'pair_id', 'date', 'period', 'holiday_name',
                    'sentiment', 'intensity', 'score', 'user_text'])
        for r in records:
            w.writerow([r['session_id'], r['pair_id'], r['date'], r['period'],
                        r.get('holiday_name', ''), r['sentiment'], r['intensity'],
                        f'{r["sentiment_score"]:.4f}', r['user_text'][:300]])
    log(f"Saved: {csv_path}")

    # ── 4. 保存会话级 CSV ──
    csv_path = os.path.join(STEP_OUT, 's0_sessions_v2.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['session_id', 'date', 'period', 'holiday_name', 'cross_day',
                    'n_replies', 'mean_sentiment', 'std_sentiment',
                    'pos_ratio', 'neg_ratio', 'dominant_sentiment', 'dominant_intensity'])
        for s in sessions:
            w.writerow([s['session_id'], s['date'], s['period'],
                        s.get('holiday_name', ''), s['cross_day'],
                        s['n_replies'],
                        f'{s["mean_sentiment"]:.4f}', f'{s["std_sentiment"]:.4f}',
                        f'{s["pos_ratio"]:.3f}', f'{s["neg_ratio"]:.3f}',
                        s['dominant_sentiment'], s['dominant_intensity']])
    log(f"Saved: {csv_path}")

    # ── 5. 执行各维度分析 ──
    log("")
    dim_s1_holiday_vs_nonholiday(records)

    log("")
    dim_s2_by_period(records)

    log("")
    dim_s3_per_holiday(records)

    log("")
    dim_s4_genre_by_holiday(records, movie_info)

    log("")
    dim_s4_session_sentiment_by_period(sessions)

    log("")
    dim_s5_per_holiday_vs_baseline(records)

    log("")
    dim_s6_keywords(records)

    log("")
    dim_s7_holiday_vs_baseline(records)

    log("")
    dim_s8_sample_replies(records)

    log("")
    log("=" * 60)
    log(f"Step 11 V2 完成！结果已保存到 {STEP_OUT}")
    log("=" * 60)

    return records, sessions


if __name__ == '__main__':
    main()
