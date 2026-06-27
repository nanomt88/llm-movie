# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文注释

"""
Step 3: Conversation Turn & Time Analysis
步骤3：会话轮次与时间分析

全部/周周期 + 会话轮次次数和时间:
  节假日 VS 非节假日 平均会话轮次对比、多轮会话(去重)平均次数和占比
  节假日 VS 工作日 VS 周末 平均会话轮次对比、多轮会话(去重)平均次数和占比
  各个节假日 VS 非节假日 平均会话轮次对比、多轮会话(去重)平均次数和占比
  各个节假日 VS 工作日 VS 周末 平均会话轮次对比、多轮会话(去重)平均次数和占比

全部/周周期 + 多轮会话时间:
  同一会话中提问平均间隔时间对比、多轮会话平均持续时间(去重)
  同上的节假日/工作日/周末/各个节假日对比

全部/周周期 + 单日/跨日会话:
  单日多会话次数对比、跨日会话次数对比
  同上的节假日/工作日/周末/各个节假日对比

Output: output/movie/step3/*.png + CSV
输出目录：output/movie/step3/
"""

import os                          # 操作系统接口，路径和目录操作
import csv                         # CSV 文件读写
from collections import defaultdict, Counter  # 默认字典和计数器
from datetime import datetime, timezone       # 日期时间与时区处理

import numpy as np                 # 数值计算
import matplotlib                  # 绘图库
matplotlib.use('Agg')              # 使用 Agg 后端（无 GUI）
import matplotlib.pyplot as plt    # pyplot
import matplotlib.ticker as ticker # 刻度格式化

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log  # 配置
from movie.step1_question_freq import (   # 从步骤1复用颜色常量
    COLOR_HOLIDAY, COLOR_NONHOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND,
)

setup_matplotlib()                         # 初始化 matplotlib（后端+字体）
STEP_OUT = STEP_DIRS[3]                    # 步骤3输出目录：output/movie/step3/
os.makedirs(STEP_OUT, exist_ok=True)       # 确保输出目录存在

# Turn count group buckets
# 轮次分组桶
TURN_GROUPS = ['1', '2-5', '5-20', '20-100', '100+']

# Turn count to bucket mapping
# 轮次数到分桶的映射
def _turn_count_to_bucket(cnt: int) -> str:
    """Map a turn count to a group label.
    将轮次数映射到分组标签。"""
    if cnt == 1:
        return '1'
    elif 2 <= cnt <= 5:
        return '2-5'
    elif 5 <= cnt <= 20:
        return '5-20'
    elif 20 <= cnt <= 100:
        return '20-100'
    else:
        return '100+'


# ═══════════════════════════════════════════════════════════════════════
#  Helper: classify session period (by its first user question)
#  辅助函数：根据首个用户提问日期分类会话的时段
# ═══════════════════════════════════════════════════════════════════════

def _session_period_series(rows: list[dict]) -> dict[str, str]:
    """
    Determine the period label for each session based on its first user question's date.
    根据每个会话中首个用户提问的日期确定该会话的时段标签。
    Args:
        rows: 所有数据行（包含提问者和系统回复）
    Returns:
        dict[session_id] -> period ('holiday'/'workday'/'weekend')
    """
    session_period = {}                        # 会话ID -> 时段
    for r in rows:                             # 遍历所有数据行
        if r['is_seeker']:                     # 仅处理用户提问行
            sid = r['session_id']              # 会话ID
            if sid not in session_period:      # 只记录该会话的第一个提问的时段
                session_period[sid] = r['period']  # 将该时段赋给这个会话
    return session_period


def _session_first_date(rows: list[dict]) -> dict[str, str]:
    """
    Get the date of first user question for each session.
    获取每个会话中第一个用户提问的日期。
    Args:
        rows: 所有数据行
    Returns:
        dict[session_id] -> date_str（日期字符串）
    """
    first_date = {}
    for r in rows:
        if r['is_seeker']:                     # 仅处理用户提问
            sid = r['session_id']
            if sid not in first_date:          # 只记录第一个提问的日期
                first_date[sid] = r['date']
    return first_date


def _compute_turn_groups(rows: list[dict], dedup: bool = False) -> dict[str, int]:
    """
    Compute session count per turn group bucket.
    计算每个轮次分组桶中的会话数量。

    Turn groups (轮次分组):
      '1': 1 turn          （1 轮）
      '2-5': 2-5 turns     （2-5 轮）
      '5-20': 5-20 turns   （5-20 轮）
      '20-100': 20-100 turns （20-100 轮）
      '100+': 100+ turns   （100 轮以上）

    If dedup=True, count unique user questions per session (deduped by content).
    如果 dedup=True，按内容去重，统计每会话的唯一用户提问数。
    If dedup=False, count all rows per session.
    如果 dedup=False，统计每会话的所有行数。

    Returns:
        {bucket_label: session_count} （{分组标签: 会话数}）
    """
    if dedup:
        session_questions = _session_user_question_counts(rows)
        counts: dict[str, int] = {}
        for sid, questions in session_questions.items():
            unique = set()
            for q in questions:
                text = q.get('proc_text', '') or q.get('raw_text', '')
                unique.add(text.strip())
            counts[sid] = len(unique)
    else:
        counts = _session_turn_counts(rows)

    result = {g: 0 for g in TURN_GROUPS}
    for cnt in counts.values():
        if cnt == 1:
            result['1'] += 1
        elif 2 <= cnt <= 5:
            result['2-5'] += 1
        elif 5 <= cnt <= 20:
            result['5-20'] += 1
        elif 20 <= cnt <= 100:
            result['20-100'] += 1
        else:
            result['100+'] += 1
    return result


def _plot_turn_group_comparison(
    stats: dict[str, dict[str, dict[str, int]]],
    title: str, filename: str):
    """
    Bar chart comparing turn group distributions across groups.
    分组柱状图：比较不同组的轮次分布。
    stats: {group_label: {'no_dedup': {bucket: count}, 'dedup': {bucket: count}}}
    """
    groups = list(stats.keys())
    buckets = TURN_GROUPS
    x = np.arange(len(buckets))
    n_groups = len(groups)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, n_groups * 1.5), 5))
    width = 0.7 / max(n_groups, 1)

    group_colors = {
        'Holiday': COLOR_HOLIDAY,
        'Non-holiday': COLOR_NONHOLIDAY,
        'Workday': COLOR_WORKDAY,
        'Weekend': COLOR_WEEKEND,
    }

    for mode_idx, (mode, ax) in enumerate([('no_dedup', ax1), ('dedup', ax2)]):
        for i, group in enumerate(groups):
            vals = [stats[group][mode].get(b, 0) for b in buckets]
            offset = (i - (n_groups - 1) / 2) * width
            color = group_colors.get(group, f'C{i}')
            bars = ax.bar(x + offset, vals, width, label=group, color=color, alpha=0.8)
            # Annotate with thin black text on small bars avoid clutter
            for j, v in enumerate(vals):
                if v > max(max(vals) * 0.05, 5):
                    ax.text(x[j] + offset, v + max(vals) * 0.01,
                            f'{v}', ha='center', va='bottom', fontsize=6)

        mode_label = 'No Dedup (All Rows)' if mode == 'no_dedup' else 'Dedup (Unique Questions)'
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, fontsize=9)
        ax.set_xlabel('Turn Group')
        ax.set_ylabel('Session Count')
        ax.set_title(mode_label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  A: Session Turn Count Analysis
#  A: 会话轮次计数分析
# ═══════════════════════════════════════════════════════════════════════

def _session_turn_counts(rows: list[dict]) -> dict[str, int]:
    """
    Count total rows (turns) per session.
    统计每个会话的总行数（即轮次数，含提问和回复）。
    Args:
        rows: 所有数据行
    Returns:
        dict[session_id] -> turn_count（轮次数）
    """
    counts = Counter()                         # 计数器
    for r in rows:                             # 遍历每一行
        counts[r['session_id']] += 1           # 该会话轮次+1
    return dict(counts)


def _session_user_question_counts(rows: list[dict]) -> dict[str, list[dict]]:
    """
    Get all user questions grouped by session.
    获取按会话分组的所有用户提问。
    Args:
        rows: 所有数据行
    Returns:
        dict[session_id] -> list of row dicts (sorted by utc_time)
        每个会话中提问行的列表，按时间排序
    """
    session_questions = defaultdict(list)      # 会话ID -> 提问行列表
    for r in rows:
        if r['is_seeker']:                     # 仅取用户提问
            session_questions[r['session_id']].append(r)
    # Sort each session's questions by time
    for sid in session_questions:
        session_questions[sid].sort(key=lambda x: x['utc_time'])  # 按UTC时间升序
    return dict(session_questions)


def _multi_turn_stats(rows: list[dict]) -> dict:
    """
    Compute multi-turn session stats.
    计算多轮会话统计指标。

    Multi-turn (deduped): same session, user questions with different content.
    多轮会话（去重后）：同一会话中，用户提问内容不同的会话。
    A session with 2+ distinct user questions is multi-turn.
    只要有2个及以上不同的用户提问即为多轮会话。

    Returns（返回字典）:
        {
            'total_sessions': int,             # 总会话数
            'multi_turn_sessions': int,        # 多轮会话数
            'multi_turn_ratio': float,         # 多轮会话占比(%)
            'avg_turns_per_session': float,    # 每会话平均轮次
            'avg_turns_multi_turn': float,     # 多轮会话平均轮次
        }
    """
    session_questions = _session_user_question_counts(rows)  # 按会话分组的提问
    turn_counts = _session_turn_counts(rows)                 # 每会话总轮次

    total_sessions = len(session_questions)                  # 总会话数
    multi_turn_count = 0                                     # 多轮会话计数
    multi_turn_turns = []                                    # 多轮会话的轮次列表

    for sid, questions in session_questions.items():         # 遍历每个会话
        # Deduplicate by question content (processed text or raw text)
        unique_questions = set()                             # 去重后的提问内容集合
        for q in questions:
            # Use proc_text (processed field content) for dedup
            text = q.get('proc_text', '') or q.get('raw_text', '')  # 取处理文本或原始文本
            unique_questions.add(text.strip())               # 去除空白后加入集合

        if len(unique_questions) >= 2:                       # 2个以上不同提问 => 多轮会话
            multi_turn_count += 1
            multi_turn_turns.append(turn_counts.get(sid, 0))

    avg_turns = np.mean(list(turn_counts.values())) if turn_counts else 0  # 平均轮次
    avg_mt_turns = np.mean(multi_turn_turns) if multi_turn_turns else 0    # 多轮平均轮次
    multi_turn_ratio = multi_turn_count / max(total_sessions, 1) * 100     # 多轮占比

    return {
        'total_sessions': total_sessions,
        'multi_turn_sessions': multi_turn_count,
        'multi_turn_ratio': multi_turn_ratio,
        'avg_turns_per_session': avg_turns,
        'avg_turns_multi_turn': avg_mt_turns,
    }


def _plot_per_holiday_turn_bars(
    holiday_agg: list, baseline_val: float, baseline_label: str,
    value_key: str, ylabel: str, title: str, filename: str, color: str,
):
    """Bar chart: each holiday's metric vs baseline.
       柱状图：每个节假日指标与基线的对比。"""
    if not holiday_agg:
        return
    names = [h['name'] for h in holiday_agg]
    values = [h[value_key] for h in holiday_agg]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.6), 6))
    x = np.arange(len(names))
    width = 0.35

    bars = ax.bar(x, values, width, label='Holiday', color=color, alpha=0.85)
    ax.axhline(y=baseline_val, color='red', linestyle='--', linewidth=1.8,
               label=f'{baseline_label} ({baseline_val:.2f})')

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(values),
                f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_per_holiday_time_combined(
    holiday_agg: list,
    interval_baseline: float, duration_baseline: float,
    baseline_label: str,
    interval_key: str, duration_key: str,
    title: str, filename: str, color: str,
):
    """Combined figure: interval (top) and duration (bottom) bar charts vs baseline.
       合并图：上半部分为平均间隔，下半部分为平均时长。"""
    if not holiday_agg:
        return
    names = [h['name'] for h in holiday_agg]
    interval_vals = [h[interval_key] for h in holiday_agg]
    duration_vals = [h[duration_key] for h in holiday_agg]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(14, len(names) * 0.6), 10),
        sharex=True,
    )

    x = np.arange(len(names))
    width = 0.35

    # ── Top: Interval ──
    bars1 = ax1.bar(x, interval_vals, width, label='Holiday', color=color, alpha=0.85)
    ax1.axhline(y=interval_baseline, color='red', linestyle='--', linewidth=1.8,
                label=f'{baseline_label} ({interval_baseline:.2f}h)')
    for bar, v in zip(bars1, interval_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(interval_vals),
                 f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax1.set_ylabel('Avg Interval (hours)')
    ax1.set_title(f'{title} — Interval')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # ── Bottom: Duration ──
    bars2 = ax2.bar(x, duration_vals, width, label='Holiday', color=color, alpha=0.85)
    ax2.axhline(y=duration_baseline, color='red', linestyle='--', linewidth=1.8,
                label=f'{baseline_label} ({duration_baseline:.2f}h)')
    for bar, v in zip(bars2, duration_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(duration_vals),
                 f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Avg Duration (hours)')
    ax2.set_title(f'{title} — Duration')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


# ── A1: 节假日 VS 非节假日 ────────────────────────────────────────────
#  A1: Holiday vs Non-Holiday Turn Groups

def dim_e1_holiday_vs_nonholiday_turns(rows: list[dict]):
    """Compare turn group distribution: holiday vs non-holiday.
    比较节假日 vs 非节假日的轮次分组分布（含去重/不去重）。"""
    log("=" * 50)
    log("E1: Holiday vs Non-Holiday Turn Groups")

    session_period = _session_period_series(rows)

    holiday_sessions = set(sid for sid, p in session_period.items() if p == 'holiday')
    non_holiday_sessions = set(sid for sid, p in session_period.items() if p != 'holiday')

    h_rows = [r for r in rows if r['session_id'] in holiday_sessions]
    nh_rows = [r for r in rows if r['session_id'] in non_holiday_sessions]

    stats = {
        'Holiday': {
            'no_dedup': _compute_turn_groups(h_rows, dedup=False),
            'dedup': _compute_turn_groups(h_rows, dedup=True),
        },
        'Non-holiday': {
            'no_dedup': _compute_turn_groups(nh_rows, dedup=False),
            'dedup': _compute_turn_groups(nh_rows, dedup=True),
        },
    }

    total_h = sum(stats['Holiday']['no_dedup'].values())
    total_nh = sum(stats['Non-holiday']['no_dedup'].values())
    log(f"  Holiday: {total_h} sessions")
    log(f"  Non-holiday: {total_nh} sessions")

    _plot_turn_group_comparison(
        stats,
        'Turn Group Distribution: Holiday vs Non-Holiday',
        'e1_holiday_vs_nonholiday_turns.png',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'e1_holiday_vs_nonholiday_turns.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'bucket', 'no_dedup_count', 'dedup_count'])
        for label, data in stats.items():
            for b in TURN_GROUPS:
                w.writerow([label, b, data['no_dedup'].get(b, 0),
                            data['dedup'].get(b, 0)])
    log(f"Saved: {csv_path}")


# ── A2: 节假日 VS 工作日 VS 周末 ──────────────────────────────────────
#  A2: Holiday vs Workday vs Weekend Turn Groups

def dim_e2_holiday_workday_weekend_turns(rows: list[dict]):
    """Compare turn group distribution: holiday vs workday vs weekend.
    比较节假日 vs 工作日 vs 周末的轮次分组分布。"""
    log("=" * 50)
    log("E2: Holiday vs Workday vs Weekend Turn Groups")

    session_period = _session_period_series(rows)
    period_sessions = {p: set(sid for sid, pp in session_period.items() if pp == p)
                       for p in ['holiday', 'workday', 'weekend']}

    stats = {}
    for p in ['holiday', 'workday', 'weekend']:
        p_rows = [r for r in rows if r['session_id'] in period_sessions[p]]
        label = p.capitalize()
        stats[label] = {
            'no_dedup': _compute_turn_groups(p_rows, dedup=False),
            'dedup': _compute_turn_groups(p_rows, dedup=True),
        }
        total = sum(stats[label]['no_dedup'].values())
        log(f"  {p}: {total} sessions")

    _plot_turn_group_comparison(
        stats,
        'Turn Group Distribution: Holiday vs Workday vs Weekend',
        'e2_holiday_workday_weekend_turns.png',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'e2_holiday_workday_weekend_turns.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'bucket', 'no_dedup_count', 'dedup_count'])
        for p in ['holiday', 'workday', 'weekend']:
            label = p.capitalize()
            for b in TURN_GROUPS:
                w.writerow([p, b, stats[label]['no_dedup'].get(b, 0),
                            stats[label]['dedup'].get(b, 0)])
    log(f"Saved: {csv_path}")


# ── A3: 各个节假日 VS 非节假日 ────────────────────────────────────────
#  A3: Per-Holiday vs Non-Holiday Turn Groups

def _holiday_turn_groups(rows: list[dict], dedup: bool = False) -> dict[str, dict[str, int]]:
    """Compute per-holiday turn group distributions.
    计算每个节假日的轮次分组分布。"""
    session_period = _session_period_series(rows)

    holiday_name_sessions = defaultdict(set)
    for r in rows:
        if r['is_seeker'] and r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_name_sessions[name].add(r['session_id'])

    result = {}
    for name, sids in holiday_name_sessions.items():
        if len(sids) < MIN_DATA_ROWS // 5:
            continue
        p_rows = [r for r in rows if r['session_id'] in sids]
        result[name] = _compute_turn_groups(p_rows, dedup=dedup)
    return result


def dim_e3_per_holiday_vs_nonholiday_turns(rows: list[dict]):
    """Per-holiday turn groups vs non-holiday baseline (heatmap).
    各节假日轮次分组 vs 非节假日基线（热力图）。"""
    log("=" * 50)
    log("E3: Per-Holiday Turn Groups vs Non-Holiday")

    session_period = _session_period_series(rows)
    nh_sessions = set(sid for sid, p in session_period.items() if p != 'holiday')
    nh_rows = [r for r in rows if r['session_id'] in nh_sessions]
    nh_no_dedup = _compute_turn_groups(nh_rows, dedup=False)
    nh_dedup = _compute_turn_groups(nh_rows, dedup=True)

    holiday_tg = _holiday_turn_groups(rows, dedup=False)
    holiday_tg_dedup = _holiday_turn_groups(rows, dedup=True)

    if not holiday_tg:
        log("  No holiday groups")
        return

    names = sorted(holiday_tg.keys())
    log(f"  Non-holiday baseline (no_dedup): {nh_no_dedup}")
    log(f"  Holidays: {len(names)} groups")

    # Heatmap: rows=holidays, cols=buckets, value = holiday_ratio - nh_ratio
    nh_total = sum(nh_no_dedup.values())
    nh_total_dedup = sum(nh_dedup.values())

    for mode, tg_dict, nh_tg, nh_total_val, suffix in [
        ('No Dedup', holiday_tg, nh_no_dedup, nh_total, 'no_dedup'),
        ('Dedup', holiday_tg_dedup, nh_dedup, nh_total_dedup, 'dedup'),
    ]:
        if not tg_dict:
            continue
        nh_ratios = {b: nh_tg.get(b, 0) / max(nh_total_val, 1) * 100 for b in TURN_GROUPS}
        matrix = np.zeros((len(names), len(TURN_GROUPS)))
        for i, name in enumerate(names):
            total = sum(tg_dict.get(name, {}).values())
            for j, b in enumerate(TURN_GROUPS):
                h_ratio = tg_dict.get(name, {}).get(b, 0) / max(total, 1) * 100
                matrix[i, j] = h_ratio - nh_ratios[b]

        fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.45), max(5, len(names) * 0.4 + 2)))
        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.1)
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(TURN_GROUPS)))
        ax.set_xticklabels(TURN_GROUPS, fontsize=9)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Turn Group')
        ax.set_title(f'Per-Holiday Turn Group Ratio — Diff from Non-Holiday ({suffix})', fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.6, label='Diff in % points')
        fig.tight_layout()
        path = os.path.join(STEP_OUT, f'e3_per_holiday_vs_nonholiday_turns_{suffix}.png')
        fig.savefig(path)
        plt.close(fig)
        log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'e3_per_holiday_vs_nonholiday_turns.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'bucket', 'count_no_dedup', 'count_dedup',
                     'nh_ratio_no_dedup', 'nh_ratio_dedup'])
        for name in names:
            for b in TURN_GROUPS:
                v_nd = holiday_tg.get(name, {}).get(b, 0)
                v_d = holiday_tg_dedup.get(name, {}).get(b, 0)
                w.writerow([name, b, v_nd, v_d,
                            f'{nh_no_dedup.get(b, 0) / max(nh_total, 1) * 100:.2f}%',
                            f'{nh_dedup.get(b, 0) / max(nh_total_dedup, 1) * 100:.2f}%'])
    log(f"Saved: {csv_path}")


# ── A4: 各个节假日 VS 工作日 VS 周末 ──────────────────────────────────
#  A4: Per-Holiday vs Workday & Weekend Turn Groups

def dim_e4_per_holiday_vs_workday_weekend_turns(rows: list[dict]):
    """Per-holiday turn groups vs workday & weekend (heatmap by mode).
    各节假日轮次分组 vs 工作日/周末（热力图）。"""
    log("=" * 50)
    log("E4: Per-Holiday Turn Groups vs Workday & Weekend")

    session_period = _session_period_series(rows)

    # Compute workday/weekend baselines
    baselines = {}
    for p in ['workday', 'weekend']:
        p_sessions = set(sid for sid, pp in session_period.items() if pp == p)
        p_rows = [r for r in rows if r['session_id'] in p_sessions]
        baselines[p] = {
            'no_dedup': _compute_turn_groups(p_rows, dedup=False),
            'dedup': _compute_turn_groups(p_rows, dedup=True),
        }
        log(f"  {p}: {sum(baselines[p]['no_dedup'].values())} sessions")

    holiday_tg = _holiday_turn_groups(rows, dedup=False)
    holiday_tg_dedup = _holiday_turn_groups(rows, dedup=True)

    if not holiday_tg:
        log("  No holiday groups")
        return

    names = sorted(holiday_tg.keys())

    for b in TURN_GROUPS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(names) * 0.35 + 1.5)))

        # no_dedup subplot
        wd_ratio_nd = baselines['workday']['no_dedup'].get(b, 0) / max(sum(baselines['workday']['no_dedup'].values()), 1) * 100
        we_ratio_nd = baselines['weekend']['no_dedup'].get(b, 0) / max(sum(baselines['weekend']['no_dedup'].values()), 1) * 100

        vals_nd_wd = []
        vals_nd_we = []
        for name in names:
            total = sum(holiday_tg.get(name, {}).values())
            h_ratio = holiday_tg.get(name, {}).get(b, 0) / max(total, 1) * 100
            vals_nd_wd.append(h_ratio - wd_ratio_nd)
            vals_nd_we.append(h_ratio - we_ratio_nd)

        x = np.arange(len(names))
        width = 0.35
        ax1.bar(x - width / 2, vals_nd_wd, width, label='vs Workday', color=COLOR_WORKDAY, alpha=0.8)
        ax1.bar(x + width / 2, vals_nd_we, width, label='vs Weekend', color=COLOR_WEEKEND, alpha=0.8)
        ax1.axhline(y=0, color='red', linestyle='-', linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax1.set_ylabel('Diff in % points')
        ax1.set_title(f'No Dedup — Bucket {b}', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(axis='y', alpha=0.3)

        # dedup subplot
        wd_ratio_d = baselines['workday']['dedup'].get(b, 0) / max(sum(baselines['workday']['dedup'].values()), 1) * 100
        we_ratio_d = baselines['weekend']['dedup'].get(b, 0) / max(sum(baselines['weekend']['dedup'].values()), 1) * 100

        vals_d_wd = []
        vals_d_we = []
        for name in names:
            total = sum(holiday_tg_dedup.get(name, {}).values())
            h_ratio = holiday_tg_dedup.get(name, {}).get(b, 0) / max(total, 1) * 100
            vals_d_wd.append(h_ratio - wd_ratio_d)
            vals_d_we.append(h_ratio - we_ratio_d)

        ax2.bar(x - width / 2, vals_d_wd, width, label='vs Workday', color=COLOR_WORKDAY, alpha=0.8)
        ax2.bar(x + width / 2, vals_d_we, width, label='vs Weekend', color=COLOR_WEEKEND, alpha=0.8)
        ax2.axhline(y=0, color='red', linestyle='-', linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax2.set_ylabel('Diff in % points')
        ax2.set_title(f'Dedup — Bucket {b}', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)

        fig.suptitle(f'Per-Holiday Turn Group "{b}" — Ratio Diff from Workday & Weekend', fontsize=12)
        fig.tight_layout()
        path = os.path.join(STEP_OUT, f'e4_per_holiday_vs_workday_weekend_turns_bucket_{b}.png')
        fig.savefig(path)
        plt.close(fig)
        log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'e4_per_holiday_vs_workday_weekend_turns.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'bucket', 'count_no_dedup', 'count_dedup',
                     'workday_ratio_no_dedup', 'weekend_ratio_no_dedup',
                     'workday_ratio_dedup', 'weekend_ratio_dedup'])
        for name in names:
            for b in TURN_GROUPS:
                v_nd = holiday_tg.get(name, {}).get(b, 0)
                v_d = holiday_tg_dedup.get(name, {}).get(b, 0)
                wd_total_nd = max(sum(baselines['workday']['no_dedup'].values()), 1)
                we_total_nd = max(sum(baselines['weekend']['no_dedup'].values()), 1)
                wd_total_d = max(sum(baselines['workday']['dedup'].values()), 1)
                we_total_d = max(sum(baselines['weekend']['dedup'].values()), 1)
                w.writerow([name, b, v_nd, v_d,
                            f'{baselines["workday"]["no_dedup"].get(b, 0) / wd_total_nd * 100:.2f}%',
                            f'{baselines["weekend"]["no_dedup"].get(b, 0) / we_total_nd * 100:.2f}%',
                            f'{baselines["workday"]["dedup"].get(b, 0) / wd_total_d * 100:.2f}%',
                            f'{baselines["weekend"]["dedup"].get(b, 0) / we_total_d * 100:.2f}%'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  B: Multi-turn Session Time Analysis
#  B: 多轮会话时间分析
# ═══════════════════════════════════════════════════════════════════════

def _session_time_metrics(rows: list[dict],
                          allow_periods: set[str] | None = None) -> dict:
    """
    Compute time-related metrics for multi-turn sessions.
    计算多轮会话的时间相关指标。

    同一会话中提问平均间隔时间:
      Within same session, only different user questions count.
      同一会话中，仅统计不同内容的用户提问之间的时间差。
      Sum of time differences (seconds) / number of intervals.
      时间差之和 / 间隔数 = 平均间隔。
      When allow_periods is set, BOTH questions in an interval pair must have
      their 'period' in allow_periods（两提问均需在本周期内）。

    多轮会话平均持续时间:
      First user question time to last user question time.
      从第一个用户提问到最后一个用户提问的时间差。
      Both must be different questions.
      首尾必须是不同的提问内容。

    Args:
        rows: data rows
        allow_periods: if set, only count intervals where both questions'
                       period ∈ allow_periods; duration requires first
                       question's period ∈ allow_periods

    Returns（返回字典）:
        {
            'avg_interval_seconds': float,      # 平均间隔时间（秒）
            'avg_duration_seconds': float,      # 平均持续时间（秒）
            'valid_interval_sessions': int,     # 有效间隔计数
            'valid_duration_sessions': int,     # 有效时长计数
        }
    """
    session_questions = _session_user_question_counts(rows)  # 按会话分组的提问

    total_intervals = 0                        # 总间隔数
    total_interval_time = 0.0                  # 总间隔时间
    total_durations = 0.0                      # 总持续时长
    valid_duration_count = 0                   # 有效持续时长计数
    valid_interval_count = 0                   # 有效间隔计数

    for sid, questions in session_questions.items():
        if len(questions) < 2:                 # 少于2个提问，无法计算间隔/时长
            continue

        # Deduplicate by content（按内容去重）
        seen_texts = {}
        unique_questions = []
        for q in questions:
            text = q.get('proc_text', '') or q.get('raw_text', '')
            text = text.strip()
            if text and text not in seen_texts:  # 跳过重复内容
                seen_texts[text] = True
                unique_questions.append(q)

        if len(unique_questions) < 2:          # 去重后少于2个，跳过
            continue

        # Sort by time（按时间排序）
        unique_questions.sort(key=lambda x: x['utc_time'])

        # Duration: first to last (if questions differ)（首尾时间差）
        # 用户第一次用户提问时间需在本周期内
        first_period = unique_questions[0].get('period', '')
        if allow_periods is not None and first_period not in allow_periods:
            continue
        first_time = unique_questions[0]['utc_time']
        last_time = unique_questions[-1]['utc_time']
        if last_time > first_time:
            total_durations += (last_time - first_time)
            valid_duration_count += 1

        # Interval time: consecutive different questions（相邻提问的时间间隔）
        # 两提问均需在本周期内
        for i in range(1, len(unique_questions)):
            if allow_periods is not None:
                q_period = unique_questions[i].get('period', '')
                prev_period = unique_questions[i - 1].get('period', '')
                if q_period not in allow_periods or prev_period not in allow_periods:
                    continue
            t_diff = unique_questions[i]['utc_time'] - unique_questions[i - 1]['utc_time']
            if t_diff > 0:                     # 仅统计时间差为正的情况
                total_interval_time += t_diff
                valid_interval_count += 1

    avg_interval = total_interval_time / max(valid_interval_count, 1)  # 平均间隔
    avg_duration = total_durations / max(valid_duration_count, 1)      # 平均时长

    return {
        'avg_interval_seconds': avg_interval,           # 平均提问间隔（秒）
        'avg_duration_seconds': avg_duration,           # 平均会话时长（秒）
        'valid_interval_sessions': valid_interval_count,  # 有效间隔计数
        'valid_duration_sessions': valid_duration_count,  # 有效时长计数
    }


def _plot_time_comparison(stats_dict: dict[str, dict], title: str, filename: str):
    """Bar chart comparing time metrics across groups.
       柱状图：比较不同组之间的时间指标。"""
    groups = list(stats_dict.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Average interval（左图：平均间隔，秒转小时）
    intervals = [stats_dict[g]['avg_interval_seconds'] / 3600 for g in groups]
    colors = [COLOR_HOLIDAY, COLOR_NONHOLIDAY][:len(groups)] if len(groups) <= 2 \
        else [COLOR_HOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND]
    ax1.bar(groups, intervals, color=colors, alpha=0.8, width=0.5)
    for i, v in enumerate(intervals):
        ax1.text(i, v + 0.5, f'{v:.1f}h', ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel('Avg Interval (hours)')
    ax1.set_title('Avg Time Between User Questions')
    ax1.grid(axis='y', alpha=0.3)

    # Average duration（右图：平均持续时间，秒转小时）
    durations = [stats_dict[g]['avg_duration_seconds'] / 3600 for g in groups]
    ax2.bar(groups, durations, color=colors, alpha=0.8, width=0.5)
    for i, v in enumerate(durations):
        ax2.text(i, v + 0.5, f'{v:.1f}h', ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('Avg Duration (hours)')
    ax2.set_title('Avg Multi-Turn Session Duration')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def dim_f1_holiday_vs_nonholiday_time(rows: list[dict]):
    """Compare session time metrics: holiday vs non-holiday.
       比较节假日 vs 非节假日的会话时间指标。"""
    log("=" * 50)
    log("F1: Holiday vs Non-Holiday Session Time Metrics")

    session_period = _session_period_series(rows)
    h_sessions = set(sid for sid, p in session_period.items() if p == 'holiday')
    nh_sessions = set(sid for sid, p in session_period.items() if p != 'holiday')

    h_rows = [r for r in rows if r['session_id'] in h_sessions]
    nh_rows = [r for r in rows if r['session_id'] in nh_sessions]

    h_time = _session_time_metrics(h_rows, allow_periods={'holiday'})          # 节假日时间指标
    nh_time = _session_time_metrics(nh_rows, allow_periods={'workday', 'weekend'})  # 非节假日时间指标

    log(f"  Holiday: avg interval {h_time['avg_interval_seconds']:.0f}s, "
        f"avg duration {h_time['avg_duration_seconds']:.0f}s")
    log(f"  Non-holiday: avg interval {nh_time['avg_interval_seconds']:.0f}s, "
        f"avg duration {nh_time['avg_duration_seconds']:.0f}s")

    _plot_time_comparison(
        {'Holiday': h_time, 'Non-holiday': nh_time},
        'Session Time Metrics: Holiday vs Non-Holiday',
        'f1_holiday_vs_nonholiday_time.png',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'f1_holiday_vs_nonholiday_time.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'avg_interval_seconds', 'avg_duration_seconds',
                     'valid_interval_sessions', 'valid_duration_sessions'])
        for label, stats in [('holiday', h_time), ('non_holiday', nh_time)]:
            w.writerow([label, f'{stats["avg_interval_seconds"]:.0f}',
                        f'{stats["avg_duration_seconds"]:.0f}',
                        stats['valid_interval_sessions'],
                        stats['valid_duration_sessions']])
    log(f"Saved: {csv_path}")


def dim_f2_holiday_workday_weekend_time(rows: list[dict]):
    """Compare session time metrics: holiday vs workday vs weekend.
       比较节假日 vs 工作日 vs 周末的会话时间指标。"""
    log("=" * 50)
    log("F2: Holiday vs Workday vs Weekend Session Time")

    session_period = _session_period_series(rows)
    stats = {}
    for p in ['holiday', 'workday', 'weekend']:
        p_sessions = set(sid for sid, pp in session_period.items() if pp == p)
        p_rows = [r for r in rows if r['session_id'] in p_sessions]
        stats[p.capitalize()] = _session_time_metrics(p_rows, allow_periods={p})
        log(f"  {p}: interval {stats[p.capitalize()]['avg_interval_seconds']:.0f}s")

    _plot_time_comparison(
        stats,
        'Session Time Metrics: Holiday vs Workday vs Weekend',
        'f2_holiday_workday_weekend_time.png',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'f2_holiday_workday_weekend_time.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'avg_interval_seconds', 'avg_duration_seconds'])
        for p in ['holiday', 'workday', 'weekend']:
            s = stats[p.capitalize()]
            w.writerow([p, f'{s["avg_interval_seconds"]:.0f}',
                        f'{s["avg_duration_seconds"]:.0f}'])
    log(f"Saved: {csv_path}")


def dim_f3_per_holiday_vs_nonholiday_time(rows: list[dict]):
    """Per-holiday time metrics vs non-holiday baseline.
       各节假日时间指标 vs 非节假日基线。"""
    log("=" * 50)
    log("F3: Per-Holiday Session Time vs Non-Holiday")

    holiday_agg_time = []                          # 各节假日时间指标列表
    session_period = _session_period_series(rows)
    # Group holiday sessions by name（按名称分组节假日会话）
    holiday_name_sessions = defaultdict(set)
    for r in rows:
        if r['is_seeker'] and r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_name_sessions[name].add(r['session_id'])

    # 非节假日基线
    nh_sessions = set(sid for sid, p in session_period.items() if p != 'holiday')
    nh_rows = [r for r in rows if r['session_id'] in nh_sessions]
    nh_time = _session_time_metrics(nh_rows, allow_periods={'workday', 'weekend'})

    for name, sids in holiday_name_sessions.items():
        if len(sids) < MIN_DATA_ROWS // 10:        # 数据太少则跳过
            continue
        p_rows = [r for r in rows if r['session_id'] in sids]
        p_time = _session_time_metrics(p_rows, allow_periods={'holiday'})
        p_time['name'] = name
        p_time['total_sessions'] = len(sids)
        holiday_agg_time.append(p_time)

    holiday_agg_time.sort(key=lambda x: x['total_sessions'], reverse=True)

    # 添加小时单位的字段（用于绘图，原始秒数保留给 CSV）
    for h in holiday_agg_time:
        h['avg_interval_hours'] = h['avg_interval_seconds'] / 3600
        h['avg_duration_hours'] = h['avg_duration_seconds'] / 3600

    nh_interval_hours = nh_time['avg_interval_seconds'] / 3600
    nh_duration_hours = nh_time['avg_duration_seconds'] / 3600

    log(f"  Non-holiday: interval {nh_interval_hours:.1f}h, "
        f"duration {nh_duration_hours:.1f}h")

    # 合并图：上半部分平均间隔，下半部分平均时长
    _plot_per_holiday_time_combined(
        holiday_agg_time,
        nh_interval_hours, nh_duration_hours,
        'Non-holiday avg',
        'avg_interval_hours', 'avg_duration_hours',
        'Per-Holiday Avg Question Interval vs Non-Holiday',
        'f3_per_holiday_vs_nonholiday_time.png',
        COLOR_HOLIDAY,
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'f3_per_holiday_vs_nonholiday_time.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'total_sessions', 'valid_interval_sessions',
                     'valid_duration_sessions', 'avg_interval_seconds',
                     'avg_duration_seconds',
                     'nh_baseline_interval', 'nh_baseline_duration'])
        for h in holiday_agg_time:
            w.writerow([h['name'], h['total_sessions'],
                        h['valid_interval_sessions'], h['valid_duration_sessions'],
                        f'{h["avg_interval_seconds"]:.0f}',
                        f'{h["avg_duration_seconds"]:.0f}',
                        f'{nh_time["avg_interval_seconds"]:.0f}',
                        f'{nh_time["avg_duration_seconds"]:.0f}'])
    log(f"Saved: {csv_path}")


def dim_f4_per_holiday_vs_workday_weekend_time(rows: list[dict]):
    """Per-holiday time metrics vs workday & weekend baselines (heatmap).
    各节假日时间指标 vs 工作日/周末基线（热力图）。"""
    log("=" * 50)
    log("F4: Per-Holiday Session Time vs Workday & Weekend")

    session_period = _session_period_series(rows)

    # Compute workday/weekend baselines
    baselines = {}
    for p in ['workday', 'weekend']:
        p_sessions = set(sid for sid, pp in session_period.items() if pp == p)
        p_rows = [r for r in rows if r['session_id'] in p_sessions]
        baselines[p] = _session_time_metrics(p_rows, allow_periods={p})
        log(f"  {p}: interval {baselines[p]['avg_interval_seconds']:.0f}s, "
            f"duration {baselines[p]['avg_duration_seconds']:.0f}s")

    # Group holiday sessions by name
    holiday_name_sessions = defaultdict(set)
    for r in rows:
        if r['is_seeker'] and r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_name_sessions[name].add(r['session_id'])

    holiday_data = []
    for name, sids in holiday_name_sessions.items():
        if len(sids) < MIN_DATA_ROWS // 10:
            continue
        p_rows = [r for r in rows if r['session_id'] in sids]
        p_time = _session_time_metrics(p_rows, allow_periods={'holiday'})
        p_time['name'] = name
        p_time['total_sessions'] = len(sids)
        holiday_data.append(p_time)

    if not holiday_data:
        log("  No holiday groups")
        return

    holiday_data.sort(key=lambda x: x['total_sessions'], reverse=True)
    names = [h['name'] for h in holiday_data]

    # Interval heatmap (rows=holidays, cols=no special split — just show diff from baselines)
    # Two subplots: interval and duration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(names) * 0.35 + 1.5)))

    # Interval: diff from workday and weekend
    wd_interval = baselines['workday']['avg_interval_seconds'] / 3600  # hours
    we_interval = baselines['weekend']['avg_interval_seconds'] / 3600

    x = np.arange(len(names))
    width = 0.35
    interval_wd = [h['avg_interval_seconds'] / 3600 - wd_interval for h in holiday_data]
    interval_we = [h['avg_interval_seconds'] / 3600 - we_interval for h in holiday_data]
    ax1.bar(x - width / 2, interval_wd, width, label='vs Workday', color=COLOR_WORKDAY, alpha=0.8)
    ax1.bar(x + width / 2, interval_we, width, label='vs Weekend', color=COLOR_WEEKEND, alpha=0.8)
    ax1.axhline(y=0, color='red', linestyle='-', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Diff in hours')
    ax1.set_title('Avg Question Interval Diff from Baseline', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Duration: diff from workday and weekend
    wd_duration = baselines['workday']['avg_duration_seconds'] / 3600
    we_duration = baselines['weekend']['avg_duration_seconds'] / 3600
    dur_wd = [h['avg_duration_seconds'] / 3600 - wd_duration for h in holiday_data]
    dur_we = [h['avg_duration_seconds'] / 3600 - we_duration for h in holiday_data]
    ax2.bar(x - width / 2, dur_wd, width, label='vs Workday', color=COLOR_WORKDAY, alpha=0.8)
    ax2.bar(x + width / 2, dur_we, width, label='vs Weekend', color=COLOR_WEEKEND, alpha=0.8)
    ax2.axhline(y=0, color='red', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Diff in hours')
    ax2.set_title('Avg Session Duration Diff from Baseline', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-Holiday Session Time Metrics — Diff from Workday & Weekend', fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'f4_per_holiday_vs_workday_weekend_time.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'f4_per_holiday_vs_workday_weekend_time.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'total_sessions', 'valid_interval_sessions',
                     'valid_duration_sessions', 'avg_interval_seconds', 'avg_duration_seconds',
                     'workday_interval', 'weekend_interval', 'workday_duration', 'weekend_duration'])
        for h in holiday_data:
            w.writerow([h['name'], h['total_sessions'],
                        h['valid_interval_sessions'], h['valid_duration_sessions'],
                        f'{h["avg_interval_seconds"]:.0f}', f'{h["avg_duration_seconds"]:.0f}',
                        f'{baselines["workday"]["avg_interval_seconds"]:.0f}',
                        f'{baselines["weekend"]["avg_interval_seconds"]:.0f}',
                        f'{baselines["workday"]["avg_duration_seconds"]:.0f}',
                        f'{baselines["weekend"]["avg_duration_seconds"]:.0f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  C: Single-day / Cross-day Session Analysis
#  C: 单日/跨日会话分析
# ═══════════════════════════════════════════════════════════════════════

def _count_days_per_period(rows: list[dict]) -> dict[str, int]:
    """Count unique dates per period (holiday / workday / weekend).
       统计每个 period 的唯一日期数。"""
    period_dates = defaultdict(set)
    for r in rows:
        if r.get('is_seeker'):
            period_dates[r['period']].add(r['date'])
    return {p: len(dates) for p, dates in period_dates.items()}


def _count_days_per_holiday_name(rows: list[dict]) -> dict[str, int]:
    """Count unique dates per holiday name (for per-day averaging).
       统计每个节假日名称的唯一日期数。"""
    holiday_dates = defaultdict(set)
    for r in rows:
        if r.get('is_seeker') and r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_dates[name].add(r['date'])
    return {name: len(dates) for name, dates in holiday_dates.items()}


def _session_day_classification(rows: list[dict]) -> dict[str, str]:
    """
    Classify sessions as 'single_day' or 'cross_day'.
    将会话分类为 'single_day'（单日）或 'cross_day'（跨日）。

    单日会话: All user questions in same day, 2+ turns
             所有用户提问在同一天内，且轮次≥2
    跨日会话: First and last user question on different days
             首个和最后一个用户提问在不同日期

    Returns dict[session_id] -> 'single_day' | 'cross_day' | 'single_question'
    """
    session_questions = _session_user_question_counts(rows)  # 按会话分组的提问
    classification = {}

    for sid, questions in session_questions.items():
        if len(questions) < 2:                      # 仅1个提问，标记为单问题会话
            classification[sid] = 'single_question'
            continue

        dates = set(q['date'] for q in questions)   # 收集该会话所有提问的日期
        if len(dates) == 1:                         # 只有一个日期 => 单日
            classification[sid] = 'single_day'
        else:                                       # 多个日期 => 跨日
            classification[sid] = 'cross_day'

    return classification


def _day_session_stats(
    rows: list[dict], session_ids: list[str]) -> dict:
    """Count single-day and cross-day sessions.
       统计单日和跨日会话的数量。"""
    p_rows = [r for r in rows if r['session_id'] in session_ids]  # 筛选数据行
    classification = _session_day_classification(p_rows)          # 分类

    single_day = sum(1 for c in classification.values() if c == 'single_day')   # 单日数
    cross_day = sum(1 for c in classification.values() if c == 'cross_day')     # 跨日数
    total = len(classification)                                                  # 总会话数

    return {
        'total_sessions': total,
        'single_day': single_day,
        'cross_day': cross_day,
        'single_day_ratio': single_day / max(total, 1) * 100,   # 单日占比
        'cross_day_ratio': cross_day / max(total, 1) * 100,     # 跨日占比
    }


def _plot_day_session_comparison(
    stats_dict: dict[str, dict], title: str, filename: str,
    ylabel: str = 'Count',
):
    """Bar chart comparing single/cross-day stats.
       柱状图：比较单日/跨日会话统计。
    Args:
        ylabel: y-axis label (default 'Count', use 'Avg per day' for per-day averages)
    """
    groups = list(stats_dict.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [COLOR_HOLIDAY, COLOR_NONHOLIDAY][:len(groups)] if len(groups) <= 2 \
        else [COLOR_HOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND]

    # Single-day count（左图：单日会话数）
    sd_vals = [stats_dict[g]['single_day'] for g in groups]
    ax1.bar(groups, sd_vals, color=colors, alpha=0.8, width=0.5)
    for i, v in enumerate(sd_vals):
        ax1.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel(ylabel)
    ax1.set_title('Single-Day Sessions (2+ turns)')
    ax1.grid(axis='y', alpha=0.3)

    # Cross-day count（右图：跨日会话数）
    cd_vals = [stats_dict[g]['cross_day'] for g in groups]
    ax2.bar(groups, cd_vals, color=colors, alpha=0.8, width=0.5)
    for i, v in enumerate(cd_vals):
        ax2.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel(ylabel)
    ax2.set_title('Cross-Day Sessions')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def dim_f5_holiday_vs_nonholiday_day_sessions(rows: list[dict]):
    """Compare per-day single/cross-day session averages: holiday vs non-holiday.
        比较节假日 vs 非节假日每天的平均单日/跨日会话数。"""
    log("=" * 50)
    log("F5: Holiday vs Non-Holiday Avg Day Sessions")

    session_period = _session_period_series(rows)
    h_sessions = set(sid for sid, p in session_period.items() if p == 'holiday')
    nh_sessions = set(sid for sid, p in session_period.items() if p != 'holiday')

    h_stats = _day_session_stats(rows, h_sessions)          # 节假日原始统计
    nh_stats = _day_session_stats(rows, nh_sessions)        # 非节假日原始统计

    # Per-day averages（每天平均会话数）
    day_counts = _count_days_per_period(rows)
    h_days = day_counts.get('holiday', 1)
    nh_days = day_counts.get('workday', 0) + day_counts.get('weekend', 1)

    h_single_per_day = h_stats['single_day'] / max(h_days, 1)
    h_cross_per_day = h_stats['cross_day'] / max(h_days, 1)
    nh_single_per_day = nh_stats['single_day'] / max(nh_days, 1)
    nh_cross_per_day = nh_stats['cross_day'] / max(nh_days, 1)

    log(f"  Holiday ({h_days} days): single {h_single_per_day:.2f}/day, "
        f"cross {h_cross_per_day:.2f}/day")
    log(f"  Non-holiday ({nh_days} days): single {nh_single_per_day:.2f}/day, "
        f"cross {nh_cross_per_day:.2f}/day")

    per_day_stats = {
        'Holiday': {'single_day': h_single_per_day, 'cross_day': h_cross_per_day},
        'Non-holiday': {'single_day': nh_single_per_day, 'cross_day': nh_cross_per_day},
    }

    _plot_day_session_comparison(
        per_day_stats,
        'Avg Single-Day & Cross-Day Sessions per Day: Holiday vs Non-Holiday',
        'f5_holiday_vs_nonholiday_day_sessions.png',
        ylabel='Avg per day',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'f5_holiday_vs_nonholiday_day_sessions.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'total_days', 'total_sessions', 'single_day',
                     'cross_day', 'single_per_day', 'cross_per_day',
                     'single_day_ratio', 'cross_day_ratio'])
        for label, stats, days, s_pd, c_pd in [
            ('holiday', h_stats, h_days, h_single_per_day, h_cross_per_day),
            ('non_holiday', nh_stats, nh_days, nh_single_per_day, nh_cross_per_day),
        ]:
            w.writerow([label, days, stats['total_sessions'], stats['single_day'],
                        stats['cross_day'], f'{s_pd:.2f}', f'{c_pd:.2f}',
                        f'{stats["single_day_ratio"]:.1f}%',
                        f'{stats["cross_day_ratio"]:.1f}%'])
    log(f"Saved: {csv_path}")


def dim_f6_holiday_workday_weekend_day_sessions(rows: list[dict]):
    """Compare per-day single/cross-day session averages: holiday vs workday vs weekend.
        比较节假日 vs 工作日 vs 周末每天的平均单日/跨日会话数。"""
    log("=" * 50)
    log("F6: Holiday vs Workday vs Weekend Avg Day Sessions")

    session_period = _session_period_series(rows)
    raw_stats = {}
    for p in ['holiday', 'workday', 'weekend']:
        p_sessions = set(sid for sid, pp in session_period.items() if pp == p)
        raw_stats[p] = _day_session_stats(rows, p_sessions)

    # Per-day averages（每天平均会话数）
    day_counts = _count_days_per_period(rows)
    per_day_stats = {}
    for p in ['holiday', 'workday', 'weekend']:
        days = day_counts.get(p, 1)
        s = raw_stats[p]
        sd_pd = s['single_day'] / max(days, 1)
        cd_pd = s['cross_day'] / max(days, 1)
        per_day_stats[p.capitalize()] = {'single_day': sd_pd, 'cross_day': cd_pd}
        log(f"  {p} ({days} days): single {sd_pd:.2f}/day, cross {cd_pd:.2f}/day")

    _plot_day_session_comparison(
        per_day_stats,
        'Avg Single-Day & Cross-Day Sessions per Day: Holiday vs Workday vs Weekend',
        'f6_holiday_workday_weekend_day_sessions.png',
        ylabel='Avg per day',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'f6_holiday_workday_weekend_day_sessions.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'total_days', 'total_sessions', 'single_day',
                     'cross_day', 'single_per_day', 'cross_per_day',
                     'single_day_ratio', 'cross_day_ratio'])
        for p in ['holiday', 'workday', 'weekend']:
            s = raw_stats[p]
            days = day_counts.get(p, 1)
            sd_pd = s['single_day'] / max(days, 1)
            cd_pd = s['cross_day'] / max(days, 1)
            w.writerow([p, days, s['total_sessions'], s['single_day'],
                        s['cross_day'], f'{sd_pd:.2f}', f'{cd_pd:.2f}',
                        f'{s["single_day_ratio"]:.1f}%',
                        f'{s["cross_day_ratio"]:.1f}%'])
    log(f"Saved: {csv_path}")


def dim_f7_per_holiday_vs_nonholiday_day_sessions(rows: list[dict]):
    """Per-holiday single/cross-day per-day averages vs non-holiday baseline.
        各节假日每天平均单日/跨日会话数 vs 非节假日基线。"""
    log("=" * 50)
    log("F7: Per-Holiday Avg Day Sessions vs Non-Holiday")

    session_period = _session_period_series(rows)
    nh_sessions = set(sid for sid, p in session_period.items() if p != 'holiday')
    nh_stats = _day_session_stats(rows, nh_sessions)

    # 非节假日总天数
    day_counts = _count_days_per_period(rows)
    nh_days = day_counts.get('workday', 0) + day_counts.get('weekend', 1)

    # 每个节假日名称的总天数
    holiday_days = _count_days_per_holiday_name(rows)

    # Group holidays by name
    holiday_name_sessions = defaultdict(set)
    for r in rows:
        if r['is_seeker'] and r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_name_sessions[name].add(r['session_id'])

    holiday_stats = []
    for name, sids in holiday_name_sessions.items():
        if len(sids) < MIN_DATA_ROWS // 10:   # 数据太少则跳过
            continue
        stats = _day_session_stats(rows, sids)
        stats['name'] = name
        stats['total_sessions'] = len(sids)
        # Per-day averages
        h_days = holiday_days.get(name, 1)
        stats['single_per_day'] = stats['single_day'] / max(h_days, 1)
        stats['cross_per_day'] = stats['cross_day'] / max(h_days, 1)
        holiday_stats.append(stats)

    holiday_stats.sort(key=lambda x: x['total_sessions'], reverse=True)

    # 非节假日 per-day 基线
    nh_single_per_day = nh_stats['single_day'] / max(nh_days, 1)
    nh_cross_per_day = nh_stats['cross_day'] / max(nh_days, 1)

    log(f"  Non-holiday ({nh_days} days): single {nh_single_per_day:.2f}/day, "
        f"cross {nh_cross_per_day:.2f}/day")

    names = [h['name'] for h in holiday_stats]
    sd_vals = [h['single_per_day'] for h in holiday_stats]
    cd_vals = [h['cross_per_day'] for h in holiday_stats]

    # 左右两张柱状图（单位统一为 per-day）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(names) * 0.6), 5))
    x = np.arange(len(names))
    width = 0.35

    # 左图：单日会话 per-day
    ax1.bar(x, sd_vals, width, label='Holiday', color=COLOR_HOLIDAY, alpha=0.85)
    ax1.axhline(y=nh_single_per_day,
                color='red', linestyle='--', linewidth=1.5,
                label=f'Non-holiday ({nh_single_per_day:.2f}/day)')
    for i, v in enumerate(sd_vals):
        ax1.text(i, v + 0.02 * max(sd_vals) if max(sd_vals) > 0 else v + 0.01,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Avg per day')
    ax1.set_title('Single-Day Sessions per Day')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # 右图：跨日会话 per-day
    ax2.bar(x, cd_vals, width, label='Holiday', color=COLOR_HOLIDAY, alpha=0.85)
    ax2.axhline(y=nh_cross_per_day,
                color='red', linestyle='--', linewidth=1.5,
                label=f'Non-holiday ({nh_cross_per_day:.2f}/day)')
    for i, v in enumerate(cd_vals):
        ax2.text(i, v + 0.02 * max(cd_vals) if max(cd_vals) > 0 else v + 0.01,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Avg per day')
    ax2.set_title('Cross-Day Sessions per Day')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-Holiday Single-Day & Cross-Day Sessions per Day vs Non-Holiday',
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'f7_per_holiday_vs_nonholiday_day_sessions.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'f7_per_holiday_vs_nonholiday_day_sessions.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'total_sessions', 'total_days',
                     'single_day', 'cross_day', 'single_per_day', 'cross_per_day',
                     'single_day_ratio', 'cross_day_ratio',
                     'nh_single_per_day', 'nh_cross_per_day'])
        for h in holiday_stats:
            h_days = holiday_days.get(h['name'], 1)
            w.writerow([h['name'], h['total_sessions'], h_days,
                        h['single_day'], h['cross_day'],
                        f'{h["single_per_day"]:.4f}', f'{h["cross_per_day"]:.4f}',
                        f'{h["single_day_ratio"]:.1f}%',
                        f'{h["cross_day_ratio"]:.1f}%',
                        f'{nh_single_per_day:.4f}', f'{nh_cross_per_day:.4f}'])
    log(f"Saved: {csv_path}")


def dim_f8_per_holiday_vs_workday_weekend_day_sessions(rows: list[dict]):
    """Per-holiday single/cross-day per-day averages vs workday & weekend baselines.
        各节假日每天平均单日/跨日会话数 vs 工作日/周末基线。"""
    log("=" * 50)
    log("F8: Per-Holiday Avg Day Sessions vs Workday & Weekend")

    session_period = _session_period_series(rows)

    # 工作日/周末基线 per-day
    day_counts = _count_days_per_period(rows)
    baselines = {}
    for p in ['workday', 'weekend']:
        p_sessions = set(sid for sid, pp in session_period.items() if pp == p)
        p_stats = _day_session_stats(rows, p_sessions)
        p_days = day_counts.get(p, 1)
        baselines[p] = {
            'single_per_day': p_stats['single_day'] / max(p_days, 1),
            'cross_per_day': p_stats['cross_day'] / max(p_days, 1),
        }
        log(f"  {p} ({p_days} days): single {baselines[p]['single_per_day']:.2f}/day, "
            f"cross {baselines[p]['cross_per_day']:.2f}/day")

    # 每个节假日名称的总天数
    holiday_days = _count_days_per_holiday_name(rows)

    # Group holidays by name
    holiday_name_sessions = defaultdict(set)
    for r in rows:
        if r['is_seeker'] and r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_name_sessions[name].add(r['session_id'])

    holiday_stats = []
    for name, sids in holiday_name_sessions.items():
        if len(sids) < MIN_DATA_ROWS // 10:
            continue
        stats = _day_session_stats(rows, sids)
        stats['name'] = name
        stats['total_sessions'] = len(sids)
        h_days = holiday_days.get(name, 1)
        stats['single_per_day'] = stats['single_day'] / max(h_days, 1)
        stats['cross_per_day'] = stats['cross_day'] / max(h_days, 1)
        holiday_stats.append(stats)

    if not holiday_stats:
        log("  No holiday groups")
        return

    holiday_stats.sort(key=lambda x: x['total_sessions'], reverse=True)
    names = [h['name'] for h in holiday_stats]

    # 柱状图：各节假日 per-day 均值 vs 工作日/周末基线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(names) * 0.6), 5))
    x = np.arange(len(names))
    width = 0.25

    # 左图：单日会话 per-day
    sd_vals = [h['single_per_day'] for h in holiday_stats]
    ax1.bar(x, sd_vals, width, label='Holiday', color=COLOR_HOLIDAY, alpha=0.85)
    ax1.axhline(y=baselines['workday']['single_per_day'],
                color=COLOR_WORKDAY, linestyle='--', linewidth=1.5,
                label=f'Workday ({baselines["workday"]["single_per_day"]:.2f}/day)')
    ax1.axhline(y=baselines['weekend']['single_per_day'],
                color=COLOR_WEEKEND, linestyle='--', linewidth=1.5,
                label=f'Weekend ({baselines["weekend"]["single_per_day"]:.2f}/day)')
    for i, v in enumerate(sd_vals):
        ax1.text(i, v + 0.02 * max(sd_vals) if max(sd_vals) > 0 else v + 0.01,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Avg per day')
    ax1.set_title('Single-Day Sessions per Day')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # 右图：跨日会话 per-day
    cd_vals = [h['cross_per_day'] for h in holiday_stats]
    ax2.bar(x, cd_vals, width, label='Holiday', color=COLOR_HOLIDAY, alpha=0.85)
    ax2.axhline(y=baselines['workday']['cross_per_day'],
                color=COLOR_WORKDAY, linestyle='--', linewidth=1.5,
                label=f'Workday ({baselines["workday"]["cross_per_day"]:.2f}/day)')
    ax2.axhline(y=baselines['weekend']['cross_per_day'],
                color=COLOR_WEEKEND, linestyle='--', linewidth=1.5,
                label=f'Weekend ({baselines["weekend"]["cross_per_day"]:.2f}/day)')
    for i, v in enumerate(cd_vals):
        ax2.text(i, v + 0.02 * max(cd_vals) if max(cd_vals) > 0 else v + 0.01,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Avg per day')
    ax2.set_title('Cross-Day Sessions per Day')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-Holiday Single-Day & Cross-Day Sessions per Day vs Workday & Weekend',
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'f8_per_holiday_vs_workday_weekend_day_sessions.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'f8_per_holiday_vs_workday_weekend_day_sessions.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'total_sessions', 'total_days',
                     'single_day', 'cross_day', 'single_per_day', 'cross_per_day',
                     'single_day_ratio', 'cross_day_ratio',
                     'workday_single_per_day', 'workday_cross_per_day',
                     'weekend_single_per_day', 'weekend_cross_per_day'])
        for h in holiday_stats:
            h_days = holiday_days.get(h['name'], 1)
            w.writerow([h['name'], h['total_sessions'], h_days,
                        h['single_day'], h['cross_day'],
                        f'{h["single_per_day"]:.4f}', f'{h["cross_per_day"]:.4f}',
                        f'{h["single_day_ratio"]:.1f}%',
                        f'{h["cross_day_ratio"]:.1f}%',
                        f'{baselines["workday"]["single_per_day"]:.4f}',
                        f'{baselines["workday"]["cross_per_day"]:.4f}',
                        f'{baselines["weekend"]["single_per_day"]:.4f}',
                        f'{baselines["weekend"]["cross_per_day"]:.4f}'])


# ═══════════════════════════════════════════════════════════════════════
#  Main（主入口）
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for Step 3: load data, run all conversation analyses.
       步骤3主入口：加载数据，运行所有会话分析维度。"""
    log("=" * 60)
    log("Step 3: Conversation Turn & Time Analysis")
    log("=" * 60)

    from movie.data_loader import load_all
    data = load_all()
    rows = data['rows']                  # 需要所有行（含系统回复）进行会话分析

    # Section e: Session turn counts（会话轮次分析 - 单日多会话平均次数对比）
    log("")
    log("-" * 40)
    log("Section A: Session Turn Analysis")
    log("-" * 40)

    dim_e1_holiday_vs_nonholiday_turns(rows)                     # e1: 节假日vs非节假日
    log("")
    dim_e2_holiday_workday_weekend_turns(rows)                   # e2: 节假日vs工作日vs周末
    log("")
    dim_e3_per_holiday_vs_nonholiday_turns(rows)                 # e3: 各节假日vs非节假日
    log("")
    dim_e4_per_holiday_vs_workday_weekend_turns(rows)            # e4: 各节假日vs工作日/周末

    # Section f: Multi-turn time（多轮会话时间分析）
    log("")
    log("-" * 40)
    log("Section B: Multi-Turn Session Time Analysis")
    log("-" * 40)

    dim_f1_holiday_vs_nonholiday_time(rows)                      # f1: 节假日vs非节假日
    log("")
    dim_f2_holiday_workday_weekend_time(rows)                    # f2: 节假日vs工作日vs周末
    log("")
    dim_f3_per_holiday_vs_nonholiday_time(rows)                  # f3: 各节假日vs非节假日
    log("")
    dim_f4_per_holiday_vs_workday_weekend_time(rows)             # f4: 各节假日vs工作日/周末

    # Section C: Single-day / Cross-day（单日/跨日会话分析 - 跨日会话平均次数对比）
    log("")
    log("-" * 40)
    log("Section C: Single-Day / Cross-Day Session Analysis")
    log("-" * 40)
    # 跨日会话
    dim_f5_holiday_vs_nonholiday_day_sessions(rows)              # C1: 节假日vs非节假日
    log("")
    dim_f6_holiday_workday_weekend_day_sessions(rows)            # C2: 节假日vs工作日vs周末
    log("")
    dim_f7_per_holiday_vs_nonholiday_day_sessions(rows)          # C3: 各节假日vs非节假日
    log("")
    dim_f8_per_holiday_vs_workday_weekend_day_sessions(rows)     # C4: 各节假日vs工作日/周末

    log("")
    log("=" * 60)
    log(f"Step 3 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()  # 独立运行时执行 main
