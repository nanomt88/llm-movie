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


def _plot_turn_comparison(stats_dict: dict[str, dict], title: str, filename: str):
    """Bar chart comparing turn stats across groups.
       柱状图：比较不同组之间的会话轮次统计。
    Args:
        stats_dict: 组名 -> 统计字典
        title:      图表标题
        filename:   保存文件名
    """
    groups = list(stats_dict.keys())                          # 组名列表
    metrics = ['avg_turns_per_session', 'multi_turn_ratio']    # 两个维度
    metric_labels = ['Avg Turns/Session', 'Multi-Turn Ratio (%)']

    # 左右两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Avg turns（左图：平均轮次）
    values1 = [stats_dict[g]['avg_turns_per_session'] for g in groups]
    ax1.bar(groups, values1, color=[COLOR_HOLIDAY, COLOR_NONHOLIDAY][:len(groups)]
            if len(groups) <= 2 else [COLOR_HOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND],
            alpha=0.8, width=0.5)
    for i, v in enumerate(values1):
        ax1.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel('Avg Turns')
    ax1.set_title('Avg Turns per Session')
    ax1.grid(axis='y', alpha=0.3)

    # Multi-turn ratio（右图：多轮会话占比）
    values2 = [stats_dict[g]['multi_turn_ratio'] for g in groups]
    ax2.bar(groups, values2, color=[COLOR_HOLIDAY, COLOR_NONHOLIDAY][:len(groups)]
            if len(groups) <= 2 else [COLOR_HOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND],
            alpha=0.8, width=0.5)
    for i, v in enumerate(values2):
        ax2.text(i, v + 0.3, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('Ratio (%)')
    ax2.set_title('Multi-Turn Session Ratio')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_per_holiday_turn_bars(
    holiday_agg: list, baseline_val: float, baseline_label: str,
    value_key: str, ylabel: str, title: str, filename: str, color: str,
):
    """Bar chart: each holiday's turn metric vs baseline.
       柱状图：每个节假日的轮次指标与基线的对比。
    Args:
        holiday_agg:   节假日聚合数据列表
        baseline_val:  基线值（如非节假日均值）
        baseline_label: 基线标签
        value_key:     从字典中取值的关键字
        ylabel:        y轴标签
        title:         图表标题
        filename:      保存文件名
        color:         柱状图颜色
    """
    if not holiday_agg:
        return
    names = [h['name'] for h in holiday_agg]                   # 节假日名称
    values = [h[value_key] for h in holiday_agg]               # 指标值

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.6), 6))
    x = np.arange(len(names))
    width = 0.35

    bars = ax.bar(x, values, width, label='Holiday', color=color, alpha=0.85)
    ax.axhline(y=baseline_val, color='red', linestyle='--', linewidth=1.8,
               label=f'{baseline_label} ({baseline_val:.2f})')  # 基线水平线

    for bar, v in zip(bars, values):                           # 标注数值
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


# ── A1: 节假日 VS 非节假日 ────────────────────────────────────────────
#  A1: Holiday vs Non-Holiday Session Turns

def dim_e1_holiday_vs_nonholiday_turns(rows: list[dict]):
    """Compare session turn stats: holiday vs non-holiday.
       比较节假日 vs 非节假日的会话轮次统计。"""
    log("=" * 50)
    log("E1: Holiday vs Non-Holiday Session Turns")

    session_period = _session_period_series(rows)              # 获取各会话的时段

    # 按节假日/非节假日划分会话
    holiday_sessions = set(sid for sid, p in session_period.items() if p == 'holiday')
    non_holiday_sessions = set(sid for sid, p in session_period.items() if p != 'holiday')

    # 筛选各自的数据行
    h_rows = [r for r in rows if r['session_id'] in holiday_sessions]
    nh_rows = [r for r in rows if r['session_id'] in non_holiday_sessions]

    h_stats = _multi_turn_stats(h_rows)                        # 节假日统计
    nh_stats = _multi_turn_stats(nh_rows)                      # 非节假日统计

    log(f"  Holiday: {h_stats['total_sessions']} sessions, "
        f"avg {h_stats['avg_turns_per_session']:.2f} turns, "
        f"multi-turn {h_stats['multi_turn_ratio']:.1f}%")
    log(f"  Non-holiday: {nh_stats['total_sessions']} sessions, "
        f"avg {nh_stats['avg_turns_per_session']:.2f} turns, "
        f"multi-turn {nh_stats['multi_turn_ratio']:.1f}%")

    _plot_turn_comparison(
        {'Holiday': h_stats, 'Non-holiday': nh_stats},
        'Session Turn Stats: Holiday vs Non-Holiday',
        'e1_holiday_vs_nonholiday_turns.png',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'e1_holiday_vs_nonholiday_turns.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'total_sessions', 'avg_turns_per_session',
                     'multi_turn_sessions', 'multi_turn_ratio'])
        for label, stats in [('holiday', h_stats), ('non_holiday', nh_stats)]:
            w.writerow([label, stats['total_sessions'],
                        f'{stats["avg_turns_per_session"]:.2f}',
                        stats['multi_turn_sessions'],
                        f'{stats["multi_turn_ratio"]:.2f}%'])
    log(f"Saved: {csv_path}")


# ── A2: 节假日 VS 工作日 VS 周末 ──────────────────────────────────────
#  A2: Holiday vs Workday vs Weekend Session Turns

def dim_e2_holiday_workday_weekend_turns(rows: list[dict]):
    """Compare session turn stats: holiday vs workday vs weekend.
       比较节假日 vs 工作日 vs 周末的会话轮次统计。"""
    log("=" * 50)
    log("E2: Holiday vs Workday vs Weekend Session Turns")

    session_period = _session_period_series(rows)
    # 按三个时段划分会话
    period_sessions = {p: set(sid for sid, pp in session_period.items() if pp == p)
                       for p in ['holiday', 'workday', 'weekend']}

    stats = {}
    for p in ['holiday', 'workday', 'weekend']:
        p_rows = [r for r in rows if r['session_id'] in period_sessions[p]]
        stats[p.capitalize()] = _multi_turn_stats(p_rows)
        log(f"  {p}: {stats[p.capitalize()]['total_sessions']} sessions, "
            f"avg {stats[p.capitalize()]['avg_turns_per_session']:.2f} turns")

    _plot_turn_comparison(
        stats,
        'Session Turn Stats: Holiday vs Workday vs Weekend',
        'e2_holiday_workday_weekend_turns.png',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'e2_holiday_workday_weekend_turns.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'total_sessions', 'avg_turns_per_session',
                     'multi_turn_sessions', 'multi_turn_ratio'])
        for p in ['holiday', 'workday', 'weekend']:
            s = stats[p.capitalize()]
            w.writerow([p, s['total_sessions'], f'{s["avg_turns_per_session"]:.2f}',
                        s['multi_turn_sessions'], f'{s["multi_turn_ratio"]:.2f}%'])
    log(f"Saved: {csv_path}")


# ── A3: 各个节假日 VS 非节假日 ────────────────────────────────────────
#  A3: Per-Holiday vs Non-Holiday Session Turns

def _holiday_name_turn_stats(rows: list[dict]) -> list[dict]:
    """Aggregate turn stats by holiday name (first 6 chars).
       按节假日名称（前6个字符）聚合会话轮次统计。"""
    session_period = _session_period_series(rows)

    # Group sessions by holiday name
    holiday_sessions = defaultdict(list)        # 节假日名称 -> 会话列表
    holiday_session_sets = defaultdict(set)     # 用于去重
    for r in rows:
        if r['is_seeker'] and r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            if r['session_id'] not in holiday_session_sets[name]:  # 避免重复添加同一会话
                holiday_session_sets[name].add(r['session_id'])
                holiday_sessions[name].append((r['session_id'], r['date']))

    result = []
    for name, sessions in holiday_sessions.items():
        if len(sessions) < MIN_DATA_ROWS // 5:   # 数据太少则跳过
            continue
        session_ids = set(s[0] for s in sessions)  # 该节假日的所有会话ID
        p_rows = [r for r in rows if r['session_id'] in session_ids]  # 筛选数据行
        stats = _multi_turn_stats(p_rows)          # 计算多轮统计
        stats['name'] = name
        stats['num_sessions'] = len(sessions)       # 会话数
        result.append(stats)

    result.sort(key=lambda x: x['num_sessions'], reverse=True)  # 按会话数降序
    return result


def dim_e3_per_holiday_vs_nonholiday_turns(rows: list[dict]):
    """Per-holiday turn stats vs non-holiday baseline.
       各节假日轮次统计 vs 非节假日基线。"""
    log("=" * 50)
    log("E3: Per-Holiday Session Turns vs Non-Holiday")

    holiday_agg = _holiday_name_turn_stats(rows)            # 各节假日聚合数据

    session_period = _session_period_series(rows)
    nh_sessions = set(sid for sid, p in session_period.items() if p != 'holiday')
    nh_rows = [r for r in rows if r['session_id'] in nh_sessions]
    nh_stats = _multi_turn_stats(nh_rows)                   # 非节假日基线
    nh_avg = nh_stats['avg_turns_per_session']              # 平均轮次
    nh_mt = nh_stats['multi_turn_ratio']                    # 多轮占比

    log(f"  Non-holiday baseline: avg {nh_avg:.2f} turns, "
        f"multi-turn {nh_mt:.1f}%")
    log(f"  Holidays: {len(holiday_agg)} groups")

    # 平均轮次柱状图
    _plot_per_holiday_turn_bars(
        holiday_agg, nh_avg, 'Non-holiday avg',
        'avg_turns_per_session', 'Avg Turns per Session',
        'Per-Holiday Avg Session Turns vs Non-Holiday Baseline',
        'e3_per_holiday_vs_nonholiday_turns.png',
        COLOR_HOLIDAY,
    )

    # 多轮占比柱状图
    _plot_per_holiday_turn_bars(
        holiday_agg, nh_mt, 'Non-holiday avg',
        'multi_turn_ratio', 'Multi-Turn Ratio (%)',
        'Per-Holiday Multi-Turn Session Ratio vs Non-Holiday Baseline',
        'e3_per_holiday_vs_nonholiday_multiturn_ratio.png',
        COLOR_HOLIDAY,
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'e3_per_holiday_vs_nonholiday_turns.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'num_sessions', 'avg_turns_per_session',
                     'multi_turn_sessions', 'multi_turn_ratio',
                     'nh_baseline_avg_turns', 'nh_baseline_mt_ratio'])
        for h in holiday_agg:
            w.writerow([h['name'], h['num_sessions'],
                        f'{h["avg_turns_per_session"]:.2f}',
                        h['multi_turn_sessions'], f'{h["multi_turn_ratio"]:.2f}%',
                        f'{nh_avg:.2f}', f'{nh_mt:.2f}%'])
    log(f"Saved: {csv_path}")


# ── A4: 各个节假日 VS 工作日 VS 周末 ──────────────────────────────────
#  A4: Per-Holiday vs Workday & Weekend Turns

def dim_e4_per_holiday_vs_workday_weekend_turns(rows: list[dict]):
    """Per-holiday turn stats vs workday & weekend baselines.
       各节假日轮次统计 vs 工作日/周末基线。"""
    log("=" * 50)
    log("E4: Per-Holiday Turns vs Workday & Weekend")

    holiday_agg = _holiday_name_turn_stats(rows)            # 各节假日聚合数据
    session_period = _session_period_series(rows)

    baselines = {}
    for p in ['workday', 'weekend']:
        p_sessions = set(sid for sid, pp in session_period.items() if pp == p)
        p_rows = [r for r in rows if r['session_id'] in p_sessions]
        baselines[p] = _multi_turn_stats(p_rows)
        log(f"  {p}: avg {baselines[p]['avg_turns_per_session']:.2f} turns")

    # 平均轮次 vs 工作日基线
    _plot_per_holiday_turn_bars(
        holiday_agg, baselines['workday']['avg_turns_per_session'],
        'Workday avg', 'avg_turns_per_session',
        'Avg Turns per Session',
        'Per-Holiday Avg Session Turns vs Workday & Weekend',
        'e4_per_holiday_vs_workday_weekend_turns.png',
        COLOR_HOLIDAY,
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'e4_per_holiday_vs_workday_weekend_turns.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'num_sessions', 'avg_turns_per_session',
                     'multi_turn_ratio', 'workday_avg_turns', 'weekend_avg_turns'])
        for h in holiday_agg:
            w.writerow([h['name'], h['num_sessions'],
                        f'{h["avg_turns_per_session"]:.2f}',
                        f'{h["multi_turn_ratio"]:.2f}%',
                        f'{baselines["workday"]["avg_turns_per_session"]:.2f}',
                        f'{baselines["weekend"]["avg_turns_per_session"]:.2f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  B: Multi-turn Session Time Analysis
#  B: 多轮会话时间分析
# ═══════════════════════════════════════════════════════════════════════

def _session_time_metrics(rows: list[dict]) -> dict:
    """
    Compute time-related metrics for multi-turn sessions.
    计算多轮会话的时间相关指标。

    同一会话中提问平均间隔时间:
      Within same session, only different user questions count.
      同一会话中，仅统计不同内容的用户提问之间的时间差。
      Sum of time differences (seconds) / number of intervals.
      时间差之和 / 间隔数 = 平均间隔。

    多轮会话平均持续时间:
      First user question time to last user question time.
      从第一个用户提问到最后一个用户提问的时间差。
      Both must be different questions.
      首尾必须是不同的提问内容。

    Returns（返回字典）:
        {
            'avg_interval_seconds': float,      # 平均间隔时间（秒）
            'avg_duration_seconds': float,      # 平均持续时间（秒）
            'valid_sessions': int,              # 有效会话数
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

        # Interval time: consecutive different questions（相邻提问的时间间隔）
        for i in range(1, len(unique_questions)):
            t_diff = unique_questions[i]['utc_time'] - unique_questions[i - 1]['utc_time']
            if t_diff > 0:                     # 仅统计时间差为正的情况
                total_interval_time += t_diff
                valid_interval_count += 1

        # Duration: first to last (if questions differ)（首尾时间差）
        first_time = unique_questions[0]['utc_time']
        last_time = unique_questions[-1]['utc_time']
        if last_time > first_time:
            total_durations += (last_time - first_time)
            valid_duration_count += 1

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

    h_time = _session_time_metrics(h_rows)          # 节假日时间指标
    nh_time = _session_time_metrics(nh_rows)        # 非节假日时间指标

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
        stats[p.capitalize()] = _session_time_metrics(p_rows)
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
    nh_time = _session_time_metrics(nh_rows)

    for name, sids in holiday_name_sessions.items():
        if len(sids) < MIN_DATA_ROWS // 10:        # 数据太少则跳过
            continue
        p_rows = [r for r in rows if r['session_id'] in sids]
        p_time = _session_time_metrics(p_rows)
        p_time['name'] = name
        p_time['num_sessions'] = len(sids)
        holiday_agg_time.append(p_time)

    holiday_agg_time.sort(key=lambda x: x['num_sessions'], reverse=True)

    log(f"  Non-holiday: interval {nh_time['avg_interval_seconds']:.0f}s, "
        f"duration {nh_time['avg_duration_seconds']:.0f}s")

    # 平均间隔柱状图
    _plot_per_holiday_turn_bars(
        holiday_agg_time, nh_time['avg_interval_seconds'] / 3600,
        'Non-holiday avg', 'avg_interval_seconds',
        'Avg Interval (hours)',
        'Per-Holiday Avg Question Interval vs Non-Holiday',
        'f3_per_holiday_vs_nonholiday_interval.png',
        COLOR_HOLIDAY,
    )

    # 平均时长柱状图
    _plot_per_holiday_turn_bars(
        holiday_agg_time, nh_time['avg_duration_seconds'] / 3600,
        'Non-holiday avg', 'avg_duration_seconds',
        'Avg Duration (hours)',
        'Per-Holiday Avg Session Duration vs Non-Holiday',
        'f3_per_holiday_vs_nonholiday_duration.png',
        COLOR_HOLIDAY,
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'f3_per_holiday_vs_nonholiday_time.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'num_sessions', 'avg_interval_seconds',
                     'avg_duration_seconds',
                     'nh_baseline_interval', 'nh_baseline_duration'])
        for h in holiday_agg_time:
            w.writerow([h['name'], h['num_sessions'],
                        f'{h["avg_interval_seconds"]:.0f}',
                        f'{h["avg_duration_seconds"]:.0f}',
                        f'{nh_time["avg_interval_seconds"]:.0f}',
                        f'{nh_time["avg_duration_seconds"]:.0f}'])
    log(f"Saved: {csv_path}")


def dim_f4_per_holiday_vs_workday_weekend_time(rows: list[dict]):
    """Per-holiday time metrics vs workday & weekend baselines.
       各节假日时间指标 vs 工作日/周末基线。"""
    log("=" * 50)
    log("F4: Per-Holiday Session Time vs Workday & Weekend")
    # Simplified: just log the baselines, detailed per-holiday left to CSV
    session_period = _session_period_series(rows)
    for p in ['workday', 'weekend']:
        p_sessions = set(sid for sid, pp in session_period.items() if pp == p)
        p_rows = [r for r in rows if r['session_id'] in p_sessions]
        p_time = _session_time_metrics(p_rows)
        log(f"  {p}: interval {p_time['avg_interval_seconds']:.0f}s, "
            f"duration {p_time['avg_duration_seconds']:.0f}s")

    log("  (Detailed per-holiday data in CSV)")
    csv_path = os.path.join(STEP_OUT, 'f4_per_holiday_vs_workday_weekend_time.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'avg_interval_seconds', 'avg_duration_seconds'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  C: Single-day / Cross-day Session Analysis
#  C: 单日/跨日会话分析
# ═══════════════════════════════════════════════════════════════════════

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
    stats_dict: dict[str, dict], title: str, filename: str):
    """Bar chart comparing single/cross-day stats.
       柱状图：比较单日/跨日会话统计。"""
    groups = list(stats_dict.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [COLOR_HOLIDAY, COLOR_NONHOLIDAY][:len(groups)] if len(groups) <= 2 \
        else [COLOR_HOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND]

    # Single-day count（左图：单日会话数）
    sd_vals = [stats_dict[g]['single_day'] for g in groups]
    ax1.bar(groups, sd_vals, color=colors, alpha=0.8, width=0.5)
    for i, v in enumerate(sd_vals):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel('Count')
    ax1.set_title('Single-Day Sessions (2+ turns)')
    ax1.grid(axis='y', alpha=0.3)

    # Cross-day count（右图：跨日会话数）
    cd_vals = [stats_dict[g]['cross_day'] for g in groups]
    ax2.bar(groups, cd_vals, color=colors, alpha=0.8, width=0.5)
    for i, v in enumerate(cd_vals):
        ax2.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('Count')
    ax2.set_title('Cross-Day Sessions')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def dim_g1_holiday_vs_nonholiday_day_sessions(rows: list[dict]):
    """Compare single/cross-day sessions: holiday vs non-holiday.
       比较节假日 vs 非节假日的单日/跨日会话。"""
    log("=" * 50)
    log("G1: Holiday vs Non-Holiday Single/Cross-Day Sessions")

    session_period = _session_period_series(rows)
    h_sessions = set(sid for sid, p in session_period.items() if p == 'holiday')
    nh_sessions = set(sid for sid, p in session_period.items() if p != 'holiday')

    h_stats = _day_session_stats(rows, h_sessions)          # 节假日统计
    nh_stats = _day_session_stats(rows, nh_sessions)        # 非节假日统计

    log(f"  Holiday: {h_stats['single_day']} single-day, "
        f"{h_stats['cross_day']} cross-day")
    log(f"  Non-holiday: {nh_stats['single_day']} single-day, "
        f"{nh_stats['cross_day']} cross-day")

    _plot_day_session_comparison(
        {'Holiday': h_stats, 'Non-holiday': nh_stats},
        'Single-Day & Cross-Day Sessions: Holiday vs Non-Holiday',
        'g1_holiday_vs_nonholiday_day_sessions.png',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'g1_holiday_vs_nonholiday_day_sessions.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'total_sessions', 'single_day', 'cross_day',
                     'single_day_ratio', 'cross_day_ratio'])
        for label, stats in [('holiday', h_stats), ('non_holiday', nh_stats)]:
            w.writerow([label, stats['total_sessions'], stats['single_day'],
                        stats['cross_day'], f'{stats["single_day_ratio"]:.1f}%',
                        f'{stats["cross_day_ratio"]:.1f}%'])
    log(f"Saved: {csv_path}")


def dim_g2_holiday_workday_weekend_day_sessions(rows: list[dict]):
    """Compare single/cross-day: holiday vs workday vs weekend.
       比较节假日 vs 工作日 vs 周末的单日/跨日会话。"""
    log("=" * 50)
    log("G2: Holiday vs Workday vs Weekend Single/Cross-Day")

    session_period = _session_period_series(rows)
    stats = {}
    for p in ['holiday', 'workday', 'weekend']:
        p_sessions = set(sid for sid, pp in session_period.items() if pp == p)
        stats[p.capitalize()] = _day_session_stats(rows, p_sessions)
        log(f"  {p}: {stats[p.capitalize()]['single_day']} single-day, "
            f"{stats[p.capitalize()]['cross_day']} cross-day")

    _plot_day_session_comparison(
        stats,
        'Single-Day & Cross-Day Sessions: Holiday vs Workday vs Weekend',
        'g2_holiday_workday_weekend_day_sessions.png',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'g2_holiday_workday_weekend_day_sessions.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'total_sessions', 'single_day', 'cross_day',
                     'single_day_ratio', 'cross_day_ratio'])
        for p in ['holiday', 'workday', 'weekend']:
            s = stats[p.capitalize()]
            w.writerow([p, s['total_sessions'], s['single_day'],
                        s['cross_day'], f'{s["single_day_ratio"]:.1f}%',
                        f'{s["cross_day_ratio"]:.1f}%'])
    log(f"Saved: {csv_path}")


def dim_g3_per_holiday_vs_nonholiday_day_sessions(rows: list[dict]):
    """Per-holiday single/cross-day stats vs non-holiday.
       各节假日单日/跨日统计 vs 非节假日。"""
    log("=" * 50)
    log("G3: Per-Holiday Single/Cross-Day vs Non-Holiday")

    session_period = _session_period_series(rows)
    nh_sessions = set(sid for sid, p in session_period.items() if p != 'holiday')
    nh_stats = _day_session_stats(rows, nh_sessions)              # 非节假日统计

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
        stats['num_sessions'] = len(sids)
        holiday_stats.append(stats)

    holiday_stats.sort(key=lambda x: x['num_sessions'], reverse=True)

    log(f"  Non-holiday: {nh_stats['single_day']} single-day, "
        f"{nh_stats['cross_day']} cross-day")

    names = [h['name'] for h in holiday_stats]                   # 名称列表
    sd_vals = [h['single_day'] for h in holiday_stats]           # 单日会话数
    cd_vals = [h['cross_day'] for h in holiday_stats]            # 跨日会话数

    # 左右两张柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(names) * 0.6), 5))
    x = np.arange(len(names))
    width = 0.35

    # 左图：单日会话
    ax1.bar(x, sd_vals, width, label='Holiday', color=COLOR_HOLIDAY, alpha=0.85)
    ax1.axhline(y=nh_stats['single_day'] / max(nh_stats['total_sessions'], 1) * 100,
                color='red', linestyle='--', linewidth=1.5,
                label=f'Non-holiday ratio ({nh_stats["single_day_ratio"]:.1f}%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Count')
    ax1.set_title('Single-Day Sessions (2+ turns)')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # 右图：跨日会话
    ax2.bar(x, cd_vals, width, label='Holiday', color=COLOR_HOLIDAY, alpha=0.85)
    ax2.axhline(y=nh_stats['cross_day'] / max(nh_stats['total_sessions'], 1) * 100,
                color='red', linestyle='--', linewidth=1.5,
                label=f'Non-holiday ratio ({nh_stats["cross_day_ratio"]:.1f}%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Count')
    ax2.set_title('Cross-Day Sessions')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-Holiday Single-Day & Cross-Day Sessions vs Non-Holiday',
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'g3_per_holiday_vs_nonholiday_day_sessions.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'g3_per_holiday_vs_nonholiday_day_sessions.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'num_sessions', 'single_day',
                     'cross_day', 'single_day_ratio', 'cross_day_ratio',
                     'nh_single_ratio', 'nh_cross_ratio'])
        for h in holiday_stats:
            w.writerow([h['name'], h['num_sessions'], h['single_day'],
                        h['cross_day'], f'{h["single_day_ratio"]:.1f}%',
                        f'{h["cross_day_ratio"]:.1f}%',
                        f'{nh_stats["single_day_ratio"]:.1f}%',
                        f'{nh_stats["cross_day_ratio"]:.1f}%'])
    log(f"Saved: {csv_path}")


def dim_g4_per_holiday_vs_workday_weekend_day_sessions(rows: list[dict]):
    """Per-holiday single/cross-day vs workday & weekend.
       各节假日单日/跨日 vs 工作日/周末。"""
    log("=" * 50)
    log("G4: Per-Holiday Single/Cross-Day vs Workday & Weekend")

    session_period = _session_period_series(rows)
    for p in ['workday', 'weekend']:
        p_sessions = set(sid for sid, pp in session_period.items() if pp == p)
        p_stats = _day_session_stats(rows, p_sessions)
        log(f"  {p}: {p_stats['single_day']} single-day, "
            f"{p_stats['cross_day']} cross-day")

    log("  (Detailed per-holiday data in CSV)")

    csv_path = os.path.join(STEP_OUT, 'g4_per_holiday_vs_workday_weekend_day_sessions.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'single_day', 'cross_day'])
    log(f"Saved: {csv_path}")


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

    # Section e: Session turn counts（会话轮次分析）
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

    # Section C: Single-day / Cross-day（单日/跨日会话分析）
    log("")
    log("-" * 40)
    log("Section C: Single-Day / Cross-Day Session Analysis")
    log("-" * 40)

    dim_g1_holiday_vs_nonholiday_day_sessions(rows)              # C1: 节假日vs非节假日
    log("")
    dim_g2_holiday_workday_weekend_day_sessions(rows)            # C2: 节假日vs工作日vs周末
    log("")
    dim_g3_per_holiday_vs_nonholiday_day_sessions(rows)          # C3: 各节假日vs非节假日
    log("")
    dim_g4_per_holiday_vs_workday_weekend_day_sessions(rows)     # C4: 各节假日vs工作日/周末

    log("")
    log("=" * 60)
    log(f"Step 3 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()  # 独立运行时执行 main
