# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文注释

"""
Step 2: Active Users Analysis
步骤2：活跃用户分析

全部/周周期 + 活跃用户数:
  节假日 VS 非节假日 平均活跃用户数对比
  节假日 VS 工作日 VS 周末 平均活跃用户数对比
  各个节假日 VS 非节假日 平均活跃用户数对比
  各个节假日 VS 工作日 VS 周末 平均活跃用户数对比

日周期-小时段 + 活跃用户数 (0-24h):
  节假日 VS 非节假日 各个时间段平均活跃用户数对比
  节假日 VS 工作日 VS 周末 各个时间段平均活跃用户数对比
  各个节假日 VS 非节假日 各个时间段平均活跃用户数对比
  各个节假日 VS 工作日 VS 周末 各个时段平均活跃用户数对比

活跃用户定义: 各统计维度内所有提问用户(user_id, is_seeker=True)排重后计数
活跃用户：在给定日期/小时内至少提问一次的唯一用户数

Output: output/movie/step2/*.png + CSV
输出目录：output/movie/step2/
"""

import os                          # 操作系统接口，路径和目录操作
import csv                         # CSV 文件读写
from collections import defaultdict, Counter  # 默认字典和计数器

import numpy as np                 # 数值计算
import matplotlib                  # 绘图库
matplotlib.use('Agg')              # 使用 Agg 后端（无 GUI）
import matplotlib.pyplot as plt    # pyplot
import matplotlib.ticker as ticker # 刻度格式化

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log  # 配置
from movie.step1_question_freq import (   # 从步骤1复用颜色常量与辅助函数
    COLOR_HOLIDAY, COLOR_NONHOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND,
    _plot_hourly_comparison,              # 通用逐小时折线图函数
    _aggregate_holiday_names,             # 节假日名称聚合函数
)

setup_matplotlib()                        # 初始化 matplotlib（后端+字体）
STEP_OUT = STEP_DIRS[2]                   # 步骤2输出目录：output/movie/step2/
os.makedirs(STEP_OUT, exist_ok=True)      # 确保输出目录存在


# ═══════════════════════════════════════════════════════════════════════
#  Helper: unique active users per day
#  辅助函数：每日独立活跃用户统计
# ═══════════════════════════════════════════════════════════════════════

def _daily_active_users(seekers: list[dict]) -> dict[str, set]:
    """
    Get unique active user_ids per date.
    获取每个日期的独立活跃用户 ID 集合。
    Args:
        seekers: 提问者数据行列表
    Returns:
        字典 {日期字符串: {用户ID集合}}
    """
    daily_users = defaultdict(set)           # 默认值为 set 的字典，自动去重
    for r in seekers:                        # 遍历每个提问
        daily_users[r['date']].add(r['user_id'])  # 将该用户加入对应日期的集合
    return dict(daily_users)


def _avg_active_users(seekers: list[dict], date_set: set) -> float:
    """
    Compute average number of unique active users per day for dates in date_set.
    计算指定日期集合中的平均每日独立活跃用户数。
    Args:
        seekers:  提问者数据行列表
        date_set: 目标日期集合
    Returns:
        日均活跃用户数，无数据时返回 0.0
    """
    daily_users = defaultdict(set)           # 每日用户集合
    for r in seekers:
        if r['date'] in date_set:            # 仅处理目标日期内的提问
            daily_users[r['date']].add(r['user_id'])  # 添加用户到该日集合

    if not daily_users:                      # 没有数据
        return 0.0
    counts = [len(u) for u in daily_users.values()]  # 每天的用户数列表
    return np.mean(counts)                   # 返回平均值


def _hourly_active_users(seekers: list[dict], date_set: set) -> list[float]:
    """
    Compute average unique active users per hour (0-23) for dates in date_set.
    计算指定日期集合中每个小时的平均独立活跃用户数。
    Args:
        seekers:  提问者数据行列表
        date_set: 目标日期集合
    Returns:
        长度为24的浮点数列表，表示每个小时的平均活跃用户数
    """
    if not date_set:                         # 空集合返回全0
        return [0.0] * 24

    # For each date-hour, track unique users（按日期-小时跟踪独立用户）
    date_hour_users = defaultdict(lambda: defaultdict(set))  # 日期->小时->用户集合
    for r in seekers:
        if r['date'] in date_set:
            date_hour_users[r['date']][r['hour']].add(r['user_id'])

    hourly_totals = [0.0] * 24               # 24小时的累计用户数
    num_dates = len(date_set)
    if num_dates == 0:
        return hourly_totals

    for date_key in date_set:                # 遍历每个日期
        dh = date_hour_users.get(date_key, {})  # 该日期的小时数据
        for h in range(24):
            hourly_totals[h] += len(dh.get(h, set()))  # 累加该小时独立用户数

    return [t / num_dates for t in hourly_totals]  # 除以天数得均值


def _avg_active_users_per_holiday_name(
    seekers: list[dict]) -> list[dict]:
    """
    Group holidays by name (first 6 chars), compute avg daily active users.
    按节假日名称（前6个字符）分组，计算日均活跃用户数。

    Returns list sorted by avg descending, each item:
      返回按日均活跃用户数降序排列的列表，每项包含：
      { 'name', 'avg_daily_users', 'num_dates', 'total_users', 'dates' }
                名称        日均活跃用户    天数     总独立用户    日期集合
    """
    # 默认字典，键为节假日名称，值为包含 date 集合和 daily_users 的字典
    holiday_groups = defaultdict(lambda: {
        'name': '', 'dates': set(), 'daily_users': defaultdict(set),
    })
    for r in seekers:                        # 遍历每个提问
        if r['period'] == 'holiday':         # 仅处理节假日
            name = r['holiday_name'][:6]     # 取前6个字符作为组名
            g = holiday_groups[name]
            g['name'] = name
            g['dates'].add(r['date'])        # 记录日期
            g['daily_users'][r['date']].add(r['user_id'])  # 记录该日用户

    result = []
    for name, g in holiday_groups.items():
        if len(g['dates']) < MIN_DATA_ROWS // 5:  # 数据天数太少则跳过
            continue
        user_counts = [len(u) for u in g['daily_users'].values()]  # 每天的用户数列表
        g['avg_daily_users'] = np.mean(user_counts)     # 日均活跃用户数
        g['num_dates'] = len(g['dates'])                # 有效天数
        # 所有日期中出现的独立用户总数（并集）
        g['total_unique_users'] = len(set().union(*g['daily_users'].values()))
        result.append(g)

    result.sort(key=lambda x: x['avg_daily_users'], reverse=True)  # 降序排列
    return result


# ═══════════════════════════════════════════════════════════════════════
#  A: Weekly period - active users
#  A: 周周期 - 活跃用户分析
# ═══════════════════════════════════════════════════════════════════════

def dim_c1_holiday_vs_nonholiday_active(seekers: list[dict]):
    """Bar chart: avg daily active users, holiday vs non-holiday.
       柱状图：节假日 vs 非节假日 日均活跃用户数对比。"""
    log("=" * 50)
    log("C1: Holiday vs Non-Holiday Avg Daily Active Users")

    # 收集日期集合
    holiday_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')

    # 计算两组日均活跃用户数
    h_avg = _avg_active_users(seekers, holiday_dates)          # 节假日
    nh_avg = _avg_active_users(seekers, non_holiday_dates)    # 非节假日

    log(f"  Holiday avg daily active users: {h_avg:.1f}")
    log(f"  Non-holiday avg daily active users: {nh_avg:.1f}")

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ['Holiday', 'Non-holiday']
    values = [h_avg, nh_avg]
    colors = [COLOR_HOLIDAY, COLOR_NONHOLIDAY]

    bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.5)
    for bar, v in zip(bars, values):           # 在柱子顶部标注数值
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{v:.1f}', ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('Avg Daily Active Users')     # y轴标签
    ax.set_title('Avg Daily Active Users: Holiday vs Non-Holiday')
    ax.grid(axis='y', alpha=0.3)                # y轴网格线
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # y轴取整

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'c1_holiday_vs_nonholiday_active.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'c1_holiday_vs_nonholiday_active.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'avg_daily_active_users'])
        w.writerow(['holiday', f'{h_avg:.2f}'])
        w.writerow(['non_holiday', f'{nh_avg:.2f}'])
    log(f"Saved: {csv_path}")


def dim_c2_holiday_workday_weekend_active(seekers: list[dict]):
    """Bar chart: avg daily active users, holiday vs workday vs weekend.
       柱状图：节假日 vs 工作日 vs 周末 日均活跃用户数对比。"""
    log("=" * 50)
    log("C2: Holiday vs Workday vs Weekend Avg Daily Active Users")

    # 按三个时段分别收集日期并计算均值
    period_dates = {}
    avgs = {}
    for p in ['holiday', 'workday', 'weekend']:
        period_dates[p] = set(r['date'] for r in seekers if r['period'] == p)
        avgs[p] = _avg_active_users(seekers, period_dates[p])
        log(f"  {p}: {avgs[p]:.1f}")

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['Holiday', 'Workday', 'Weekend']
    values = [avgs['holiday'], avgs['workday'], avgs['weekend']]
    colors = [COLOR_HOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND]

    bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{v:.1f}', ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('Avg Daily Active Users')
    ax.set_title('Avg Daily Active Users: Holiday vs Workday vs Weekend')
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'c2_holiday_workday_weekend_active.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'c2_holiday_workday_weekend_active.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'avg_daily_active_users'])
        for p in ['holiday', 'workday', 'weekend']:
            w.writerow([p, f'{avgs[p]:.2f}'])
    log(f"Saved: {csv_path}")


def dim_c3_per_holiday_vs_nonholiday_active(seekers: list[dict]):
    """Bar chart: each holiday name avg daily active users vs non-holiday.
       柱状图：各节假日名称日均活跃用户数 vs 非节假日基线。"""
    log("=" * 50)
    log("C3: Per-Holiday Avg Daily Active Users vs Non-Holiday")

    # 聚合各节假日数据
    holiday_agg = _avg_active_users_per_holiday_name(seekers)

    # 计算非节假日基线
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_avg = _avg_active_users(seekers, non_holiday_dates)
    log(f"  Non-holiday baseline: {nh_avg:.1f} active users/day")

    if not holiday_agg:
        log("  WARN: No holiday data")
        return

    names = [h['name'] for h in holiday_agg]            # 节假日名称列表
    avgs = [h['avg_daily_users'] for h in holiday_agg]  # 各节假日日均活跃用户数

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.6), 6))
    x = np.arange(len(names))
    width = 0.35

    bars = ax.bar(x, avgs, width, label='Holiday', color=COLOR_HOLIDAY, alpha=0.85)
    # 非节假日基线（红色虚线）
    ax.axhline(y=nh_avg, color='red', linestyle='--', linewidth=1.8,
               label=f'Non-holiday avg ({nh_avg:.1f})')

    for i, (bar, v, h) in enumerate(zip(bars, avgs, holiday_agg)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{v:.1f}\n(n={h["num_dates"]}d)',
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Avg Daily Active Users')
    ax.set_title('Per-Holiday Avg Daily Active Users vs Non-Holiday Baseline')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'c3_per_holiday_vs_nonholiday_active.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'c3_per_holiday_vs_nonholiday_active.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'avg_daily_active_users', 'num_dates',
                     'total_unique_users', 'non_holiday_baseline'])
        for h in holiday_agg:
            w.writerow([h['name'], f'{h["avg_daily_users"]:.2f}', h['num_dates'],
                        h['total_unique_users'], f'{nh_avg:.2f}'])
    log(f"Saved: {csv_path}")


def dim_c4_per_holiday_vs_workday_weekend_active(seekers: list[dict]):
    """Bar: each holiday name vs workday/weekend baselines for active users.
       柱状图：各节假日 vs 工作日/周末 日均活跃用户数对比。"""
    log("=" * 50)
    log("C4: Per-Holiday Active Users vs Workday & Weekend")

    # 聚合各节假日数据
    holiday_agg = _avg_active_users_per_holiday_name(seekers)

    # 计算工作日和周末基线
    workday_dates = set(r['date'] for r in seekers if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seekers if r['period'] == 'weekend')
    wd_avg = _avg_active_users(seekers, workday_dates)   # 工作日日均活跃用户
    we_avg = _avg_active_users(seekers, weekend_dates)   # 周末日均活跃用户

    log(f"  Workday: {wd_avg:.1f}, Weekend: {we_avg:.1f}")

    if not holiday_agg:
        log("  WARN: No holiday data")
        return

    names = [h['name'] for h in holiday_agg]
    avgs = [h['avg_daily_users'] for h in holiday_agg]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.6), 6))
    x = np.arange(len(names))
    width = 0.25

    ax.bar(x - width, avgs, width, label='Holiday', color=COLOR_HOLIDAY, alpha=0.85)
    ax.axhline(y=wd_avg, color=COLOR_WORKDAY, linestyle='--', linewidth=1.5,
               label=f'Workday avg ({wd_avg:.1f})')     # 工作日基线（黄色虚线）
    ax.axhline(y=we_avg, color=COLOR_WEEKEND, linestyle='--', linewidth=1.5,
               label=f'Weekend avg ({we_avg:.1f})')     # 周末基线（青色虚线）

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Avg Daily Active Users')
    ax.set_title('Per-Holiday Avg Daily Active Users vs Workday & Weekend')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'c4_per_holiday_vs_workday_weekend_active.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'c4_per_holiday_vs_workday_weekend_active.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'avg_daily_active_users', 'num_dates',
                     'total_unique_users', 'workday_baseline', 'weekend_baseline'])
        for h in holiday_agg:
            w.writerow([h['name'], f'{h["avg_daily_users"]:.2f}', h['num_dates'],
                        h['total_unique_users'], f'{wd_avg:.2f}', f'{we_avg:.2f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  B: Hourly active users
#  B: 小时段活跃用户分析
# ═══════════════════════════════════════════════════════════════════════

def dim_d1_hourly_holiday_vs_nonholiday_active(seekers: list[dict]):
    """Line chart: hourly avg active users, holiday vs non-holiday.
       折线图：节假日 vs 非节假日 逐小时平均活跃用户数。"""
    log("=" * 50)
    log("D1: Hourly Active Users - Holiday vs Non-Holiday")

    holiday_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')

    h_hourly = _hourly_active_users(seekers, holiday_dates)          # 节假日逐小时
    nh_hourly = _hourly_active_users(seekers, non_holiday_dates)    # 非节假日逐小时

    # 使用步骤1复用的通用折线图函数
    _plot_hourly_comparison(
        {'Holiday': h_hourly, 'Non-holiday': nh_hourly},
        'Hourly Avg Active Users: Holiday vs Non-Holiday',
        os.path.join(STEP_OUT, 'd1_hourly_holiday_vs_nonholiday_active.png'),
        {'Holiday': COLOR_HOLIDAY, 'Non-holiday': COLOR_NONHOLIDAY},
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'd1_hourly_holiday_vs_nonholiday_active.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['hour', 'holiday_avg', 'non_holiday_avg'])
        for h in range(24):
            w.writerow([h, f'{h_hourly[h]:.4f}', f'{nh_hourly[h]:.4f}'])
    log(f"Saved: {csv_path}")


def dim_d2_hourly_holiday_workday_weekend_active(seekers: list[dict]):
    """Line chart: hourly avg active users, holiday vs workday vs weekend.
       折线图：节假日 vs 工作日 vs 周末 逐小时平均活跃用户数。"""
    log("=" * 50)
    log("D2: Hourly Active Users - Holiday vs Workday vs Weekend")

    period_dates = {}
    hourly_data = {}
    for p in ['holiday', 'workday', 'weekend']:
        period_dates[p] = set(r['date'] for r in seekers if r['period'] == p)
        hourly_data[p.capitalize()] = _hourly_active_users(seekers, period_dates[p])

    _plot_hourly_comparison(
        hourly_data,
        'Hourly Avg Active Users: Holiday vs Workday vs Weekend',
        os.path.join(STEP_OUT, 'd2_hourly_holiday_workday_weekend_active.png'),
        {'Holiday': COLOR_HOLIDAY, 'Workday': COLOR_WORKDAY, 'Weekend': COLOR_WEEKEND},
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'd2_hourly_holiday_workday_weekend_active.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['hour', 'holiday_avg', 'workday_avg', 'weekend_avg'])
        for h in range(24):
            w.writerow([h, f'{hourly_data["Holiday"][h]:.4f}',
                        f'{hourly_data["Workday"][h]:.4f}',
                        f'{hourly_data["Weekend"][h]:.4f}'])
    log(f"Saved: {csv_path}")


def dim_d3_per_holiday_hourly_active_vs_nonholiday(seekers: list[dict]):
    """Heatmap: each holiday name x hour, active users diff from non-holiday.
       热力图：各节假日逐小时活跃用户数与非节假日基线的差值。"""
    log("=" * 50)
    log("D3: Per-Holiday Hourly Active Users Heatmap vs Non-Holiday")

    # 按节假日名称聚合
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_groups[name].append(r)

    # 过滤数据量太少的节假日
    holiday_groups = {k: v for k, v in holiday_groups.items()
                     if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  WARN: No holiday data")
        return

    # 非节假日逐小时基线
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_hourly = _hourly_active_users(seekers, non_holiday_dates)

    group_names = sorted(holiday_groups.keys())
    matrix = np.zeros((len(group_names), 24))  # 差值矩阵

    for i, name in enumerate(group_names):
        group_dates = set(r['date'] for r in holiday_groups[name])
        h_hourly = _hourly_active_users(holiday_groups[name], group_dates)
        for h in range(24):
            matrix[i, h] = h_hourly[h] - nh_hourly[h]  # 节假日 - 非节假日差值

    # 热力图
    fig, ax = plt.subplots(figsize=(16, max(6, len(group_names) * 0.4 + 2)))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=8)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names, fontsize=8)
    ax.set_xlabel('Hour of Day (UTC)')
    ax.set_title('Per-Holiday Hourly Active Users: Difference from Non-Holiday\n'
                 '(Red=more active, Blue=less active)', fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.6, label='Active Users Diff')

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd3_per_holiday_hourly_active_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'd3_per_holiday_hourly_active.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name'] + [str(h) for h in range(24)])
        for i, name in enumerate(group_names):
            w.writerow([name] + [f'{matrix[i, h]:.4f}' for h in range(24)])
    log(f"Saved: {csv_path}")


def dim_d4_per_holiday_hourly_active_vs_workday_weekend(seekers: list[dict]):
    """Dual heatmap: each holiday hourly active users diff from workday & weekend.
       双热力图：各节假日逐小时活跃用户数与工作日/周末基线的差值。"""
    log("=" * 50)
    log("D4: Per-Holiday Hourly Active Users vs Workday & Weekend")

    # 按节假日名称聚合
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_groups[name].append(r)

    holiday_groups = {k: v for k, v in holiday_groups.items()
                     if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  WARN: No holiday data")
        return

    # 工作日和周末逐小时基线
    workday_dates = set(r['date'] for r in seekers if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seekers if r['period'] == 'weekend')
    wd_hourly = _hourly_active_users(seekers, workday_dates)
    we_hourly = _hourly_active_users(seekers, weekend_dates)

    group_names = sorted(holiday_groups.keys())
    matrix_wd = np.zeros((len(group_names), 24))  # 与工作日差值矩阵
    matrix_we = np.zeros((len(group_names), 24))  # 与周末差值矩阵

    for i, name in enumerate(group_names):
        group_dates = set(r['date'] for r in holiday_groups[name])
        h_hourly = _hourly_active_users(holiday_groups[name], group_dates)
        for h in range(24):
            matrix_wd[i, h] = h_hourly[h] - wd_hourly[h]  # 节假日-工作日
            matrix_we[i, h] = h_hourly[h] - we_hourly[h]  # 节假日-周末

    # 上下排列两张热力图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, max(8, len(group_names) * 0.7 + 2)))

    # 上半图：vs 工作日
    vmax1 = max(abs(matrix_wd.min()), abs(matrix_wd.max()), 0.01)
    im1 = ax1.imshow(matrix_wd, cmap='RdBu_r', aspect='auto', vmin=-vmax1, vmax=vmax1)
    ax1.set_xticks(range(24))
    ax1.set_xticklabels(range(24), fontsize=7)
    ax1.set_yticks(range(len(group_names)))
    ax1.set_yticklabels(group_names, fontsize=7)
    ax1.set_title('Diff: Holiday Active Users - Workday Baseline', fontsize=10)
    fig.colorbar(im1, ax=ax1, shrink=0.5, label='Diff')

    # 下半图：vs 周末
    vmax2 = max(abs(matrix_we.min()), abs(matrix_we.max()), 0.01)
    im2 = ax2.imshow(matrix_we, cmap='RdBu_r', aspect='auto', vmin=-vmax2, vmax=vmax2)
    ax2.set_xticks(range(24))
    ax2.set_xticklabels(range(24), fontsize=7)
    ax2.set_yticks(range(len(group_names)))
    ax2.set_yticklabels(group_names, fontsize=7)
    ax2.set_xlabel('Hour of Day (UTC)')
    ax2.set_title('Diff: Holiday Active Users - Weekend Baseline', fontsize=10)
    fig.colorbar(im2, ax=ax2, shrink=0.5, label='Diff')

    fig.suptitle('Per-Holiday Hourly Active Users vs Workday & Weekend Baselines',
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'd4_per_holiday_hourly_active_vs_workday_weekend.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'd4_per_holiday_hourly_active_vs_workday_weekend.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name'] + [str(h) for h in range(24)])
        for i, name in enumerate(group_names):
            diff_wd = [f'{matrix_wd[i, h]:.4f}' for h in range(24)]
            w.writerow([f'{name}_vs_workday'] + diff_wd)
            diff_we = [f'{matrix_we[i, h]:.4f}' for h in range(24)]
            w.writerow([f'{name}_vs_weekend'] + diff_we)
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main（主入口）
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for Step 2: load data, run all active user analyses.
       步骤2主入口：加载数据，运行所有活跃用户分析维度。"""
    log("=" * 60)
    log("Step 2: Active Users Analysis")
    log("=" * 60)

    # Load data（加载数据）
    from movie.data_loader import load_all
    data = load_all()
    seekers = data['seekers']

    # Section A: Weekly period（周周期分析）
    log("")
    log("-" * 40)
    log("Section A: Weekly Period - Active Users")
    log("-" * 40)

    dim_c1_holiday_vs_nonholiday_active(seekers)                       # C1: 节假日vs非节假日
    log("")
    dim_c2_holiday_workday_weekend_active(seekers)                     # C2: 节假日vs工作日vs周末
    log("")
    dim_c3_per_holiday_vs_nonholiday_active(seekers)                   # C3: 各节假日vs非节假日
    log("")
    dim_c4_per_holiday_vs_workday_weekend_active(seekers)              # C4: 各节假日vs工作日/周末

    # Section B: Hourly（小时段分析）
    log("")
    log("-" * 40)
    log("Section B: Hourly - Active Users")
    log("-" * 40)

    dim_d1_hourly_holiday_vs_nonholiday_active(seekers)                # D1: 节假日vs非节假日逐小时
    log("")
    dim_d2_hourly_holiday_workday_weekend_active(seekers)              # D2: 节假日vs工作日vs周末逐小时
    log("")
    dim_d3_per_holiday_hourly_active_vs_nonholiday(seekers)            # D3: 各节假日vs非节假日逐小时差值
    log("")
    dim_d4_per_holiday_hourly_active_vs_workday_weekend(seekers)       # D4: 各节假日vs工作日/周末逐小时差值

    log("")
    log("=" * 60)
    log(f"Step 2 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()  # 独立运行时执行 main
