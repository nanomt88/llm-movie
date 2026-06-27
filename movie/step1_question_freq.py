# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文注释

"""
Step 1: Question Frequency & Hourly Access Analysis
步骤1：提问频率与小时段访问分析

全部/周周期 + 提问次数:
  节假日 VS 非节假日 平均提问次数对比 (pie)
  节假日 VS 工作日 VS 周末 平均提问次数对比 (pie)
  各个节假日 VS 非节假日 平均提问次数对比 (grouped bar)
  各个节假日 VS 工作日 VS 周末 平均提问次数对比 (grouped bar)

日周期-小时段 + 访问次数:
  节假日 VS 非节假日 各个时间段(0-24h)平均提问次数对比
  节假日 VS 工作日 VS 周末 各个时间段平均提问次数对比
  各个节假日 VS 非节假日 各个时间段平均提问次数对比 (heatmap)
  各个节假日 VS 工作日 VS 周末 各个时间段平均提问次数对比 (heatmap)

Output: output/movie/step1/*.png + CSV
输出目录：output/movie/step1/，包含 PNG 图表和 CSV 数据文件
"""

import os                          # 操作系统接口，用于路径拼接和目录创建
import csv                         # CSV 文件读写，用于保存数值结果
from collections import defaultdict, Counter  # 默认字典和计数器
from datetime import datetime, timezone       # 日期时间与时区处理

import numpy as np                 # 数值计算库，用于均值等计算
import matplotlib                  # 绘图库
matplotlib.use('Agg')              # 使用 Agg 后端（无 GUI），适用于服务器环境
import matplotlib.pyplot as plt    # pyplot 模块，用于生成图表
import matplotlib.ticker as ticker # 坐标轴刻度格式化

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log  # 共享配置
from movie.data_loader import (    # 数据加载模块
    load_conversations, load_holiday_definitions, load_holiday_workday_adjustments,
    tag_period,                    # 时段标记函数
)

setup_matplotlib()                 # 初始化 matplotlib（Agg 后端 + 中文字体）
STEP_OUT = STEP_DIRS[1]            # 步骤1的输出目录：output/movie/step1/
os.makedirs(STEP_OUT, exist_ok=True)  # 确保输出目录存在


# ── Chart color scheme（图表配色方案）────────────────────────────────────────
COLOR_HOLIDAY = '#ff6b6b'          # 节假日：红色
COLOR_NONHOLIDAY = '#74b9ff'       # 非节假日：蓝色
COLOR_WORKDAY = '#feca57'          # 工作日：黄色
COLOR_WEEKEND = '#48dbfb'          # 周末：青色
HOLIDAY_CMAP = 'Set2'              # 节假日组柱状图使用的色图


# ═══════════════════════════════════════════════════════════════════════
#  Helper: compute avg questions per day for a group
#  辅助函数：计算指定日期集合的日均提问数
# ═══════════════════════════════════════════════════════════════════════

def _avg_daily_questions(seekers: list[dict], date_set: set) -> float:
    """
    Compute average number of questions per day for dates in date_set.
    计算指定日期集合的平均每日提问数。
    Args:
        seekers:  提问者数据行列表（is_seeker=True 的行）
        date_set: 目标日期集合（如所有节假日日期）
    Returns:
        日均提问数（浮点数），无数据时返回 0.0
    """
    daily_counts = Counter()                                 # 每日提问数计数器
    for r in seekers:                                       # 遍历每个提问
        if r['date'] in date_set:                            # 如果该提问在目标日期内
            daily_counts[r['date']] += 1                     # 对应日期计数+1
    if not daily_counts:                                     # 没有数据
        return 0.0
    return np.mean(list(daily_counts.values()))              # 返回每日计数的平均值


def _grouped_daily_questions(seekers: list[dict]) -> dict:
    """
    Group daily question counts by period.
    按时段（节假日/工作日/周末）分组每天的提问数。

    Returns:
        {
            'holiday': {date: count, ...},     # 节假日每天提问数
            'workday': {date: count, ...},     # 工作日每天提问数
            'weekend': {date: count, ...},     # 周末每天提问数
            'non_holiday': {date: count, ...}, # 非节假日每天提问数（工作日+周末）
        }
    """
    # 初始化四个 defaultdict，键为日期，值为整数（默认0）
    groups = {
        'holiday': defaultdict(int),        # 节假日每日提问数
        'workday': defaultdict(int),        # 工作日每日提问数
        'weekend': defaultdict(int),        # 周末每日提问数
        'non_holiday': defaultdict(int),    # 非节假日每日提问数
    }
    for r in seekers:                      # 遍历每个提问
        d = r['date']                      # 提问日期
        if r['period'] == 'holiday':       # 节假日提问
            groups['holiday'][d] += 1
            groups['non_holiday'][d] += 0  # 确保 non_holiday 中不含此日期
        elif r['period'] == 'workday':     # 工作日提问
            groups['workday'][d] += 1
            groups['non_holiday'][d] += 1
        elif r['period'] == 'weekend':     # 周末提问
            groups['weekend'][d] += 1
            groups['non_holiday'][d] += 1
    return groups


# ═══════════════════════════════════════════════════════════════════════
#  A: 节假日 VS 非节假日 平均提问次数 (Pie)
#  A: Holiday vs Non-Holiday Average Daily Questions
# ═══════════════════════════════════════════════════════════════════════

def dim_a1_holiday_vs_nonholiday_pie(seekers: list[dict]):
    """
    Two independent bar charts: left=holiday vs non-holiday, right=holiday vs workday vs weekend.
    两个独立柱状图：左侧=节假日 vs 非节假日，右侧=节假日 vs 工作日 vs 周末 日均提问数对比。
    Args:
        seekers: 提问者数据行列表
    """
    log("=" * 50)
    log("A1: Holiday vs Non-Holiday vs Workday vs Weekend Avg Daily Questions (Bar)")

    # 收集各组日期集合
    holiday_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    workday_dates = set(r['date'] for r in seekers if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seekers if r['period'] == 'weekend')

    # 计算各组日均提问数
    h_avg = _avg_daily_questions(seekers, holiday_dates)
    nh_avg = _avg_daily_questions(seekers, non_holiday_dates)
    wd_avg = _avg_daily_questions(seekers, workday_dates)
    we_avg = _avg_daily_questions(seekers, weekend_dates)

    log(f"  Holiday avg daily: {h_avg:.1f} (from {len(holiday_dates)} days)")
    log(f"  Non-holiday avg daily: {nh_avg:.1f} (from {len(non_holiday_dates)} days)")
    log(f"  Workday avg daily: {wd_avg:.1f} (from {len(workday_dates)} days)")
    log(f"  Weekend avg daily: {we_avg:.1f} (from {len(weekend_dates)} days)")

    # 两个独立柱状图：左=节假日 vs 非节假日，右=节假日 vs 工作日 vs 周末
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：Holiday vs Non-holiday
    groups_2 = ['Holiday', 'Non-holiday']
    vals_2 = [h_avg, nh_avg]
    colors_2 = [COLOR_HOLIDAY, COLOR_NONHOLIDAY]
    bars1 = ax1.bar(groups_2, vals_2, color=colors_2, alpha=0.8, width=0.5,
                    edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars1, vals_2):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=11)
    ax1.set_ylabel('Avg Daily Questions')
    ax1.set_title('Holiday vs Non-Holiday', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 右图：Holiday vs Workday vs Weekend
    groups_3 = ['Holiday', 'Workday', 'Weekend']
    vals_3 = [h_avg, wd_avg, we_avg]
    colors_3 = [COLOR_HOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND]
    bars2 = ax2.bar(groups_3, vals_3, color=colors_3, alpha=0.8, width=0.5,
                    edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars2, vals_3):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=11)
    ax2.set_ylabel('Avg Daily Questions')
    ax2.set_title('Holiday vs Workday vs Weekend', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle('Average Daily Questions by Period', fontsize=13)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'a1_holiday_vs_nonholiday_bar.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV（保存数值结果）
    csv_path = os.path.join(STEP_OUT, 'a1_holiday_vs_nonholiday.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'avg_daily_questions', 'num_days'])
        w.writerow(['holiday', f'{h_avg:.2f}', len(holiday_dates)])
        w.writerow(['non_holiday', f'{nh_avg:.2f}', len(non_holiday_dates)])
        w.writerow(['workday', f'{wd_avg:.2f}', len(workday_dates)])
        w.writerow(['weekend', f'{we_avg:.2f}', len(weekend_dates)])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  A2: 节假日 VS 工作日 VS 周末 平均提问次数 (Pie)
#  A2: Holiday vs Workday vs Weekend Average Daily Questions
# ═══════════════════════════════════════════════════════════════════════

def dim_a2_holiday_workday_weekend_pie(seekers: list[dict]):
    """
    Merged into A1 bar chart (a1_holiday_vs_nonholiday_bar.png).
    This function is kept for compatibility — data is now included in dim_a1's merged chart.
    Args:
        seekers: 提问者数据行列表
    """
    log("=" * 50)
    log("A2: (merged into A1 bar chart — see a1_holiday_vs_nonholiday_bar.png)")

    # 按三个时段分别收集日期集合
    period_dates = {}
    for p in ['holiday', 'workday', 'weekend']:
        period_dates[p] = set(r['date'] for r in seekers if r['period'] == p)

    avgs = {}
    for p in ['holiday', 'workday', 'weekend']:
        avgs[p] = _avg_daily_questions(seekers, period_dates[p])
        log(f"  {p} avg daily: {avgs[p]:.1f} (from {len(period_dates[p])} days)")

    # CSV only — chart merged into A1
    csv_path = os.path.join(STEP_OUT, 'a2_holiday_workday_weekend.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'avg_daily_questions', 'num_days'])
        for p in ['holiday', 'workday', 'weekend']:
            w.writerow([p, f'{avgs[p]:.2f}', len(period_dates[p])])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  A3: 各个节假日 VS 非节假日 平均提问次数 (Grouped Bar)
#  A3: Per-Holiday vs Non-Holiday Average Daily Questions
# ═══════════════════════════════════════════════════════════════════════

def _aggregate_holiday_names(seekers: list[dict]) -> list[dict]:
    """
    Group holiday data by holiday name (first 6 chars), aggregate across years.
    按节假日名称（前6个字符）分组，跨年聚合，返回按提问总数降序排列的列表。

    Args:
        seekers: 提问者数据行列表
    Returns:
        每个节假日名称的聚合统计列表，按总提问数降序排列
        每个元素包含：名称、总提问数、日期集合、每日计数、日均提问数
    """
    # Per holiday name stats（每个节假日名称的统计）
    holiday_stats = defaultdict(lambda: {
        'name': '',                 # 节假日名称
        'total_questions': 0,       # 总提问数
        'dates': set(),             # 出现该节假日的日期集合
        'daily_counts': defaultdict(int),  # 每天提问数
    })

    for r in seekers:              # 遍历每个提问
        if r['period'] == 'holiday':        # 仅处理节假日提问
            name = r['holiday_name'][:6]    # 取前6个字符作为组名（如 "春节"）
            entry = holiday_stats[name]
            entry['name'] = name
            entry['total_questions'] += 1   # 总提问数+1
            entry['dates'].add(r['date'])   # 记录日期
            entry['daily_counts'][r['date']] += 1  # 每日计数+1

    # Compute avg daily questions per holiday name
    result = []
    for name, data in holiday_stats.items():
        if len(data['dates']) < MIN_DATA_ROWS // 5:  # 如果数据天数太少则跳过
            continue
        daily_vals = list(data['daily_counts'].values())  # 所有每日计数
        data['avg_daily'] = np.mean(daily_vals)           # 日均提问数
        data['num_dates'] = len(data['dates'])            # 有效天数
        result.append(data)

    result.sort(key=lambda x: x['total_questions'], reverse=True)  # 按总提问数降序
    log(f"Aggregated {len(result)} unique holiday names (first 6 chars)", "A3")
    return result


def dim_a3_per_holiday_vs_nonholiday(seekers: list[dict]):
    """
    Merged two-panel figure: top = per-holiday vs non-holiday baseline,
    bottom = per-holiday vs workday & weekend baselines.
    合并双面板图：上部分=各节假日 vs 非节假日基线，下部分=各节假日 vs 工作日/周末基线。
    Args:
        seekers: 提问者数据行列表
    """
    log("=" * 50)
    log("A3+A4: Per-Holiday Avg Daily Questions — Merged (top: vs Non-Holiday, bottom: vs Workday/Weekend)")

    holiday_agg = _aggregate_holiday_names(seekers)
    if not holiday_agg:
        log("  WARN: No holiday data")
        return

    # 计算非节假日基线
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_avg = _avg_daily_questions(seekers, non_holiday_dates)

    # 计算工作日和周末基线
    workday_dates = set(r['date'] for r in seekers if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seekers if r['period'] == 'weekend')
    wd_avg = _avg_daily_questions(seekers, workday_dates)
    we_avg = _avg_daily_questions(seekers, weekend_dates)

    log(f"  Non-holiday baseline: {nh_avg:.1f}, Workday: {wd_avg:.1f}, Weekend: {we_avg:.1f}")

    names = [h['name'] for h in holiday_agg]
    h_avgs = [h['avg_daily'] for h in holiday_agg]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(names) * 0.6), 10))
    x = np.arange(len(names))
    width = 0.35

    # Top subplot: per-holiday vs non-holiday
    bars1 = ax1.bar(x, h_avgs, width, label='Holiday', color=COLOR_HOLIDAY, alpha=0.85)
    ax1.axhline(y=nh_avg, color='red', linestyle='--', linewidth=1.8,
                label=f'Non-holiday avg ({nh_avg:.1f})')
    for i, (bar, v, d) in enumerate(zip(bars1, h_avgs, holiday_agg)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1f}\n(n={d["num_dates"]}d)', ha='center', va='bottom', fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Avg Daily Questions')
    ax1.set_title('Per-Holiday vs Non-Holiday Baseline', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Bottom subplot: per-holiday vs workday & weekend
    ax2.bar(x, h_avgs, width, label='Holiday', color=COLOR_HOLIDAY, alpha=0.85)
    ax2.axhline(y=wd_avg, color=COLOR_WORKDAY, linestyle='--', linewidth=1.5,
                label=f'Workday avg ({wd_avg:.1f})')
    ax2.axhline(y=we_avg, color=COLOR_WEEKEND, linestyle='--', linewidth=1.5,
                label=f'Weekend avg ({we_avg:.1f})')
    for i, bar in enumerate(bars1):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{h_avgs[i]:.1f}', ha='center', va='bottom', fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Avg Daily Questions')
    ax2.set_title('Per-Holiday vs Workday & Weekend Baselines', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-Holiday Avg Daily Questions', fontsize=14)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'a3_a4_per_holiday_merged.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV — A3 data
    csv_a3 = os.path.join(STEP_OUT, 'a3_per_holiday_vs_nonholiday.csv')
    with open(csv_a3, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'avg_daily_questions', 'num_dates',
                     'total_questions', 'non_holiday_baseline'])
        for h in holiday_agg:
            w.writerow([h['name'], f'{h["avg_daily"]:.2f}', h['num_dates'],
                        h['total_questions'], f'{nh_avg:.2f}'])
    log(f"Saved: {csv_a3}")

    # CSV — A4 data
    csv_a4 = os.path.join(STEP_OUT, 'a4_per_holiday_vs_workday_weekend.csv')
    with open(csv_a4, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'avg_daily_questions', 'num_dates',
                     'total_questions', 'workday_baseline', 'weekend_baseline'])
        for h in holiday_agg:
            w.writerow([h['name'], f'{h["avg_daily"]:.2f}', h['num_dates'],
                        h['total_questions'], f'{wd_avg:.2f}', f'{we_avg:.2f}'])
    log(f"Saved: {csv_a4}")


# ═══════════════════════════════════════════════════════════════════════
#  A4: 各个节假日 VS 工作日 VS 周末 平均提问次数 (Grouped Bar)
#  A4: Per-Holiday vs Workday vs Weekend Average Daily Questions
# ═══════════════════════════════════════════════════════════════════════

def dim_a4_per_holiday_vs_workday_weekend(seekers: list[dict]):
    """
    Merged into A3 chart (a3_a4_per_holiday_merged.png — bottom subplot).
    CSV output still produced here independently.
    Args:
        seekers: 提问者数据行列表
    """
    log("=" * 50)
    log("A4: (merged into A3 chart bottom subplot — see a3_a4_per_holiday_merged.png)")

    holiday_agg = _aggregate_holiday_names(seekers)

    workday_dates = set(r['date'] for r in seekers if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seekers if r['period'] == 'weekend')
    wd_avg = _avg_daily_questions(seekers, workday_dates)
    we_avg = _avg_daily_questions(seekers, weekend_dates)

    log(f"  Workday baseline: {wd_avg:.1f}, Weekend baseline: {we_avg:.1f}")

    # CSV only — chart merged into A3
    csv_path = os.path.join(STEP_OUT, 'a4_per_holiday_vs_workday_weekend.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'avg_daily_questions', 'num_dates',
                     'total_questions', 'workday_baseline', 'weekend_baseline'])
        for h in holiday_agg:
            w.writerow([h['name'], f'{h["avg_daily"]:.2f}', h['num_dates'],
                        h['total_questions'], f'{wd_avg:.2f}', f'{we_avg:.2f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  B: Hourly Access Frequency
#  B: 小时段访问频率分析
# ═══════════════════════════════════════════════════════════════════════

def _hourly_avg(seekers: list[dict], date_set: set) -> list[float]:
    """
    Compute average questions per hour (0-23) for dates in date_set.
    计算指定日期集合中每个小时的平均提问数。
    Args:
        seekers:  提问者数据行列表
        date_set: 目标日期集合
    Returns:
        长度为24的浮点数列表，表示每个小时的平均提问数
    """
    if not date_set:                        # 空日期集合返回全0
        return [0.0] * 24

    # For each date, count questions per hour
    date_hour_counts = defaultdict(lambda: defaultdict(int))  # 日期 -> 小时 -> 计数
    for r in seekers:
        if r['date'] in date_set:
            date_hour_counts[r['date']][r['hour']] += 1      # 该日期该小时计数+1

    # Average across all dates in set（跨所有日期求平均）
    hourly_totals = [0.0] * 24
    num_dates = len(date_set)
    if num_dates == 0:
        return hourly_totals

    for date_key in date_set:              # 遍历每个日期
        for h in range(24):                # 遍历24小时
            hourly_totals[h] += date_hour_counts[date_key].get(h, 0)  # 累加每小时计数

    return [t / num_dates for t in hourly_totals]  # 除以天数得到均值


def _plot_hourly_comparison(
    hourly_data: dict[str, list[float]],  # 传入数据：标签 -> 24小时数据列表
    title: str,                           # 图表标题
    filename: str,                        # 保存文件名
    colors: dict[str, str],               # 标签 -> 颜色映射
):
    """Generic helper to plot hourly line chart comparisons.
       通用辅助函数：绘制小时段折线对比图。
    Args:
        hourly_data: 标签到24个数据的字典
        title:       图表标题
        filename:    保存的文件名
        colors:      标签到颜色的映射
    """
    fig, ax = plt.subplots(figsize=(12, 5))  # 12x5 英寸的宽图
    hours = list(range(24))                   # x轴：0-23小时

    for label, values in hourly_data.items():  # 为每个标签画一条折线
        ax.plot(hours, values, 'o-', label=label, color=colors.get(label),
                linewidth=2, markersize=4, alpha=0.85)  # 'o-' 表示带圆点的折线

    ax.set_xlabel('Hour of Day (UTC)')        # x轴：一天中的小时（UTC时区）
    ax.set_ylabel('Avg Questions per Hour per Day')  # y轴：每小时每日平均提问数
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(0, 24, 2))            # x轴刻度：每2小时一个
    ax.legend(fontsize=10)                    # 图例
    ax.grid(axis='y', alpha=0.3)              # y轴网格线
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # y轴取整

    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


# ── B1: 节假日 VS 非节假日 小时段 ─────────────────────────────────────
#  B1: Hourly comparison: Holiday vs Non-holiday

def dim_b1_hourly_holiday_vs_nonholiday(seekers: list[dict]):
    """Merged two-panel figure: top = holiday vs non-holiday, bottom = holiday vs workday vs weekend.
    合并双面板折线图：上部分=节假日 vs 非节假日，下部分=节假日 vs 工作日 vs 周末。"""
    log("=" * 50)
    log("B1+B2: Hourly Distribution — Merged (top: Holiday vs Non-Holiday, bottom: Holiday vs Workday vs Weekend)")

    # 收集各组日期集合
    holiday_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    workday_dates = set(r['date'] for r in seekers if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seekers if r['period'] == 'weekend')

    # 计算各组逐小时均值
    h_hourly = _hourly_avg(seekers, holiday_dates)
    nh_hourly = _hourly_avg(seekers, non_holiday_dates)
    wd_hourly = _hourly_avg(seekers, workday_dates)
    we_hourly = _hourly_avg(seekers, weekend_dates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    hours = list(range(24))

    # Top: holiday vs non-holiday
    ax1.plot(hours, h_hourly, 'o-', label='Holiday', color=COLOR_HOLIDAY,
             linewidth=2, markersize=4, alpha=0.85)
    ax1.plot(hours, nh_hourly, 'o-', label='Non-holiday', color=COLOR_NONHOLIDAY,
             linewidth=2, markersize=4, alpha=0.85)
    ax1.set_ylabel('Avg Questions/hr/day')
    ax1.set_title('Hourly Avg Questions: Holiday vs Non-Holiday', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Bottom: holiday vs workday vs weekend
    ax2.plot(hours, h_hourly, 'o-', label='Holiday', color=COLOR_HOLIDAY,
             linewidth=2, markersize=4, alpha=0.85)
    ax2.plot(hours, wd_hourly, 'o-', label='Workday', color=COLOR_WORKDAY,
             linewidth=2, markersize=4, alpha=0.85)
    ax2.plot(hours, we_hourly, 'o-', label='Weekend', color=COLOR_WEEKEND,
             linewidth=2, markersize=4, alpha=0.85)
    ax2.set_xlabel('Hour of Day (UTC)')
    ax2.set_ylabel('Avg Questions/hr/day')
    ax2.set_title('Hourly Avg Questions: Holiday vs Workday vs Weekend', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle('Hourly Question Distribution', fontsize=14)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'b1_b2_hourly_merged.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV — B1
    csv_b1 = os.path.join(STEP_OUT, 'b1_hourly_holiday_vs_nonholiday.csv')
    with open(csv_b1, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['hour', 'holiday_avg', 'non_holiday_avg'])
        for h in range(24):
            w.writerow([h, f'{h_hourly[h]:.4f}', f'{nh_hourly[h]:.4f}'])
    log(f"Saved: {csv_b1}")

    # CSV — B2
    csv_b2 = os.path.join(STEP_OUT, 'b2_hourly_holiday_workday_weekend.csv')
    with open(csv_b2, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['hour', 'holiday_avg', 'workday_avg', 'weekend_avg'])
        for h in range(24):
            w.writerow([h, f'{h_hourly[h]:.4f}', f'{wd_hourly[h]:.4f}', f'{we_hourly[h]:.4f}'])
    log(f"Saved: {csv_b2}")


# ── B2: 节假日 VS 工作日 VS 周末 小时段 ──────────────────────────────
#  B2: Hourly comparison: Holiday vs Workday vs Weekend

def dim_b2_hourly_holiday_workday_weekend(seekers: list[dict]):
    """Merged into B1 chart (b1_b2_hourly_merged.png — bottom subplot).
    CSV output still produced here independently."""
    log("=" * 50)
    log("B2: (merged into B1 chart bottom subplot — see b1_b2_hourly_merged.png)")

    period_dates = {}
    for p in ['holiday', 'workday', 'weekend']:
        period_dates[p] = set(r['date'] for r in seekers if r['period'] == p)

    hourly_data = {}
    for p in ['holiday', 'workday', 'weekend']:
        hourly_data[p.capitalize()] = _hourly_avg(seekers, period_dates[p])

    # CSV only — chart merged into B1
    csv_path = os.path.join(STEP_OUT, 'b2_hourly_holiday_workday_weekend.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['hour', 'holiday_avg', 'workday_avg', 'weekend_avg'])
        for h in range(24):
            w.writerow([h, f'{hourly_data["Holiday"][h]:.4f}',
                        f'{hourly_data["Workday"][h]:.4f}',
                        f'{hourly_data["Weekend"][h]:.4f}'])
    log(f"Saved: {csv_path}")


# ── B3: 各个节假日 VS 非节假日 小时段 (Heatmap) ──────────────────────
#  B3: Per-Holiday hourly vs Non-Holiday (Heatmap with difference)

def dim_b3_per_holiday_hourly_vs_nonholiday(seekers: list[dict]):
    """
    Heatmap: each holiday name (row) x hour (col) showing the difference
    between holiday hourly avg and non-holiday hourly baseline.
    热力图：每个节假日名称在各小时与非节假日基线的差值。
    行=节假日名称，列=小时(0-23)，颜色=差值。
    Args:
        seekers: 提问者数据行列表
    """
    log("=" * 50)
    log("B3: Per-Holiday Hourly Distribution vs Non-Holiday (Heatmap)")

    # Aggregate holidays by name（按节假日名称聚合）
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['period'] == 'holiday':
            name = r['holiday_name'][:6]   # 取前6个字符作为组名
            holiday_groups[name].append(r)  # 添加到对应组

    # Filter: enough data（过滤掉数据量太少的节假日）
    holiday_groups = {k: v for k, v in holiday_groups.items()
                     if len(v) >= MIN_DATA_ROWS}  # 提问数 ≥ MIN_DATA_ROWS
    if not holiday_groups:
        log("  WARN: No holiday groups with sufficient data")
        return

    # Non-holiday hourly baseline（非节假日逐小时基线）
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_hourly = _hourly_avg(seekers, non_holiday_dates)

    # Per-holiday hourly values（计算每个节假日的逐小时值）
    group_names = sorted(holiday_groups.keys())  # 节假日名称排序
    matrix = np.zeros((len(group_names), 24))    # 矩阵：行=节假日，列=小时

    for i, name in enumerate(group_names):
        group_dates = set(r['date'] for r in holiday_groups[name])  # 该节假日的日期集合
        h_hourly = _hourly_avg(holiday_groups[name], group_dates)   # 该节假日的逐小时均值
        for h in range(24):
            # Difference: holiday avg - non-holiday avg（节假日均值减去非节假日基线）
            matrix[i, h] = h_hourly[h] - nh_hourly[h]

    # Heatmap（绘制热力图）
    fig, ax = plt.subplots(figsize=(16, max(6, len(group_names) * 0.4 + 2)))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)  # 最大绝对值，用于对称色阶
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    # cmap='RdBu_r': 红蓝反向色图（红=正向差，蓝=负向差）

    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=8)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names, fontsize=8)
    ax.set_xlabel('Hour of Day (UTC)')
    ax.set_title('Per-Holiday Hourly Question Frequency: Difference from Non-Holiday Baseline\n'
                 '(Red=more on holiday, Blue=less)', fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.6, label='Avg Questions Diff')  # 颜色条

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'b3_per_holiday_hourly_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'b3_per_holiday_hourly_vs_nonholiday.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name'] + [str(h) for h in range(24)])  # 表头：名称 + 0-23小时
        for i, name in enumerate(group_names):
            w.writerow([name] + [f'{matrix[i, h]:.4f}' for h in range(24)])
    log(f"Saved: {csv_path}")


# ── B4: 各个节假日 VS 工作日 VS 周末 小时段 (Line chart) ────────────
#  B4: Per-Holiday hourly vs Workday & Weekend (Dual heatmap)

def dim_b4_per_holiday_hourly_vs_workday_weekend(seekers: list[dict]):
    """
    For each holiday (name grouped), plot hourly avg questions
    with workday/weekend baselines.
    Since 20+ holidays would make one chart too crowded,
    we save subplots and a summary heatmap.
    双热力图：每个节假日 vs 工作日基线和 vs 周末基线的逐小时差值。
    上半图=节假日-工作日差值，下半图=节假日-周末差值。
    Args:
        seekers: 提问者数据行列表
    """
    log("=" * 50)
    log("B4: Per-Holiday Hourly Distribution vs Workday & Weekend")

    # Aggregate holidays by name
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_groups[name].append(r)

    # 过滤数据量太少的节假日
    holiday_groups = {k: v for k, v in holiday_groups.items()
                     if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  WARN: No holiday groups with sufficient data")
        return

    # Workday and weekend hourly baselines
    workday_dates = set(r['date'] for r in seekers if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seekers if r['period'] == 'weekend')
    wd_hourly = _hourly_avg(seekers, workday_dates)   # 工作日逐小时基线
    we_hourly = _hourly_avg(seekers, weekend_dates)   # 周末逐小时基线

    group_names = sorted(holiday_groups.keys())

    # Heatmap: holiday hourly - workday baseline（节假日-工作日差值矩阵）
    matrix_wd = np.zeros((len(group_names), 24))
    # Heatmap: holiday hourly - weekend baseline（节假日-周末差值矩阵）
    matrix_we = np.zeros((len(group_names), 24))

    for i, name in enumerate(group_names):
        group_dates = set(r['date'] for r in holiday_groups[name])
        h_hourly = _hourly_avg(holiday_groups[name], group_dates)
        for h in range(24):
            matrix_wd[i, h] = h_hourly[h] - wd_hourly[h]  # 与工作日差值
            matrix_we[i, h] = h_hourly[h] - we_hourly[h]  # 与周末差值

    # Save both heatmaps in one figure（上下排列两张热力图）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, max(8, len(group_names) * 0.7 + 2)))

    # 上半图：与工作日差值
    vmax1 = max(abs(matrix_wd.min()), abs(matrix_wd.max()), 0.01)
    im1 = ax1.imshow(matrix_wd, cmap='RdBu_r', aspect='auto', vmin=-vmax1, vmax=vmax1)
    ax1.set_xticks(range(24))
    ax1.set_xticklabels(range(24), fontsize=7)
    ax1.set_yticks(range(len(group_names)))
    ax1.set_yticklabels(group_names, fontsize=7)
    ax1.set_title('Diff: Holiday Avg - Workday Baseline', fontsize=10)
    fig.colorbar(im1, ax=ax1, shrink=0.5, label='Diff')

    # 下半图：与周末差值
    vmax2 = max(abs(matrix_we.min()), abs(matrix_we.max()), 0.01)
    im2 = ax2.imshow(matrix_we, cmap='RdBu_r', aspect='auto', vmin=-vmax2, vmax=vmax2)
    ax2.set_xticks(range(24))
    ax2.set_xticklabels(range(24), fontsize=7)
    ax2.set_yticks(range(len(group_names)))
    ax2.set_yticklabels(group_names, fontsize=7)
    ax2.set_xlabel('Hour of Day (UTC)')
    ax2.set_title('Diff: Holiday Avg - Weekend Baseline', fontsize=10)
    fig.colorbar(im2, ax=ax2, shrink=0.5, label='Diff')

    fig.suptitle('Per-Holiday Hourly Frequency: Difference from Workday & Weekend',
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'b4_per_holiday_hourly_vs_workday_weekend.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'b4_per_holiday_hourly_vs_workday_weekend.csv')
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
    """Main entry point for Step 1: load data, run all analysis dimensions.
       步骤1主入口：加载数据，运行所有分析维度。"""
    log("=" * 60)
    log("Step 1: Question Frequency & Hourly Access Analysis")
    log("=" * 60)

    # Load data（加载数据）
    from movie.data_loader import load_all                    # 导入数据加载函数
    data = load_all()                                          # 加载所有数据
    seekers = data['seekers']                                  # 提取提问者数据

    # ── Section A: Weekly period question frequency ──
    # 周周期：按天统计的提问频率
    log("")
    log("-" * 40)
    log("Section A: Weekly Period - Question Frequency")
    log("-" * 40)

    dim_a1_holiday_vs_nonholiday_pie(seekers)                  # A1: 饼图-节假日vs非节假日
    log("")
    dim_a2_holiday_workday_weekend_pie(seekers)                # A2: 饼图-节假日vs工作日vs周末
    log("")
    dim_a3_per_holiday_vs_nonholiday(seekers)                  # A3: 柱状图-各节假日vs非节假日
    log("")
    dim_a4_per_holiday_vs_workday_weekend(seekers)             # A4: 柱状图-各节假日vs工作日vs周末

    # ── Section B: Hourly access frequency ──
    # 日周期：按小时统计的访问频率
    log("")
    log("-" * 40)
    log("Section B: Hourly Period - Access Frequency")
    log("-" * 40)

    dim_b1_hourly_holiday_vs_nonholiday(seekers)               # B1: 折线图-节假日vs非节假日逐小时
    log("")
    dim_b2_hourly_holiday_workday_weekend(seekers)             # B2: 折线图-节假日vs工作日vs周末逐小时
    log("")
    dim_b3_per_holiday_hourly_vs_nonholiday(seekers)           # B3: 热力图-各节假日vs非节假日逐小时差值
    log("")
    dim_b4_per_holiday_hourly_vs_workday_weekend(seekers)      # B4: 双热力图-各节假日vs工作日/周末逐小时差值

    log("")
    log("=" * 60)
    log(f"Step 1 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()  # 独立运行时执行 main
