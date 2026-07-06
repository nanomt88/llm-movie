# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文等非 ASCII 字符
"""
Step 4: Age Distribution Analysis
步骤 4：年龄分布分析

全部/周周期 + 年龄:
  - 节假日 VS 非节假日 各年龄段活跃用户数对比
  - 节假日 VS 工作日 VS 周末 各年龄段活跃用户数对比
  - 各个节假日 VS 非节假日 各年龄段活跃用户数对比
  - 各个节假日 VS 工作日 VS 周末 各年龄段活跃用户数对比

日周期-小时段 + 年龄 (0-24h):
  - 节假日 VS 非节假日 各时间段各年龄段活跃用户数对比
  - 节假日 VS 工作日 VS 周末 各时间段各年龄段活跃用户数对比
  - 各个节假日 VS 非节假日 各时间段各年龄段活跃用户数对比
  - 各个节假日 VS 工作日 VS 周末 各时间段各年龄段活跃用户数对比

Output: output/movie/step4/*.png + CSV
输出：output/movie/step4/ 目录下的 PNG 图片和 CSV 文件
"""

import os  # 操作系统接口，用于文件和路径操作
import csv  # CSV 文件读写模块
from collections import defaultdict, Counter  # defaultdict：带默认值的字典；Counter：计数工具

import numpy as np  # 数值计算库，用于数组和矩阵运算
import matplotlib  # 数据可视化基础库
matplotlib.use('Agg')  # 使用非交互式后端 Agg（适用于无 GUI 环境的图片生成）
import matplotlib.pyplot as plt  # pyplot 接口，用于绘制图表
import matplotlib.ticker as ticker  # 坐标轴刻度格式控制

from movie.config import STEP_DIRS, MIN_DATA_ROWS, AGE_SEGMENTS, setup_matplotlib, log  # 导入配置：步骤输出目录、最小数据行数、年龄段定义、matplotlib 初始化、日志函数
from movie.utils.plotting import (annotate_heatmap,
                                   COLOR_HOLIDAY, COLOR_NONHOLIDAY,
                                   COLOR_WORKDAY, COLOR_WEEKEND)  # 统一配色

setup_matplotlib()  # 初始化 matplotlib 样式（字体等）
STEP_OUT = STEP_DIRS[4]  # 步骤 4 的输出目录路径（output/movie/step4/）
os.makedirs(STEP_OUT, exist_ok=True)  # 创建输出目录（如果已存在则不报错）


# Colors for age segments
# 各年龄段的配色方案
AGE_COLORS = {
    '<18': '#7bed9f',     # 18 岁以下：浅绿色
    '18-25': '#70a1ff',   # 18-25 岁：浅蓝色
    '26-35': '#ffa502',   # 26-35 岁：橙色
    '36-50': '#eccc68',   # 36-50 岁：土黄色
    '50+': '#ff6b81',     # 50 岁以上：粉红色
    'unknown': '#ced6e0',  # 年龄未知：浅灰色
}


# ═══════════════════════════════════════════════════════════════════════
#  Helper: age mapping  辅助函数：年龄映射
# ═══════════════════════════════════════════════════════════════════════

def _get_user_age(user_ages: dict, user_id: str) -> str:
    """Get age segment for a user_id, defaulting to 'unknown'.
    获取指定用户的年龄段，如果找不到则返回 'unknown'。"""
    return user_ages.get(user_id, 'unknown')  # 从 user_ages 字典中查找用户年龄，默认返回 'unknown'


def _compute_user_daily_session_counts(
    seekers: list[dict],
) -> dict[str, dict[str, set[str]]]:
    """
    For each user, compute {date: set of session_ids}.
    计算每位用户每天使用的不同会话（session）数量。

    Returns: {user_id: {date: set_of_session_ids}}
    返回值：{用户 ID: {日期: 会话 ID 集合}}
    """
    user_daily: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for r in seekers:
        user_daily[r['user_id']][r['date']].add(r['session_id'])
    return {uid: dict(dates) for uid, dates in user_daily.items()}


def _age_multi_session_avg_daily(
    seekers: list[dict], dates_set: set[str], user_ages: dict,
    min_sessions: int = 2,
) -> dict[str, float]:
    """
    For a set of dates, compute avg daily count of users (by age segment)
    who had >= min_sessions distinct sessions in the same day.

    对于给定的日期集合，计算每天（按年龄段）至少有 min_sessions 个不同会话的用户平均人数。
    Returns: {age_segment: avg_daily_count}
    返回值：{年龄段: 日均用户数}
    """
    user_daily = _compute_user_daily_session_counts(seekers)
    age_daily_counts: dict[str, list[int]] = defaultdict(list)

    for date in sorted(dates_set):
        age_on_date: dict[str, int] = defaultdict(int)
        for uid, dates_dict in user_daily.items():
            if date in dates_dict and len(dates_dict[date]) >= min_sessions:
                seg = _get_user_age(user_ages, uid)
                age_on_date[seg] += 1
        for seg in AGE_SEGMENTS:
            age_daily_counts[seg].append(age_on_date.get(seg, 0))

    result = {}
    for seg in AGE_SEGMENTS:
        vals = age_daily_counts.get(seg, [0])
        result[seg] = sum(vals) / max(len(vals), 1)
    return result


def _age_cross_day_avg_daily(
    seekers: list[dict], dates_set: set[str], user_ages: dict,
    min_days: int = 2,
) -> dict[str, float]:
    """
    For a set of dates, compute avg daily count of users (by age segment)
    who have sessions spanning >= min_days distinct dates.

    对于给定的日期集合，计算每天（按年龄段）在 min_days 天以上有会话的用户平均人数。
    Returns: {age_segment: avg_daily_count}
    返回值：{年龄段: 日均用户数}
    """
    user_daily = _compute_user_daily_session_counts(seekers)

    # Identify cross-day users: appear on >= min_days distinct dates
    cross_day_users: set[str] = set()
    for uid, dates_dict in user_daily.items():
        active_dates = set(dates_dict.keys())
        if len(active_dates) >= min_days:
            cross_day_users.add(uid)

    age_daily_counts: dict[str, list[int]] = defaultdict(list)

    for date in sorted(dates_set):
        age_on_date: dict[str, int] = defaultdict(int)
        for uid in cross_day_users:
            if uid in user_daily and date in user_daily[uid]:
                seg = _get_user_age(user_ages, uid)
                age_on_date[seg] += 1
        for seg in AGE_SEGMENTS:
            age_daily_counts[seg].append(age_on_date.get(seg, 0))

    result = {}
    for seg in AGE_SEGMENTS:
        vals = age_daily_counts.get(seg, [0])
        result[seg] = sum(vals) / max(len(vals), 1)
    return result


def _age_active_users(
    seekers: list[dict], date_set: set, user_ages: dict) -> dict[str, int]:
    """
    Count unique active users by age segment for dates in date_set.
    统计指定日期集合中各年龄段的独立活跃用户数。

    Returns: dict[age_segment] -> count
    返回：字典[年龄段] -> 人数
    """
    age_users = defaultdict(set)  # 每个年龄段对应一个用户 ID 集合（自动去重）
    for r in seekers:  # 遍历所有用户提问记录
        if r['date'] in date_set:  # 如果该记录日期在目标日期集合中
            age = _get_user_age(user_ages, r['user_id'])  # 获取该用户的年龄段
            age_users[age].add(r['user_id'])  # 将该用户 ID 加入对应年龄段的集合

    return {seg: len(age_users.get(seg, set())) for seg in AGE_SEGMENTS}  # 返回每个年龄段的独立用户数


def _age_hourly_active_users(
    seekers: list[dict], date_set: set, user_ages: dict) -> dict[str, list[float]]:
    """
    Compute hourly avg active users by age segment.
    计算每个年龄段的逐小时平均活跃用户数。

    Returns: dict[age_segment] -> list[24 floats] (avg per hour per day)
    返回：字典[年龄段] -> 24 个浮点数（每小时每天平均值）
    """
    if not date_set:  # 如果日期集合为空
        return {seg: [0.0] * 24 for seg in AGE_SEGMENTS}  # 返回每个年龄段的全零列表

    # Per date-hour, per age: set of users
    # 按日期-小时-年龄段存储用户 ID 集合
    dh_age_users = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))  # 三层嵌套字典：日期 -> 小时 -> 年龄段 -> 用户集合
    for r in seekers:  # 遍历所有用户提问记录
        if r['date'] in date_set:  # 如果记录日期在目标日期集合中
            age = _get_user_age(user_ages, r['user_id'])  # 获取该用户的年龄段
            dh_age_users[r['date']][r['hour']][age].add(r['user_id'])  # 按日期-小时-年龄段记录用户

    result = {}  # 存储结果
    num_dates = len(date_set)  # 日期总数，用于计算平均值
    for seg in AGE_SEGMENTS:  # 遍历每个年龄段
        hourly = [0.0] * 24  # 初始化 24 小时计数列表
        for d in date_set:  # 遍历每个日期
            dh = dh_age_users.get(d, {})  # 获取该日期的小时数据
            for h in range(24):  # 遍历 0-23 小时
                hourly[h] += len(dh.get(h, {}).get(seg, set()))  # 累加该小时该年龄段的用户数
        result[seg] = [t / num_dates for t in hourly]  # 除以天数得到每小时平均活跃用户数

    return result  # 返回各年龄段逐小时平均活跃用户数


# ═══════════════════════════════════════════════════════════════════════
#  G: Multi-session analysis  G：多会话分析
# ═══════════════════════════════════════════════════════════════════════


# def dim_g1_multi_session_same_day_age(seekers: list[dict], user_ages: dict):
#     """G1: Avg daily users with >=2 sessions on the same day, by age.
#     G1: 长期活跃用户 各年龄段的单日内多次会话(单日多session)平均人数。"""
#     log("=" * 50)
#     log("G1: Multi-Session Same-Day Users by Age (Avg Daily)")
#
#     holiday_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')
#     non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
#
#     h_age = _age_multi_session_avg_daily(seekers, holiday_dates, user_ages)
#     nh_age = _age_multi_session_avg_daily(seekers, non_holiday_dates, user_ages)
#
#     _plot_age_grouped_bars(
#         {'Holiday': h_age, 'Non-holiday': nh_age},
#         'Avg Daily Multi-Session (≥2) Users by Age: Holiday vs Non-Holiday',
#         'g1_multi_session_same_day_age.png',
#     )
#
#     csv_path = os.path.join(STEP_OUT, 'g1_multi_session_same_day_age.csv')
#     with open(csv_path, 'w', encoding='utf-8', newline='') as f:
#         w = csv.writer(f)
#         w.writerow(['age_segment', 'holiday_avg', 'non_holiday_avg'])
#         for seg in AGE_SEGMENTS:
#             w.writerow([seg, f'{h_age.get(seg, 0):.2f}', f'{nh_age.get(seg, 0):.2f}'])
#     log(f"Saved: {csv_path}")
#
#
# def dim_g2_cross_day_multi_session_age(seekers: list[dict], user_ages: dict):
#     """G2: Avg daily users with sessions across >=2 days, by age.
#     G2: 长期活跃用户 各年龄段的跨天多次会话(跨多天多session)平均人数。"""
#     log("=" * 50)
#     log("G2: Cross-Day Multi-Session Users by Age (Avg Daily)")
#
#     holiday_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')
#     non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
#
#     h_age = _age_cross_day_avg_daily(seekers, holiday_dates, user_ages)
#     nh_age = _age_cross_day_avg_daily(seekers, non_holiday_dates, user_ages)
#
#     _plot_age_grouped_bars(
#         {'Holiday': h_age, 'Non-holiday': nh_age},
#         'Avg Daily Cross-Day (≥2 Days) Users by Age: Holiday vs Non-Holiday',
#         'g2_cross_day_multi_session_age.png',
#     )
#
#     csv_path = os.path.join(STEP_OUT, 'g2_cross_day_multi_session_age.csv')
#     with open(csv_path, 'w', encoding='utf-8', newline='') as f:
#         w = csv.writer(f)
#         w.writerow(['age_segment', 'holiday_avg', 'non_holiday_avg'])
#         for seg in AGE_SEGMENTS:
#             w.writerow([seg, f'{h_age.get(seg, 0):.2f}', f'{nh_age.get(seg, 0):.2f}'])
#     log(f"Saved: {csv_path}")


# def dim_g3_per_holiday_multi_session_age(seekers: list[dict], user_ages: dict):
#     """G3: Per-holiday multi-session same-day avg daily by age.
#     G3: 各节假日年龄段的单日多session日均人数对比。"""
#     log("=" * 50)
#     log("G3: Per-Holiday Multi-Session Same-Day Users (Avg Daily)")
#
#     holiday_groups = defaultdict(list)
#     for r in seekers:
#         if r['period'] == 'holiday':
#             name = r['holiday_name'][:6]
#             holiday_groups[name].append(r)
#     holiday_groups = {k: v for k, v in holiday_groups.items() if len(v) >= MIN_DATA_ROWS}
#     if not holiday_groups:
#         log("  No holiday groups")
#         return
#
#     names = sorted(holiday_groups.keys())
#     n = len(names)
#
#     matrix = np.zeros((len(AGE_SEGMENTS), n))
#     for j, name in enumerate(names):
#         dates = set(r['date'] for r in holiday_groups[name])
#         age_avg = _age_multi_session_avg_daily(holiday_groups[name], dates, user_ages)
#         for i, seg in enumerate(AGE_SEGMENTS):
#             matrix[i, j] = age_avg.get(seg, 0)
#
#     fig, ax = plt.subplots(figsize=(max(10, n * 0.6), max(5, len(AGE_SEGMENTS) * 0.5)))
#     im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
#     ax.set_xticks(range(n))
#     ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
#     ax.set_yticks(range(len(AGE_SEGMENTS)))
#     ax.set_yticklabels(AGE_SEGMENTS, fontsize=8)
#     ax.set_title('Avg Daily Multi-Session (≥2) Users per Holiday by Age', fontsize=11)
#     fig.colorbar(im, ax=ax, shrink=0.6, label='Avg Daily Users')
#     fig.tight_layout()
#     path = os.path.join(STEP_OUT, 'g3_per_holiday_multi_session_age.png')
#     fig.savefig(path)
#     plt.close(fig)
#     log(f"Saved: {path}")
#
#     csv_path = os.path.join(STEP_OUT, 'g3_per_holiday_multi_session_age.csv')
#     with open(csv_path, 'w', encoding='utf-8', newline='') as f:
#         w = csv.writer(f)
#         w.writerow(['age_segment'] + names)
#         for i, seg in enumerate(AGE_SEGMENTS):
#             w.writerow([seg] + [f'{matrix[i, j]:.2f}' for j in range(n)])
#     log(f"Saved: {csv_path}")


# def dim_g4_per_holiday_cross_day_age(seekers: list[dict], user_ages: dict):
#     """G4: Per-holiday cross-day session avg daily by age.
#     G4: 各节假日年龄段跨天多session日均人数对比。"""
#     log("=" * 50)
#     log("G4: Per-Holiday Cross-Day Users (Avg Daily)")
#
#     holiday_groups = defaultdict(list)
#     for r in seekers:
#         if r['period'] == 'holiday':
#             name = r['holiday_name'][:6]
#             holiday_groups[name].append(r)
#     holiday_groups = {k: v for k, v in holiday_groups.items() if len(v) >= MIN_DATA_ROWS}
#     if not holiday_groups:
#         log("  No holiday groups")
#         return
#
#     names = sorted(holiday_groups.keys())
#     n = len(names)
#
#     matrix = np.zeros((len(AGE_SEGMENTS), n))
#     for j, name in enumerate(names):
#         dates = set(r['date'] for r in holiday_groups[name])
#         age_avg = _age_cross_day_avg_daily(holiday_groups[name], dates, user_ages)
#         for i, seg in enumerate(AGE_SEGMENTS):
#             matrix[i, j] = age_avg.get(seg, 0)
#
#     fig, ax = plt.subplots(figsize=(max(10, n * 0.6), max(5, len(AGE_SEGMENTS) * 0.5)))
#     im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
#     ax.set_xticks(range(n))
#     ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
#     ax.set_yticks(range(len(AGE_SEGMENTS)))
#     ax.set_yticklabels(AGE_SEGMENTS, fontsize=8)
#     ax.set_title('Avg Daily Cross-Day (≥2 Days) Users per Holiday by Age', fontsize=11)
#     fig.colorbar(im, ax=ax, shrink=0.6, label='Avg Daily Users')
#     fig.tight_layout()
#     path = os.path.join(STEP_OUT, 'g4_per_holiday_cross_day_age.png')
#     fig.savefig(path)
#     plt.close(fig)
#     log(f"Saved: {path}")
#
#     csv_path = os.path.join(STEP_OUT, 'g4_per_holiday_cross_day_age.csv')
#     with open(csv_path, 'w', encoding='utf-8', newline='') as f:
#         w = csv.writer(f)
#         w.writerow(['age_segment'] + names)
#         for i, seg in enumerate(AGE_SEGMENTS):
#             w.writerow([seg] + [f'{matrix[i, j]:.2f}' for j in range(n)])
#     log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  A: Weekly period - age distribution  A：周周期 - 年龄分布
# ═══════════════════════════════════════════════════════════════════════

def _plot_age_grouped_bars(
    stats: dict[str, dict[str, float]], title: str, filename: str):
    """Grouped bar chart: age segments across groups.
    分组柱状图：显示不同分组中各年龄段的日均活跃用户数对比。"""
    groups = list(stats.keys())  # 获取分组名称列表（如 ['Holiday', 'Non-holiday']）
    age_segments = [s for s in AGE_SEGMENTS if s != 'unknown']  # 排除 'unknown' 年龄段

    fig, ax = plt.subplots(figsize=(10, 6))  # 创建画布和坐标轴，尺寸 10×6 英寸
    x = np.arange(len(age_segments))  # x 轴位置：每个年龄段一个位置
    width = 0.8 / max(len(groups), 1)  # 每个分组的柱宽，确保所有分组柱状图能并排显示

    for i, group in enumerate(groups):  # 遍历每个分组
        vals = [stats[group].get(seg, 0) for seg in age_segments]  # 获取该分组在各年龄段的值
        offset = (i - (len(groups) - 1) / 2) * width  # 计算该分组柱子的偏移量以实现并排
        bars = ax.bar(x + offset, vals, width, label=group,  # 绘制柱状图
                      color=[COLOR_HOLIDAY, COLOR_NONHOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND][i],  # 按顺序分配颜色
                      alpha=0.8)  # 透明度 0.8

    ax.set_xticks(x)  # 设置 x 轴刻度位置
    ax.set_xticklabels(age_segments, fontsize=10)  # 设置 x 轴标签（年龄段名称），字号 10
    ax.set_ylabel('Avg Daily Active Users')  # y 轴标签：日均活跃用户数
    ax.set_title(title, fontsize=12)  # 图表标题，字号 12
    ax.legend(fontsize=9)  # 图例，字号 9
    ax.grid(axis='y', alpha=0.3)  # 显示 y 方向网格线，透明度 0.3

    fig.tight_layout()  # 自动调整布局，避免元素重叠
    path = os.path.join(STEP_OUT, filename)  # 拼接输出文件路径
    fig.savefig(path)  # 保存图片到文件
    plt.close(fig)  # 关闭图形以释放内存
    log(f"Saved: {path}")  # 日志记录保存信息


def _plot_age_grouped_bars_two_row(
    stats_top: dict[str, dict[str, float]],
    stats_bottom: dict[str, dict[str, float]],
    title_top: str, title_bottom: str,
    top_colors: list[str], bottom_colors: list[str],
    title: str, filename: str):
    """Two-row grouped bar chart: top and bottom subplots.
    两行分组柱状图：上半部分和下半部分各一组并排柱。"""
    age_segments = [s for s in AGE_SEGMENTS if s != 'unknown']
    x = np.arange(len(age_segments))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # Top subplot
    groups_top = list(stats_top.keys())
    width = 0.8 / max(len(groups_top), 1)
    for i, group in enumerate(groups_top):
        vals = [stats_top[group].get(seg, 0) for seg in age_segments]
        offset = (i - (len(groups_top) - 1) / 2) * width
        ax1.bar(x + offset, vals, width, label=group,
                color=top_colors[i], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(age_segments, fontsize=10)
    ax1.set_ylabel('Avg Daily Active Users')
    ax1.set_title(title_top, fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Bottom subplot
    groups_bottom = list(stats_bottom.keys())
    width = 0.8 / max(len(groups_bottom), 1)
    for i, group in enumerate(groups_bottom):
        vals = [stats_bottom[group].get(seg, 0) for seg in age_segments]
        offset = (i - (len(groups_bottom) - 1) / 2) * width
        ax2.bar(x + offset, vals, width, label=group,
                color=bottom_colors[i], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(age_segments, fontsize=10)
    ax2.set_ylabel('Avg Daily Active Users')
    ax2.set_title(title_bottom, fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  H1: 节假日 VS 非节假日 年龄段活跃用户 (Bar)
#  H1: Holiday vs Non-Holiday Age Group Active Users
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】双面板柱状图: 左=holiday vs non-holiday, 右=holiday vs workday vs weekend
#   每面板各年龄段分组: 10s, 20s, 30s, 40s, 50s, 60s+
# 【统计口径】
#   user_ages: {user_id: age_group} 映射
#   _age_daily_active(seekers, date_set, user_ages) → {age_group: avg_daily_users}
#   遍历各组日期 → 逐天统计各 age_group 的活跃用户数 → 平均
# 【输出文件】PNG: h1_h2_age_merged.png, CSV: h1_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_h1_holiday_vs_nonholiday_age(seekers: list[dict], user_ages: dict):
    """Compare avg daily active users by age: holiday vs non-holiday.
    对比节假日 vs 非节假日的各年龄段日均活跃用户数。"""
    log("=" * 50)  # 日志：分隔线
    log("H1: Holiday vs Non-Holiday Avg Daily Active Users by Age")  # 日志：分析标题

    holiday_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')  # 获取所有节假日日期集合
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')  # 获取所有非节假日日期集合
    num_h = max(len(holiday_dates), 1)
    num_nh = max(len(non_holiday_dates), 1)

    h_age = _age_active_users(seekers, holiday_dates, user_ages)  # 计算节假日各年龄段活跃用户数
    nh_age = _age_active_users(seekers, non_holiday_dates, user_ages)  # 计算非节假日各年龄段活跃用户数

    # 转换为日均值
    h_age_per_day = {seg: count / num_h for seg, count in h_age.items()}
    nh_age_per_day = {seg: count / num_nh for seg, count in nh_age.items()}

    _plot_age_grouped_bars(  # 绘制分组柱状图
        {'Holiday': h_age_per_day, 'Non-holiday': nh_age_per_day},  # 数据：节假日 vs 非节假日
        'Avg Daily Active Users by Age: Holiday vs Non-Holiday',  # 图表标题
        'h1_holiday_vs_nonholiday_age.png',  # 输出文件名
    )

    csv_path = os.path.join(STEP_OUT, 'h1_holiday_vs_nonholiday_age.csv')  # CSV 输出文件路径
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:  # 以 UTF-8 编码打开 CSV 文件写入
        w = csv.writer(f)  # 创建 CSV 写入器
        w.writerow(['age_segment', 'holiday_avg_daily', 'non_holiday_avg_daily'])  # 写入表头
        for seg in AGE_SEGMENTS:  # 遍历每个年龄段
            w.writerow([seg, f'{h_age_per_day.get(seg, 0):.2f}',
                        f'{nh_age_per_day.get(seg, 0):.2f}'])  # 写入该年龄段的数据行
    log(f"Saved: {csv_path}")  # 日志记录 CSV 保存信息


# ═══════════════════════════════════════════════════════════════════════
#  H2: 节假日 VS 工作日 VS 周末 年龄段活跃用户
#  H2: Holiday vs Workday vs Weekend Age Group
# ═══════════════════════════════════════════════════════════════════════
# 【输出文件】CSV: h2_holiday_workday_weekend_age.csv (图片已合并到 H1 下面板)
# 【统计口径】3组 × 6年龄段
# ═══════════════════════════════════════════════════════════════════════

def dim_h2_holiday_workday_weekend_age(seekers: list[dict], user_ages: dict):
    """Compare avg daily active users by age: holiday vs workday vs weekend.
    对比节假日 vs 工作日 vs 周末的各年龄段日均活跃用户数。"""
    log("=" * 50)  # 日志：分隔线
    log("H2: Holiday vs Workday vs Weekend Avg Daily Active Users by Age")  # 日志：分析标题

    period_dates = {}  # 存储各周期的日期集合
    period_age = {}  # 存储各周期各年龄段的日均活跃用户数
    for p in ['holiday', 'workday', 'weekend']:  # 遍历三种周期
        period_dates[p] = set(r['date'] for r in seekers if r['period'] == p)  # 获取该周期的所有日期
        raw = _age_active_users(seekers, period_dates[p], user_ages)  # 计算该周期各年龄段活跃用户数
        num_days = max(len(period_dates[p]), 1)
        period_age[p.capitalize()] = {seg: count / num_days for seg, count in raw.items()}  # 转为日均值
        total = sum(period_age[p.capitalize()].values())  # 计算该周期总日均活跃用户数
        log(f"  {p}: {total:.2f} avg daily active users")  # 日志输出总数

    _plot_age_grouped_bars(  # 绘制分组柱状图
        period_age,  # 数据：Holiday / Workday / Weekend
        'Avg Daily Active Users by Age: Holiday vs Workday vs Weekend',  # 图表标题
        'h2_holiday_workday_weekend_age.png',  # 输出文件名
    )

    csv_path = os.path.join(STEP_OUT, 'h2_holiday_workday_weekend_age.csv')  # CSV 输出文件路径
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:  # 以 UTF-8 编码打开 CSV 文件写入
        w = csv.writer(f)  # 创建 CSV 写入器
        w.writerow(['age_segment', 'holiday_avg_daily', 'workday_avg_daily', 'weekend_avg_daily'])  # 写入表头
        for seg in AGE_SEGMENTS:  # 遍历每个年龄段
            w.writerow([seg,
                        f'{period_age["Holiday"].get(seg, 0):.2f}',
                        f'{period_age["Workday"].get(seg, 0):.2f}',
                        f'{period_age["Weekend"].get(seg, 0):.2f}'])  # 写入各周期的日均值
    log(f"Saved: {csv_path}")  # 日志记录 CSV 保存信息


# ═══════════════════════════════════════════════════════════════════════
#  H1+H2 合并渲染 (双面板)
#  H1+H2 Combined: Dual Panel Age Chart Renderer
# ═══════════════════════════════════════════════════════════════════════
# 【作用】将 H1 和 H2 的数据合并渲染到一张双面板图中
# 【图表】上=holiday vs non-holiday, 下=holiday vs workday vs weekend
# 【代码逻辑】收集两组数据 → fig, (ax1, ax2) → 分别绘制柱状图 → 保存 PNG
# ═══════════════════════════════════════════════════════════════════════

def dim_h1_h2_combined(seekers: list[dict], user_ages: dict):
    """H1+H2 combined: Holiday vs Non-Holiday (top) + Holiday vs Workday vs Weekend (bottom).
    H1+H2 合并图：上半部分节假日 vs 非节假日，下半部分节假日 vs 工作日 vs 周末。"""
    log("=" * 50)
    log("H1+H2: Combined Age Distribution")

    # H1 stats: holiday vs non-holiday
    holiday_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    num_h = max(len(holiday_dates), 1)
    num_nh = max(len(non_holiday_dates), 1)

    h_age = _age_active_users(seekers, holiday_dates, user_ages)
    nh_age = _age_active_users(seekers, non_holiday_dates, user_ages)

    h_per_day = {seg: count / num_h for seg, count in h_age.items()}
    nh_per_day = {seg: count / num_nh for seg, count in nh_age.items()}

    # H2 stats: holiday vs workday vs weekend
    period_age = {}
    for p in ['holiday', 'workday', 'weekend']:
        dates = set(r['date'] for r in seekers if r['period'] == p)
        raw = _age_active_users(seekers, dates, user_ages)
        nd = max(len(dates), 1)
        period_age[p.capitalize()] = {seg: count / nd for seg, count in raw.items()}

    _plot_age_grouped_bars_two_row(
        {'Holiday': h_per_day, 'Non-holiday': nh_per_day},
        period_age,
        'Holiday vs Non-Holiday',
        'Holiday vs Workday vs Weekend',
        [COLOR_HOLIDAY, COLOR_NONHOLIDAY],
        [COLOR_HOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND],
        'Avg Daily Active Users by Age',
        'h1_h2_combined_age.png',
    )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'h1_h2_combined_age.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['age_segment',
                    'holiday_avg_daily', 'non_holiday_avg_daily',
                    'workday_avg_daily', 'weekend_avg_daily'])
        age_segs_plot = [s for s in AGE_SEGMENTS if s != 'unknown']
        for seg in age_segs_plot:
            w.writerow([seg,
                        f'{h_per_day.get(seg, 0):.2f}',
                        f'{nh_per_day.get(seg, 0):.2f}',
                        f'{period_age["Workday"].get(seg, 0):.2f}',
                        f'{period_age["Weekend"].get(seg, 0):.2f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  H3: 各节假日年龄段活跃用户 VS 非节假日基线 (Stacked Bar)
#  H3: Per-Holiday Age Distribution vs Non-Holiday
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】堆叠柱状图(每节假日 × 每年龄段) + 非节假日基线
# 【统计口径】节假日按名称聚合 → _age_daily_active() 计算各年龄段日均
#   _age_grouped_bar() 辅助绘制
# 【输出文件】PNG: h3_h4_per_holiday_age_merged.png, CSV: h3_*.csv
# 【特殊说明】使用毛玻璃效果(alpha=0.85)区分假期自身与基线
# ═══════════════════════════════════════════════════════════════════════

def dim_h3_per_holiday_vs_nonholiday_age(seekers: list[dict], user_ages: dict):
    """
    For each holiday name, show age distribution as stacked bars (per-day),
    comparing with non-holiday baseline.
    对每个节假日分别显示各年龄段的日均活跃用户数（堆叠柱状图），并与非节假日基线对比。
    """
    log("=" * 50)  # 日志：分隔线
    log("H3: Per-Holiday Age Distribution vs Non-Holiday (Avg Daily)")  # 日志：分析标题

    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')  # 获取所有非节假日日期集合
    num_nh = max(len(non_holiday_dates), 1)
    nh_age = _age_active_users(seekers, non_holiday_dates, user_ages)  # 计算非节假日各年龄段活跃用户数作为基线
    # 转为日均值
    nh_age_per_day = {seg: count / num_nh for seg, count in nh_age.items()}

    # Group holidays by name
    # 按节假日名称分组
    holiday_groups = defaultdict(list)  # 节假日名称 -> 记录列表
    for r in seekers:  # 遍历所有记录
        if r['period'] == 'holiday':  # 如果是节假日记录
            name = r['holiday_name'][:6]  # 取节假日名称前 6 个字符作为分组键（避免过长）
            holiday_groups[name].append(r)  # 将该记录加入对应节假日分组

    holiday_groups = {k: v for k, v in holiday_groups.items()  # 过滤：只保留数据行数达到最小要求的节假日
                     if len(v) >= MIN_DATA_ROWS}

    if not holiday_groups:  # 如果没有符合条件的节假日分组
        log("  WARN: No holiday groups with sufficient data")  # 日志：警告信息
        return  # 提前返回

    # Per-holiday stacked bar vs non-holiday
    # 每个节假日的堆叠柱状图，与非节假日基线对比
    names = sorted(holiday_groups.keys())  # 获取排序后的节假日名称列表
    age_segs_plot = [s for s in AGE_SEGMENTS if s != 'unknown']  # 排除 'unknown' 年龄段

    num_holidays = len(names)  # 节假日数量
    fig_height = max(6, num_holidays * 0.4)  # 动态计算图表高度
    bar_w = 0.6  # 统一柱宽，保证左右子图柱状大小一致
    fig, axes = plt.subplots(1, 2, figsize=(max(16, num_holidays * 0.7), fig_height))  # 创建左右两个子图

    # Left: holiday age counts (avg daily)
    # 左图：各节假日的各年龄段日均活跃用户数（堆叠柱状图）
    ax1 = axes[0]  # 第一个子图（左侧）
    x = np.arange(num_holidays)  # x 轴位置：每个节假日一个位置
    bottom = np.zeros(num_holidays)  # 堆叠基线初始化为全零
    for seg in age_segs_plot:  # 遍历每个年龄段
        vals = []  # 存储该年龄段在各节假日的日均值
        for name in names:  # 遍历每个节假日
            group_dates = set(r['date'] for r in holiday_groups[name])  # 获取该节假日的日期集合
            num_dates = max(len(group_dates), 1)
            age_counts = _age_active_users(holiday_groups[name], group_dates, user_ages)  # 计算该节假日各年龄段活跃用户数
            vals.append(age_counts.get(seg, 0) / num_dates)  # 转为日均值
        ax1.bar(x, vals, width=bar_w, bottom=bottom, label=seg, color=AGE_COLORS[seg], alpha=0.8)  # 在该位置堆叠绘制
        bottom += vals  # 更新堆叠基线
    ax1.set_xticks(x)  # 设置 x 轴刻度位置
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)  # 设置 x 轴标签（节假日名称），旋转 45 度
    ax1.set_xlim(-0.5, num_holidays - 0.5)
    ax1.set_ylabel('Avg Daily Active Users')  # y 轴标签
    ax1.set_title('Per-Holiday Avg Daily Active Users by Age', fontsize=11)  # 子图标题
    ax1.legend(fontsize=8)  # 图例

    # Right: non-holiday age counts (avg daily)
    # 右图：非节假日的各年龄段日均活跃用户数（作为基线对比）
    ax2 = axes[1]  # 第二个子图（右侧）
    nh_vals = [nh_age_per_day.get(seg, 0) for seg in age_segs_plot]  # 获取非节假日各年龄段日均值
    ax2.bar([0], [sum(nh_vals)], width=bar_w, color='#ced6e0', alpha=0.5, label='Total')  # 绘制非节假日总日均值柱状图（背景色）
    bottom2 = 0  # 堆叠基线的初始值
    for i, seg in enumerate(age_segs_plot):  # 遍历每个年龄段
        ax2.bar([0], [nh_vals[i]], width=bar_w, bottom=[bottom2],  # 在该柱子上堆叠
                label=seg, color=AGE_COLORS[seg], alpha=0.8)
        bottom2 += nh_vals[i]  # 更新堆叠基线
    ax2.set_xticks([0])  # x 轴刻度位置
    ax2.set_xticklabels(['Non-holiday'], fontsize=9)  # x 轴标签
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylabel('Avg Daily Active Users')  # y 轴标签
    ax2.set_title('Non-Holiday Baseline: Avg Daily Active Users by Age', fontsize=11)  # 子图标题
    ax2.legend(fontsize=8)  # 图例
    ax2.grid(axis='y', alpha=0.3)  # y 方向网格线

    fig.suptitle('Per-Holiday Age Distribution vs Non-Holiday Baseline (Avg Daily)', fontsize=13)  # 总标题
    fig.tight_layout()  # 自动调整布局
    path = os.path.join(STEP_OUT, 'h3_per_holiday_vs_nonholiday_age.png')  # 输出文件路径
    fig.savefig(path)  # 保存图片
    plt.close(fig)  # 关闭图形
    log(f"Saved: {path}")  # 日志记录保存信息

    # CSV
    # 输出 CSV 文件
    csv_path = os.path.join(STEP_OUT, 'h3_per_holiday_vs_nonholiday_age.csv')  # CSV 文件路径
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:  # 写入 CSV
        w = csv.writer(f)  # 创建 CSV 写入器
        header = ['holiday_name', 'num_dates'] + age_segs_plot + ['total_avg_daily']  # 表头
        w.writerow(header)  # 写入表头
        for name in names:  # 遍历每个节假日
            group_dates = set(r['date'] for r in holiday_groups[name])  # 获取该节假日日期集合
            num_dates = max(len(group_dates), 1)
            age_counts = _age_active_users(holiday_groups[name], group_dates, user_ages)  # 计算各年龄段人数
            row = [name, num_dates] + [f'{age_counts.get(seg, 0) / num_dates:.2f}' for seg in age_segs_plot]  # 构建日均值数据行
            total_avg = sum(age_counts.get(seg, 0) for seg in age_segs_plot) / num_dates
            row.append(f'{total_avg:.2f}')  # 添加总日均值
            w.writerow(row)  # 写入行
        # Non-holiday baseline
        # 非节假日基线行
        row = ['non_holiday_baseline', num_nh] + [f'{nh_age_per_day.get(seg, 0):.2f}' for seg in age_segs_plot]
        row.append(f'{sum(nh_age_per_day.get(seg, 0) for seg in age_segs_plot):.2f}')  # 添加总日均值
        w.writerow(row)  # 写入行
    log(f"Saved: {csv_path}")  # 日志记录 CSV 保存信息


# ═══════════════════════════════════════════════════════════════════════
#  H4: 各节假日 VS 工作日/周末 年龄段活跃用户
#  H4: Per-Holiday Age vs Workday & Weekend
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】已合并到 H3 图中（显示工作日/周末基线）
# 【输出文件】CSV: h4_per_holiday_vs_workday_weekend_age.csv
#   图片已合并到 H3
# ═══════════════════════════════════════════════════════════════════════

def dim_h4_per_holiday_vs_workday_weekend_age(seekers: list[dict], user_ages: dict):
    """Per-holiday age avg daily active users vs workday & weekend.
    每个节假日各年龄段日均活跃用户数与工作日、周末的对比。"""
    log("=" * 50)  # 日志：分隔线
    log("H4: Per-Holiday Age vs Workday & Weekend (Avg Daily)")  # 日志：分析标题

    # Compute workday/weekend baselines (avg daily)
    workday_dates = set(r['date'] for r in seekers if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seekers if r['period'] == 'weekend')
    wd_age = _age_active_users(seekers, workday_dates, user_ages)
    we_age = _age_active_users(seekers, weekend_dates, user_ages)
    num_wd = max(len(workday_dates), 1)
    num_we = max(len(weekend_dates), 1)

    for p, pa in [('workday', wd_age), ('weekend', we_age)]:
        total = sum(pa.values())
        log(f"  {p}: {total} active users by age (total across all days)")

    # Group seekers by holiday
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_groups[name].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items() if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  No holiday groups")
        return

    names = sorted(holiday_groups.keys())
    n = len(names)
    age_segs_plot = [s for s in AGE_SEGMENTS if s != 'unknown']

    # Build matrices: rows=age segments, cols=holidays, value = holiday_avg - baseline_avg
    matrix_wd = np.zeros((len(age_segs_plot), n))
    matrix_we = np.zeros((len(age_segs_plot), n))
    csv_data = []
    for j, name in enumerate(names):
        group_dates = set(r['date'] for r in holiday_groups[name])
        num_dates = max(len(group_dates), 1)
        h_age = _age_active_users(holiday_groups[name], group_dates, user_ages)
        for i, seg in enumerate(age_segs_plot):
            h_avg = h_age.get(seg, 0) / num_dates
            matrix_wd[i, j] = h_avg - (wd_age.get(seg, 0) / num_wd)
            matrix_we[i, j] = h_avg - (we_age.get(seg, 0) / num_we)

    # Dual heatmap: top=vs Workday, bottom=vs Weekend
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, n * 0.55), max(7, len(age_segs_plot) * 0.6 + 2)))

    vmax1 = max(abs(matrix_wd.min()), abs(matrix_wd.max()), 0.01)
    im1 = ax1.imshow(matrix_wd, cmap='RdBu_r', aspect='auto', vmin=-vmax1, vmax=vmax1)
    annotate_heatmap(ax1, matrix_wd, fmt='.1f', fs=6)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax1.set_yticks(range(len(age_segs_plot)))
    ax1.set_yticklabels(age_segs_plot, fontsize=8)
    ax1.set_title('Diff: Holiday Avg Daily - Workday Baseline', fontsize=10)
    fig.colorbar(im1, ax=ax1, shrink=0.5, label='Diff')

    vmax2 = max(abs(matrix_we.min()), abs(matrix_we.max()), 0.01)
    im2 = ax2.imshow(matrix_we, cmap='RdBu_r', aspect='auto', vmin=-vmax2, vmax=vmax2)
    annotate_heatmap(ax2, matrix_we, fmt='.1f', fs=6)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax2.set_yticks(range(len(age_segs_plot)))
    ax2.set_yticklabels(age_segs_plot, fontsize=8)
    ax2.set_xlabel('Holiday')
    ax2.set_title('Diff: Holiday Avg Daily - Weekend Baseline', fontsize=10)
    fig.colorbar(im2, ax=ax2, shrink=0.5, label='Diff')

    fig.suptitle('Per-Holiday Avg Daily Active Users by Age — Diff from Workday & Weekend', fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'h4_per_holiday_vs_workday_weekend_age.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'h4_per_holiday_vs_workday_weekend_age.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name'] + age_segs_plot + ['workday_avg', 'weekend_avg'])
        for j, name in enumerate(names):
            group_dates = set(r['date'] for r in holiday_groups[name])
            h_age = _age_active_users(holiday_groups[name], group_dates, user_ages)
            num_dates = max(len(group_dates), 1)
            row = [name] + [f'{h_age.get(seg, 0) / num_dates:.2f}' for seg in age_segs_plot]
            row.append(f'{sum(wd_age.values()) / num_wd:.2f}')
            row.append(f'{sum(we_age.values()) / num_we:.2f}')
            w.writerow(row)
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  B: Hourly age analysis  B：逐小时年龄分析
# ═══════════════════════════════════════════════════════════════════════

def _plot_age_hourly_lines(
    hourly_age: dict[str, dict[str, list[float]]],
    title: str, filename: str):
    """
    Line chart: for each age segment, show hourly active users across groups.
    折线图：对每个年龄段，展示不同分组（节假日/非节假日等）的逐小时活跃用户数变化。
    hourly_age: {group_label: {age_seg: [24 floats]}}
    hourly_age: {分组标签: {年龄段: [24 个浮点数]}}
    """
    age_segs_plot = [s for s in AGE_SEGMENTS if s != 'unknown']  # 排除 'unknown' 年龄段
    groups = list(hourly_age.keys())  # 获取分组名称列表
    hours = list(range(24))  # 小时范围 0-23

    fig, axes = plt.subplots(len(age_segs_plot), 1, figsize=(12, 3 * len(age_segs_plot)),  # 每个年龄段一个子图，垂直排列
                             sharex=True)  # 所有子图共享 x 轴

    if len(age_segs_plot) == 1:  # 如果只有一个年龄段
        axes = [axes]  # 将 axes 包装为列表，确保可以按索引访问

    for idx, seg in enumerate(age_segs_plot):  # 遍历每个年龄段
        ax = axes[idx]  # 获取对应的子图
        for group in groups:  # 遍历每个分组
            vals = hourly_age[group].get(seg, [0.0] * 24)  # 获取该分组该年龄段的逐小时数据
            ax.plot(hours, vals, 'o-', label=group, linewidth=1.5, markersize=3,  # 绘制折线图，带圆形标记
                    alpha=0.85)  # 透明度 0.85
        ax.set_ylabel(f'{seg}\n(avg/hr)')  # y 轴标签：年龄段 + 每小时间
        ax.set_title(f'Age: {seg}', fontsize=10)  # 子图标题
        ax.legend(fontsize=7, loc='upper right')  # 图例，放置在右上角
        ax.grid(axis='y', alpha=0.3)  # y 方向网格线
        if idx == len(age_segs_plot) - 1:  # 如果是最后一个子图（底部）
            ax.set_xlabel('Hour of Day (UTC)')  # 设置 x 轴标签

    fig.suptitle(title, fontsize=13)  # 总标题
    fig.tight_layout()  # 自动调整布局
    path = os.path.join(STEP_OUT, filename)  # 输出文件路径
    fig.savefig(path)  # 保存图片
    plt.close(fig)  # 关闭图形
    log(f"Saved: {path}")  # 日志记录保存信息


# ═══════════════════════════════════════════════════════════════════════
#  I1: 逐小时年龄段活跃用户: 节假日 VS 非节假日 (Line)
#  I1: Hourly Age: Holiday vs Non-Holiday
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】多面板折线图: 每个年龄段一个子图, 2组曲线对比
# 【统计口径】24小时 × 6年龄段 × 2组(holiday/non-holiday)
#   _hourly_age_active(seekers, date_set, user_ages) → {seg: [24 values]}
# 【输出文件】PNG: i1_i2_hourly_age_merged.png, CSV: i1_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_i1_hourly_holiday_vs_nonholiday_age(seekers: list[dict], user_ages: dict):
    """Hourly active users by age: holiday vs non-holiday.
    节假日 vs 非节假日的逐小时各年龄段活跃用户数对比。"""
    log("=" * 50)  # 日志：分隔线
    log("I1: Hourly Age Distribution - Holiday vs Non-Holiday")  # 日志：分析标题

    holiday_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')  # 节假日日期集合
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')  # 非节假日日期集合

    h_hourly = _age_hourly_active_users(seekers, holiday_dates, user_ages)  # 计算节假日逐小时各年龄段活跃用户数
    nh_hourly = _age_hourly_active_users(seekers, non_holiday_dates, user_ages)  # 计算非节假日逐小时各年龄段活跃用户数

    _plot_age_hourly_lines(  # 绘制逐小时折线图
        {'Holiday': h_hourly, 'Non-holiday': nh_hourly},  # 数据
        'Hourly Active Users by Age: Holiday vs Non-Holiday',  # 图表标题
        'i1_hourly_holiday_vs_nonholiday_age.png',  # 输出文件名
    )

    csv_path = os.path.join(STEP_OUT, 'i1_hourly_holiday_vs_nonholiday_age.csv')  # CSV 文件路径
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:  # 写入 CSV
        w = csv.writer(f)  # 创建 CSV 写入器
        w.writerow(['hour'] + [f'holiday_{seg}' for seg in AGE_SEGMENTS]  # 写入表头：小时 + 节假日各年龄段 + 非节假日各年龄段
                    + [f'non_holiday_{seg}' for seg in AGE_SEGMENTS])
        for h in range(24):  # 遍历 0-23 小时
            row = [h]  # 行数据以小时数开头
            for seg in AGE_SEGMENTS:  # 遍历每个年龄段
                row.append(f'{h_hourly.get(seg, [0]*24)[h]:.4f}')  # 添加节假日数据（保留 4 位小数）
            for seg in AGE_SEGMENTS:  # 遍历每个年龄段
                row.append(f'{nh_hourly.get(seg, [0]*24)[h]:.4f}')  # 添加非节假日数据（保留 4 位小数）
            w.writerow(row)  # 写入该小时的数据行
    log(f"Saved: {csv_path}")  # 日志记录 CSV 保存信息


# ═══════════════════════════════════════════════════════════════════════
#  I2: 逐小时年龄段活跃用户: 节假日 VS 工作日 VS 周末
#  I2: Hourly Age: Holiday vs Workday vs Weekend
# ═══════════════════════════════════════════════════════════════════════
# 【输出文件】CSV: i2_hourly_age_workday_weekend.csv (图片已合并到 I1 下面板)
# ═══════════════════════════════════════════════════════════════════════

def dim_i2_hourly_holiday_workday_weekend_age(seekers: list[dict], user_ages: dict):
    """Hourly age: holiday vs workday vs weekend.
    节假日 vs 工作日 vs 周末的逐小时各年龄段活跃用户数对比。"""
    log("=" * 50)  # 日志：分隔线
    log("I2: Hourly Age - Holiday vs Workday vs Weekend")  # 日志：分析标题

    hourly_data = {}  # 存储各周期的逐小时数据
    for p in ['holiday', 'workday', 'weekend']:  # 遍历三种周期
        p_dates = set(r['date'] for r in seekers if r['period'] == p)  # 获取该周期的日期集合
        hourly_data[p.capitalize()] = _age_hourly_active_users(  # 计算逐小时数据，键名首字母大写
            seekers, p_dates, user_ages)

    _plot_age_hourly_lines(  # 绘制逐小时折线图
        hourly_data,  # 数据：Holiday / Workday / Weekend
        'Hourly Active Users by Age: Holiday vs Workday vs Weekend',  # 图表标题
        'i2_hourly_holiday_workday_weekend_age.png',  # 输出文件名
    )

    csv_path = os.path.join(STEP_OUT, 'i2_hourly_holiday_workday_weekend_age.csv')  # CSV 文件路径
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:  # 写入 CSV
        w = csv.writer(f)  # 创建 CSV 写入器
        w.writerow(['hour'] + [f'holiday_{seg}' for seg in AGE_SEGMENTS]  # 表头：小时 + 节假日/工作日/周末各年龄段
                    + [f'workday_{seg}' for seg in AGE_SEGMENTS]
                    + [f'weekend_{seg}' for seg in AGE_SEGMENTS])
        for h in range(24):  # 遍历 0-23 小时
            row = [h]  # 以小时数开头
            for seg in AGE_SEGMENTS:  # 节假日各年龄段
                row.append(f'{hourly_data["Holiday"].get(seg, [0]*24)[h]:.4f}')
            for seg in AGE_SEGMENTS:  # 工作日各年龄段
                row.append(f'{hourly_data["Workday"].get(seg, [0]*24)[h]:.4f}')
            for seg in AGE_SEGMENTS:  # 周末各年龄段
                row.append(f'{hourly_data["Weekend"].get(seg, [0]*24)[h]:.4f}')
            w.writerow(row)  # 写入该小时的数据行
    log(f"Saved: {csv_path}")  # 日志记录 CSV 保存信息


# ═══════════════════════════════════════════════════════════════════════
#  I3(旧版): 各节假日逐小时年龄段差值 (Heatmap Per Segment)
#  I3(OLD): Per-Holiday Hourly Age Diff per Age Segment
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】每个年龄段一张热力图，垂直排列
# 【说明】旧版实现，每个年龄段独立热力图，保留用于对比
# ═══════════════════════════════════════════════════════════════════════

def dim_i3_per_holiday_hourly_age_old(seekers: list[dict], user_ages: dict):
    """
    Per-holiday hourly age distribution vs non-holiday.
    Uses heatmap per age segment.
    每个节假日的逐小时年龄分布 vs 非节假日基线。
    对每个年龄段生成热力图，显示与基线的差值。
    """
    log("=" * 50)  # 日志：分隔线
    log("I3: Per-Holiday Hourly Age Heatmaps vs Non-Holiday")  # 日志：分析标题

    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')  # 非节假日日期集合
    nh_hourly = _age_hourly_active_users(seekers, non_holiday_dates, user_ages)  # 非节假日逐小时基线

    holiday_groups = defaultdict(list)  # 按节假日名称分组
    for r in seekers:  # 遍历所有记录
        if r['period'] == 'holiday':  # 如果是节假日记录
            name = r['holiday_name'][:6]  # 取前 6 字符作为分组键
            holiday_groups[name].append(r)  # 加入对应分组

    holiday_groups = {k: v for k, v in holiday_groups.items()  # 过滤：只保留数据量足够的节假日
                     if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:  # 如果没有符合条件的节假日
        log("  WARN: No holiday groups")  # 日志警告
        return  # 提前返回

    group_names = sorted(holiday_groups.keys())  # 排序后的节假日名称列表
    age_segs_plot = [s for s in AGE_SEGMENTS if s != 'unknown']  # 排除 'unknown' 年龄段

    for seg in age_segs_plot:  # 对每个年龄段生成一张热力图
        matrix = np.zeros((len(group_names), 24))  # 创建矩阵：行=节假日数，列=24 小时
        for i, name in enumerate(group_names):  # 遍历每个节假日
            group_dates = set(r['date'] for r in holiday_groups[name])  # 获取该节假日的日期集合
            h_hourly = _age_hourly_active_users(  # 计算该节假日逐小时数据
                holiday_groups[name], group_dates, user_ages)
            seg_vals = h_hourly.get(seg, [0.0] * 24)  # 该年龄段在该节假日的逐小时数据
            nh_vals = nh_hourly.get(seg, [0.0] * 24)  # 该年龄段在非节假日的逐小时基线
            for h in range(24):  # 遍历 24 小时
                matrix[i, h] = seg_vals[h] - nh_vals[h]  # 计算差值（节假日 - 非节假日）

        fig, ax = plt.subplots(figsize=(16, max(5, len(group_names) * 0.35 + 2)))  # 创建图表面板
        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)  # 计算对称颜色范围的最大绝对值
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)  # 绘制热力图，红蓝配色
        annotate_heatmap(ax, matrix, fmt='.1f', fs=6)

        ax.set_xticks(range(24))  # x 轴刻度：0-23 小时
        ax.set_xticklabels(range(24), fontsize=8)  # x 轴标签
        ax.set_yticks(range(len(group_names)))  # y 轴刻度：各节假日
        ax.set_yticklabels(group_names, fontsize=8)  # y 轴标签：节假日名称
        ax.set_xlabel('Hour of Day (UTC)')  # x 轴标签
        ax.set_title(f'Age {seg}: Per-Holiday Hourly Active Users Diff from Non-Holiday',  # 子图标题
                     fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.6, label='Diff')  # 添加颜色条

        fig.tight_layout()  # 自动调整布局
        safe_seg = seg.replace('<', 'lt_').replace('>', 'gt_').replace('+', 'p')  # 替换文件名中的非法字符
        path = os.path.join(STEP_OUT, f'i3_age_{safe_seg}_hourly_heatmap.png')  # 输出文件路径
        fig.savefig(path)  # 保存图片
        plt.close(fig)  # 关闭图形
        log(f"Saved: {path}")  # 日志记录保存信息

    csv_path = os.path.join(STEP_OUT, 'i3_per_holiday_hourly_age.csv')  # CSV 文件路径
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:  # 写入 CSV
        w = csv.writer(f)  # 创建 CSV 写入器
        w.writerow(['holiday_name', 'age_segment', 'hour', 'diff_from_nonholiday'])  # 表头
        for name in group_names:  # 遍历每个节假日
            group_dates = set(r['date'] for r in holiday_groups[name])  # 获取日期集合
            h_hourly = _age_hourly_active_users(  # 计算该节假日逐小时数据
                holiday_groups[name], group_dates, user_ages)
            for seg in age_segs_plot:  # 遍历每个年龄段
                for h in range(24):  # 遍历 24 小时
                    diff = h_hourly.get(seg, [0.0]*24)[h] - nh_hourly.get(seg, [0.0]*24)[h]  # 计算差值
                    w.writerow([name, seg, h, f'{diff:.4f}'])  # 写入该行
    log(f"Saved: {csv_path}")  # 日志记录 CSV 保存信息


# ═══════════════════════════════════════════════════════════════════════
#  I3(新版): 各节假日逐小时年龄段差值 (Stacked Heatmap)
#  I3: Per-Holiday Hourly Age Diff (Stacked)
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】垂直堆叠热力图: 行=年龄段×节假日(每个年龄段先堆叠所有节假日,再切换到下一年龄段)
#   列=0-23小时, 值=节假日与非节假日的绝对值差值
# 【输出文件】PNG: i3_i4_per_holiday_hourly_age_merged.png, CSV: i3_*.csv
# 【特殊说明】所有年龄段共享同一色标
# ═══════════════════════════════════════════════════════════════════════

def dim_i3_per_holiday_hourly_age(seekers: list[dict], user_ages: dict):
    """
    Per-holiday hourly age distribution vs non-holiday.
    Single vertically-stacked heatmap: rows = age_segment × holiday, cols = 24h.
    每个节假日的逐小时年龄分布 vs 非节假日基线（垂直堆叠热力图，所有年龄段共享色标）。
    """
    log("=" * 50)
    log("I3: Per-Holiday Hourly Age Heatmaps vs Non-Holiday (Stacked)")

    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_hourly = _age_hourly_active_users(seekers, non_holiday_dates, user_ages)

    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_groups[name].append(r)

    holiday_groups = {k: v for k, v in holiday_groups.items()
                     if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  WARN: No holiday groups")
        return

    group_names = sorted(holiday_groups.keys())
    age_segs_plot = [s for s in AGE_SEGMENTS if s != 'unknown']

    # ── 预计算所有 (age_seg, holiday) 的逐小时差值 ──
    # diff_data: {seg: {name: [24 floats]}}
    diff_data = {}
    for seg in age_segs_plot:
        diff_data[seg] = {}
        nh_vals = nh_hourly.get(seg, [0.0] * 24)
        for name in group_names:
            group_dates = set(r['date'] for r in holiday_groups[name])
            h_hourly = _age_hourly_active_users(holiday_groups[name], group_dates, user_ages)
            seg_vals = h_hourly.get(seg, [0.0] * 24)
            diff_data[seg][name] = [seg_vals[h] - nh_vals[h] for h in range(24)]

    # ── 构建垂直堆叠矩阵：行 = age_seg × holiday, 列 = 24h ──
    n_holidays = len(group_names)
    n_segs = len(age_segs_plot)
    total_rows = n_segs * n_holidays
    matrix = np.zeros((total_rows, 24))
    y_labels = []

    for si, seg in enumerate(age_segs_plot):
        for hi, name in enumerate(group_names):
            row_idx = si * n_holidays + hi
            matrix[row_idx] = diff_data[seg][name]
            y_labels.append(f'{seg}  {name}')

    # ── 全局共享色标范围 ──
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)

    # ── 绘图 ──
    fig_h = max(8, total_rows * 0.28 + 3)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    annotate_heatmap(ax, matrix, fmt='.1f', fs=6)

    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=8)
    ax.set_xlabel('Hour of Day (UTC)')
    ax.set_yticks(range(total_rows))
    ax.set_yticklabels(y_labels, fontsize=7)

    # ── 白色分隔线区分不同年龄段 ──
    for si in range(1, n_segs):
        y = si * n_holidays - 0.5
        ax.axhline(y, color='white', linewidth=2)

    fig.colorbar(im, ax=ax, shrink=0.6, label='Diff (Holiday - Non-Holiday)')
    fig.suptitle('Per-Holiday Hourly Active Users by Age — Diff from Non-Holiday', fontsize=13)
    fig.tight_layout()

    path = os.path.join(STEP_OUT, 'i3_per_holiday_hourly_age_stacked.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV（保持不变） ──
    csv_path = os.path.join(STEP_OUT, 'i3_per_holiday_hourly_age.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'age_segment', 'hour', 'diff_from_nonholiday'])
        for name in group_names:
            for seg in age_segs_plot:
                for h in range(24):
                    w.writerow([name, seg, h, f'{diff_data[seg][name][h]:.4f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  I4: 各节假日逐小时年龄段 VS 工作日/周末 (折线图)
#  I4: Per-Holiday Hourly Age vs Workday & Weekend
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】每个年龄段一张子图，每条曲线=节假日某天 vs 工作日/周末基线
# 【输出文件】CSV: i4_per_holiday_hourly_age_vs_workday_weekend.csv
#   图片已合并到 I3
# ═══════════════════════════════════════════════════════════════════════

def dim_i4_per_holiday_hourly_age_vs_workday_weekend(
    seekers: list[dict], user_ages: dict):
    """Per-holiday hourly age avg daily active vs workday & weekend (line charts).
    每个节假日逐小时各年龄段日均活跃用户数 vs 工作日和周末（折线图）。"""
    log("=" * 50)  # 日志：分隔线
    log("I4: Per-Holiday Hourly Age vs Workday & Weekend")  # 日志：分析标题

    # Compute workday/weekend baselines (hourly avg daily)
    workday_dates = set(r['date'] for r in seekers if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seekers if r['period'] == 'weekend')
    wd_hourly = _age_hourly_active_users(seekers, workday_dates, user_ages)
    we_hourly = _age_hourly_active_users(seekers, weekend_dates, user_ages)

    for p, ph in [('workday', wd_hourly), ('weekend', we_hourly)]:
        total = sum(sum(v) for v in ph.values())
        log(f"  {p}: {total:.2f} avg hourly active users across age segments")

    # Group seekers by holiday
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_groups[name].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items() if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  No holiday groups")
        return

    names = sorted(holiday_groups.keys())
    age_segs_plot = [s for s in AGE_SEGMENTS if s != 'unknown']
    hours = list(range(24))

    # For each age segment, plot holiday hourly lines vs workday/weekend baselines
    for seg in age_segs_plot:
        fig, ax = plt.subplots(figsize=(12, 5))

        # Workday and weekend baselines
        ax.plot(hours, wd_hourly.get(seg, [0.0]*24), '-', color='#888888', linewidth=1.5,
                label='Workday', alpha=0.7)
        ax.plot(hours, we_hourly.get(seg, [0.0]*24), '--', color='#aaaaaa', linewidth=1.5,
                label='Weekend', alpha=0.7)

        # Each holiday as a separate line
        holiday_colors = plt.cm.tab20(np.linspace(0, 1, len(names)))
        for j, name in enumerate(names):
            group_dates = set(r['date'] for r in holiday_groups[name])
            h_hourly = _age_hourly_active_users(holiday_groups[name], group_dates, user_ages)
            ax.plot(hours, h_hourly.get(seg, [0.0]*24), 'o-', color=holiday_colors[j],
                    linewidth=1.2, markersize=2.5, label=name, alpha=0.85)

        ax.set_xlabel('Hour of Day (UTC)')
        ax.set_ylabel('Avg Active Users')
        ax.set_title(f'Hourly Active Users: Age Segment "{seg}" — Holidays vs Workday/Weekend', fontsize=11)
        ax.legend(fontsize=7, loc='upper right', ncol=min(4, len(names)+2))
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(range(24))
        fig.tight_layout()
        safe_seg = seg.replace('+', 'plus').replace('<', 'lt')
        path = os.path.join(STEP_OUT, f'i4_age_{safe_seg}_hourly_lines.png')
        fig.savefig(path)
        plt.close(fig)
        log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'i4_per_holiday_hourly_age_vs_workday_weekend.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'age_segment', 'hour', 'holiday_avg', 'workday_avg', 'weekend_avg'])
        for name in names:
            group_dates = set(r['date'] for r in holiday_groups[name])
            h_hourly = _age_hourly_active_users(holiday_groups[name], group_dates, user_ages)
            for seg in age_segs_plot:
                for h in range(24):
                    w.writerow([name, seg, h,
                                f'{h_hourly.get(seg, [0.0]*24)[h]:.4f}',
                                f'{wd_hourly.get(seg, [0.0]*24)[h]:.4f}',
                                f'{we_hourly.get(seg, [0.0]*24)[h]:.4f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main  主函数入口
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None):
    log("=" * 60)  # 日志：分隔CCCCDDDD线
    log("Step 4: Age Distribution Analysis")  # 日志：步骤标题
    log("=" * 60)  # 日志：分隔线

    if data is None:
        from movie.data_loader import load_all  # 延迟导入数据加载函数
        data = load_all()  # 加载所有数据
    seekers = data['seekers']  # 获取用户提问记录列表
    user_ages = data['user_ages']  # 获取用户年龄映射字典

    log(f"Loaded age segments for {len(user_ages)} users")  # 日志：显示已加载的用户年龄数量

    # Section G: Multi-Session Analysis
    # G 部分：多会话分析
    log("")
    log("-" * 40)
    log("Section G: Multi-Session Analysis by Age")
    log("-" * 40)

    # dim_g1_multi_session_same_day_age(seekers, user_ages)  # G1：单日多session
    # log("")
    # dim_g2_cross_day_multi_session_age(seekers, user_ages)  # G2：跨天多session
    # log("")
    # dim_g3_per_holiday_multi_session_age(seekers, user_ages)  # G3：各节假日单日多session
    # log("")
    # dim_g4_per_holiday_cross_day_age(seekers, user_ages)  # G4：各节假日跨天多session

    # Section H: Weekly
    # A 部分：周周期分析
    log("")  # 日志：空行
    log("-" * 40)  # 日志：分隔线
    log("Section H: Weekly - Age Distribution")  # 日志：部分标题
    log("-" * 40)  # 日志：分隔线

    dim_h1_h2_combined(seekers, user_ages)  # A1+A2：合并图 节假日vs非节假日 + 节假日vs工作日vs周末
    log("")
    dim_h3_per_holiday_vs_nonholiday_age(seekers, user_ages)  # A3：每个节假日 vs 非节假日
    log("")
    dim_h4_per_holiday_vs_workday_weekend_age(seekers, user_ages)  # A4：每个节假日 vs 工作日和周末

    # Section B: Hourly
    # B 部分：逐小时分析
    log("")
    log("-" * 40)
    log("Section B: Hourly - Age Distribution")
    log("-" * 40)

    dim_i1_hourly_holiday_vs_nonholiday_age(seekers, user_ages)  # B1：逐小时节假日 vs 非节假日
    log("")
    dim_i2_hourly_holiday_workday_weekend_age(seekers, user_ages)  # B2：逐小时三类周期
    log("")
    dim_i3_per_holiday_hourly_age(seekers, user_ages)  # B3：逐小时各个节假日 vs 非节假日
    log("")
    dim_i4_per_holiday_hourly_age_vs_workday_weekend(seekers, user_ages)  # B4：逐小时各个节假日 vs 工作日和周末

    log("")
    log("=" * 60)
    log(f"Step 4 complete! Results saved to {STEP_OUT}")  # 日志：步骤完成
    log("=" * 60)


if __name__ == '__main__':  # 如果该文件作为脚本直接运行
    main()  # 调用主函数入口
