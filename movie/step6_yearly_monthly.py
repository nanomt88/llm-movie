# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文注释

"""
Step 6: Yearly Daily Question Count Analysis
步骤6：年度逐日提问量分析

对比各年份间每天的访问次数差异，以折线图按天绘制。
横轴：月份（仅标出每月1号）
纵轴：每日提问数
每年一条折线，按年天叠加展示，展示每年每天的访问次数差异。

Output: output/movie/step6/*.png + CSV
输出目录：output/movie/step6/
"""

import os                          # 操作系统接口，路径和目录操作
import csv                         # CSV 文件读写
from collections import defaultdict  # 默认字典
from datetime import datetime, timedelta  # 日期时间处理
import calendar                    # 日历工具（判断闰年）

import numpy as np                 # 数值计算
import matplotlib                  # 绘图库
matplotlib.use('Agg')              # 使用 Agg 后端（无 GUI）
import matplotlib.pyplot as plt    # pyplot
import matplotlib.ticker as ticker # 刻度格式化

from movie.config import STEP_DIRS, setup_matplotlib, log  # 配置
from movie.data_loader import load_all                      # 数据加载


setup_matplotlib()                        # 初始化 matplotlib（后端+字体）
STEP_OUT = STEP_DIRS[6]                   # 步骤6输出目录：output/movie/step6/
os.makedirs(STEP_OUT, exist_ok=True)      # 确保输出目录存在


# ── Color palette for years（年份配色方案）─────────────────────────────
# 使用一组易于区分的颜色，每个年份分配一种
YEAR_COLORS = {
    2018: '#1f77b4',   # 蓝色
    2019: '#ff7f0e',   # 橙色
    2020: '#2ca02c',   # 绿色
    2021: '#d62728',   # 红色
    2022: '#9467bd',   # 紫色
}


def _daily_questions_by_year(seekers: list[dict]) -> dict[int, dict[str, int]]:
    """
    For each year, map every date to its question count (0 for missing dates).
    对于每个年份，建立每天 -> 提问数的映射（缺失日期补 0）。

    Returns:
        {year: {date_str: question_count}}
        {年份: {日期字符串: 提问数}}
    """
    # Count questions per date
    # 统计每个日期的提问数
    date_counts: dict[str, int] = defaultdict(int)
    for r in seekers:                             # 遍历每条用户提问
        date_counts[r['date']] += 1                # 该日期提问数 +1

    # Extract year ranges from data
    # 从数据中提取年份范围
    all_dates = sorted(date_counts.keys())
    if not all_dates:
        return {}
    start_year = int(all_dates[0][:4])
    end_year = int(all_dates[-1][:4])

    # Build full daily series per year (including 0-count days)
    # 为每个年份构建完整的逐日序列（含无数据的天，补 0）
    result: dict[int, dict[str, int]] = {}
    for y in range(start_year, end_year + 1):
        days_in_year = 366 if calendar.isleap(y) else 365  # 当年天数
        year_data: dict[str, int] = {}
        for doy in range(1, days_in_year + 1):
            dt = datetime(y, 1, 1) + timedelta(days=doy - 1)
            date_str = dt.strftime('%Y-%m-%d')
            year_data[date_str] = date_counts.get(date_str, 0)  # 有数据取数据，无数据补 0
        result[y] = year_data

    return result


def dim_y1_yearly_monthly_questions(seekers: list[dict]):
    """
    Line chart: yearly daily question counts (one line per year, day-by-day).
    折线图：各年份每天的提问数对比，按天绘制，每年一条线。
    Args:
        seekers: 提问者数据行列表
    """
    log("=" * 50)
    log("Y1: Yearly Daily Question Counts")

    # Compute per-date counts per year
    yearly_data = _daily_questions_by_year(seekers)

    if not yearly_data:
        log("  No data available")
        return

    years = sorted(yearly_data.keys())

    # Log summary per year
    for y in years:
        all_counts = list(yearly_data[y].values())
        total = sum(all_counts)
        days = len([c for c in all_counts if c > 0])
        log(f"  {y}: {total} questions across {days} days with activity, "
            f"{len(all_counts)} total days")

    # ── Line Chart ──────────────────────────────────────────────────
    # 折线图（按年叠加，x 轴 = 当年第几天 1~365/366）
    fig, ax = plt.subplots(figsize=(14, 6))

    # Month start positions on day-of-year axis (non-leap reference)
    # 各月 1 号在年天轴上的位置（以平年为基准）
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for y in years:
        date_strs = sorted(yearly_data[y].keys())
        # Convert date strings to day-of-year (1-based)
        # 将日期字符串转为年天序号（1 起）
        doy_list = []
        counts = []
        for d in date_strs:
            dt = datetime.strptime(d, '%Y-%m-%d')
            doy = dt.timetuple().tm_yday
            doy_list.append(doy)
            counts.append(yearly_data[y][d])

        color = YEAR_COLORS.get(y, '#333333')
        ax.plot(doy_list, counts, '-', color=color, linewidth=0.8,
                label=str(y), alpha=0.85)

    # X-axis: month-start ticks (aligned to day-of-year space)
    # x 轴：仅显示每月 1 号
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_xlim(1, 366)
    ax.set_xlabel('Month')
    ax.set_ylabel('Daily Questions')
    ax.set_title('Daily Questions by Year (Day-of-Year Overlay)', fontsize=13)
    ax.legend(title='Year', fontsize=10, title_fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'y1_yearly_monthly_questions.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(STEP_OUT, 'y1_yearly_monthly_questions.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['year', 'date', 'day_of_year', 'questions'])
        for y in years:
            for date_str in sorted(yearly_data[y].keys()):
                cnt = yearly_data[y][date_str]
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                doy = dt.timetuple().tm_yday
                w.writerow([y, date_str, doy, cnt])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main  主函数入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for Step 6: load data, run yearly daily analysis.
       步骤6主入口：加载数据，运行年度逐日分析。"""
    log("=" * 60)
    log("Step 6: Yearly Daily Question Count Analysis")
    log("=" * 60)

    data = load_all()                              # 加载所有数据
    seekers = data['seekers']                      # 获取用户提问记录列表

    dim_y1_yearly_monthly_questions(seekers)       # Y1：年度逐日提问量

    log("")
    log("=" * 60)
    log(f"Step 6 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':                         # 如果该文件作为脚本直接运行
    main()                                         # 调用主函数入口
