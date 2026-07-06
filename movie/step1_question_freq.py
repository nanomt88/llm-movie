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

全部/周周期 + 单词数分组:
  节假日 VS 非节假日 VS 工作日 VS 周末 单词数分布 (grouped bar)
  各个节假日 VS 非节假日 VS 工作日 VS 周末 单词数分布 (grouped bar)

日周期-小时段 + 访问次数:
  节假日 VS 非节假日 各个时间段(0-24h)平均提问次数对比
  节假日 VS 工作日 VS 周末 各个时间段平均提问次数对比
  各个节假日 VS 非节假日 各个时间段平均提问次数对比 (heatmap)
  各个节假日 VS 工作日 VS 周末 各个时间段平均提问次数对比 (heatmap)

Output: output/movie/step1/*.png + CSV (A5-A6 added for word-length analysis)
输出目录：output/movie/step1/，包含 PNG 图表和 CSV 数据文件（A5-A6 新增单词数分析）
"""

import os  # 操作系统接口，用于路径拼接和目录创建
import re  # 正则表达式，用于文本清理
import csv  # CSV 文件读写，用于保存数值结果
from collections import defaultdict, Counter  # 默认字典和计数器
from datetime import datetime, timezone  # 日期时间与时区处理

import numpy as np  # 数值计算库，用于均值等计算
import matplotlib  # 绘图库

matplotlib.use('Agg')  # 使用 Agg 后端（无 GUI），适用于服务器环境
import matplotlib.pyplot as plt  # pyplot 模块，用于生成图表
import matplotlib.ticker as ticker  # 坐标轴刻度格式化

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log  # 共享配置
from movie.utils.plotting import (annotate_heatmap,  # 热力图标注
                                   COLOR_HOLIDAY, COLOR_NONHOLIDAY,
                                   COLOR_WORKDAY, COLOR_WEEKEND,
                                   HOLIDAY_CMAP)  # 统一配色
from movie.data_loader import (  # 数据加载模块
    load_conversations, load_holiday_definitions, load_holiday_workday_adjustments,
    tag_period,  # 时段标记函数
)

setup_matplotlib()  # 初始化 matplotlib（Agg 后端 + 中文字体）
STEP_OUT = STEP_DIRS[1]  # 步骤1的输出目录：output/movie/step1/
os.makedirs(STEP_OUT, exist_ok=True)  # 确保输出目录存在

# ── Word length bucket definitions（单词数分组定义）─────────────────────
WORD_LEN_BUCKETS = [(1, 10), (11, 30), (31, 100), (101, float('inf'))]
WORD_LEN_LABELS = ['1-10', '11-30', '31-100', '100+']
BUCKET_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']


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
    daily_counts = Counter()  # 每日提问数计数器
    for r in seekers:  # 遍历每个提问
        if r['date'] in date_set:  # 如果该提问在目标日期内
            daily_counts[r['date']] += 1  # 对应日期计数+1
    if not daily_counts:  # 没有数据
        return 0.0
    return np.mean(list(daily_counts.values()))  # 返回每日计数的平均值


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
        'holiday': defaultdict(int),  # 节假日每日提问数
        'workday': defaultdict(int),  # 工作日每日提问数
        'weekend': defaultdict(int),  # 周末每日提问数
        'non_holiday': defaultdict(int),  # 非节假日每日提问数
    }
    for r in seekers:  # 遍历每个提问
        d = r['date']  # 提问日期
        if r['period'] == 'holiday':  # 节假日提问
            groups['holiday'][d] += 1
            groups['non_holiday'][d] += 0  # 确保 non_holiday 中不含此日期
        elif r['period'] == 'workday':  # 工作日提问
            groups['workday'][d] += 1
            groups['non_holiday'][d] += 1
        elif r['period'] == 'weekend':  # 周末提问
            groups['weekend'][d] += 1
            groups['non_holiday'][d] += 1
    return groups


# ═══════════════════════════════════════════════════════════════════════
#  Helper: clean & count English words (for word-length analysis)
#  辅助函数：清理文本并统计英文单词数（供单词数分析使用）
# ═══════════════════════════════════════════════════════════════════════

def _clean_word_count(text: str) -> int:
    """
    Count English words after removing movie IDs, special chars, emojis.
    统计英文单词数，去除电影ID、特殊字符、表情符号等。
    Args:
        text: 原始文本（通常为 proc_text 字段）
    Returns:
        英文单词数（仅保留 a-zA-Z 字符构成的单词）
    """
    if not text:
        return 0
    # Remove movie IDs (tt followed by digits) which appear in proc_text
    text = re.sub(r'tt\d+', ' ', text)
    # Keep only letters and spaces — strips digits, punctuation, emojis
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        return 0
    return len(cleaned.split())


def _word_len_distribution(seekers: list[dict], date_set: set) -> list[float]:
    """
    Compute per-day average questions in each word-length bucket for a date set.
    计算指定日期集合中每个单词数分组的日均提问数。
    Args:
        seekers:  提问者数据行列表
        date_set: 目标日期集合
    Returns:
        [avg_1_10, avg_11_30, avg_31_100, avg_100+] 四个分组的日均提问数
    """
    daily_bucket_counts = defaultdict(lambda: [0, 0, 0, 0])  # date -> [c1, c2, c3, c4]

    for r in seekers:
        if r['date'] not in date_set:
            continue
        wc = _clean_word_count(r.get('proc_text', ''))
        if wc == 0:
            continue
        # Find which bucket this word count falls into
        for i, (lo, hi) in enumerate(WORD_LEN_BUCKETS):
            if lo <= wc <= hi:
                daily_bucket_counts[r['date']][i] += 1
                break

    num_days = len(date_set)
    if num_days == 0:
        return [0.0, 0.0, 0.0, 0.0]

    totals = [0.0, 0.0, 0.0, 0.0]
    for day_buckets in daily_bucket_counts.values():
        for i in range(4):
            totals[i] += day_buckets[i]

    return [t / num_days for t in totals]


'''
# ═══════════════════════════════════════════════════════════════════════
#  A: 节假日 VS 非节假日 平均提问次数 (Pie)
#  A: Holiday vs Non-Holiday Average Daily Questions
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   双面板柱状图（1行2列）
#   - 左图: 节假日(Holiday) vs 非节假日(Non-holiday) 日均提问数对比
#   - 右图: 节假日(Holiday) vs 工作日(Workday) vs 周末(Weekend) 日均提问数对比
# 
# 【统计口径】
#   指标说明:
#     - "日均提问数" = 该组内各天的提问数列表取均值 (np.mean)
#     - 注意: 是"每天提问数的均值", 不是"总提问数/总天数", 前者更能反映典型日水平
#   分组方式:
#     - 按 data_loader 中 tag_period() 打标的 r['period'] 字段分组
#     - holiday: 法定节假日日期
#     - workday: 非节假日的周一至周五
#     - weekend: 非节假日的周六周日
#     - non_holiday: workday + weekend 的并集
# 
# 【坐标轴】
#   X轴: 分组类别（每根柱子对应一个分组）
#   Y轴: 日均提问数（用 MaxNLocator(integer=True) 强制显示整数刻度）
# 
# 【输出文件】
#   PNG: a1_holiday_vs_nonholiday_bar.png
#   CSV: a1_holiday_vs_nonholiday.csv (含: 分组名, 日均提问数, 天数)
# 
# 【特殊说明】
#   - 柱顶用 ax.text() 手动标注数值 (居中对齐, 离柱顶 0.3 单位)
#   - 柱宽 0.5, 白色边框线宽 0.5, alpha=0.8
#   - 配色方案: 假日红(#ff6b6b) / 非假日蓝(#74b9ff) / 工作日黄(#feca57) / 周末青(#48dbfb)
# 
# 【代码中处理逻辑】
#   1. 数据分组阶段
#      遍历 seekers 列表, 根据 r['period'] 取值将日期分入四个 set:
#        holiday_dates     ← 收集 r['period'] == 'holiday' 的 r['date']
#        non_holiday_dates ← 收集 r['period'] != 'holiday' 的 r['date']
#        workday_dates     ← 收集 r['period'] == 'workday' 的 r['date']
#        weekend_dates     ← 收集 r['period'] == 'weekend' 的 r['date']
#      每个 set 中的日期天然去重（set 特性, 同一天多个提问只算一个日期）
# 
#   2. 日均值计算阶段 (核心函数 _avg_daily_questions)
#       输入: seekers 提问行列表, date_set 目标日期集合
#       过程:
#        a) 初始化 Counter() → daily_counts
#        b) 遍历整个 seekers:
#           若 r['date'] in date_set → daily_counts[r['date']] += 1
#        c) 若 daily_counts 为空 → 返回 0.0
#        d) 否则 → np.mean(list(daily_counts.values())) 返回 float 均值
#       示例: 某组日期集合有3天, 每天提问数分别为 [5, 3, 7]
#             则日均值 = (5+3+7)/3 = 5.0, 而非 (5+3+7)/总天数N
# 
#   3. 图表渲染阶段
#       左图 (ax1):
#        - ax1.bar(['Holiday', 'Non-holiday'], [h_avg, nh_avg], color=[红, 蓝])
#        - 遍历 bars, 在每根柱子 x+width/2 位置, y+0.3 位置标注数值
#       右图 (ax2):
#        - ax2.bar(['Holiday', 'Workday', 'Weekend'], [h_avg, wd_avg, we_avg], color=[红, 黄, 青])
#        - 同样 ax.text() 标注
#       两图共享:
#        - set_ylabel('Avg Daily Questions')
#        - grid(axis='y', alpha=0.3) 水平网格线
#        - yaxis.set_major_locator(MaxNLocator(integer=True)) 整数刻度
# 
#   4. 输出保存阶段
#        - fig.savefig(path) 输出 PNG
#        - csv.writer 写入 CSV, 每行: [group, avg_daily_questions, num_days]
# ═══════════════════════════════════════════════════════════════════════════
'''


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

# ═══════════════════════════════════════════════════════════════════════
#  A2: 节假日 VS 工作日 VS 周末 平均提问次数 (Pie)
#  A2: Holiday vs Workday vs Weekend Average Daily Questions
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   独立柱状图（1面板）：节假日(Holiday) vs 工作日(Workday) vs 周末(Weekend) 日均提问数对比
#   注意：目前已合并到 A1 的右图，此处仅输出 CSV，不再生成独立图片
# 
# 【统计口径】
#   指标说明:
#     - "日均提问数" = 该组内各天的提问数列表取均值 (np.mean)
#   分组方式:
#     - holiday: 法定节假日日期（按 r['period'] == 'holiday' 筛选）
#     - workday: 非节假日的周一至周五（按 r['period'] == 'workday' 筛选）
#     - weekend: 非节假日的周六周日（按 r['period'] == 'weekend' 筛选）
# 
# 【坐标轴】
#   X轴: 三个分组标签（Holiday, Workday, Weekend）
#   Y轴: 日均提问数（整数刻度，MaxNLocator）
# 
# 【输出文件】
#   CSV: a2_holiday_workday_weekend.csv (含: 分组名, 日均提问数, 天数)
#   注意: 图片已合并到 A1，不再单独输出 PNG
# 
# 【特殊说明】
#   - 此函数仅保留 CSV 输出，图表已合并到 dim_a1 的右子图
#   - 代码路径与 dim_a1 基本一致，只是分组从2组变为3组
# 
# 【代码中处理逻辑】
#   1. 数据分组阶段
#      遍历 seekers, 根据 r['period'] 取值收集日期到三个 set:
#        holiday_dates ← r['period'] == 'holiday'
#        workday_dates ← r['period'] == 'workday'
#        weekend_dates ← r['period'] == 'weekend'
# 
#   2. 日均值计算
#       对每个 period_dates[p] 调用 _avg_daily_questions(seekers, date_set)
#       该函数内部: Counter() 统计每天提问数 → np.mean(...) 求均值
# 
#   3. 输出 CSV
#       写入 [group, avg_daily_questions, num_days] 三列
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
        'name': '',  # 节假日名称
        'total_questions': 0,  # 总提问数
        'dates': set(),  # 出现该节假日的日期集合
        'daily_counts': defaultdict(int),  # 每天提问数
    })

    for r in seekers:  # 遍历每个提问
        if r['period'] == 'holiday':  # 仅处理节假日提问
            name = r['holiday_name'][:6]  # 取前6个字符作为组名（如 "春节"）
            entry = holiday_stats[name]
            entry['name'] = name
            entry['total_questions'] += 1  # 总提问数+1
            entry['dates'].add(r['date'])  # 记录日期
            entry['daily_counts'][r['date']] += 1  # 每日计数+1

    # Compute avg daily questions per holiday name
    result = []
    for name, data in holiday_stats.items():
        if len(data['dates']) < MIN_DATA_ROWS // 5:  # 如果数据天数太少则跳过
            continue
        daily_vals = list(data['daily_counts'].values())  # 所有每日计数
        data['avg_daily'] = np.mean(daily_vals)  # 日均提问数
        data['num_dates'] = len(data['dates'])  # 有效天数
        result.append(data)

    result.sort(key=lambda x: x['total_questions'], reverse=True)  # 按总提问数降序
    log(f"Aggregated {len(result)} unique holiday names (first 6 chars)", "A3")
    return result


# ═══════════════════════════════════════════════════════════════════════
#  A3: 各个节假日 VS 非节假日 平均提问次数 (Grouped Bar)
#  A3: Per-Holiday vs Non-Holiday Average Daily Questions
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   双面板柱状图（2行1列），纵向排列
#   - 上部分: 各节假日柱状图 + 非节假日基线（红色虚线）
#   - 下部分: 各节假日柱状图 + 工作日/周末基线（黄色/青色虚线）
# 
# 【统计口径】
#   指标说明:
#     - 节假日按名称（取前6个字符）跨年聚合，如 "春节" 合并多年数据
#     - 日均提问数 = 组内每天提问数取均值
#     - 非节假日基线 = 所有非节假日（workday+weekend）日期集合的日均值
#     - 工作日/周末基线分别计算
#   数据过滤:
#     - 节假日数据天数 < MIN_DATA_ROWS//5 时跳过（数据太少不可靠）
# 
# 【坐标轴】
#   X轴: 各节假日名称（旋转45度显示）
#   Y轴: 日均提问数
#   辅助线: 非节假日/工作日/周末基线作为水平虚线
# 
# 【输出文件】
#   PNG: a3_a4_per_holiday_merged.png（合并输出）
#   CSV: a3_per_holiday_vs_nonholiday.csv（A3数据）
#   CSV: a4_per_holiday_vs_workday_weekend.csv（A4数据）
# 
# 【特殊说明】
#   - A3 和 A4 已合并到一个双面板图中输出
#   - _aggregate_holiday_names() 辅助函数完成节假日名称聚合
#   - 基线条用 axhline() 绘制
# 
# 【代码中处理逻辑】
#   1. 节假日聚合阶段 (_aggregate_holiday_names)
#      数据结构: defaultdict(lambda: {'name','total_questions','dates':set,'daily_counts':defaultdict(int)})
#      遍历 seekers, 对 r['period']=='holiday' 的行:
#       - name = r['holiday_name'][:6] 取前6字符
#       - 累加 total_questions, 添加日期到 dates set, 递增 daily_counts[date]
#      筛选: 去除 dates 数量 < MIN_DATA_ROWS//5 的节假日
#      排序: 按 total_questions 降序排列
# 
#   2. 基线计算
#      non_holiday_dates = 所有 r['period']!='holiday' 的日期 set
#      nh_avg = _avg_daily_questions(seekers, non_holiday_dates)
#      类似计算 wd_avg 和 we_avg
# 
#   3. 图表渲染
#      ax1 (上): ax1.bar(x, h_avgs) + ax1.axhline(y=nh_avg)
#      ax2 (下): ax2.bar(x, h_avgs) + ax2.axhline(y=wd_avg) + ax2.axhline(y=we_avg)
#      柱顶标注: ax.text() 显示数值和天数 n
# 
#   4. CSV 输出
#      A3: holiday_name, avg_daily_questions, num_dates, total_questions, non_holiday_baseline
#      A4: 同上 + workday_baseline + weekend_baseline
# ═══════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════
#  A4: 各个节假日 VS 工作日 VS 周末 平均提问次数 (Grouped Bar)
#  A4: Per-Holiday vs Workday vs Weekend Average Daily Questions
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   已合并到 A3 图的下半部分，此处仅输出 CSV
# 
# 【统计口径】
#   节假日按名称聚合，计算日均提问数
#   对比基线: 工作日日均 / 周末日均
# 
# 【输出文件】
#   CSV: a4_per_holiday_vs_workday_weekend.csv
# 
# 【特殊说明】
#   图片输出已在 dim_a3 中完成，本函数仅保留 CSV 输出以保证数据独立可用
# 
# 【代码中处理逻辑】
#   1. 复用 _aggregate_holiday_names() 聚合节假日数据
#   2. 计算 workday_dates / weekend_dates 的日均值
#   3. CSV 写入每行: [holiday_name, avg, num_dates, total, wd_baseline, we_baseline]
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
#  A5: 节假日 VS 非节假日 / VS 工作日周末 平均提问单词数分组 (Grouped Bar)
#  A5: Holiday vs Non-Holiday / vs Workday-Weekend — Word Length
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  A5: 节假日 VS 非节假日 / VS 工作日周末 单词数分布 (Grouped Bar)
#  A5: Holiday vs Non-Holiday / vs Workday-Weekend Word Length Distribution
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   双面板分组柱状图（1行2列）
#   - 左图: Holiday vs Non-holiday 四组单词数分段日均提问数对比
#   - 右图: Holiday vs Workday vs Weekend 四组单词数分段日均提问数对比
# 
# 【统计口径】
#   指标说明:
#     - 单词数: 对 proc_text 清洗后（去电影ID、去特殊字符）的英文单词数
#     - 使用 _clean_word_count(text) 函数清洗和计数
#     - 四个分组: WORD_LEN_BUCKETS = [(1,10), (11,30), (31,100), (101,inf)]
#     - "日均提问数" = 每个分组桶的日均提问数
#   分组方式:
#     - 同 A1: holiday / non_holiday / workday / weekend
# 
# 【坐标轴】
#   X轴: 四个单词数分组标签（1-10, 11-30, 31-100, 100+）
#   Y轴: 日均提问数（每组桶中的日均值）
# 
# 【输出文件】
#   PNG: a5_holiday_vs_nonholiday_length.png
#   CSV: a5_holiday_vs_nonholiday_length.csv
# 
# 【特殊说明】
#   - 单词数 = 0 的提问被跳过
#   - 每个桶的配色方案: 红(#e41a1c) / 蓝(#377eb8) / 绿(#4daf4a) / 紫(#984ea3)
#   - 柱顶标注数值（小字体 8pt/7pt）
# 
# 【代码中处理逻辑】
#   1. 日期收集与 A1 完全相同: 四个 set
# 
#   2. 单词数分布计算 (_word_len_distribution)
#      输入: seekers + date_set
#      数据结构: daily_bucket_counts = defaultdict(lambda: [0,0,0,0])
#       键=日期, 值=[c1_10, c11_30, c31_100, c100+] 四元素列表
#      处理流程:
#       a) 遍历 seekers, 仅处理 r['date'] in date_set 的行
#       b) 调用 _clean_word_count(r.get('proc_text','')) 得到 wc
#       c) 跳过 wc==0 的行
#       d) 遍历 WORD_LEN_BUCKETS 找到所属桶, daily_bucket_counts[date][i] += 1
#       e) 遍历所有日期, 累加每个桶的值, 最后除以 date_set 大小得到日均值
#      返回: [avg_1_10, avg_11_30, avg_31_100, avg_100+]
# 
#   3. 辅助函数 _clean_word_count
#      a) re.sub(r'tt\d+', ' ', text) — 移除电影ID "tt1234567"
#      b) re.sub(r'[^a-zA-Z\s]', ' ', text) — 只保留字母和空格
#      c) re.sub(r'\s+', ' ', cleaned).strip() — 合并多余空格
#      d) len(cleaned.split()) — 按空格拆分计单词数
# 
#   4. 图表渲染
#      左图: 2组 × 4桶 = 8根柱子, 错位排列 (width_2=0.35)
#      右图: 3组 × 4桶 = 12根柱子, 错位排列 (width_3=0.25)
#      柱顶标注 f'{v:.1f}' 小字体
# ═══════════════════════════════════════════════════════════════════════

def dim_a5_holiday_length(seekers: list[dict]):
    """
    Two-panel bar chart: left=holiday vs non-holiday, right=holiday vs workday vs weekend,
    showing avg daily questions by word-length groups (1-10, 11-30, 31-100, 100+).
    双面板柱状图：左=节假日 vs 非节假日，右=节假日 vs 工作日 vs 周末
    展示各单词数分组的日均提问数。
    Args:
        seekers: 提问者数据行列表
    """
    log("=" * 50)
    log("A5: Holiday vs Non-Holiday vs Workday vs Weekend — Word Length Distribution")

    holiday_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    workday_dates = set(r['date'] for r in seekers if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seekers if r['period'] == 'weekend')

    h_dist = _word_len_distribution(seekers, holiday_dates)
    nh_dist = _word_len_distribution(seekers, non_holiday_dates)
    wd_dist = _word_len_distribution(seekers, workday_dates)
    we_dist = _word_len_distribution(seekers, weekend_dates)

    log(f"  Holiday: {[f'{v:.2f}' for v in h_dist]}")
    log(f"  Non-holiday: {[f'{v:.2f}' for v in nh_dist]}")
    log(f"  Workday: {[f'{v:.2f}' for v in wd_dist]}")
    log(f"  Weekend: {[f'{v:.2f}' for v in we_dist]}")
    log(f"  Holiday dates: {len(holiday_dates)}, Non-holiday: {len(non_holiday_dates)}, "
        f"Workday: {len(workday_dates)}, Weekend: {len(weekend_dates)}")

    x = np.arange(4)
    width_2 = 0.35
    width_3 = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: Holiday vs Non-holiday ──
    b1 = ax1.bar(x - width_2 / 2, h_dist, width_2, label='Holiday',
                 color=COLOR_HOLIDAY, alpha=0.85, edgecolor='white', linewidth=0.5)
    b2 = ax1.bar(x + width_2 / 2, nh_dist, width_2, label='Non-holiday',
                 color=COLOR_NONHOLIDAY, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, v in zip(b1, h_dist):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    for bar, v in zip(b2, nh_dist):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(WORD_LEN_LABELS)
    ax1.set_ylabel('Avg Daily Questions')
    ax1.set_title('Holiday vs Non-Holiday', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # ── Right: Holiday vs Workday vs Weekend ──
    b3 = ax2.bar(x - width_3, h_dist, width_3, label='Holiday',
                 color=COLOR_HOLIDAY, alpha=0.85, edgecolor='white', linewidth=0.5)
    b4 = ax2.bar(x, wd_dist, width_3, label='Workday',
                 color=COLOR_WORKDAY, alpha=0.85, edgecolor='white', linewidth=0.5)
    b5 = ax2.bar(x + width_3, we_dist, width_3, label='Weekend',
                 color=COLOR_WEEKEND, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, v in zip(b3, h_dist):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    for bar, v in zip(b4, wd_dist):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    for bar, v in zip(b5, we_dist):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(WORD_LEN_LABELS)
    ax2.set_ylabel('Avg Daily Questions')
    ax2.set_title('Holiday vs Workday vs Weekend', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Avg Daily Questions by Word Count Group', fontsize=13)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'a5_holiday_vs_nonholiday_length.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'a5_holiday_vs_nonholiday_length.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'num_days', 'bucket_1_10', 'bucket_11_30',
                    'bucket_31_100', 'bucket_100+'])
        for label, nd, dist in [
            ('holiday', len(holiday_dates), h_dist),
            ('non_holiday', len(non_holiday_dates), nh_dist),
            ('workday', len(workday_dates), wd_dist),
            ('weekend', len(weekend_dates), we_dist),
        ]:
            w.writerow([label, nd] + [f'{v:.4f}' for v in dist])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  A6: 各个节假日 VS 非节假日 / VS 工作日周末 单词数分布 (Grouped Bar)
#  A6: Per-Holiday vs Non-Holiday / vs Workday-Weekend — Word Length
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  A6: 各个节假日 VS 非节假日 / VS 工作日周末 单词数分布 (Grouped Bar)
#  A6: Per-Holiday vs Non-Holiday / vs Workday-Weekend Word Length
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   双面板分组柱状图（2行1列）
#   - 上部分: 各节假日在四个单词数分段的日均提问数 + 非节假日基线（虚线）
#   - 下部分: 各节假日在四个单词数分段的日均提问数 + 工作日/周末基线（点线）
# 
# 【统计口径】
#   同 A5: 单词数分四个桶计算日均提问数
#   节假日名称同 A3/A4 方式聚合: 取前6字符
#   基线: 非节假日/工作日/周末分别计算四桶的日均值
# 
# 【坐标轴】
#   X轴: 各节假日名称（旋转45度）
#   Y轴: 日均提问数
#   每个节假日显示4根柱子（4种颜色对应4个桶）
# 
# 【输出文件】
#   PNG: a6_per_holiday_length.png
#   CSV: a6_per_holiday_length.csv (含节假日+基线完整数据)
# 
# 【特殊说明】
#   - 使用来自 matplotlib 的 Patch 和 Line2D 构建自定义图例
#   - 基线用虚线(上部: --) 和点线(下部: : 和 -.) 区分
#   - 柱宽 0.18，每节假日4根柱子紧密排列
# 
# 【代码中处理逻辑】
#   1. 节假日聚合: 复用 _aggregate_holiday_names()
#      对每个节假日, 调用 _word_len_distribution(seekers, h['dates'])
#      得到 h_dists: list of [4-element dist] per holiday
# 
#   2. 基线计算
#      non_holiday_dates / workday_dates / weekend_dates 分别计算四桶分布
#      nh_dist/wd_dist/we_dist = _word_len_distribution(seekers, date_set)
# 
#   3. 图表渲染 (双面板)
#      每面板: 对 i=0..3 四个桶循环, 每组4根柱子
#        ax.bar(x + (i-1.5)*width, vals, width, label=WORD_LEN_LABELS[i], color=BUCKET_COLORS[i])
#      基线: axhline(y=dist[i], color=BUCKET_COLORS[i], linestyle=...)
#      图例: 用 Patch(颜色块) + Line2D(线) 组合显示
# 
#   4. CSV 输出
#      每行: [holiday_name, num_dates, b_1_10, b_11_30, b_31_100, b_100+,
#             nh_1_10, ..., wd_1_10, ..., we_1_10, ...]
#      一行包含节假日自身 + 三种基线的完整四桶数据
# ═══════════════════════════════════════════════════════════════════════

def dim_a6_per_holiday_length(seekers: list[dict]):
    """
    Two-panel figure: top=per-holiday vs non-holiday, bottom=per-holiday vs workday & weekend,
    showing word-length distribution across 4 buckets.
    For each holiday, 4 grouped bars (one per bucket) with baseline reference lines.
    双面板图：上=各节假日 vs 非节假日，下=各节假日 vs 工作日&周末 单词数分布对比。
    Args:
        seekers: 提问者数据行列表
    """
    log("=" * 50)
    log("A6: Per-Holiday Word Length Distribution — (top: vs Non-Holiday, bottom: vs Workday/Weekend)")

    holiday_agg = _aggregate_holiday_names(seekers)
    if not holiday_agg:
        log("  WARN: No holiday data")
        return

    # Baselines
    non_holiday_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    workday_dates = set(r['date'] for r in seekers if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seekers if r['period'] == 'weekend')
    nh_dist = _word_len_distribution(seekers, non_holiday_dates)
    wd_dist = _word_len_distribution(seekers, workday_dates)
    we_dist = _word_len_distribution(seekers, weekend_dates)

    # Per-holiday distributions
    names = []
    h_dists = []  # list of [4-element dist] per holiday
    for h in holiday_agg:
        names.append(h['name'])
        h_dists.append(_word_len_distribution(seekers, h['dates']))

    num_h = len(names)
    log(f"  {num_h} holidays")
    log(f"  Baselines — nh: {[f'{v:.2f}' for v in nh_dist]}, "
        f"wd: {[f'{v:.2f}' for v in wd_dist]}, "
        f"we: {[f'{v:.2f}' for v in we_dist]}")

    x = np.arange(num_h)
    width = 0.18  # 4 bars per holiday group

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, num_h * 0.6), 10))

    # ── Top: Per-holiday vs Non-holiday ──
    for i in range(4):
        vals = [d[i] for d in h_dists]
        ax1.bar(x + (i - 1.5) * width, vals, width,
                label=WORD_LEN_LABELS[i], color=BUCKET_COLORS[i],
                alpha=0.85, edgecolor='white', linewidth=0.3)
    # Non-holiday baselines as dashed lines
    for i in range(4):
        ax1.axhline(y=nh_dist[i], color=BUCKET_COLORS[i], linestyle='--',
                    linewidth=1.5, alpha=0.6)
    # Add a custom legend entry for baselines
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = (
            [Patch(facecolor=BUCKET_COLORS[i], alpha=0.85, label=WORD_LEN_LABELS[i])
             for i in range(4)]
            + [Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Non-holiday baseline')]
    )
    ax1.legend(handles=legend_elements, fontsize=8, ncol=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Avg Daily Questions')
    ax1.set_title('Per-Holiday Word Length Distribution vs Non-Holiday (dashed)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # ── Bottom: Per-holiday vs Workday & Weekend ──
    for i in range(4):
        vals = [d[i] for d in h_dists]
        ax2.bar(x + (i - 1.5) * width, vals, width,
                label=WORD_LEN_LABELS[i], color=BUCKET_COLORS[i],
                alpha=0.85, edgecolor='white', linewidth=0.3)
    for i in range(4):
        ax2.axhline(y=wd_dist[i], color=BUCKET_COLORS[i], linestyle=':',
                    linewidth=1.5, alpha=0.6)
        ax2.axhline(y=we_dist[i], color=BUCKET_COLORS[i], linestyle='-.',
                    linewidth=1.5, alpha=0.6)
    legend_elements2 = (
            [Patch(facecolor=BUCKET_COLORS[i], alpha=0.85, label=WORD_LEN_LABELS[i])
             for i in range(4)]
            + [Line2D([0], [0], color='gray', linestyle=':', linewidth=1.5, label='Workday baseline'),
               Line2D([0], [0], color='gray', linestyle='-.', linewidth=1.5, label='Weekend baseline')]
    )
    ax2.legend(handles=legend_elements2, fontsize=8, ncol=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Avg Daily Questions')
    ax2.set_title('Per-Holiday Word Length Distribution vs Workday(:) / Weekend(-.) Baseline', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-Holiday Avg Daily Questions by Word Count Group', fontsize=14)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'a6_per_holiday_length.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'a6_per_holiday_length.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'num_dates',
                    'b_1_10', 'b_11_30', 'b_31_100', 'b_100+',
                    'nh_1_10', 'nh_11_30', 'nh_31_100', 'nh_100+',
                    'wd_1_10', 'wd_11_30', 'wd_31_100', 'wd_100+',
                    'we_1_10', 'we_11_30', 'we_31_100', 'we_100+'])
        for name, dist, holiday in zip(names, h_dists, holiday_agg):
            w.writerow([name, holiday['num_dates'],
                        f'{dist[0]:.4f}', f'{dist[1]:.4f}', f'{dist[2]:.4f}', f'{dist[3]:.4f}',
                        f'{nh_dist[0]:.4f}', f'{nh_dist[1]:.4f}', f'{nh_dist[2]:.4f}', f'{nh_dist[3]:.4f}',
                        f'{wd_dist[0]:.4f}', f'{wd_dist[1]:.4f}', f'{wd_dist[2]:.4f}', f'{wd_dist[3]:.4f}',
                        f'{we_dist[0]:.4f}', f'{we_dist[1]:.4f}', f'{we_dist[2]:.4f}', f'{we_dist[3]:.4f}'])
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
    if not date_set:  # 空日期集合返回全0
        return [0.0] * 24

    # For each date, count questions per hour
    date_hour_counts = defaultdict(lambda: defaultdict(int))  # 日期 -> 小时 -> 计数
    for r in seekers:
        if r['date'] in date_set:
            date_hour_counts[r['date']][r['hour']] += 1  # 该日期该小时计数+1

    # Average across all dates in set（跨所有日期求平均）
    hourly_totals = [0.0] * 24
    num_dates = len(date_set)
    if num_dates == 0:
        return hourly_totals

    for date_key in date_set:  # 遍历每个日期
        for h in range(24):  # 遍历24小时
            hourly_totals[h] += date_hour_counts[date_key].get(h, 0)  # 累加每小时计数

    return [t / num_dates for t in hourly_totals]  # 除以天数得到均值


def _plot_hourly_comparison(
        hourly_data: dict[str, list[float]],  # 传入数据：标签 -> 24小时数据列表
        title: str,  # 图表标题
        filename: str,  # 保存文件名
        colors: dict[str, str],  # 标签 -> 颜色映射
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
    hours = list(range(24))  # x轴：0-23小时

    for label, values in hourly_data.items():  # 为每个标签画一条折线
        ax.plot(hours, values, 'o-', label=label, color=colors.get(label),
                linewidth=2, markersize=4, alpha=0.85)  # 'o-' 表示带圆点的折线

    ax.set_xlabel('Hour of Day (UTC)')  # x轴：一天中的小时（UTC时区）
    ax.set_ylabel('Avg Questions per Hour per Day')  # y轴：每小时每日平均提问数
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(0, 24, 2))  # x轴刻度：每2小时一个
    ax.legend(fontsize=10)  # 图例
    ax.grid(axis='y', alpha=0.3)  # y轴网格线
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # y轴取整

    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  B1: 节假日 VS 非节假日 / VS 工作日周末 小时段提问量 (折线图)
#  B1: Hourly Question Count: Holiday vs Non-holiday
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   双面板折线图（2行1列）
#   - 上部分: Holiday vs Non-holiday 24小时段提问数曲线对比
#   - 下部分: Holiday vs Workday vs Weekend 24小时段提问数曲线对比
# 
# 【统计口径】
#   - 按小时段汇总: 对每行 r['proc_time'], 提取 HH 部分作为 hour_key
#   - 分 holiday/non_holiday/workday/weekend 四组
#   - 使用 _hourly_stats() 辅助函数计算各组 24 小时的均值
#   - 跳过 proc_time 为空的行
# 
# 【坐标轴】
#   X轴: 0-23 小时段
#   Y轴: 平均提问数
#   图例: 两组（holiday/非节假日 或 holiday/workday/weekend）
# 
# 【输出文件】
#   PNG: b1_b2_hourly_holiday_merged.png（与 B2 合并输出）
#   CSV: b1_hourly_holiday_vs_nonholiday.csv
# 
# 【特殊说明】
#   - 已与 B2 合并到一个双面板图中，B1 在上、B2 在下
#   - 24小时制: X周从0到23标记
#   - 用不同的颜色和风格区分组别（实线/虚线/点线等）
# 
# 【代码中处理逻辑】
#   1. 调用 _hourly_stats() 分别计算 hourly_non_holiday 和 hourly_holiday
#      返回: list of 24 小时均值
# 
#   2. _hourly_stats(seekers, date_set, timerange=(0,24))
#      内部: 遍历 date_set 中的各天，统计每个小时出现的次数
#      → 求所有天数的逐小时均值
# 
#   3. 上面板: 两条曲线
#      x = np.arange(24)
#      ax1.plot(x, h_vals, 'o-', color='#e41a1c', label='Holiday')
#      ax1.plot(x, nh_vals, 's--', color='#377eb8', label='Non-Holiday')
# 
#   4. CSV 输出: [hour, holiday_avg, non_holiday_avg]
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
#  B2: 节假日 VS 工作日 VS 周末 小时段 (已合并到 B1)
#  B2: Holiday vs Workday vs Weekend Hourly (merged into B1)
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   图表已合并到 B1 的下面板，此处仅输出 CSV
# 
# 【统计口径】
#   同 B1: 24小时段，分 holiday/workday/weekend 三组
# 
# 【输出文件】
#   CSV: b2_hourly_holiday_workday_weekend.csv
# 
# 【特殊说明】
#   图片输出已在 dim_b1 中完成，本函数仅保留 CSV 输出
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
#  B3: 各个节假日 VS 非节假日 小时段 (Heatmap)
#  B3: Per-Holiday Hourly Difference: Holiday minus Non-Holiday
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   热力图（Heatmap）
#   - 行: 各节假日名称
#   - 列: 0-23 小时
#   - 颜色值: 节假日小时均值 - 非节假日小时基线
# 
# 【统计口径】
#   节假日聚合同 A3/A6: 取前6字符
#   非节假日基线 = _hourly_avg(seekers, non_holiday_dates)
#   每个节假日的 24 小时均值 = _hourly_avg(seekers, h_dates)
#   差值 = h_avg[h] - nh_avg[h] (可能为负数)
# 
# 【坐标轴】
#   X轴: 0-23 小时
#   Y轴: 节假日名称
#   颜色条: 差值（symlog 处理宽动态范围，红色=正/蓝=负）
# 
# 【输出文件】
#   PNG: b3_b4_per_holiday_hourly_merged.png（B3 在上面板，B4 在下面板）
#   CSV: b3_per_holiday_hourly_vs_nonholiday.csv
# 
# 【特殊说明】
#   - 已与 B4 合并输出到一个双面板图中
#   - 使用 sns.heatmap 渲染
#   - symlog 映射处理正负值和宽范围的差异
# 
# 【代码中处理逻辑】
#   1. 聚合节假日数据 (_aggregate_holiday_names → _hourly_avg 计算)
#   2. 非节假日基线计算
#   3. DataFrame 构建: rows=holidays, cols=h0..h23
#   4. 差值计算: diff = holiday_avg - non_holiday_baseline
#   5. sns.heatmap(diff_df, ...) 渲染
# ═══════════════════════════════════════════════════════════════════════

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
            name = r['holiday_name'][:6]  # 取前6个字符作为组名
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
    matrix = np.zeros((len(group_names), 24))  # 矩阵：行=节假日，列=小时

    for i, name in enumerate(group_names):
        group_dates = set(r['date'] for r in holiday_groups[name])  # 该节假日的日期集合
        h_hourly = _hourly_avg(holiday_groups[name], group_dates)  # 该节假日的逐小时均值
        for h in range(24):
            # Difference: holiday avg - non-holiday avg（节假日均值减去非节假日基线）
            matrix[i, h] = h_hourly[h] - nh_hourly[h]

    # Heatmap（绘制热力图）
    fig, ax = plt.subplots(figsize=(16, max(6, len(group_names) * 0.4 + 2)))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)  # 最大绝对值，用于对称色阶
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    annotate_heatmap(ax, matrix, fmt='.1f', fs=6)
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


# ═══════════════════════════════════════════════════════════════════════
#  B4: 各个节假日 VS 工作日 VS 周末 小时段 (Dual Heatmap)
#  B4: Per-Holiday Hourly vs Workday & Weekend Dual Heatmap
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   双热力图（B3 已在上面板，B4 在下面板 - 含两张热力图）
#   第一张: 每小时(holiday - workday) 差值
#   第二张: 每小时(holiday - weekend) 差值
# 
# 【统计口径】
#   同 B3 逻辑计算差值
#   workday_baseline / weekend_baseline = _hourly_avg() 计算
# 
# 【坐标轴】
#   X轴: 0-23 小时
#   Y轴: 节假日名称
#   颜色条: 差值（symlog）
# 
# 【输出文件】
#   PNG: 已合并到 B3 的 b3_b4_per_holiday_hourly_merged.png（下方两张热力图）
#   CSV: b4_per_holiday_hourly_vs_workday_weekend.csv
# 
# 【特殊说明】
#   图片已合并到 B3 输出，本函数仅保留 CSV 输出
# ═══════════════════════════════════════════════════════════════════════

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
    wd_hourly = _hourly_avg(seekers, workday_dates)  # 工作日逐小时基线
    we_hourly = _hourly_avg(seekers, weekend_dates)  # 周末逐小时基线

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
    annotate_heatmap(ax1, matrix_wd, fmt='.1f', fs=6)
    ax1.set_xticks(range(24))
    ax1.set_xticklabels(range(24), fontsize=7)
    ax1.set_yticks(range(len(group_names)))
    ax1.set_yticklabels(group_names, fontsize=7)
    ax1.set_title('Diff: Holiday Avg - Workday Baseline', fontsize=10)
    fig.colorbar(im1, ax=ax1, shrink=0.5, label='Diff')

    # 下半图：与周末差值
    vmax2 = max(abs(matrix_we.min()), abs(matrix_we.max()), 0.01)
    im2 = ax2.imshow(matrix_we, cmap='RdBu_r', aspect='auto', vmin=-vmax2, vmax=vmax2)
    annotate_heatmap(ax2, matrix_we, fmt='.1f', fs=6)
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

def main(data: dict = None):
    """Main entry point for Step 1: load data, run all analysis dimensions.
       步骤1主入口：加载数据，运行所有分析维度。"""
    log("=" * 60)
    log("Step 1: Question Frequency & Hourly Access Analysis")
    log("=" * 60)

    if data is None:
        # Load data（加载数据）
        from movie.data_loader import load_all  # 导入数据加载函数
        data = load_all()  # 加载所有数据
    seekers = data['seekers']  # 提取提问者数据

    # ── Section A: Weekly period question frequency ──
    # 周周期：按天统计的提问频率
    log("")
    log("-" * 40)
    log("Section A: Weekly Period - Question Frequency")
    log("-" * 40)

    dim_a1_holiday_vs_nonholiday_pie(seekers)  # A1: 节假日vs非节假日 / 节假日vs工作日vs周末 日均提问数对比 (柱状图)
    log("")
    dim_a2_holiday_workday_weekend_pie(seekers)  # A2: 节假日vs工作日vs周末 日均提问数对比 (仅CSV, 已合并到A1)
    log("")
    dim_a3_per_holiday_vs_nonholiday(seekers)  # A3: 各节假日vs非节假日 / vs工作日&周末 日均提问数对比 (柱状图+基线)
    log("")
    dim_a4_per_holiday_vs_workday_weekend(seekers)  # A4: 各节假日vs工作日&周末 日均提问数对比 (仅CSV, 已合并到A3)

    # ── Section A5-A6: Word length analysis ──
    # 周周期：提问单词数分组分析
    log("")
    log("-" * 40)
    log("Section A5-A6: Weekly Period - Word Length Distribution")
    log("-" * 40)

    dim_a5_holiday_length(seekers)  # A5: 节假日vs非节假日/工作日/周末 单词数(1-10/11-30/31-100/100+)分布对比 (分组柱状图)
    log("")
    dim_a6_per_holiday_length(seekers)  # A6: 各节假日单词数分布 vs 非节假日/工作日/周末基线 (双面板分组柱状图+基线)

    # ── Section B: Hourly access frequency ──
    # 日周期：按小时统计的访问频率
    log("")
    log("-" * 40)
    log("Section B: Hourly Period - Access Frequency")
    log("-" * 40)

    dim_b1_hourly_holiday_vs_nonholiday(seekers)  # B1: 节假日vs非节假日 / 节假日vs工作日vs周末 逐小时(0-23h)平均提问数 (折线图)
    log("")
    dim_b2_hourly_holiday_workday_weekend(seekers)  # B2: 节假日vs工作日vs周末 逐小时平均提问数 (仅CSV, 已合并到B1)
    log("")
    dim_b3_per_holiday_hourly_vs_nonholiday(seekers)  # B3: 各节假日逐小时 vs 非节假日基线 差值热力图
    log("")
    dim_b4_per_holiday_hourly_vs_workday_weekend(seekers)  # B4: 各节假日逐小时 vs 工作日/周末基线 双差值热力图

    log("")
    log("=" * 60)
    log(f"Step 1 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()  # 独立运行时执行 main
