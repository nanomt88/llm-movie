# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文等非 ASCII 字符
"""
Step 5: Movie Genre Analysis
步骤 5：电影类型分析

全部/周周期 + 影片类型:
  - 节假日 VS 非节假日 各类型影片被提及频次对比
  - 节假日 VS 工作日 VS 周末 各类型影片被提及频次对比
  - 各个节假日 VS 非节假日 各类型影片被提及频次对比
  - 各个节假日 VS 工作日 VS 周末 各类型影片被提及频次对比
  - 各个节假日 × 各类型影片 提及次数热力图

日周期-小时段 + 影片类型 (0-24h):
  - 节假日 VS 非节假日 各时间段各类型影片被提及频次对比
  - 节假日 VS 工作日 VS 周末 各时间段各类型影片被提及频次对比
  - 各个节假日 VS 非节假日 各时间段各类型影片被提及频次对比
  - 各个节假日 VS 工作日 VS 周末 各时间段各类型影片被提及频次对比

Genre is extracted from system replies in the same conversation (matching conv_id prefix).
影片类型从同一会话中的系统回复中提取（基于 conv_id 前缀匹配）。
Output: output/movie/step5/*.png + CSV
输出：output/movie/step5/ 目录下的 PNG 图片和 CSV 文件
"""

import os  # 操作系统接口，用于文件和路径操作
import csv  # CSV 文件读写模块
import re  # 正则表达式模块，用于字符串匹配
from collections import defaultdict, Counter  # defaultdict：带默认值的字典；Counter：计数工具

import numpy as np  # 数值计算库，用于数组和矩阵运算
import pandas as pd  # 数据分析库，用于透视表和数据处理
import seaborn as sns  # 统计可视化库，用于热力图绘制
import matplotlib  # 数据可视化基础库
matplotlib.use('Agg')  # 使用非交互式后端 Agg（适用于无 GUI 环境的图片生成）
import matplotlib.pyplot as plt  # pyplot 接口，用于绘制图表
import matplotlib.ticker as ticker  # 坐标轴刻度格式控制

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log  # 导入配置：步骤输出目录、最小数据行数、matplotlib 初始化、日志函数
from movie.utils.plotting import (annotate_heatmap,
                                   COLOR_HOLIDAY, COLOR_NONHOLIDAY,
                                   COLOR_WORKDAY, COLOR_WEEKEND)  # 统一配色

setup_matplotlib()  # 初始化 matplotlib 样式（字体等）
STEP_OUT = STEP_DIRS[5]  # 步骤 5 的输出目录路径（output/movie/step5/）
os.makedirs(STEP_OUT, exist_ok=True)  # 创建输出目录（如果已存在则不报错）


TT_PATTERN = re.compile(r'\b(tt\d+)\b')  # 正则：匹配 IMDb 电影 ID（tt 后跟数字，如 tt1234567）

# Genre color palette (up to ~20 genres)
# 影片类型颜色调色板（最多支持约 20 种类型）
GENRE_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',  # 红、蓝、绿、橙、紫
    '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#16a085',  # 青、橙黄、深蓝、黄、深青
    '#c0392b', '#2980b9', '#27ae60', '#d35400', '#8e44ad',  # 深红、深蓝、深绿、深橙、深紫
    '#2c3e50', '#d4ac0d', '#7f8c8d', '#95a5a6', '#bdc3c7',  # 深灰、金、灰、浅灰、银灰
]

# Top N genres to show
# 显示前 N 个最热门的影片类型
TOP_N_GENRES = 20


# ═══════════════════════════════════════════════════════════════════════
#  Helper: extract genres for each user seeker record
#  辅助函数：为每条用户提问记录提取影片类型
# ═══════════════════════════════════════════════════════════════════════
# 【功能】
#   遍历所有行（包括用户提问和系统回复），为每个用户提问记录找到关联的电影类型
#   电影类型信息存储在 movie_info 字典中（从 movie_info.csv 加载）
#   关联方式:
#     - conv_id 格式: {session_id}_{current_turn}/{total_turns}，如 t3_rt7enj_1/14
#     - 从 conv_id 中解析 session_id（最后一个 _ 之前）和 turn_num（/ 之前）
#     - 系统回复与用户提问按 (session_id, turn_num) 精确配对
#     - 只取同一轮次的系统回复，不会跨轮次污染
#     - 从系统回复的 processed_raw 中提取 tt 开头的电影 ID
#     - 用电影 ID 在 movie_info 中查找对应的 genres 列表
# 
# 【数据结构】
#   conv_system: dict[tuple[str, str], list[str]]
#     键=(session_id, turn_num) — 精确到会话的某一轮次
#     值=该轮次系统回复的 processed_raw 文本列表（通常只有一条）
# 
#   seeker_genres 输出:
#     在原始记录上添加 'genres' 字段（set 类型）
#     如果未找到任何类型，默认为 {'unknown'}
#     每条原始记录（一个 seeker 行） → 一条输出记录（genres 追加）
# 
# 【性能说明】
#   遍历 all_rows 一次构建 conv_system
#   遍历 seekers 一次匹配类型
#   使用 set 去重: 每个电影 ID 只计一次，每种类型只计一次
#   使用 (session_id, turn_num) 二元组作为 key 而非简单字符串
# ═══════════════════════════════════════════════════════════════════════

def _extract_movie_ids(processed_field_str: str) -> list[str]:
    """Extract tt... movie IDs from a processed field string.
    从处理后的字段字符串中提取 tt... 格式的电影 ID。"""
    return TT_PATTERN.findall(processed_field_str)  # 使用正则查找所有 tt+数字 模式的匹配项


def _build_seeker_genres(
    seekers: list[dict],
    all_rows: list[dict],
    movie_info: dict,
) -> list[dict]:
    """
    Augment seekers with genre info by matching system replies in the
    same conversation turn.  Returns list of dicts with 'genres' key added.
    通过匹配同一会话同一轮次中的系统回复，为用户提问记录增加影片类型信息。

    规则8：从系统回复中提取电影ID，在 movie_info.json 中查找电影类型。
    使用公共函数 build_conv_system / get_system_movie_ids。
    """
    from movie.utils.text import build_conv_system, get_system_movie_ids

    # 使用公共函数构建系统回复映射表
    conv_system = build_conv_system(all_rows)

    result = []
    for r in seekers:
        conv_id = r.get('conv_id', '')
        # 规则8：从系统回复中提取电影 ID
        movie_ids = get_system_movie_ids(conv_id, conv_system)

        # 收集这些电影 ID 对应的类型
        genres_found = set()
        for mid in movie_ids:
            info = movie_info.get(mid, {})
            if isinstance(info, dict):
                genre_list = info.get('genres', []) or []
                if genre_list:
                    genres_found.update(g.strip() for g in genre_list if g.strip())

        rec = dict(r)
        rec['genres'] = genres_found if genres_found else {'unknown'}
        result.append(rec)

    return result


# ── Genre Mention Counter 类型提及统计 ──────────────────────────────────
# 【功能】统计指定日期集合中各电影类型的提及次数
#   输入: seeker_genres（增强后的用户提问记录）+ date_set（目标日期集合）
#   输出: {genre: int} 每个类型被提及的总次数
#   注意: 每条记录中每种类型只计一次（使用 set 去重，Counter 累加）
#         但不同类型的记录数不同，同一类型在不同记录中多次出现时累加
# 【调用者】
#   J1/J2: 计算节假日/工作日/周末各类型日均提及（除以天数）
#   J3/J4/J5: 计算各节假日内各类型的提及次数
# ═══════════════════════════════════════════════════════════════════════

def _genre_mention_counts(
    seeker_genres: list[dict], date_set: set) -> dict[str, int]:
    """Count genre mentions (unique per record) for dates in date_set.
    统计指定日期集合中各电影类型的提及次数（每条记录每种类型只计一次）。"""
    counter: Counter = Counter()  # 创建 Counter 计数器
    for r in seeker_genres:  # 遍历每条带类型信息的用户提问记录
        if r['date'] in date_set:  # 如果记录日期在目标日期集合中
            for g in r.get('genres', {'unknown'}):  # 遍历该记录涉及的所有影片类型
                counter[g] += 1  # 对应类型计数加 1
    return dict(counter)  # 返回字典格式的计数结果


# ── Hourly Genre Mention Counter 逐小时类型提及统计 ────────────────────
# 【功能】计算指定日期集合中每个类型在 0-23 小时的日均提及次数
#   输入: seeker_genres + date_set
#   输出: {genre: [24 floats]} 每个类型一个 24 元素列表
#         每个 float = 该小时的总提及数 / 总天数（日平均值）
# 【数据结构】
#   dh_genre_counter: dict[str, dict[tuple[int, str], int]]
#     键路径: date → (hour, genre) → count
#     示例: {'2022-01-01': {(14, 'Drama'): 3, (15, 'Comedy'): 1}}
#   hour_genre_total: dict[str, list[int]]
#     键=类型, 值=[0]*24, 跨日期累加每小时计数
# 【核心逻辑】
#   phase 1: 按日期-小时-类型统计原始计数
#   phase 2: 跨日期聚合，累加至 hour_genre_total[g][h]
#   phase 3: 除以 date_set 大小得日均值
# ═══════════════════════════════════════════════════════════════════════

def _genre_hourly_mention_counts(
    seeker_genres: list[dict], date_set: set) -> dict[str, list[float]]:
    """
    Avg hourly genre mention counts for dates in date_set.
    计算指定日期集合中各电影类型在每个小时的平均提及次数。
    Returns: dict[genre] -> list[24 floats] (avg per hour per day)
    返回：字典[类型] -> 24 个浮点数（每小时每天平均值）
    """
    if not date_set:  # 如果日期集合为空
        return {}  # 返回空字典

    # Per date-hour: Counter of genre mentions
    # 按日期-小时统计类型提及次数
    dh_genre_counter = defaultdict(lambda: defaultdict(int))  # 两层嵌套字典：日期 -> (小时, 类型) -> 次数
    for r in seeker_genres:  # 遍历每条带类型信息的记录
        if r['date'] in date_set:  # 如果记录日期在目标日期集合中
            h = r['hour']  # 获取该记录的小时数
            for g in r.get('genres', {'unknown'}):  # 遍历该记录涉及的所有影片类型
                dh_genre_counter[r['date']][(h, g)] += 1  # 对应日期-小时-类型计数加 1

    # Aggregate over dates
    # 跨日期汇总
    hour_genre_total = defaultdict(lambda: [0] * 24)  # 每种类型对应 24 小时的累加次数
    for d in date_set:  # 遍历每个日期
        for (h, g), count in dh_genre_counter[d].items():  # 遍历该日期下每个小时-类型组合
            hour_genre_total[g][h] += count  # 累加到对应类型对应小时

    num_dates = len(date_set)  # 日期总数，用于计算平均值
    result = {}  # 存储结果
    for g, vals in hour_genre_total.items():  # 遍历每种类型
        result[g] = [v / num_dates for v in vals]  # 除以天数得到每小时平均提及次数
    return result  # 返回结果


# ═══════════════════════════════════════════════════════════════════════
#  Helper: Grouped Bar Chart for Genre Mention Stats
#  辅助函数：电影类型日均提及分组柱状图
# ═══════════════════════════════════════════════════════════════════════
# 【功能】绘制多分组对比柱状图，X轴=类型名称，每组类型有多根柱子
#   用于 J1（2组: Holiday/Non-holiday）和 J2（3组: Holiday/Workday/Weekend）
#   自动确定热门类型、分配颜色、标注轴标签和图例
#
# 【参数】
#   stats:   {'group_name': {genre: avg_daily, ...}, ...}
#   title:   图表标题
#   filename: 输出 PNG 文件名（不含路径）
#   top_n:   显示前 N 个热门类型（默认 TOP_N_GENRES=20）
#
# 【返回值】
#   top_genres: list[str] 前 N 个类型名称列表，供 CSV 输出使用
#
# 【渲染流程】
#   1. 合并所有分组中的类型，按总提及数排序取前 top_n
#   2. 对每个分组计算柱子偏移量 (group_idx - (n-1)/2) * width
#   3. 使用 ax.bar() 绘制分组柱子，颜色按序分配
#   4. X轴标签旋转45度，添加图例和网格线
#   5. 保存 PNG 至 STEP_OUT
# ═══════════════════════════════════════════════════════════════════════

def _plot_genre_grouped_bars(
    stats: dict[str, dict[str, float]],
    title: str, filename: str,
    top_n: int = TOP_N_GENRES):
    """Grouped bar chart comparing genre avg daily mention counts.
    分组柱状图：比较不同分组中各电影类型的日均提及次数。"""
    # Find top genres by total mentions
    # 找出总提及次数最多的前 N 个类型
    all_genres = set()  # 收集所有出现的类型
    for group in stats.values():  # 遍历每个分组的数据
        all_genres.update(group.keys())  # 将该分组的类型加入集合
    genre_totals = {g: sum(stats[grp].get(g, 0) for grp in stats)  # 计算每个类型在所有分组的总提及次数
                    for g in all_genres}
    top_genres = sorted(genre_totals, key=genre_totals.get, reverse=True)[:top_n]  # 按总次数排序，取前 N 个

    if not top_genres:  # 如果没有有效类型
        log("  No genres to plot")  # 日志提示
        return  # 提前返回

    groups = list(stats.keys())  # 获取分组名称列表
    fig, ax = plt.subplots(figsize=(12, 6))  # 创建画布，尺寸 12×6 英寸
    x = np.arange(len(top_genres))  # x 轴位置：每种类型一个位置
    width = 0.8 / max(len(groups), 1)  # 每个分组的柱宽

    group_colors = [COLOR_HOLIDAY, COLOR_NONHOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND]  # 分组颜色列表

    for i, group in enumerate(groups):  # 遍历每个分组
        vals = [stats[group].get(g, 0) for g in top_genres]  # 获取该分组在各类型的值
        offset = (i - (len(groups) - 1) / 2) * width  # 计算偏移量以实现并排
        ax.bar(x + offset, vals, width, label=group,  # 绘制柱状图
               color=group_colors[i % len(group_colors)], alpha=0.8)  # 分配颜色

    ax.set_xticks(x)  # 设置 x 轴刻度位置
    ax.set_xticklabels(top_genres, rotation=45, ha='right', fontsize=9)  # 设置 x 轴标签（类型名称），旋转 45 度
    ax.set_ylabel('Avg Daily Mentions')  # y 轴标签：日均提及次数
    ax.set_title(title, fontsize=12)  # 图表标题
    ax.legend(fontsize=9)  # 图例
    ax.grid(axis='y', alpha=0.3)  # y 方向网格线

    fig.tight_layout()  # 自动调整布局
    path = os.path.join(STEP_OUT, filename)  # 拼接输出文件路径
    fig.savefig(path)  # 保存图片
    plt.close(fig)  # 关闭图形
    log(f"Saved: {path}")  # 日志记录保存信息

    return top_genres  # 返回前 N 个类型的列表，供 CSV 输出使用


# ═══════════════════════════════════════════════════════════════════════
#  J1: 节假日 VS 非节假日 电影类型日均提及 (Grouped Bar)
#  J1: Holiday vs Non-Holiday Genre Avg Daily Mentions
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   分组柱状图：横轴=电影类型，纵轴=日均提及次数
#   每个类型有两根柱子并排：节假日(Holiday) vs 非节假日(Non-holiday)
#   柱宽自动根据分组数调整，颜色使用 COLOR_HOLIDAY/COLOR_NONHOLIDAY
# 
# 【统计口径】
#   指标说明:
#     - "日均提及次数" = 该类型在指定日期集合中的总提及次数 / 天数
#     - 注意: 每条记录中每种类型只计一次（_genre_mention_counts 中使用 Counter）
#     - 只显示 TOP_N_GENRES(20) 个最常见的类型
#   分组方式:
#     - holiday: r['period'] == 'holiday' 的日期集合
#     - non_holiday: r['period'] != 'holiday' 的日期集合（工作日+周末）
#   数据来源:
#     - seeker_genres 由 _build_seeker_genres() 构建
#     - 电影类型通过匹配同一会话（conv_id 前缀）中系统回复的电影 ID 提取
# 
# 【坐标轴】
#   X轴: 电影类型名称（按总提及数降序排列，旋转45度显示，字号9）
#   Y轴: 日均提及次数（网格线 alpha=0.3）
#   图例: 右上角，字号9
# 
# 【输出文件】
#   PNG: j1_holiday_vs_nonholiday_genre.png（通过 _plot_genre_grouped_bars 绘制）
#   CSV: j1_holiday_vs_nonholiday_genre.csv（含: genre, holiday_avg_daily, non_holiday_avg_daily）
# 
# 【特殊说明】
#   - 柱状图通过辅助函数 _plot_genre_grouped_bars() 统一绘制
#   - 该函数可复用给 J2（三组对比），通过参数 stats dict 区分
#   - 类型排序逻辑: 按所有分组的总提及数降序取前 TOP_N_GENRES 个
#   - 柱子偏移: offset = (i - (len(groups)-1)/2) * width 实现并排
#   - 注意: 这里的"日均"是总提及数/天数，而非 A1 的"每天提问数的均值"
# 
# 【代码中处理逻辑】
#   1. 日期收集阶段
#      遍历 seeker_genres, 根据 r['period'] 收集日期：
#        holiday_dates ← r['period'] == 'holiday' 的 r['date'] 集合（set 去重）
#        non_holiday_dates ← r['period'] != 'holiday' 的 r['date'] 集合
#      天数上限保护: num_h = max(len(dates), 1) 避免除零
# 
#   2. 提及次数统计 (_genre_mention_counts)
#      输入: seeker_genres（增强后的提问记录列表）+ date_set（目标日期集合）
#      数据结构: Counter（字典子类，自动处理不存在的键）
#      处理流程:
#       a) 遍历 seeker_genres 每条记录 r
#       b) 如果 r['date'] not in date_set → 跳过
#       c) 对 r['genres'] 集合中的每个类型 g，counter[g] += 1
#      注意: 使用 set 存储类型，每条记录每种类型最多计1次
#      返回: {genre: int} 类型-原始提及次数 字典
# 
#   3. 日均值计算
#      h_avg[g] = h_genre[g] / num_holiday_dates
#      nh_avg[g] = nh_genre[g] / num_non_holiday_dates
#     注意: 分母是"天数"而非"记录数"，体现每天平均提及强度
# 
#   4. 图表渲染 (_plot_genre_grouped_bars)
#      - 合并所有分组中的类型，按总提及数降序排序
#      - 截取前 TOP_N_GENRES 个类型
#      - 对每个分组 i, 计算偏移量并绘制 bar
#      - X 轴刻度和标签通过 set_xticks/set_xticklabels 设置
#      - 自动调整布局后保存 PNG
#      - 返回 top_genres 列表供 CSV 输出使用分组排序
# 
#   5. CSV 输出
#      表头: ['genre', 'holiday_avg_daily', 'non_holiday_avg_daily']
#      排序: 按 holiday+non_holiday 日均值之和降序
#      每行: 类型名, 节假日日均值, 非节假日日均值（format '.2f'）
# ═══════════════════════════════════════════════════════════════════════

def dim_j1_holiday_vs_nonholiday_genre(seeker_genres: list[dict]):
    """Compare genre avg daily mentions: holiday vs non-holiday.
    对比节假日与非节假日的电影类型日均提及次数。"""
    log("=" * 50)  # 日志：分隔线
    log("J1: Holiday vs Non-Holiday Genre Avg Daily Mentions")  # 日志：分析标题

    holiday_dates = set(r['date'] for r in seeker_genres if r['period'] == 'holiday')  # 节假日日期集合
    non_holiday_dates = set(r['date'] for r in seeker_genres if r['period'] != 'holiday')  # 非节假日日期集合
    num_h = max(len(holiday_dates), 1)
    num_nh = max(len(non_holiday_dates), 1)

    h_genre = _genre_mention_counts(seeker_genres, holiday_dates)  # 计算节假日各类型提及总次数
    nh_genre = _genre_mention_counts(seeker_genres, non_holiday_dates)  # 计算非节假日各类型提及总次数

    # 转为日均值
    h_avg = {g: c / num_h for g, c in h_genre.items()}
    nh_avg = {g: c / num_nh for g, c in nh_genre.items()}

    top_genres = _plot_genre_grouped_bars(  # 绘制分组柱状图
        {'Holiday': h_avg, 'Non-holiday': nh_avg},  # 数据
        'Genre Avg Daily Mentions: Holiday vs Non-Holiday',  # 图表标题
        'j1_holiday_vs_nonholiday_genre.png',  # 输出文件名
    )

    csv_path = os.path.join(STEP_OUT, 'j1_holiday_vs_nonholiday_genre.csv')  # CSV 文件路径
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:  # 写入 CSV
        w = csv.writer(f)  # 创建 CSV 写入器
        w.writerow(['genre', 'holiday_avg_daily', 'non_holiday_avg_daily'])  # 写入表头
        all_genres = sorted(set(list(h_genre.keys()) + list(nh_genre.keys())),
                           key=lambda g: h_avg.get(g, 0) + nh_avg.get(g, 0),  # 按日均值排序
                           reverse=True)
        for g in all_genres:  # 遍历每个类型
            w.writerow([g, f'{h_avg.get(g, 0):.2f}', f'{nh_avg.get(g, 0):.2f}'])
    log(f"Saved: {csv_path}")  # 日志记录 CSV 保存信息


# ═══════════════════════════════════════════════════════════════════════
#  J2: 节假日 VS 工作日 VS 周末 电影类型日均提及 (Grouped Bar)
#  J2: Holiday vs Workday vs Weekend Genre Avg Daily Mentions
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   分组柱状图：横轴=电影类型，纵轴=日均提及次数
#   每个类型有三根柱子并排：Holiday(红) / Workday(黄) / Weekend(青)
#   通过辅助函数 _plot_genre_grouped_bars() 绘制
# 
# 【统计口径】
#   指标说明:
#     - "日均提及次数" = 该类型在指定分组日期集合中的总提及次数 / 天数
#     - 每种类型每条记录只计一次（Counter + set 去重）
#     - 只显示 TOP_N_GENRES(20) 个最常见的类型
#   分组方式:
#     - holiday: r['period'] == 'holiday' 的日期集合
#     - workday: r['period'] == 'workday' 的日期集合
#     - weekend: r['period'] == 'weekend' 的日期集合
#   数据来源:
#     - 同 J1，seeker_genres 来自 _build_seeker_genres()
# 
# 【坐标轴】
#   X轴: 类型名称（按总提及数降序，旋转45度）
#   Y轴: 日均提及次数
#   图例: 右上角，三种颜色对应三个分组
# 
# 【输出文件】
#   PNG: j2_holiday_workday_weekend_genre.png（独立图片，未合并到 J1）
#   CSV: j2_holiday_workday_weekend_genre.csv（含: genre, holiday, workday, weekend 日均值）
# 
# 【特殊说明】
#   - 与 J1 使用相同的 _plot_genre_grouped_bars() 渲染，但传入3组数据
#   - 使用 period_genre 字典存储每个分组的原始计数和日均值
#   - CSV 排序逻辑: 所有类型按三组日均值之和降序排列
#   - 注意: 这与 J1 使用独立 PNG 输出（J1 也曾试图合并但各自保留）
# 
# 【代码中处理逻辑】
#   1. 数据分组与日均值计算
#      遍历 ['holiday', 'workday', 'weekend']:
#       a) p_dates = r['period'] == p 的日期集合
#       b) raw = _genre_mention_counts(seeker_genres, p_dates) → {genre: int}
#       c) avg = {genre: count / max(len(p_dates), 1)}
#      period_avg 结构: {'Holiday': {g: avg, ...}, 'Workday': {...}, 'Weekend': {...}}
# 
#   2. 图表渲染
#      与 J1 完全相同的 _plot_genre_grouped_bars 调用
#      groups = ['Holiday', 'Workday', 'Weekend'] → 柱宽 = 0.8/3 ≈ 0.27
#      颜色: [红, 黄, 青] 对应 COLOR_HOLIDAY / COLOR_WORKDAY / COLOR_WEEKEND
# 
#   3. CSV 输出
#      表头: ['genre', 'holiday_avg_daily', 'workday_avg_daily', 'weekend_avg_daily']
#      所有类型合并去重后按三组日均值之和降序排列
#      每行: 类型名 + 三组日均值
# ═══════════════════════════════════════════════════════════════════════

def dim_j2_holiday_workday_weekend_genre(seeker_genres: list[dict]):
    """Compare genre avg daily mentions: holiday vs workday vs weekend.
    对比节假日 vs 工作日 vs 周末的电影类型日均提及次数。"""
    log("=" * 50)  # 日志：分隔线
    log("J2: Holiday vs Workday vs Weekend Genre Avg Daily Mentions")  # 日志：分析标题

    period_genre = {}  # 存储各周期的类型提及次数
    period_avg = {}  # 存储各周期的日均提及次数
    for p in ['holiday', 'workday', 'weekend']:  # 遍历三种周期
        p_dates = set(r['date'] for r in seeker_genres if r['period'] == p)  # 获取该周期的日期集合
        raw = _genre_mention_counts(seeker_genres, p_dates)  # 计算类型提及总次数
        num_days = max(len(p_dates), 1)
        period_genre[p.capitalize()] = raw
        period_avg[p.capitalize()] = {g: c / num_days for g, c in raw.items()}
        total = sum(period_avg[p.capitalize()].values())  # 计算总日均值
        log(f"  {p}: {total:.2f} avg daily genre mentions")  # 日志输出

    _plot_genre_grouped_bars(  # 绘制分组柱状图
        period_avg,  # 数据：Holiday / Workday / Weekend（日均值）
        'Genre Avg Daily Mentions: Holiday vs Workday vs Weekend',  # 图表标题
        'j2_holiday_workday_weekend_genre.png',  # 输出文件名
    )

    csv_path = os.path.join(STEP_OUT, 'j2_holiday_workday_weekend_genre.csv')  # CSV 文件路径
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:  # 写入 CSV
        w = csv.writer(f)  # 创建 CSV 写入器
        w.writerow(['genre', 'holiday_avg_daily', 'workday_avg_daily', 'weekend_avg_daily'])  # 表头
        all_g = set()  # 收集所有类型
        for grp in period_avg.values():  # 遍历每个分组
            all_g.update(grp.keys())  # 将该分组的类型加入集合
        sorted_g = sorted(all_g,  # 按日均值降序排序
                         key=lambda g: sum(period_avg[p].get(g, 0)
                                           for p in ['Holiday', 'Workday', 'Weekend']),
                         reverse=True)
        for g in sorted_g:  # 遍历每个类型
            w.writerow([g,
                        f'{period_avg["Holiday"].get(g, 0):.2f}',
                        f'{period_avg["Workday"].get(g, 0):.2f}',
                        f'{period_avg["Weekend"].get(g, 0):.2f}'])
    log(f"Saved: {csv_path}")  # 日志记录 CSV 保存信息


# ═══════════════════════════════════════════════════════════════════════
#  J3: 各节假日电影类型分布 VS 非节假日基线 (Heatmap)
#  J3: Per-Holiday Genre Distribution vs Non-Holiday
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   热力图（RdBu_r 配色）：行=节假日，列=电影类型，单元格值=日均提及次数差值
#   正值（红色）= 节假日高出非节假日，负值（蓝色）= 节假日低于非节假日
#   颜色刻度对称：vmax = max(|matrix.min()|, |matrix.max()|)
# 
# 【统计口径】
#   指标说明:
#     - 单元格值 = holiday_avg_daily - non_holiday_avg_daily
#     - 非节假日基线: 所有 r['period'] != 'holiday' 的日期集合的日均提及次数
#     - 节假日按名称（前6字符）分组，跨年聚合（如"春节"合并多年）
#   数据过滤:
#     - 节假日需满足 len(records) >= MIN_DATA_ROWS 才有足够数据量
#     - 类型取全局 TOP_N_GENRES(20) 个
# 
# 【坐标轴】
#   X轴: 电影类型名称（TOP_N_GENRES 个，按全局总提及数降序，旋转45度，字号8）
#   Y轴: 节假日名称（按名称字母排序，字号8）
#   颜色条: label='Avg Daily Mention Diff', shrink=0.6
#   标题: 红=假日更多, 蓝=假日更少
# 
# 【输出文件】
#   PNG: j3_per_holiday_vs_nonholiday_genre.png（独立热力图）
#   CSV: j3_per_holiday_vs_nonholiday_genre.csv
#         （含每个节假日×类型的日均值，最后一行为 non_holiday_baseline）
# 
# 【特殊说明】
#   - 使用 ax.imshow() 绘制热力图（非 sns.heatmap），通过 annotate_heatmap 显示数值
#   - 矩阵值 = 节假日日均值 - 非节假日日均值，而非原始值
#   - CSV 最后一行包含非节假日基线数据，便于比较
#   - 与 step1 中 A3/A4 的柱状图+基线不同，此处用颜色直观展示偏离程度
# 
# 【代码中处理逻辑】
#   1. 非节假日基线计算
#      non_holiday_dates = 所有非节假日的日期集合
#      nh_genre = _genre_mention_counts(seeker_genres, non_holiday_dates)
#      nh_avg[g] = nh_genre[g] / num_nh_dates → 非节假日日均提及基线
# 
#   2. 节假日分组聚合
#      遍历 seeker_genres, 对 r['period']=='holiday' 的行:
#       - name = r['holiday_name'][:6] 取前6字符作为分组键
#       - defaultdict(list) → holiday_groups[name].append(r)
#      过滤: 去除 len(values) < MIN_DATA_ROWS 的分组
#      排序: names = sorted(holiday_groups.keys()) 按名称字母序
# 
#   3. 全局类型排序
#      Counter 累加所有类型提及次数
#      取 most_common(TOP_N_GENRES) 作为矩阵的列标签
# 
#   4. 矩阵构建
#      matrix 维度: len(names) × len(top_genres)
#      对每个节假日 i:
#       - 取该节假日的日期集合 → group_dates
#       - gc = _genre_mention_counts(records, group_dates)
#       - 对每个类型 j: matrix[i,j] = gc[g]/num_dates - nh_avg[g]
# 
#   5. 热力图渲染
#      imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
#      annotate_heatmap(ax, matrix, fmt='.1f', fs=6) 标注数值
#      colorbar(label='Avg Daily Mention Diff') 颜色条
# 
#   6. CSV 输出
#      表头: ['holiday_name', 'num_dates'] + top_genres + ['total_avg_daily']
#      每节假日一行: 名称, 天数, 每个类型的日均值, 总日均值
#      末行: non_holiday_baseline, 非节假日天数, 基线数据
# ═══════════════════════════════════════════════════════════════════════

def dim_j3_per_holiday_vs_nonholiday_genre(seeker_genres: list[dict]):
    """Per-holiday genre distribution vs non-holiday baseline (HEATMAP).
    每个节假日的电影类型分布热力图 vs 非节假日基线。"""
    log("=" * 50)  # 日志：分隔线
    log("J3: Per-Holiday Genre Distribution vs Non-Holiday (Heatmap)")  # 日志：分析标题

    non_holiday_dates = set(r['date'] for r in seeker_genres if r['period'] != 'holiday')  # 非节假日日期集合
    # 计算非节假日基线（日均）
    num_nh_dates = len(non_holiday_dates)
    nh_genre = _genre_mention_counts(seeker_genres, non_holiday_dates)
    nh_avg = {g: c / max(num_nh_dates, 1) for g, c in nh_genre.items()}

    holiday_groups = defaultdict(list)  # 按节假日名称分组
    for r in seeker_genres:  # 遍历所有带类型信息的记录
        if r['period'] == 'holiday':  # 如果是节假日记录
            name = r['holiday_name'][:6]  # 取前 6 字符作为分组键
            holiday_groups[name].append(r)  # 加入对应分组

    holiday_groups = {k: v for k, v in holiday_groups.items()  # 过滤：只保留数据量足够的节假日
                     if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:  # 如果没有符合条件的节假日
        log("  No holiday groups with sufficient data")  # 日志提示
        return  # 提前返回

    names = sorted(holiday_groups.keys())  # 排序后的节假日名称列表

    # Find top genres globally
    all_genre_totals: Counter = Counter()
    for r in seeker_genres:
        for g in r.get('genres', {'unknown'}):
            all_genre_totals[g] += 1
    top_genres = [g for g, _ in all_genre_totals.most_common(TOP_N_GENRES)]

    # Build matrix: rows=holidays, cols=genres, value = (holiday_avg_daily - non_holiday_avg_daily)
    matrix = np.zeros((len(names), len(top_genres)))
    for i, name in enumerate(names):
        group_dates = set(r['date'] for r in holiday_groups[name])
        num_dates = max(len(group_dates), 1)
        gc = _genre_mention_counts(holiday_groups[name], group_dates)
        for j, g in enumerate(top_genres):
            h_avg = gc.get(g, 0) / num_dates
            matrix[i, j] = h_avg - nh_avg.get(g, 0)

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(14, len(top_genres) * 0.55), max(6, len(names) * 0.4 + 2)))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    annotate_heatmap(ax, matrix, fmt='.1f', fs=6)

    ax.set_xticks(range(len(top_genres)))
    ax.set_xticklabels(top_genres, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Genre')
    ax.set_title('Per-Holiday Genre Distribution: Difference from Non-Holiday Baseline\n(Red=more on holiday, Blue=less)',
                 fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.6, label='Avg Daily Mention Diff')

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'j3_per_holiday_vs_nonholiday_genre.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV（日均值）
    csv_path = os.path.join(STEP_OUT, 'j3_per_holiday_vs_nonholiday_genre.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        header = ['holiday_name', 'num_dates'] + top_genres + ['total_avg_daily']
        w.writerow(header)
        for name in names:
            group_dates = set(r['date'] for r in holiday_groups[name])
            num_dates = max(len(group_dates), 1)
            gc = _genre_mention_counts(holiday_groups[name], group_dates)
            row = [name, num_dates] + [f'{gc.get(g, 0) / num_dates:.2f}' for g in top_genres]
            total_avg = sum(gc.get(g, 0) for g in top_genres) / num_dates
            row.append(f'{total_avg:.2f}')
            w.writerow(row)
        row = ['non_holiday_baseline', num_nh_dates] + [f'{nh_avg.get(g, 0):.2f}' for g in top_genres]
        row.append(f'{sum(nh_avg.get(g, 0) for g in top_genres):.2f}')
        w.writerow(row)
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  J4: 各节假日 VS 工作日/周末 电影类型分布 (Dual Heatmap)
#  J4: Per-Holiday Genre vs Workday & Weekend
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   双面板热力图（2行1列，RdBu_r 配色）
#   - 上部分: 节假日日均提及 - 工作日基线（差值）
#   - 下部分: 节假日日均提及 - 周末基线（差值）
#   行=节假日，列=TOP_N_GENRES 个类型
#   每张图有独立的颜色刻度（对称范围，vmax 各自计算）
# 
# 【统计口径】
#   指标说明:
#     - 上面板值 = holiday_avg_daily - workday_baseline
#     - 下面板值 = holiday_avg_daily - weekend_baseline
#     - 工作日基线: r['period'] == 'workday' 的日期集合的日均提及
#     - 周末基线: r['period'] == 'weekend' 的日期集合的日均提及
#   数据过滤:
#     - 同 J3: 节假日按名称聚合，需满足 len(records) >= MIN_DATA_ROWS
# 
# 【坐标轴】
#   X轴: 电影类型（TOP_N_GENRES 个，旋转45度，上字号7/下字号7）
#   Y轴: 节假日名称（按字母序，字号7）
#   每个子图有独立颜色条，label='Diff'
#   标题分别标注 "Diff: Holiday - Workday Baseline" 和 "Diff: Holiday - Weekend Baseline"
# 
# 【输出文件】
#   PNG: j4_per_holiday_vs_workday_weekend_genre.png（独立双面板热力图）
#   CSV: j4_per_holiday_vs_workday_weekend_genre.csv（含节假日×类型日均值+基线数据）
# 
# 【特殊说明】
#   - 两张热力图共享相同的节假日行顺序，便于左右对照
#   - 与 J3 相比，此次使用工作日和周末两个独立基线而非一个非节假日基线
#   - 两张图使用独立 vmax（各自的最大绝对值），适合颜色分布不同的场景
# 
# 【代码中处理逻辑】
#   1. 基线计算
#      workday_dates = r['period'] == 'workday' 的日期集合
#      weekend_dates = r['period'] == 'weekend' 的日期集合
#      wd_genre = _genre_mention_counts(seeker_genres, workday_dates)
#      we_genre = _genre_mention_counts(seeker_genres, weekend_dates)
#      wd_avg[g] = wd_genre[g] / max(len(workday_dates), 1)
#      we_avg[g] = we_genre[g] / max(len(weekend_dates), 1)
# 
#   2. 节假日分组（同 J3）
#      按 holiday_name[:6] 分组，过滤数据不足的组
#      排序: names = sorted(holiday_groups.keys())
# 
#   3. 矩阵构建（两个独立矩阵）
#      matrix_wd: 节假日日均 - 工作日基准（差值）
#      matrix_we: 节假日日均 - 周末基准（差值）
#      每个矩阵: len(names) × TOP_N_GENRES
# 
#   4. 双面板热力图渲染
#      上子图 (ax1): imshow(matrix_wd, ...) + annotate_heatmap
#      下子图 (ax2): imshow(matrix_we, ...) + annotate_heatmap
#      各自设置 x/y 轴刻度和标签，颜色条
#      总标题: 'Per-Holiday Genre Avg Daily Mentions: Diff from Workday & Weekend'
# 
#   5. CSV 输出
#      表头: ['holiday_name', 'num_dates'] + ['{g}_avg_daily'] + ['workday_avg', 'weekend_avg']
#      每节假日一行: 名称, 天数, 每个类型日均值, 工作日均值, 周末均值
# ═══════════════════════════════════════════════════════════════════════

def dim_j4_per_holiday_vs_workday_weekend_genre(seeker_genres: list[dict]):
    """Per-holiday genre avg daily mention vs workday & weekend baselines.
    每个节假日的类型日均推荐次数 vs 工作日和周末基线对比。"""
    log("=" * 50)  # 日志：分隔线
    log("J4: Per-Holiday Genre Avg Daily vs Workday & Weekend")  # 日志：分析标题

    # 计算工作日和周末基线（日均）
    workday_dates = set(r['date'] for r in seeker_genres if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seeker_genres if r['period'] == 'weekend')
    wd_genre = _genre_mention_counts(seeker_genres, workday_dates)
    we_genre = _genre_mention_counts(seeker_genres, weekend_dates)
    num_wd = max(len(workday_dates), 1)
    num_we = max(len(weekend_dates), 1)

    log(f"  Workday: {sum(wd_genre.values())} total genre mentions from {len(workday_dates)} days")
    log(f"  Weekend: {sum(we_genre.values())} total genre mentions from {len(weekend_dates)} days")

    # 按节假日名称分组
    holiday_groups = defaultdict(list)
    for r in seeker_genres:
        if r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_groups[name].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items() if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  No holiday groups")
        return

    names = sorted(holiday_groups.keys())

    # 计算每个节假日每个类型的日均次数
    all_genre_totals: Counter = Counter()
    for r in seeker_genres:
        for g in r.get('genres', {'unknown'}):
            all_genre_totals[g] += 1
    top_genres = [g for g, _ in all_genre_totals.most_common(TOP_N_GENRES)]

    # Heatmap: rows=holidays, cols=genres, value = holiday_avg - workday_baseline
    matrix_wd = np.zeros((len(names), len(top_genres)))
    matrix_we = np.zeros((len(names), len(top_genres)))
    for i, name in enumerate(names):
        group_dates = set(r['date'] for r in holiday_groups[name])
        num_dates = max(len(group_dates), 1)
        gc = _genre_mention_counts(holiday_groups[name], group_dates)
        for j, g in enumerate(top_genres):
            h_avg = gc.get(g, 0) / num_dates
            matrix_wd[i, j] = h_avg - (wd_genre.get(g, 0) / num_wd)
            matrix_we[i, j] = h_avg - (we_genre.get(g, 0) / num_we)

    # 双热力图：上=vs Workday, 下=vs Weekend
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(top_genres) * 0.55),
                                                    max(8, len(names) * 0.7 + 2)))

    vmax1 = max(abs(matrix_wd.min()), abs(matrix_wd.max()), 0.01)
    im1 = ax1.imshow(matrix_wd, cmap='RdBu_r', aspect='auto', vmin=-vmax1, vmax=vmax1)
    annotate_heatmap(ax1, matrix_wd, fmt='.1f', fs=6)
    ax1.set_xticks(range(len(top_genres)))
    ax1.set_xticklabels(top_genres, rotation=45, ha='right', fontsize=7)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=7)
    ax1.set_title('Diff: Holiday Avg Daily - Workday Baseline', fontsize=10)
    fig.colorbar(im1, ax=ax1, shrink=0.5, label='Diff')

    vmax2 = max(abs(matrix_we.min()), abs(matrix_we.max()), 0.01)
    im2 = ax2.imshow(matrix_we, cmap='RdBu_r', aspect='auto', vmin=-vmax2, vmax=vmax2)
    annotate_heatmap(ax2, matrix_we, fmt='.1f', fs=6)
    ax2.set_xticks(range(len(top_genres)))
    ax2.set_xticklabels(top_genres, rotation=45, ha='right', fontsize=7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=7)
    ax2.set_xlabel('Genre')
    ax2.set_title('Diff: Holiday Avg Daily - Weekend Baseline', fontsize=10)
    fig.colorbar(im2, ax=ax2, shrink=0.5, label='Diff')

    fig.suptitle('Per-Holiday Genre Avg Daily Mentions: Difference from Workday & Weekend', fontsize=12)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'j4_per_holiday_vs_workday_weekend_genre.png')
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV（日均值）
    csv_path = os.path.join(STEP_OUT, 'j4_per_holiday_vs_workday_weekend_genre.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'num_dates'] + [f'{g}_avg_daily' for g in top_genres] + ['workday_avg', 'weekend_avg'])
        for i, name in enumerate(names):
            group_dates = set(r['date'] for r in holiday_groups[name])
            num_dates = max(len(group_dates), 1)
            gc = _genre_mention_counts(holiday_groups[name], group_dates)
            row = [name, num_dates] + [f'{gc.get(g, 0) / num_dates:.2f}' for g in top_genres]
            row.append(f'{sum(wd_genre.values()) / num_wd:.2f}')
            row.append(f'{sum(we_genre.values()) / num_we:.2f}')
            w.writerow(row)
    log(f"Saved: {csv_path}")

# ═══════════════════════════════════════════════════════════════════════
#  J5: 节假日 × 电影类型 热力图 (Heatmap)
#  J5: Per-Holiday Genre Mention Heatmap
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   热力图（sns.heatmap，YlOrRd 配色）：行=电影类型，列=节假日，值=日均提及次数
#   与 J3/J4 不同，此处展示绝对量而非差值
#   Y轴（类型）和 X轴（节假日）均按总日均提及数降序排列
# 
# 【统计口径】
#   指标说明:
#     - 单元格值 = 该节假日内该类型的日均提及次数（总提及数/天数）
#     - 每条记录每种类型只计一次（set 去重）
#   数据过滤:
#     - 节假日按名称（前6字符）聚合，需满足 len(records) >= MIN_DATA_ROWS
#     - 类型取全局 TOP_N_GENRES(20) 个
# 
# 【坐标轴】
#   X轴: 节假日名称（按所有类型总日均提及数降序，旋转30度，ha='right'）
#   Y轴: 电影类型名称（按所有节假日总日均提及数降序，不旋转）
#   颜色条: label='Avg Daily Mentions'
# 
# 【输出文件】
#   PNG: j5_per_holiday_genre_heatmap.png（DPI=150, bbox_inches='tight'）
#   CSV: j5_per_holiday_genre_heatmap.csv（透视表格式，类型为行、节假日为列）
# 
# 【特殊说明】
#   - 使用 pandas pivot_table 构建矩阵（aggfunc='sum', fill_value=0）
#   - 双向排序: pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
#   - 使用 sns.heatmap 而非 imshow，自带 annot=True 显示数值
#   - 透明度: linewidths=0.5 网格线，便于区分单元格
#   - 与 J3/J4 核心区别：展示绝对值，可直观看出哪个节假日哪种类型被讨论最多
# 
# 【代码中处理逻辑】
#   1. 节假日分组
#      按 holiday_name[:6] 分组（同 J3/J4）
#      过滤数据不足的分组
# 
#   2. 热点类型确定
#      Counter 统计全量 seeker_genres 中各类型提及次数
#      取 most_common(TOP_N_GENRES) 作为矩阵行标签
# 
#   3. 数据行构建
#      对每个节假日 × 每个类型:
#        计算该节假日内该类型的日均提及次数
#      每条数据: {genre, holiday, count}
#      存入列表 data_rows
# 
#   4. 透视表构建
#      df = pd.DataFrame(data_rows)
#      pivot = df.pivot_table(index='genre', columns='holiday',
#                             values='count', aggfunc='sum', fill_value=0)
# 
#   5. 双向排序
#      Y轴降序: pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
#      X轴降序: pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]
#      使得最热门的类型和节假日排在最前面
# 
#   6. 热力图渲染
#      sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
#                  linewidths=0.5, cbar_kws={'label': 'Avg Daily Mentions'})
#      annot=True 自动在每个单元格显示数值
#      DPI=150 高质量输出
# 
#   7. CSV 输出
#      直接使用 pivot.to_csv() 保存透视表
#      编码: utf-8-sig，float_format='%.2f'
# ═══════════════════════════════════════════════════════════════════════

def dim_j5_per_holiday_genre_heatmap(seeker_genres: list[dict]):
    """
    Heatmap: x=holidays, y=genres, values=avg daily mention counts (per day).
    Both axes sorted descending by total avg daily mentions.
    热力图：横轴为各个节假日，纵轴为各种电影类型，单元格值=日均提及次数。
    X/Y 轴均按日均提及总数降序排列。
    Args:
        seeker_genres: 带电影类型信息的用户提问记录列表
    """
    log("=" * 50)
    log("J5: Per-Holiday Genre Avg Daily Heatmap")

    # ── 按节假日名称分组 ──
    # 按节假日名称（前 6 字符）分组
    holiday_groups = defaultdict(list)
    for r in seeker_genres:
        if r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_groups[name].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items()
                      if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  No holiday groups with sufficient data")
        return

    names = sorted(holiday_groups.keys())
    log(f"  {len(names)} holidays with sufficient data")

    # ── 确定全局热门类型（前 TOP_N_GENRES 个）──
    all_genre_totals: Counter = Counter()
    for r in seeker_genres:
        for g in r.get('genres', {'unknown'}):
            all_genre_totals[g] += 1
    top_genres = [g for g, _ in all_genre_totals.most_common(TOP_N_GENRES)]
    log(f"  {len(top_genres)} genres")

    # ── 构建透视表：行=类型，列=节假日，值=日均提及次数 ──
    data_rows = []
    for name in names:
        group_dates = set(r['date'] for r in holiday_groups[name])
        num_dates = max(len(group_dates), 1)
        gc = _genre_mention_counts(holiday_groups[name], group_dates)
        for g in top_genres:
            data_rows.append({
                'genre': g,
                'holiday': name,
                'count': gc.get(g, 0) / num_dates,  # 转为日均值
            })

    df = pd.DataFrame(data_rows)
    pivot = df.pivot_table(
        index='genre',
        columns='holiday',
        values='count',
        aggfunc='sum',
        fill_value=0,
    )

    # ── 双轴降序排列 ──
    # Y 轴（类型）：按所有节假日总日均提及次数降序
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    # X 轴（节假日）：按所有类型总日均提及次数降序
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]

    log(f"  Heatmap size: {pivot.shape[0]} genres x {pivot.shape[1]} holidays")

    # ── 绘制热力图 ──
    fig, ax = plt.subplots(figsize=(
        max(8, pivot.shape[1] * 1.2),
        max(5, pivot.shape[0] * 0.55),
    ))
    sns.heatmap(
        pivot, annot=True, fmt='.1f', cmap='YlOrRd',
        linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Avg Daily Mentions'},
    )
    ax.set_title('Per-Holiday Genre Avg Daily Mentions', fontsize=14, pad=16)
    ax.set_xlabel('Holiday', fontsize=11)
    ax.set_ylabel('Genre', fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(STEP_OUT, 'j5_per_holiday_genre_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")

    # ── CSV（日均值）──
    csv_path = os.path.join(STEP_OUT, 'j5_per_holiday_genre_heatmap.csv')
    pivot.to_csv(csv_path, encoding='utf-8-sig', float_format='%.2f')
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  B: Hourly genre analysis  B：逐小时类型分析
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  Helper: Multi-Panel Hourly Genre Line Chart
#  辅助函数：逐小时电影类型提及多面板折线图
# ═══════════════════════════════════════════════════════════════════════
# 【功能】为每个热门类型绘制一个子图，每个子图显示多组归一化折线
#   用于 K1（2组: Holiday/Non-holiday）和 K2（3组: Holiday/Workday/Weekend）
#   纵轴统一归一化为百分比（消除绝对量差异，聚焦时间分布模式）
#
# 【参数】
#   hourly_genre: {'group_label': {genre: [24 floats]}, ...}
#     其中 24 floats 为 0-23 小时日均提及次数
#   title:   图表总标题
#   filename: 输出 PNG 文件名
#   top_n:   显示前 N 个热门类型子图（默认 6）
#
# 【返回值】无
#
# 【渲染流程】
#   1. 合并所有分组所有类型，计算总提及数，排序取前 top_n 个
#   2. subplots(top_n, 1, sharex=True, figsize=(12, 2.5*top_n+1))
#   3. 对每个类型子图:
#     a) 遍历每个分组, 取该分组该类型的 [24] 原始值
#     b) 计算总和, 归一化为百分比: vals_pct = [v/total*100]
#     c) ax.plot(hours, vals_pct, 'o-', label=group)
#     d) 设置 ylabel=类型名(%/hr), 图例右上角
#   4. 底部子图: set_xlabel('Hour of Day (UTC)')
#   5. fig.tight_layout() → 保存 PNG
#   6. 注意: 如果 top_n==1, axes 需包装为列表处理
# ═══════════════════════════════════════════════════════════════════════

def _plot_genre_hourly_lines(
    hourly_genre: dict[str, dict[str, list[float]]],
    title: str, filename: str,
    top_n: int = 6):
    """
    Line chart for top N genres: hourly mention counts across groups.
    前 N 个热门类型的折线图：显示不同分组的逐小时提及次数变化。
    hourly_genre: {group_label: {genre: [24 floats]}}
    hourly_genre: {分组标签: {类型: [24 个浮点数]}}
    """
    # Determine top N genres across all groups
    # 找出所有分组中总提及次数最多的前 N 个类型
    all_genres = set()  # 收集所有类型
    for group_data in hourly_genre.values():  # 遍历每个分组的数据
        all_genres.update(group_data.keys())  # 将分组中的类型加入集合
    genre_totals = {  # 计算每个类型在所有分组的总提及次数
        g: sum(sum(vals) for gd in hourly_genre.values()
               for genre, vals in gd.items() if genre == g)
        for g in all_genres
    }
    top_genres = sorted(genre_totals, key=genre_totals.get, reverse=True)[:top_n]  # 排序取前 N 个

    groups = list(hourly_genre.keys())  # 获取分组名称列表
    hours = list(range(24))  # 小时范围 0-23

    fig, axes = plt.subplots(len(top_genres), 1, figsize=(12, 2.5 * len(top_genres) + 1),  # 每种类型一个子图
                             sharex=True)  # 所有子图共享 x 轴
    if len(top_genres) == 1:  # 如果只有一种类型
        axes = [axes]  # 将 axes 包装为列表

    for idx, g in enumerate(top_genres):  # 遍历每个热门类型
        ax = axes[idx]  # 获取对应的子图
        for grp in groups:  # 遍历每个分组
            vals = hourly_genre[grp].get(g, [0.0] * 24)  # 获取该分组该类型的逐小时数据
            mention_total = sum(vals)  # 计算总提及次数
            # Normalize to percentage if there are mentions
            # 如果有提及，则归一化为百分比
            if mention_total > 0:  # 避免除零
                vals_pct = [v / mention_total * 100 for v in vals]  # 将每个小时的值转为百分比
                ax.plot(hours, vals_pct, 'o-', label=grp, linewidth=1.5,  # 绘制折线图
                        markersize=3, alpha=0.85)  # 圆形标记，透明度 0.85
        ax.set_ylabel(f'{g[:20]}\n(%/hr)')  # y 轴标签：类型名称（截断前 20 字符）+ 每小时百分比
        ax.set_title(f'Genre: {g}', fontsize=10)  # 子图标题
        ax.legend(fontsize=7, loc='upper right')  # 图例，右上角
        ax.grid(axis='y', alpha=0.3)  # y 方向网格线
        if idx == len(top_genres) - 1:  # 如果是最后一个子图（底部）
            ax.set_xlabel('Hour of Day (UTC)')  # 设置 x 轴标签

    fig.suptitle(title, fontsize=13)  # 总标题
    fig.tight_layout()  # 自动调整布局
    path = os.path.join(STEP_OUT, filename)  # 输出文件路径
    fig.savefig(path)  # 保存图片
    plt.close(fig)  # 关闭图形
    log(f"Saved: {path}")  # 日志记录保存信息


# ═══════════════════════════════════════════════════════════════════════
#  K1: 逐小时类型提及分布: 节假日 VS 非节假日 (Multi-Line)
#  K1: Hourly Genre: Holiday vs Non-Holiday
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   多面板折线图（垂直排列）：每个热门类型一个子图，2组曲线
#   - 红色曲线: 节假日 (Holiday)
#   - 蓝色曲线: 非节假日 (Non-holiday)
#   纵轴归一化为百分比（每小时值占该类型全天总值的百分比）
#   子图数量 = min(top_n=6, 实际热门类型数)
# 
# 【统计口径】
#   指标说明:
#     - 原始值: _genre_hourly_mention_counts() 返回 {genre: [24 floats]}
#     - 每个 float = 该小时在该组的总提及数 / 天数（日均每小时提及数）
#     - 归一化: 每小时百分比 = 该小时值 / 24小时总和 × 100
#     - 归一化目的：消除不同分组间绝对量的差异，聚焦"时间分布模式"差异
#   分组:
#     - holiday: r['period'] == 'holiday' 的日期集合
#     - non_holiday: r['period'] != 'holiday' 的日期集合
# 
# 【坐标轴】
#   X轴: 小时 (0-23, UTC)，所有子图共享
#   Y轴: 每小时提及百分比 (%/hr)，每个子图独立
#   图例: 右上角，区分 Holiday / Non-holiday
#   线型: 'o-'（实线+圆形标记），alpha=0.85
# 
# 【输出文件】
#   PNG: k1_hourly_holiday_vs_nonholiday_genre.png（多面板折线图）
#   CSV: k1_hourly_holiday_vs_nonholiday_genre.csv（24行×所有类型原始值）
# 
# 【特殊说明】
#   - 使用 _plot_genre_hourly_lines() 辅助函数统一绘制
#   - 子图数量自动适应: 如果 top_n=1，将 axes 包装为列表处理
#   - 热门类型取各分组总提及数之和最大的 TOP_N(6) 个
#   - 归一化使用百分比而非原始值，可直观比较时间分布模式
# 
# 【代码中处理逻辑】
#   1. 日期收集
#      holiday_dates = r['period'] == 'holiday' 的日期 set
#      non_holiday_dates = r['period'] != 'holiday' 的日期 set
# 
#   2. 逐小时类型计数 (_genre_hourly_mention_counts)
#      数据结构: dh_genre_counter = defaultdict(lambda: defaultdict(int))
#       键路径: date → (hour, genre) → count
#      处理流程:
#       a) 遍历 seeker_genres: 如果 r['date'] in date_set
#       b) dh_genre_counter[r['date']][(h, g)] += 1
#      聚合:
#       hour_genre_total = defaultdict(lambda: [0]*24)
#      遍历 date_set 中每个日期 d:
#       对 d 下每个 (h, g) 组合: hour_genre_total[g][h] += count
#      日均值: result[g][h] = hour_genre_total[g][h] / len(date_set)
#      返回: {genre: [24 floats]}
# 
#   3. 热门类型排序 (_plot_genre_hourly_lines)
#      合并所有分组所有类型，计算总提及数
#      取前 top_n=6 个类型绘制子图
# 
#   4. 折线图渲染 (_plot_genre_hourly_lines)
#      subplots(n, 1, sharex=True) 垂直排列子图
#      每个子图:
#       - 遍历 groups, 获取该组该类型的 [24] 数据
#       - 计算 total = sum(vals), 如果 >0 则归一化
#       - plot(hours, vals_pct, 'o-', label=group)
#       - 设置 ylabel=g[:20] + '(%/hr)', 图例右上角
#      底部的子图: set_xlabel('Hour of Day (UTC)')
# 
#   5. CSV 输出
#      扁平格式: 24行, 每行=[hour, holiday_g1, holiday_g2, ..., non_holiday_g1, ...]
#      值格式: '.4f'
# ═══════════════════════════════════════════════════════════════════════

def dim_k1_hourly_holiday_vs_nonholiday_genre(seeker_genres: list[dict]):
    """Hourly genre mention distribution: holiday vs non-holiday.
    节假日 vs 非节假日的逐小时类型提及分布。"""
    log("=" * 50)  # 日志：分隔线
    log("K1: Hourly Genre - Holiday vs Non-Holiday")  # 日志：分析标题

    holiday_dates = set(r['date'] for r in seeker_genres if r['period'] == 'holiday')  # 节假日日期集合
    non_holiday_dates = set(r['date'] for r in seeker_genres if r['period'] != 'holiday')  # 非节假日日期集合

    h_hourly = _genre_hourly_mention_counts(seeker_genres, holiday_dates)  # 节假日逐小时类型提及
    nh_hourly = _genre_hourly_mention_counts(seeker_genres, non_holiday_dates)  # 非节假日逐小时类型提及

    _plot_genre_hourly_lines(  # 绘制逐小时折线图
        {'Holiday': h_hourly, 'Non-holiday': nh_hourly},  # 数据
        'Hourly Genre Mention Distribution: Holiday vs Non-Holiday',  # 图表标题
        'k1_hourly_holiday_vs_nonholiday_genre.png',  # 输出文件名
    )

    csv_path = os.path.join(STEP_OUT, 'k1_hourly_holiday_vs_nonholiday_genre.csv')  # CSV 文件路径
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:  # 写入 CSV
        w = csv.writer(f)  # 创建 CSV 写入器
        all_g = sorted(set(h_hourly.keys()) | set(nh_hourly.keys()))  # 合并所有类型并排序
        w.writerow(['hour'] + [f'holiday_{g}' for g in all_g]  # 表头：小时 + 节假日各类型 + 非节假日各类型
                    + [f'non_holiday_{g}' for g in all_g])
        for h in range(24):  # 遍历 0-23 小时
            row = [h]  # 以小时数开头
            for g in all_g:  # 节假日各类型
                row.append(f'{h_hourly.get(g, [0]*24)[h]:.4f}')
            for g in all_g:  # 非节假日各类型
                row.append(f'{nh_hourly.get(g, [0]*24)[h]:.4f}')
            w.writerow(row)  # 写入该小时的数据行
    log(f"Saved: {csv_path}")  # 日志记录 CSV 保存信息


# ═══════════════════════════════════════════════════════════════════════
#  K2: 逐小时类型提及分布: 节假日 VS 工作日 VS 周末 (Multi-Line)
#  K2: Hourly Genre: Holiday vs Workday vs Weekend
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   多面板折线图（垂直排列）：每个热门类型一个子图，3组曲线
#   - 红色: 节假日 (Holiday)
#   - 黄色: 工作日 (Workday)
#   - 青色: 周末 (Weekend)
#   纵轴归一化为百分比，子图数量 = min(6, 实际热门类型数)
#   与 K1 使用相同的辅助函数 _plot_genre_hourly_lines()，数据改为3组
# 
# 【统计口径】
#   指标说明:
#     - 原始值: _genre_hourly_mention_counts() 返回 {genre: [24 floats]}
#     - 归一化百分比 = 每小时值 / 24小时总和 × 100
#     - 每个子图显示一种类型的3条曲线对比
#   分组:
#     - holiday: r['period'] == 'holiday' 日期，红
#     - workday: r['period'] == 'workday' 日期，黄
#     - weekend: r['period'] == 'weekend' 日期，青
# 
# 【坐标轴】
#   X轴: 小时 (0-23, UTC)，所有子图共享
#   Y轴: 每小时提及百分比 (%/hr)
#   图例: 右上角，3种曲线对应 Holiday/Workday/Weekend
# 
# 【输出文件】
#   PNG: k2_hourly_holiday_workday_weekend_genre.png（独立多面板折线图）
#   CSV: k2_hourly_holiday_workday_weekend_genre.csv（24行×所有类型×3组原始值）
# 
# 【特殊说明】
#   - 与 K1 输出独立的 PNG（非合并）
#   - 所有逻辑与 K1 相同，仅分组从2组变为3组
#   - top_n 同样为6个热门类型
# 
# 【代码中处理逻辑】
#   1. 三组逐小时数据计算
#      对 ['holiday', 'workday', 'weekend'] 分别:
#       - p_dates = r['period'] == p 的日期 set
#       - hourly_data[p.capitalize()] = _genre_hourly_mention_counts(seeker_genres, p_dates)
# 
#   2. 图表渲染
#      与 K1 完全相同的 _plot_genre_hourly_lines() 调用
#      输入: {Holiday: {...}, Workday: {...}, Weekend: {...}}
#      top_n=6 → 最多6个子图
#      每个子图绘制3条归一化曲线
# 
#   3. CSV 输出
#      表头: [hour, holiday_g1, ..., workday_g1, ..., weekend_g1, ...]
#      24行，每行该小时下所有分组×所有类型的原始值（非百分比）
# ═══════════════════════════════════════════════════════════════════════

def dim_k2_hourly_holiday_workday_weekend_genre(seeker_genres: list[dict]):
    """Hourly genre: holiday vs workday vs weekend.
    节假日 vs 工作日 vs 周末的逐小时类型提及分布。"""
    log("=" * 50)  # 日志：分隔线
    log("K2: Hourly Genre - Holiday vs Workday vs Weekend")  # 日志：分析标题

    hourly_data = {}  # 存储各周期的逐小时类型数据
    for p in ['holiday', 'workday', 'weekend']:  # 遍历三种周期
        p_dates = set(r['date'] for r in seeker_genres if r['period'] == p)  # 获取日期集合
        hourly_data[p.capitalize()] = _genre_hourly_mention_counts(seeker_genres, p_dates)  # 计算逐小时类型数据

    _plot_genre_hourly_lines(  # 绘制逐小时折线图
        hourly_data,  # 数据：Holiday / Workday / Weekend
        'Hourly Genre Mention Distribution: Holiday vs Workday vs Weekend',  # 图表标题
        'k2_hourly_holiday_workday_weekend_genre.png',  # 输出文件名
    )

    csv_path = os.path.join(STEP_OUT, 'k2_hourly_holiday_workday_weekend_genre.csv')  # CSV 文件路径
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:  # 写入 CSV
        w = csv.writer(f)  # 创建 CSV 写入器
        all_g = set()  # 收集所有类型
        for gd in hourly_data.values():  # 遍历每周期数据
            all_g.update(gd.keys())  # 将类型加入集合
        all_g = sorted(all_g)  # 排序
        w.writerow(['hour'] + [f'holiday_{g}' for g in all_g]  # 表头：节假日 + 工作日 + 周末各类型
                    + [f'workday_{g}' for g in all_g]
                    + [f'weekend_{g}' for g in all_g])
        for h in range(24):  # 遍历 24 小时
            row = [h]  # 以小时数开头
            for g in all_g:  # 节假日
                row.append(f'{hourly_data["Holiday"].get(g, [0]*24)[h]:.4f}')
            for g in all_g:  # 工作日
                row.append(f'{hourly_data["Workday"].get(g, [0]*24)[h]:.4f}')
            for g in all_g:  # 周末
                row.append(f'{hourly_data["Weekend"].get(g, [0]*24)[h]:.4f}')
            w.writerow(row)  # 写入该小时的数据行
    log(f"Saved: {csv_path}")  # 日志记录 CSV 保存信息


# ═══════════════════════════════════════════════════════════════════════
#  K3: 各节假日逐小时类型差值 (Heatmap per Genre)
#  K3: Per-Holiday Hourly Genre Diff from Non-Holiday
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   每个热门类型生成一张独立热力图（RdBu_r 配色）
#   行=节假日，列=0-23小时，单元格值=该节假日该小时 vs 非节假日基线的日均提及差值
#   正值（红色）= 节假日高于非节假日，负值（蓝色）= 节假日低于非节假日
#   每个类型独立保存一个 PNG 文件
# 
# 【统计口径】
#   指标说明:
#     - 单元格值 = h_hourly[g][h] - nh_hourly[g][h]
#     - h_hourly: 节假日每小时日均提及数（_genre_hourly_mention_counts）
#     - nh_hourly: 非节假日每小时日均提及数（同函数计算基线）
#     - 单元格值 > 0: 该节假日该小时该类型被讨论得比平时多
#     - 单元格值 < 0: 该节假日该小时该类型被讨论得比平时少
#   数据过滤:
#     - 节假日按名称聚合，需满足 len(records) >= MIN_DATA_ROWS
#     - 只对 TOP_N_GENRES(20) 个热门类型生成热力图
# 
# 【坐标轴】
#   X轴: 小时 (0-23 UTC)，标签字号8
#   Y轴: 节假日名称，标签字号8
#   颜色条: label='Diff (avg/hr)', shrink=0.6
#   标题: Genre "{g}": Per-Holiday Hourly Diff from Non-Holiday
# 
# 【输出文件】
#   PNG: k3_genre_{g[:10]}_hourly_heatmap.png（每个热门类型一张）
#   CSV: k3_per_holiday_hourly_genre.csv（长格式: holiday, genre, hour, diff）
# 
# 【特殊说明】
#   - 每种类型独立图片，文件名用类型前10字符
#   - 颜色刻度对称: vmax = max(|matrix.min()|, |matrix.max()|)
#   - 使用 ax.imshow() 绘制 + annotate_heatmap() 标注数值
#   - 与 K4 的区别: K3 使用非节假日基线，K4 分别使用工作日/周末基线
# 
# 【代码中处理逻辑】
#   1. 非节假日小时级基线
#      non_holiday_dates = r['period'] != 'holiday' 的日期 set
#      nh_hourly = _genre_hourly_mention_counts(seeker_genres, non_holiday_dates)
#      返回: {genre: [24 floats]} 每个类型每小时日均提及数
# 
#   2. 节假日分组（同 J3）
#      按 holiday_name[:6] 分组，过滤数据不足的组
# 
#   3. 全局类型排序
#      Counter 统计全量类型，取 TOP_N_GENRES 个
# 
#   4. 逐类型热力图生成（循环 top_genres）
#      对每个类型 g:
#      a) 创建 matrix: len(names) × 24，初始化为0
#      b) 对每个节假日 i:
#         - 计算该节假日的逐小时数据 h_hourly
#         - 对每小时 h: matrix[i,h] = h_hourly[g][h] - nh_hourly[g][h]
#      c) imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
#      d) annotate_heatmap(ax, matrix, fmt='.1f', fs=6)
#      e) 保存为 k3_genre_{g[:10]}_hourly_heatmap.png
# 
#   5. CSV 输出（长格式，所有类型合并）
#      表头: ['holiday_name', 'genre', 'hour', 'diff_from_nonholiday']
#      每节假日 × 每类型 × 每小时 一条记录
#      适合后续导入 pandas 做进一步分析
# ═══════════════════════════════════════════════════════════════════════

def dim_k3_per_holiday_hourly_genre(seeker_genres: list[dict]):
    """Per-holiday hourly genre heatmap vs non-holiday.
    每个节假日的逐小时类型热力图，与非节假日基线对比。"""
    log("=" * 50)  # 日志：分隔线
    log("K3: Per-Holiday Hourly Genre Heatmap")  # 日志：分析标题

    non_holiday_dates = set(r['date'] for r in seeker_genres if r['period'] != 'holiday')  # 非节假日日期集合
    nh_hourly = _genre_hourly_mention_counts(seeker_genres, non_holiday_dates)  # 非节假日逐小时基线

    holiday_groups = defaultdict(list)  # 按节假日名称分组
    for r in seeker_genres:  # 遍历所有记录
        if r['period'] == 'holiday':  # 如果是节假日记录
            name = r['holiday_name'][:6]  # 取前 6 字符作为分组键
            holiday_groups[name].append(r)  # 加入对应分组

    holiday_groups = {k: v for k, v in holiday_groups.items()  # 过滤：只保留数据量足够的节假日
                     if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:  # 如果没有符合条件的节假日
        log("  No holiday groups")  # 日志提示
        return  # 提前返回

    names = sorted(holiday_groups.keys())  # 排序后的节假日名称列表

    # Top 6 genres for heatmaps
    # 取前 TOP_N_GENRES 个热门类型生成热力图
    all_genre_totals: Counter = Counter()  # 全局计数器
    for r in seeker_genres:  # 遍历所有记录
        for g in r.get('genres', {'unknown'}):  # 遍历每个类型
            all_genre_totals[g] += 1  # 计数
    top_genres = [g for g, _ in all_genre_totals.most_common(TOP_N_GENRES)]  # 取前 TOP_N_GENRES 个

    for g in top_genres:  # 对每个热门类型生成一张热力图
        matrix = np.zeros((len(names), 24))  # 创建矩阵：行=节假日数，列=24 小时
        for i, name in enumerate(names):  # 遍历每个节假日
            group_dates = set(r['date'] for r in holiday_groups[name])  # 获取日期集合
            h_hourly = _genre_hourly_mention_counts(  # 计算该节假日逐小时类型数据
                holiday_groups[name], group_dates)
            for h in range(24):  # 遍历 24 小时
                matrix[i, h] = h_hourly.get(g, [0.0]*24)[h] - nh_hourly.get(g, [0.0]*24)[h]  # 计算差值

        fig, ax = plt.subplots(figsize=(16, max(4, len(names) * 0.3 + 2)))  # 创建图表
        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)  # 对称颜色范围的最大绝对值
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)  # 绘制热力图，红蓝配色
        annotate_heatmap(ax, matrix, fmt='.1f', fs=6)

        ax.set_xticks(range(24))  # x 轴刻度 0-23
        ax.set_xticklabels(range(24), fontsize=8)  # x 轴标签
        ax.set_yticks(range(len(names)))  # y 轴刻度：各节假日
        ax.set_yticklabels(names, fontsize=8)  # y 轴标签
        ax.set_xlabel('Hour of Day (UTC)')  # x 轴标签
        ax.set_title(f'Genre "{g}": Per-Holiday Hourly Diff from Non-Holiday', fontsize=11)  # 子图标题
        fig.colorbar(im, ax=ax, shrink=0.6, label='Diff (avg/hr)')  # 颜色条

        fig.tight_layout()  # 自动调整布局
        path = os.path.join(STEP_OUT, f'k3_genre_{g[:10]}_hourly_heatmap.png')  # 输出文件路径，类型名取前 10 字符
        fig.savefig(path)  # 保存图片
        plt.close(fig)  # 关闭图形
        log(f"Saved: {path}")  # 日志记录保存信息

    csv_path = os.path.join(STEP_OUT, 'k3_per_holiday_hourly_genre.csv')  # CSV 文件路径
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:  # 写入 CSV
        w = csv.writer(f)  # 创建 CSV 写入器
        w.writerow(['holiday_name', 'genre', 'hour', 'diff_from_nonholiday'])  # 表头
        for name in names:  # 遍历每个节假日
            group_dates = set(r['date'] for r in holiday_groups[name])  # 获取日期集合
            h_hourly = _genre_hourly_mention_counts(  # 计算该节假日逐小时类型数据
                holiday_groups[name], group_dates)
            for g in top_genres:  # 遍历每个热门类型
                for h in range(24):  # 遍历 24 小时
                    diff = (h_hourly.get(g, [0.0]*24)[h]  # 节假日该小时数据
                            - nh_hourly.get(g, [0.0]*24)[h])  # 减去非节假日基线
                    w.writerow([name, g, h, f'{diff:.4f}'])  # 写入该行
    log(f"Saved: {csv_path}")  # 日志记录 CSV 保存信息


# ═══════════════════════════════════════════════════════════════════════
#  K4: 各节假日逐小时类型 VS 工作日/周末 (Heatmap)
#  K4: Per-Holiday Hourly Genre vs Workday & Weekend
# ═══════════════════════════════════════════════════════════════════════
#  K4: 各节假日逐小时类型 VS 工作日/周末 (Dual Heatmap per Genre)
#  K4: Per-Holiday Hourly Genre vs Workday & Weekend
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】
#   每个热门类型生成一张双面板热力图（2行1列，RdBu_r 配色）
#   - 上面板: 节假日逐小时日均 - 工作日基线（差值）
#   - 下面板: 节假日逐小时日均 - 周末基线（差值）
#   行=节假日，列=0-23小时
#   每张图有独立颜色刻度（对称范围）
# 
# 【统计口径】
#   指标说明:
#     - 上面板值 = holiday_hourly_avg - workday_hourly_baseline
#     - 下面板值 = holiday_hourly_avg - weekend_hourly_baseline
#     - 工作日基线: r['period'] == 'workday' 的日期集合的逐小时日均值
#     - 周末基线: r['period'] == 'weekend' 的日期集合的逐小时日均值
#   数据过滤:
#     - 节假日按名称聚合，需满足 len(records) >= MIN_DATA_ROWS
#     - 只对前 6 个热门类型生成热力图（TOP_N_GENRES=6，缩窄范围便于阅读）
# 
# 【坐标轴】
#   X轴: 小时 (0-23 UTC)，标签字号7
#   Y轴: 节假日名称，标签字号7
#   每个子图独立颜色条，label='Diff'
#   标题分别标注 Diff: Holiday - Workday/Weekend Baseline
# 
# 【输出文件】
#   PNG: k4_genre_{safe_g[:10]}_hourly_heatmap.png（每个热门类型一张双面板图）
#   CSV: k4_per_holiday_hourly_genre_vs_workday_weekend.csv
#        （长格式: holiday, genre, hour, holiday_avg, workday_avg, weekend_avg）
# 
# 【特殊说明】
#   - 与 K3 的核心区别: K3 使用非节假日基线，K4 分别使用工作日/周末基线
#   - 与 K3 的另一个区别: 只取前6个类型（非 TOP_N_GENRES=20），避免图片太多
#   - 文件名中类型名做安全处理: 空格→下划线，斜杠→下划线
#   - 每个类型独立双面板 PNG，便于观察各类型在不同基线下节假日效应
# 
# 【代码中处理逻辑】
#   1. 工作/周末小时级基线
#      workday_dates = r['period']=='workday' 的日期 set
#      weekend_dates = r['period']=='weekend' 的日期 set
#      wd_hourly = _genre_hourly_mention_counts(seeker_genres, workday_dates)
#      we_hourly = _genre_hourly_mention_counts(seeker_genres, weekend_dates)
# 
#   2. 节假日分组（同 J3/J4）
#      按 holiday_name[:6] 分组，过滤
# 
#   3. 类型选取（影响每个类型一张图的数量）
#      Counter 统计全量类型，取前6个（非TOP_N_GENRES=20）
#      因为每张图是双面板 PNG，6 张图已经足够代表性
# 
#   4. 逐类型双面板热力图（循环 top_genres[:6]）
#      对每个类型 g:
#      a) 创建 matrix_wd / matrix_we: len(names) × 24
#      b) 对每个节假日 i:
#         - 计算该节假日的逐小时数据 h_hourly
#         - 对每小时 h:
#           matrix_wd[i,h] = h_hourly[g][h] - wd_hourly[g][h]
#           matrix_we[i,h] = h_hourly[g][h] - we_hourly[g][h]
#      c) subplots(2, 1) 创建双面板
#      d) 上面板: imshow(matrix_wd) + annotate_heatmap
#      e) 下面板: imshow(matrix_we) + annotate_heatmap
#      f) 各自设置 x/y 轴刻度和颜色条
#      g) safe_g = g.replace(' ', '_').replace('/', '_')[:10]
#      h) 保存为 k4_genre_{safe_g}_hourly_heatmap.png
# 
#   5. CSV 输出（长格式）
#      表头: ['holiday_name', 'genre', 'hour', 'holiday_avg', 'workday_avg', 'weekend_avg']
#      每节假日 × 每类型 × 每小时 一条记录
#      包含节假日原始值的绝对值（holiday_avg），方便同时对比绝对值和差值
# ═══════════════════════════════════════════════════════════════════════

def dim_k4_per_holiday_hourly_genre_vs_workday_weekend(
    seeker_genres: list[dict]):
    """Per-holiday hourly genre vs workday & weekend (heatmap).
    每个节假日的逐小时类型数据 vs 工作日和周末（热力图）。"""
    log("=" * 50)  # 日志：分隔线
    log("K4: Per-Holiday Hourly Genre vs Workday & Weekend")  # 日志：分析标题

    # 计算工作日和周末基线的逐小时数据
    workday_dates = set(r['date'] for r in seeker_genres if r['period'] == 'workday')
    weekend_dates = set(r['date'] for r in seeker_genres if r['period'] == 'weekend')
    wd_hourly = _genre_hourly_mention_counts(seeker_genres, workday_dates)
    we_hourly = _genre_hourly_mention_counts(seeker_genres, weekend_dates)

    for p, ph in [('workday', wd_hourly), ('weekend', we_hourly)]:
        total = sum(sum(v) for v in ph.values())
        log(f"  {p}: {total:.2f} avg hourly genre mentions")

    # 按节假日名称分组
    holiday_groups = defaultdict(list)
    for r in seeker_genres:
        if r['period'] == 'holiday':
            name = r['holiday_name'][:6]
            holiday_groups[name].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items() if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  No holiday groups")
        return

    names = sorted(holiday_groups.keys())

    # Top genres
    all_genre_totals: Counter = Counter()
    for r in seeker_genres:
        for g in r.get('genres', {'unknown'}):
            all_genre_totals[g] += 1
    top_genres = [g for g, _ in all_genre_totals.most_common(6)]  # 取前 6 个做热力图（太多难以阅读）

    for g in top_genres:
        # Matrix: rows=holidays, cols=24 hours, value = holiday_avg - workday_baseline
        matrix_wd = np.zeros((len(names), 24))
        matrix_we = np.zeros((len(names), 24))
        for i, name in enumerate(names):
            group_dates = set(r['date'] for r in holiday_groups[name])
            h_hourly = _genre_hourly_mention_counts(holiday_groups[name], group_dates)
            for h in range(24):
                matrix_wd[i, h] = h_hourly.get(g, [0.0]*24)[h] - wd_hourly.get(g, [0.0]*24)[h]
                matrix_we[i, h] = h_hourly.get(g, [0.0]*24)[h] - we_hourly.get(g, [0.0]*24)[h]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, max(6, len(names) * 0.6 + 2)))

        vmax1 = max(abs(matrix_wd.min()), abs(matrix_wd.max()), 0.01)
        im1 = ax1.imshow(matrix_wd, cmap='RdBu_r', aspect='auto', vmin=-vmax1, vmax=vmax1)
        annotate_heatmap(ax1, matrix_wd, fmt='.1f', fs=6)
        ax1.set_xticks(range(24))
        ax1.set_xticklabels(range(24), fontsize=7)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=7)
        ax1.set_title(f'Genre "{g}" — Diff: Holiday - Workday Baseline', fontsize=10)
        fig.colorbar(im1, ax=ax1, shrink=0.5, label='Diff')

        vmax2 = max(abs(matrix_we.min()), abs(matrix_we.max()), 0.01)
        im2 = ax2.imshow(matrix_we, cmap='RdBu_r', aspect='auto', vmin=-vmax2, vmax=vmax2)
        annotate_heatmap(ax2, matrix_we, fmt='.1f', fs=6)
        ax2.set_xticks(range(24))
        ax2.set_xticklabels(range(24), fontsize=7)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=7)
        ax2.set_xlabel('Hour of Day (UTC)')
        ax2.set_title(f'Genre "{g}" — Diff: Holiday - Weekend Baseline', fontsize=10)
        fig.colorbar(im2, ax=ax2, shrink=0.5, label='Diff')

        fig.suptitle(f'Per-Holiday Hourly Genre: "{g}" — Difference from Workday & Weekend', fontsize=12)
        fig.tight_layout()
        safe_g = g.replace(' ', '_').replace('/', '_')[:10]
        path = os.path.join(STEP_OUT, f'k4_genre_{safe_g}_hourly_heatmap.png')
        fig.savefig(path)
        plt.close(fig)
        log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'k4_per_holiday_hourly_genre_vs_workday_weekend.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['holiday_name', 'genre', 'hour', 'holiday_avg', 'workday_avg', 'weekend_avg'])
        for name in names:
            group_dates = set(r['date'] for r in holiday_groups[name])
            h_hourly = _genre_hourly_mention_counts(holiday_groups[name], group_dates)
            for g in top_genres:
                for h in range(24):
                    w.writerow([name, g, h,
                                f'{h_hourly.get(g, [0.0]*24)[h]:.4f}',
                                f'{wd_hourly.get(g, [0.0]*24)[h]:.4f}',
                                f'{we_hourly.get(g, [0.0]*24)[h]:.4f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main  主函数入口
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None):
    log("=" * 60)  # 日志：分隔线
    log("Step 5: Movie Genre Analysis")  # 日志：步骤标题
    log("=" * 60)  # 日志：分隔线

    if data is None:
        from movie.data_loader import load_all  # 延迟导入数据加载函数
        data = load_all()  # 加载所有数据
    seekers = data['seekers']  # 获取用户提问记录列表
    rows = data['rows']  # 获取所有行（包括系统回复）
    movie_info = data['movie_info']  # 获取电影信息字典（含类型信息）

    log(f"Building seeker-genre associations from {len(rows)} total rows ...")  # 日志：开始构建关联
    seeker_genres = _build_seeker_genres(seekers, rows, movie_info)  # 为用户提问记录增加类型信息
    log(f"Built {len(seeker_genres)} seeker records with genre info")  # 日志：构建完成

    # Section A: Weekly
    # A 部分：周周期分析
    log("")
    log("-" * 40)
    log("Section A: Weekly - Genre Distribution")
    log("-" * 40)

    dim_j1_holiday_vs_nonholiday_genre(seeker_genres)  # A1：节假日 vs 非节假日
    log("")
    dim_j2_holiday_workday_weekend_genre(seeker_genres)  # A2：节假日 vs 工作日 vs 周末
    log("")
    dim_j3_per_holiday_vs_nonholiday_genre(seeker_genres)  # A3：每个节假日 vs 非节假日
    log("")
    dim_j4_per_holiday_vs_workday_weekend_genre(seeker_genres)  # A4：每个节假日 vs 工作日和周末
    log("")
    dim_j5_per_holiday_genre_heatmap(seeker_genres)            # J5：热力图-节假日×电影类型
    # B 部分：逐小时分析
    log("")
    log("-" * 40)
    log("Section B: Hourly - Genre Distribution")
    log("-" * 40)

    dim_k1_hourly_holiday_vs_nonholiday_genre(seeker_genres)  # B1：逐小时节假日 vs 非节假日
    log("")
    dim_k2_hourly_holiday_workday_weekend_genre(seeker_genres)  # B2：逐小时三类周期
    log("")
    dim_k3_per_holiday_hourly_genre(seeker_genres)  # B3：逐小时各个节假日 vs 非节假日
    log("")
    dim_k4_per_holiday_hourly_genre_vs_workday_weekend(seeker_genres)  # B4：逐小时各个节假日 vs 工作日和周末

    log("")
    log("=" * 60)
    log(f"Step 5 complete! Results saved to {STEP_OUT}")  # 日志：步骤完成
    log("=" * 60)


if __name__ == '__main__':  # 如果该文件作为脚本直接运行
    main()  # 调用主函数入口
