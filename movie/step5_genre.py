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
import matplotlib  # 数据可视化基础库
matplotlib.use('Agg')  # 使用非交互式后端 Agg（适用于无 GUI 环境的图片生成）
import matplotlib.pyplot as plt  # pyplot 接口，用于绘制图表
import matplotlib.ticker as ticker  # 坐标轴刻度格式控制

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log  # 导入配置：步骤输出目录、最小数据行数、matplotlib 初始化、日志函数
from movie.step1_question_freq import (  # 从步骤 1 导入颜色常量
    COLOR_HOLIDAY, COLOR_NONHOLIDAY, COLOR_WORKDAY, COLOR_WEEKEND,  # 节假日/非节假日/工作日/周末的颜色
)

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
    same conversation.  Returns list of dicts with 'genre' key added,
    one entry per seeker x movie_id found.
    通过匹配同一会话中的系统回复，为用户提问记录增加影片类型信息。
    返回字典列表，每个字典增加了 'genres' 键。
    每条记录对应该用户提问中提到的每部电影。
    """
    # Build conv_id -> list of system reply processed fields
    # 构建会话 ID -> 系统回复的处理后文本列表
    # conv_id base: first 9 chars of session_id + rest after last '_'
    # conv_id 格式：会话ID_当前轮次/总轮次
    # Actually conv_id format: {session_id}_{current_turn}/{total_turns}
    # conv_id 格式实际为：{session_id}_{current_turn}/{total_turns}
    # We match by session_id (first 9 chars of conv_id)
    # 我们通过 conv_id 的前 9 个字符（session_id）进行匹配
    conv_system = defaultdict(list)  # 会话基标识符 -> 系统回复文本列表
    for row in all_rows:  # 遍历所有行（包括用户提问和系统回复）
        # row from load_all -> conv_id, processed_raw
        # 从 load_all 加载的行包含 conv_id 和 processed_raw 字段
        conv_id = row.get('conv_id', '')  # 获取会话 ID
        processed = row.get('processed_raw', row.get('processed', ''))  # 获取处理后文本（优先使用 processed_raw）
        is_seeker = row.get('is_seeker', False)  # 是否为用户提问（True 为用户，False 为系统）

        # SYSTEM reply (not seeker)
        # 只处理系统回复（不是用户提问）
        if not is_seeker and processed:  # 如果是系统回复且有文本内容
            # Extract base session from conv_id: everything before last '_'
            # 从 conv_id 中提取基会话标识：最后一个 '_' 之前的部分
            base_id = conv_id.rsplit('_', 1)[0] if '_' in conv_id else conv_id
            conv_system[base_id].append(processed)  # 将该回复文本加入对应的会话列表

    result = []  # 存储增加了影片类型信息的记录
    for r in seekers:  # 遍历每条用户提问记录
        conv_id = r.get('conv_id', '')  # 获取会话 ID
        base_id = conv_id.rsplit('_', 1)[0] if '_' in conv_id else conv_id  # 提取基会话标识
        system_msgs = conv_system.get(base_id, [])  # 获取该会话中所有系统回复文本

        # Collect all movie IDs from system replies in this conversation
        # 收集该会话所有系统回复中提到的电影 ID
        movie_ids = set()  # 使用集合自动去重
        for msg in system_msgs:  # 遍历每条系统回复
            movie_ids.update(_extract_movie_ids(str(msg)))  # 提取其中的电影 ID 并加入集合

        # Gather genres for these movie IDs
        # 收集这些电影 ID 对应的类型
        genres_found = set()  # 使用集合自动去重
        for mid in movie_ids:  # 遍历每部电影
            info = movie_info.get(mid, {})  # 从电影信息字典中查找该电影的信息
            if isinstance(info, dict):  # 如果电影信息是字典类型
                genre_list = info.get('genres', []) or []  # 获取电影类型列表，如果为 None 则取空列表
                if genre_list:  # 如果有类型信息
                    genres_found.update(g.strip() for g in genre_list if g.strip())  # 去除空格后加入集合

        rec = dict(r)  # 复制原始记录（避免修改原数据）
        rec['genres'] = genres_found if genres_found else {'unknown'}  # 如果找到类型则使用，否则标记为 'unknown'
        result.append(rec)  # 加入结果列表

    return result  # 返回扩展后的记录列表


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
#  A: Weekly period - genre  A：周周期 - 类型分析
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
    ax1.set_xticks(range(len(top_genres)))
    ax1.set_xticklabels(top_genres, rotation=45, ha='right', fontsize=7)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=7)
    ax1.set_title('Diff: Holiday Avg Daily - Workday Baseline', fontsize=10)
    fig.colorbar(im1, ax=ax1, shrink=0.5, label='Diff')

    vmax2 = max(abs(matrix_we.min()), abs(matrix_we.max()), 0.01)
    im2 = ax2.imshow(matrix_we, cmap='RdBu_r', aspect='auto', vmin=-vmax2, vmax=vmax2)
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
#  B: Hourly genre analysis  B：逐小时类型分析
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
        ax1.set_xticks(range(24))
        ax1.set_xticklabels(range(24), fontsize=7)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=7)
        ax1.set_title(f'Genre "{g}" — Diff: Holiday - Workday Baseline', fontsize=10)
        fig.colorbar(im1, ax=ax1, shrink=0.5, label='Diff')

        vmax2 = max(abs(matrix_we.min()), abs(matrix_we.max()), 0.01)
        im2 = ax2.imshow(matrix_we, cmap='RdBu_r', aspect='auto', vmin=-vmax2, vmax=vmax2)
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

def main():
    log("=" * 60)  # 日志：分隔线
    log("Step 5: Movie Genre Analysis")  # 日志：步骤标题
    log("=" * 60)  # 日志：分隔线

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

    # Section B: Hourly
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
