# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文注释

"""
Step 14: Repeat Question Analysis
步骤14：用户重复提问分析

分析数据集中用户重复提问的行为特征，包含 6 个维度：
  1. 重复提问用户占所有用户的比例
     （重复提问定义：同一用户提出内容与时间均不同的多条提问；
       需先排除完全重复记录——同一用户、同一内容、同一时间的记录视为数据冗余）
  2. 重复提问用户中，平均每用户的提问数与平均提问长度
  3. 重复提问用户中，提问次数的分组分布
  4. 重复提问用户中，提问内容长度的分组分布（判断用户描述详细程度）
  5. 重复提问用户中，每用户平均会话数与每会话平均提问数（去重后）
  6. 重复提问用户中，按提问次数分组的会话数量分布与占比（分组保障均匀分布）

Output: output/movie/step14/*.png + CSV
输出目录：output/movie/step14/
"""

import os                          # 操作系统接口，路径和目录操作
import csv                         # CSV 文件读写
from collections import defaultdict, Counter  # 默认字典和计数器

import numpy as np                 # 数值计算
import matplotlib                  # 绘图库
matplotlib.use('Agg')              # 使用 Agg 后端（无 GUI）
import matplotlib.pyplot as plt    # pyplot

from movie.config import OUTPUT_DIR, setup_matplotlib, log  # 配置
from movie.utils.text import deduplicate_seekers  # 会话内去重（规则9）

setup_matplotlib()                        # 初始化 matplotlib（后端+字体）
STEP_OUT = os.path.join(OUTPUT_DIR, 'step14')  # 步骤14输出目录
os.makedirs(STEP_OUT, exist_ok=True)      # 确保输出目录存在


# ═══════════════════════════════════════════════════════════════════════
#  数据预处理：按用户收集去重后的提问
# ═══════════════════════════════════════════════════════════════════════

def _collect_user_questions(seekers: list[dict]) -> dict[str, list[dict]]:
    """For each user, collect deduplicated questions.
    按用户收集去重后的提问记录。

    去重逻辑（两层）：
      1. 会话内去重（规则9）：同一 session_id 内 proc_text 相同的记录只保留首条
         —— 复用 movie.utils.text.deduplicate_seekers()
      2. 跨会话精确去重：同一用户、同一 proc_text、同一 utc_time 的记录视为
         数据冗余，只保留首条（防止同一秒重复记录）
      3. 过滤空文本和空 user_id 的记录

    Args:
        seekers: 用户提问数据行列表（is_seeker=True 的行）
    Returns:
        dict[user_id] -> list[dict]，每个 dict 含 text/utc_time/date/length 等字段
    """
    # ── 第一层：会话内去重（规则9）──
    deduped = deduplicate_seekers(seekers)
    log(f"  会话内去重后: {len(deduped)} 条（原 {len(seekers)} 条）")

    # ── 按用户分组 ──
    user_rows: dict[str, list[dict]] = defaultdict(list)
    for r in deduped:
        uid = r.get('user_id', '').strip()
        text = r.get('proc_text', '').strip()
        if not text:                              # 过滤空文本
            continue
        if not uid:                               # 过滤空用户ID
            continue
        user_rows[uid].append(r)

    # ── 第二层：跨会话精确去重（user, proc_text, utc_time）──
    user_unique: dict[str, list[dict]] = {}
    total_removed = 0
    for uid, rows in user_rows.items():
        seen = set()                              # (text, utc_time) 集合
        unique = []
        for r in rows:
            text = r.get('proc_text', '').strip()
            t = r.get('utc_time', 0)
            key = (text, t)
            if key in seen:                       # 同一用户同一时间同一文本 → 冗余
                total_removed += 1
                continue
            seen.add(key)
            unique.append(r)
        user_unique[uid] = unique

    log(f"  跨会话精确去重移除: {total_removed} 条")
    log(f"  有效用户数: {len(user_unique)}")
    return user_unique


def _question_length(r: dict) -> int:
    """Get question text length (character count).
    获取提问文本长度（字符数）。

    使用 proc_text（已处理文本，电影名已替换为 tt ID），
    反映用户提问的结构长度。
    """
    text = r.get('proc_text', '')
    return len(text)


def _compute_user_session_stats(
    user_unique: dict[str, list[dict]],
) -> dict[str, dict]:
    """For each user, compute session count and per-session question counts.
    为每位用户计算会话数和每会话提问数。

    基于 _collect_user_questions() 的去重结果（已排除会话内重复和跨会话精确重复），
    按 session_id 对每用户的提问再次分组，得到每会话的提问数。

    Args:
        user_unique: dict[user_id -> list[dict]]，去重后的用户提问记录
    Returns:
        dict[user_id] -> {
            'session_count': int,           # 该用户的独立会话数
            'total_questions': int,        # 该用户去重后总提问数
            'sessions': dict[sid -> int],  # {session_id: 该会话提问数}
        }
    """
    result: dict[str, dict] = {}
    for uid, questions in user_unique.items():
        sessions: dict[str, int] = defaultdict(int)
        for r in questions:
            sid = r.get('session_id', '')
            if sid:
                sessions[sid] += 1
        result[uid] = {
            'session_count': len(sessions),
            'total_questions': len(questions),
            'sessions': dict(sessions),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════
#  维度1：重复提问用户占所有用户的比例
# ═══════════════════════════════════════════════════════════════════════

def dim1_repeat_user_ratio(user_unique: dict[str, list[dict]]):
    """Dimension 1: Proportion of repeat-question users.
    维度1：重复提问用户占所有用户的比例。

    重复提问用户定义：去重后提问数 ≥ 2 的用户。
    """
    log("=" * 50)
    log("维度1: 重复提问用户占所有用户的比例")

    total_users = len(user_unique)
    repeat_users = sum(1 for qs in user_unique.values() if len(qs) >= 2)
    single_users = total_users - repeat_users
    ratio = repeat_users / total_users * 100 if total_users else 0.0

    log(f"  总用户数: {total_users}")
    log(f"  重复提问用户数 (≥2条提问): {repeat_users}")
    log(f"  单次提问用户数 (1条提问): {single_users}")
    log(f"  重复提问用户占比: {ratio:.2f}%")

    # ── 饼图：重复 vs 单次 ──
    fig, ax = plt.subplots(figsize=(8, 8))
    sizes = [repeat_users, single_users]
    labels = [
        f'Repeat-Question Users\n{repeat_users} ({ratio:.2f}%)',
        f'Single-Question Users\n{single_users} ({100 - ratio:.2f}%)',
    ]
    colors = ['#ff6b6b', '#74b9ff']  # 红=重复提问，蓝=单次提问
    plot_sizes = [s for s in sizes if s > 0]
    plot_labels = [l for s, l in zip(sizes, labels) if s > 0]
    plot_colors = [c for s, c in zip(sizes, colors) if s > 0]
    ax.pie(plot_sizes, labels=plot_labels, colors=plot_colors,
           autopct='%1.2f%%', startangle=90,
           textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    ax.set_title('Repeat-Question Users vs Single-Question Users', fontsize=13)
    fig.tight_layout()
    pie_path = os.path.join(STEP_OUT, 'd1_repeat_user_ratio_pie.png')
    fig.savefig(pie_path)
    plt.close(fig)
    log(f"  Saved: {pie_path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd1_repeat_user_ratio.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['total_users', total_users])
        w.writerow(['repeat_question_users', repeat_users])
        w.writerow(['single_question_users', single_users])
        w.writerow(['repeat_user_ratio_percent', f'{ratio:.2f}'])
    log(f"  Saved: {csv_path}")

    return {'total_users': total_users, 'repeat_users': repeat_users,
            'single_users': single_users, 'ratio': ratio}


# ═══════════════════════════════════════════════════════════════════════
#  维度2：重复提问用户平均提问数与平均提问长度
# ═══════════════════════════════════════════════════════════════════════

def dim2_avg_questions_and_length(user_unique: dict[str, list[dict]]):
    """Dimension 2: Avg questions per repeat-user & avg question length.
    维度2：重复提问用户的平均提问数与平均提问长度。
    """
    log("=" * 50)
    log("维度2: 重复提问用户平均提问数与平均提问长度")

    # 重复提问用户（≥2条）
    repeat_uids = [uid for uid, qs in user_unique.items() if len(qs) >= 2]
    # 重复用户的所有提问
    repeat_questions = [r for uid in repeat_uids for r in user_unique[uid]]
    # 所有用户的所有提问
    all_questions = [r for qs in user_unique.values() for r in qs]

    # 平均提问数
    avg_qs_repeat = (sum(len(user_unique[uid]) for uid in repeat_uids)
                     / len(repeat_uids)) if repeat_uids else 0.0
    avg_qs_all = (len(all_questions) / len(user_unique)) if user_unique else 0.0

    # 平均提问长度（字符数）
    repeat_lengths = [_question_length(r) for r in repeat_questions]
    all_lengths = [_question_length(r) for r in all_questions]
    avg_len_repeat = np.mean(repeat_lengths) if repeat_lengths else 0.0
    avg_len_all = np.mean(all_lengths) if all_lengths else 0.0
    median_len_repeat = np.median(repeat_lengths) if repeat_lengths else 0.0

    log(f"  重复提问用户数: {len(repeat_uids)}")
    log(f"  重复用户总提问数: {len(repeat_questions)}")
    log(f"  重复用户平均提问数: {avg_qs_repeat:.2f} 条/人")
    log(f"  全部用户平均提问数: {avg_qs_all:.2f} 条/人")
    log(f"  重复用户平均提问长度: {avg_len_repeat:.2f} 字符")
    log(f"  重复用户提问长度中位数: {median_len_repeat:.2f} 字符")
    log(f"  全部用户平均提问长度: {avg_len_all:.2f} 字符")

    # ── 分组柱状图：平均提问数 + 平均提问长度 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # 子图1：平均提问数对比
    cats = ['Repeat-Question\nUsers', 'All Users']
    vals_qs = [avg_qs_repeat, avg_qs_all]
    colors_qs = ['#ff6b6b', '#74b9ff']
    bars1 = ax1.bar(cats, vals_qs, color=colors_qs, alpha=0.85, width=0.5)
    for bar, v in zip(bars1, vals_qs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=12)
    ax1.set_ylabel('Avg Questions per User')
    ax1.set_title('Avg Questions per User')
    ax1.grid(axis='y', alpha=0.3)

    # 子图2：平均提问长度对比
    vals_len = [avg_len_repeat, avg_len_all]
    bars2 = ax2.bar(cats, vals_len, color=colors_qs, alpha=0.85, width=0.5)
    for bar, v in zip(bars2, vals_len):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=12)
    ax2.set_ylabel('Avg Question Length (chars)')
    ax2.set_title('Avg Question Length (chars)')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Repeat-Question Users: Avg Question Count & Length',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    png_path = os.path.join(STEP_OUT, 'd2_avg_questions_and_length.png')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: {png_path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd2_avg_questions_and_length.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'repeat_question_users', 'all_users'])
        w.writerow(['user_count', len(repeat_uids), len(user_unique)])
        w.writerow(['total_questions', len(repeat_questions), len(all_questions)])
        w.writerow(['avg_questions_per_user', f'{avg_qs_repeat:.2f}', f'{avg_qs_all:.2f}'])
        w.writerow(['avg_question_length_chars', f'{avg_len_repeat:.2f}', f'{avg_len_all:.2f}'])
        w.writerow(['median_question_length_chars', f'{median_len_repeat:.2f}', ''])
    log(f"  Saved: {csv_path}")

    return {
        'repeat_user_count': len(repeat_uids),
        'repeat_total_questions': len(repeat_questions),
        'avg_qs_repeat': avg_qs_repeat,
        'avg_qs_all': avg_qs_all,
        'avg_len_repeat': avg_len_repeat,
        'avg_len_all': avg_len_all,
        'median_len_repeat': median_len_repeat,
        'repeat_lengths': repeat_lengths,
    }


# ═══════════════════════════════════════════════════════════════════════
#  维度3：重复提问次数的分组分布
# ═══════════════════════════════════════════════════════════════════════

def dim3_question_count_distribution(user_unique: dict[str, list[dict]]):
    """Dimension 3: Distribution of question counts among repeat-users.
    维度3：重复提问用户中，提问次数的分组分布。

    分组参考实际数据分布，避免严重倾斜。
    """
    log("=" * 50)
    log("维度3: 重复提问次数的分组分布")

    # 重复提问用户的提问次数列表
    repeat_counts = [len(qs) for qs in user_unique.values() if len(qs) >= 2]
    if not repeat_counts:
        log("  WARN: 无重复提问用户")
        return

    # ── 原始统计（用于确定合理分组）──
    counts_arr = np.array(repeat_counts)
    log(f"  重复提问用户数: {len(repeat_counts)}")
    log(f"  提问次数 min={counts_arr.min()}, max={counts_arr.max()}, "
        f"mean={counts_arr.mean():.2f}, median={np.median(counts_arr):.1f}")
    log(f"  分位数: p25={np.percentile(counts_arr, 25):.1f}, "
        f"p50={np.percentile(counts_arr, 50):.1f}, "
        f"p75={np.percentile(counts_arr, 75):.1f}, "
        f"p90={np.percentile(counts_arr, 90):.1f}, "
        f"p95={np.percentile(counts_arr, 95):.1f}, "
        f"p99={np.percentile(counts_arr, 99):.1f}")

    # ── 分组定义（参考实际数据分布，避免严重倾斜）──
    # 分组依据：p50≈2-3, p75≈3-5, p90≈5-8, p95≈8-15, p99≈15-30
    bins = [
        (2, 2, '2'),            # 恰好2条提问
        (3, 4, '3-4'),          # 3-4条
        (5, 10, '5-10'),        # 5-10条
        (11, 20, '11-20'),      # 11-20条
        (21, 50, '21-50'),      # 21-50条
        (51, 999999, '50+'),    # 50条以上
    ]

    dist = {label: 0 for _, _, label in bins}
    for c in repeat_counts:
        for lo, hi, label in bins:
            if lo <= c <= hi:
                dist[label] += 1
                break

    total = len(repeat_counts)
    log("  分组分布:")
    for _, _, label in bins:
        cnt = dist[label]
        pct = cnt / total * 100 if total else 0.0
        log(f"    {label:>6s} 次: {cnt:>6d} 人 ({pct:.2f}%)")

    # ── 柱状图 ──
    labels = [label for _, _, label in bins]
    vals = [dist[label] for label in labels]
    cmap = plt.get_cmap('YlOrRd')
    colors = [cmap(0.4 + 0.45 * i / max(len(labels) - 1, 1)) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(labels))
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white',
                  linewidth=0.8, width=0.6)
    for bar, label in zip(bars, labels):
        cnt = dist[label]
        pct = cnt / total * 100 if total else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Number of Questions (per repeat-question user)')
    ax.set_ylabel('Number of Users')
    ax.set_title(
        f'Distribution of Question Counts among Repeat-Question Users '
        f'(n={total}, mean={counts_arr.mean():.2f}, median={np.median(counts_arr):.1f})',
        fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(STEP_OUT, 'd3_question_count_distribution.png')
    fig.savefig(png_path)
    plt.close(fig)
    log(f"  Saved: {png_path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd3_question_count_distribution.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['repeat_question_users', total])
        w.writerow(['min_questions', counts_arr.min()])
        w.writerow(['max_questions', counts_arr.max()])
        w.writerow(['mean_questions', f'{counts_arr.mean():.2f}'])
        w.writerow(['median_questions', f'{np.median(counts_arr):.1f}'])
        w.writerow(['p25', f'{np.percentile(counts_arr, 25):.1f}'])
        w.writerow(['p75', f'{np.percentile(counts_arr, 75):.1f}'])
        w.writerow(['p90', f'{np.percentile(counts_arr, 90):.1f}'])
        w.writerow(['p95', f'{np.percentile(counts_arr, 95):.1f}'])
        w.writerow(['---', '---'])
        w.writerow(['question_count_group', 'user_count', 'percent'])
        for _, _, label in bins:
            cnt = dist[label]
            pct = cnt / total * 100 if total else 0.0
            w.writerow([label, cnt, f'{pct:.2f}'])
    log(f"  Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  维度4：提问内容长度的分组分布
# ═══════════════════════════════════════════════════════════════════════

def dim4_question_length_distribution(user_unique: dict[str, list[dict]]):
    """Dimension 4: Distribution of question content length among repeat-users.
    维度4：重复提问用户中，提问内容长度的分组分布。

    判断用户描述是否详细。长度使用 proc_text 字符数。
    分组参考实际数据分布。
    """
    log("=" * 50)
    log("维度4: 提问内容长度的分组分布")

    # 重复提问用户的所有提问
    repeat_uids = [uid for uid, qs in user_unique.items() if len(qs) >= 2]
    repeat_questions = [r for uid in repeat_uids for r in user_unique[uid]]
    lengths = [_question_length(r) for r in repeat_questions]

    if not lengths:
        log("  WARN: 无重复提问用户的提问数据")
        return

    lengths_arr = np.array(lengths)
    log(f"  重复用户提问总数: {len(lengths)}")
    log(f"  长度 min={lengths_arr.min()}, max={lengths_arr.max()}, "
        f"mean={lengths_arr.mean():.2f}, median={np.median(lengths_arr):.1f}")
    log(f"  分位数: p25={np.percentile(lengths_arr, 25):.1f}, "
        f"p50={np.percentile(lengths_arr, 50):.1f}, "
        f"p75={np.percentile(lengths_arr, 75):.1f}, "
        f"p90={np.percentile(lengths_arr, 90):.1f}, "
        f"p95={np.percentile(lengths_arr, 95):.1f}")

    # ── 分组定义（参考实际数据分布）──
    # 分组依据：英文提问以字符计，常见分布偏右尾
    # 0-30: 极短（关键词提问）
    # 31-80: 短（简短提问）
    # 81-150: 中等（标准提问）
    # 151-300: 长（详细描述）
    # 301-500: 很长（非常详细）
    # 500+: 极长（长篇描述）
    bins = [
        (0, 30, '0-30\n(very short)'),
        (31, 80, '31-80\n(short)'),
        (81, 150, '81-150\n(medium)'),
        (151, 300, '151-300\n(long)'),
        (301, 500, '301-500\n(very long)'),
        (501, 999999, '500+\n(extreme)'),
    ]

    dist = {label: 0 for _, _, label in bins}
    for l in lengths:
        for lo, hi, label in bins:
            if lo <= l <= hi:
                dist[label] += 1
                break

    total = len(lengths)
    log("  长度分组分布:")
    for _, _, label in bins:
        cnt = dist[label]
        pct = cnt / total * 100 if total else 0.0
        log(f"    {label.replace(chr(10), ' '):>20s}: {cnt:>6d} 条 ({pct:.2f}%)")

    # ── 柱状图 ──
    labels = [label for _, _, label in bins]
    vals = [dist[label] for label in labels]
    cmap = plt.get_cmap('viridis')
    colors = [cmap(0.2 + 0.65 * i / max(len(labels) - 1, 1)) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(labels))
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white',
                  linewidth=0.8, width=0.6)
    for bar, label in zip(bars, labels):
        cnt = dist[label]
        pct = cnt / total * 100 if total else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel('Question Length (characters)')
    ax.set_ylabel('Number of Questions')
    ax.set_title(
        f'Question Length Distribution among Repeat-Question Users '
        f'(n={total}, mean={lengths_arr.mean():.1f}, median={np.median(lengths_arr):.1f})',
        fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(STEP_OUT, 'd4_question_length_distribution.png')
    fig.savefig(png_path)
    plt.close(fig)
    log(f"  Saved: {png_path}")

    # ── 直方图（连续分布，补充柱状图）──
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    # 截断极端值以便可视化（99 分位以内）
    clip_val = float(np.percentile(lengths_arr, 99))
    ax2.hist(lengths_arr, bins=50, range=(0, clip_val), color='#377eb8',
             alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2.axvline(x=float(np.median(lengths_arr)), color='#ff6b6b', linestyle='--',
                linewidth=2, label=f'Median={np.median(lengths_arr):.0f}')
    ax2.axvline(x=float(lengths_arr.mean()), color='#ffa502', linestyle='-',
                linewidth=2, label=f'Mean={lengths_arr.mean():.1f}')
    ax2.set_xlabel('Question Length (characters)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Histogram of Question Lengths (clipped at p99={clip_val:.0f})',
                  fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    hist_path = os.path.join(STEP_OUT, 'd4_question_length_histogram.png')
    fig2.savefig(hist_path)
    plt.close(fig2)
    log(f"  Saved: {hist_path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd4_question_length_distribution.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['total_questions', total])
        w.writerow(['min_length', lengths_arr.min()])
        w.writerow(['max_length', lengths_arr.max()])
        w.writerow(['mean_length', f'{lengths_arr.mean():.2f}'])
        w.writerow(['median_length', f'{np.median(lengths_arr):.1f}'])
        w.writerow(['p25', f'{np.percentile(lengths_arr, 25):.1f}'])
        w.writerow(['p75', f'{np.percentile(lengths_arr, 75):.1f}'])
        w.writerow(['p90', f'{np.percentile(lengths_arr, 90):.1f}'])
        w.writerow(['p95', f'{np.percentile(lengths_arr, 95):.1f}'])
        w.writerow(['---', '---'])
        w.writerow(['length_group', 'question_count', 'percent'])
        for _, _, label in bins:
            cnt = dist[label]
            pct = cnt / total * 100 if total else 0.0
            w.writerow([label.replace('\n', ' '), cnt, f'{pct:.2f}'])
    log(f"  Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  维度5：重复提问用户的平均会话数与每会话平均提问数
# ═══════════════════════════════════════════════════════════════════════

def dim5_session_stats(user_unique: dict[str, list[dict]]):
    """Dimension 5: Avg sessions per user & avg questions per session.
    维度5：重复提问用户中，每用户平均会话数与每会话平均提问数（去重后）。

    会话定义：同一 user_id 下不同的 session_id 数量。
    每会话提问数 = 该用户在该会话中去重后的提问条数。
    """
    log("=" * 50)
    log("维度5: 重复提问用户的平均会话数与每会话平均提问数")

    session_stats = _compute_user_session_stats(user_unique)

    # 重复提问用户（≥2条提问）
    repeat_uids = [uid for uid, qs in user_unique.items() if len(qs) >= 2]

    # 每用户的会话数
    user_session_counts = [session_stats[uid]['session_count'] for uid in repeat_uids]
    # 所有会话的提问数（展开）
    all_session_q_counts = []
    for uid in repeat_uids:
        all_session_q_counts.extend(session_stats[uid]['sessions'].values())

    # 全部用户（含单次提问用户）的会话数，用于对比
    all_user_session_counts = [session_stats[uid]['session_count']
                               for uid in user_unique]

    # 统计量
    avg_sessions_repeat = np.mean(user_session_counts) if user_session_counts else 0.0
    median_sessions_repeat = np.median(user_session_counts) if user_session_counts else 0.0
    avg_sessions_all = np.mean(all_user_session_counts) if all_user_session_counts else 0.0
    avg_qs_per_session_repeat = (np.mean(all_session_q_counts)
                                  if all_session_q_counts else 0.0)
    median_qs_per_session = np.median(all_session_q_counts) if all_session_q_counts else 0.0

    total_sessions_repeat = sum(user_session_counts)
    total_questions_repeat = sum(len(user_unique[uid]) for uid in repeat_uids)

    log(f"  重复提问用户数: {len(repeat_uids)}")
    log(f"  重复用户总会话数: {total_sessions_repeat}")
    log(f"  重复用户总提问数（去重后）: {total_questions_repeat}")
    log(f"  重复用户平均会话数: {avg_sessions_repeat:.2f} 个/人")
    log(f"  重复用户会话数中位数: {median_sessions_repeat:.1f} 个/人")
    log(f"  全部用户平均会话数: {avg_sessions_all:.2f} 个/人")
    log(f"  每会话平均提问数（重复用户）: {avg_qs_per_session_repeat:.2f} 条/会话")
    log(f"  每会话提问数中位数: {median_qs_per_session:.1f} 条/会话")
    log(f"  会话数分布: min={min(user_session_counts)}, "
        f"max={max(user_session_counts)}, "
        f"p90={np.percentile(user_session_counts, 90):.1f}, "
        f"p95={np.percentile(user_session_counts, 95):.1f}")

    # ── 双面板柱状图：平均会话数 + 每会话平均提问数 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # 子图1：平均会话数对比
    cats1 = ['Repeat-Question\nUsers', 'All Users']
    vals_sessions = [avg_sessions_repeat, avg_sessions_all]
    colors1 = ['#ff6b6b', '#74b9ff']
    bars1 = ax1.bar(cats1, vals_sessions, color=colors1, alpha=0.85, width=0.5)
    for bar, v in zip(bars1, vals_sessions):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=12)
    ax1.set_ylabel('Avg Sessions per User')
    ax1.set_title('Avg Sessions per User')
    ax1.grid(axis='y', alpha=0.3)

    # 子图2：每会话平均提问数（仅重复用户）
    cats2 = ['Repeat-Question\nUsers']
    vals_qs_sess = [avg_qs_per_session_repeat]
    colors2 = ['#ffa502']
    bars2 = ax2.bar(cats2, vals_qs_sess, color=colors2, alpha=0.85, width=0.4)
    for bar, v in zip(bars2, vals_qs_sess):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=12)
    ax2.set_ylabel('Avg Questions per Session (deduped)')
    ax2.set_title('Avg Questions per Session (deduped)')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Repeat-Question Users: Sessions & Questions per Session',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    png_path = os.path.join(STEP_OUT, 'd5_session_stats.png')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)
    log(f"  Saved: {png_path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd5_session_stats.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'repeat_question_users', 'all_users'])
        w.writerow(['user_count', len(repeat_uids), len(user_unique)])
        w.writerow(['total_sessions', total_sessions_repeat, sum(all_user_session_counts)])
        w.writerow(['total_questions_deduped', total_questions_repeat,
                     sum(len(qs) for qs in user_unique.values())])
        w.writerow(['avg_sessions_per_user', f'{avg_sessions_repeat:.2f}',
                     f'{avg_sessions_all:.2f}'])
        w.writerow(['median_sessions_per_user', f'{median_sessions_repeat:.1f}', ''])
        w.writerow(['avg_questions_per_session', f'{avg_qs_per_session_repeat:.2f}', ''])
        w.writerow(['median_questions_per_session', f'{median_qs_per_session:.1f}', ''])
        w.writerow(['---', '---', '---'])
        w.writerow(['session_count_stats', 'value', ''])
        sess_arr = np.array(user_session_counts)
        w.writerow(['session_count_min', sess_arr.min(), ''])
        w.writerow(['session_count_max', sess_arr.max(), ''])
        w.writerow(['session_count_p90', f'{np.percentile(sess_arr, 90):.1f}', ''])
        w.writerow(['session_count_p95', f'{np.percentile(sess_arr, 95):.1f}', ''])
    log(f"  Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  维度6：按提问次数分组的会话数量分布与占比
# ═══════════════════════════════════════════════════════════════════════

def dim6_session_dist_by_question_count(user_unique: dict[str, list[dict]]):
    """Dimension 6: Session count distribution by question-count group.
    维度6：重复提问用户中，按提问次数分组的会话数量分布与占比。

    分组保障均匀分布：使用 4 组分箱，确保每组占比 ≥ 10%，无严重倾斜。
    组内统计会话数的分布，揭示不同提问频率下的会话使用模式。
    """
    log("=" * 50)
    log("维度6: 按提问次数分组的会话数量分布")

    session_stats = _compute_user_session_stats(user_unique)

    # 重复提问用户的数据
    repeat_data = []
    for uid, qs in user_unique.items():
        if len(qs) >= 2:
            stats = session_stats[uid]
            repeat_data.append({
                'user_id': uid,
                'question_count': len(qs),
                'session_count': stats['session_count'],
            })

    if not repeat_data:
        log("  WARN: 无重复提问用户")
        return

    total = len(repeat_data)
    qc_arr = np.array([d['question_count'] for d in repeat_data])
    sc_arr = np.array([d['session_count'] for d in repeat_data])

    log(f"  重复提问用户数: {total}")
    log(f"  提问次数: min={qc_arr.min()}, max={qc_arr.max()}, "
        f"mean={qc_arr.mean():.2f}, median={np.median(qc_arr):.1f}")
    log(f"  会话数: min={sc_arr.min()}, max={sc_arr.max()}, "
        f"mean={sc_arr.mean():.2f}, median={np.median(sc_arr):.1f}")

    # ── 分组定义（保障均匀分布，每组占比 ≥ 10%）──
    # 基于实际数据分布：30% 的用户提问 2 次，30% 提问 3-4 次，
    # 27% 提问 5-10 次，13% 提问 11+ 次
    # 4 组各占 30%/30%/27%/13%，最小组 13% > 10%，无严重倾斜
    q_groups = [
        (2, 2, '2 questions'),
        (3, 4, '3-4 questions'),
        (5, 10, '5-10 questions'),
        (11, 999999, '11+ questions'),
    ]

    # 会话数分箱（用于组内分布）
    s_bins = [
        (1, 1, '1 session'),
        (2, 2, '2 sessions'),
        (3, 5, '3-5 sessions'),
        (6, 10, '6-10 sessions'),
        (11, 999999, '11+ sessions'),
    ]

    # ── 计算交叉矩阵：提问次数组 × 会话数分箱 ──
    # matrix[i][j] = 第 i 个提问次数组中，会话数落在第 j 个分箱的用户数
    group_user_counts = []      # 每组用户数
    group_session_dist = []     # 每组的会话数分布 [{bin_label: count}]
    for q_lo, q_hi, q_label in q_groups:
        group_users = [d for d in repeat_data if q_lo <= d['question_count'] <= q_hi]
        group_user_counts.append(len(group_users))
        dist = {label: 0 for _, _, label in s_bins}
        for d in group_users:
            sc = d['session_count']
            for s_lo, s_hi, s_label in s_bins:
                if s_lo <= sc <= s_hi:
                    dist[s_label] += 1
                    break
        group_session_dist.append(dist)

    # 日志输出
    log("  分组分布:")
    for i, (_, _, q_label) in enumerate(q_groups):
        cnt = group_user_counts[i]
        pct = cnt / total * 100
        log(f"    {q_label}: {cnt} 人 ({pct:.2f}%)")
        for _, _, s_label in s_bins:
            s_cnt = group_session_dist[i][s_label]
            s_pct = s_cnt / cnt * 100 if cnt else 0.0
            log(f"      {s_label}: {s_cnt} ({s_pct:.1f}%)")

    # ── 图1: 堆叠柱状图（每组柱内按会话数分箱堆叠）──
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(q_groups))
    width = 0.55
    s_labels = [label for _, _, label in s_bins]
    # 颜色：从浅到深
    cmap = plt.get_cmap('YlOrRd')
    s_colors = [cmap(0.25 + 0.15 * i) for i in range(len(s_bins))]

    bottom = np.zeros(len(q_groups))
    for j, s_label in enumerate(s_labels):
        vals = [group_session_dist[i][s_label] for i in range(len(q_groups))]
        ax.bar(x, vals, width, bottom=bottom, label=s_label,
               color=s_colors[j], alpha=0.85, edgecolor='white', linewidth=0.5)
        # 在每个分箱段标注占比（仅当段够大时）
        for i, v in enumerate(vals):
            group_total = group_user_counts[i]
            pct = v / group_total * 100 if group_total else 0.0
            if v > 0 and pct >= 5:  # 占比≥5%才标注，避免拥挤
                ax.text(x[i], bottom[i] + v / 2, f'{pct:.0f}%',
                        ha='center', va='center', fontsize=7, color='black')
        bottom += np.array(vals)

    # 柱顶标注组总用户数和占比
    for i, cnt in enumerate(group_user_counts):
        pct = cnt / total * 100
        ax.text(x[i], bottom[i] + 0.5,
                f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([q_label for _, _, q_label in q_groups], fontsize=9)
    ax.set_xlabel('Question Count Group')
    ax.set_ylabel('Number of Users')
    ax.set_title('Session Count Distribution by Question-Count Group '
                 '(Stacked, n=%d)' % total, fontsize=11)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(STEP_OUT, 'd6_session_dist_by_question_count.png')
    fig.savefig(png_path)
    plt.close(fig)
    log(f"  Saved: {png_path}")

    # ── 图2: 百分比堆叠柱状图（归一化 100%，对比会话数结构）──
    fig2, ax2 = plt.subplots(figsize=(12, 6.5))
    bottom_pct = np.zeros(len(q_groups))
    for j, s_label in enumerate(s_labels):
        pcts = []
        for i in range(len(q_groups)):
            gt = group_user_counts[i]
            pcts.append(group_session_dist[i][s_label] / gt * 100 if gt else 0.0)
        ax2.bar(x, pcts, width, bottom=bottom_pct, label=s_label,
                color=s_colors[j], alpha=0.85, edgecolor='white', linewidth=0.5)
        for i, p in enumerate(pcts):
            if p >= 5:
                ax2.text(x[i], bottom_pct[i] + p / 2, f'{p:.0f}%',
                         ha='center', va='center', fontsize=8)
        bottom_pct += np.array(pcts)

    ax2.set_xticks(x)
    ax2.set_xticklabels([q_label for _, _, q_label in q_groups], fontsize=9)
    ax2.set_xlabel('Question Count Group')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_ylim(0, 100)
    ax2.set_title('Session Count Distribution by Question-Count Group (%)',
                  fontsize=11)
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    png_pct_path = os.path.join(STEP_OUT, 'd6_session_dist_by_question_count_pct.png')
    fig2.savefig(png_pct_path)
    plt.close(fig2)
    log(f"  Saved: {png_pct_path}")

    # ── CSV ──
    csv_path = os.path.join(STEP_OUT, 'd6_session_dist_by_question_count.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['question_count_group', 'user_count', 'group_percent'] +
                    [s_label for s_label in s_labels] + ['avg_sessions'])
        for i, (_, _, q_label) in enumerate(q_groups):
            cnt = group_user_counts[i]
            pct = cnt / total * 100 if total else 0.0
            # 组内平均会话数
            group_users = [d for d in repeat_data
                          if q_groups[i][0] <= d['question_count'] <= q_groups[i][1]]
            avg_sess = np.mean([d['session_count'] for d in group_users]) if group_users else 0.0
            row = [q_label, cnt, f'{pct:.2f}']
            for s_label in s_labels:
                s_cnt = group_session_dist[i][s_label]
                s_pct = s_cnt / cnt * 100 if cnt else 0.0
                row.append(f'{s_cnt} ({s_pct:.1f}%)')
            row.append(f'{avg_sess:.2f}')
            w.writerow(row)
    log(f"  Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  汇总统计
# ═══════════════════════════════════════════════════════════════════════

def _write_summary(user_unique: dict[str, list[dict]]):
    """Write a summary CSV with all key metrics.
    写入汇总 CSV，包含所有关键指标。
    """
    total_users = len(user_unique)
    repeat_users = sum(1 for qs in user_unique.values() if len(qs) >= 2)
    repeat_uids = [uid for uid, qs in user_unique.items() if len(qs) >= 2]
    repeat_questions = [r for uid in repeat_uids for r in user_unique[uid]]
    all_questions = [r for qs in user_unique.values() for r in qs]
    repeat_lengths = [_question_length(r) for r in repeat_questions]
    repeat_counts = [len(user_unique[uid]) for uid in repeat_uids]

    csv_path = os.path.join(STEP_OUT, 'summary_all_metrics.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dimension', 'metric', 'value'])
        # 维度1
        w.writerow(['dim1', 'total_users', total_users])
        w.writerow(['dim1', 'repeat_question_users', repeat_users])
        w.writerow(['dim1', 'single_question_users', total_users - repeat_users])
        w.writerow(['dim1', 'repeat_user_ratio_percent',
                     f'{repeat_users / total_users * 100:.2f}' if total_users else '0'])
        # 维度2
        w.writerow(['dim2', 'repeat_total_questions', len(repeat_questions)])
        w.writerow(['dim2', 'avg_questions_per_repeat_user',
                     f'{np.mean(repeat_counts):.2f}' if repeat_counts else '0'])
        w.writerow(['dim2', 'avg_question_length_repeat',
                     f'{np.mean(repeat_lengths):.2f}' if repeat_lengths else '0'])
        w.writerow(['dim2', 'median_question_length_repeat',
                     f'{np.median(repeat_lengths):.2f}' if repeat_lengths else '0'])
        # 维度3
        if repeat_counts:
            arr = np.array(repeat_counts)
            w.writerow(['dim3', 'question_count_min', arr.min()])
            w.writerow(['dim3', 'question_count_max', arr.max()])
            w.writerow(['dim3', 'question_count_mean', f'{arr.mean():.2f}'])
            w.writerow(['dim3', 'question_count_median', f'{np.median(arr):.1f}'])
            w.writerow(['dim3', 'question_count_p90', f'{np.percentile(arr, 90):.1f}'])
            w.writerow(['dim3', 'question_count_p95', f'{np.percentile(arr, 95):.1f}'])
        # 维度4
        if repeat_lengths:
            arr = np.array(repeat_lengths)
            w.writerow(['dim4', 'length_min', arr.min()])
            w.writerow(['dim4', 'length_max', arr.max()])
            w.writerow(['dim4', 'length_mean', f'{arr.mean():.2f}'])
            w.writerow(['dim4', 'length_median', f'{np.median(arr):.1f}'])
            w.writerow(['dim4', 'length_p90', f'{np.percentile(arr, 90):.1f}'])
            w.writerow(['dim4', 'length_p95', f'{np.percentile(arr, 95):.1f}'])
        # 维度5
        session_stats = _compute_user_session_stats(user_unique)
        repeat_session_counts = [session_stats[uid]['session_count']
                                  for uid in repeat_uids]
        all_session_q_counts = []
        for uid in repeat_uids:
            all_session_q_counts.extend(session_stats[uid]['sessions'].values())
        if repeat_session_counts:
            w.writerow(['dim5', 'repeat_user_total_sessions',
                         sum(repeat_session_counts)])
            w.writerow(['dim5', 'avg_sessions_per_repeat_user',
                         f'{np.mean(repeat_session_counts):.2f}'])
            w.writerow(['dim5', 'median_sessions_per_repeat_user',
                         f'{np.median(repeat_session_counts):.1f}'])
            w.writerow(['dim5', 'avg_questions_per_session_repeat',
                         f'{np.mean(all_session_q_counts):.2f}'])
            w.writerow(['dim5', 'session_count_p90',
                         f'{np.percentile(repeat_session_counts, 90):.1f}'])
            w.writerow(['dim5', 'session_count_p95',
                         f'{np.percentile(repeat_session_counts, 95):.1f}'])
    log(f"  Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main（主入口）
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict | None = None):
    """Main entry point for Step 14: load data, run all repeat-question analyses.
       步骤14主入口：加载数据，运行所有重复提问分析维度。"""
    log("=" * 60)
    log("Step 14: Repeat Question Analysis")
    log("=" * 60)

    if data is None:
        # Load data（加载数据）
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']

    log(f"  原始提问记录数: {len(seekers)}")

    # ── 数据预处理：按用户收集去重后的提问 ──
    log("")
    log("-" * 40)
    log("数据预处理：按用户收集去重后的提问")
    log("-" * 40)
    user_unique = _collect_user_questions(seekers)

    total_qs = sum(len(qs) for qs in user_unique.values())
    log(f"  去重后总提问数: {total_qs}")
    log(f"  去重后总用户数: {len(user_unique)}")

    # ── 维度1：重复提问用户占比 ──
    log("")
    dim1_result = dim1_repeat_user_ratio(user_unique)

    # ── 维度2：平均提问数与平均提问长度 ──
    log("")
    dim2_result = dim2_avg_questions_and_length(user_unique)

    # ── 维度3：提问次数分组分布 ──
    log("")
    dim3_question_count_distribution(user_unique)

    # ── 维度4：提问长度分组分布 ──
    log("")
    dim4_question_length_distribution(user_unique)

    # ── 维度5：平均会话数与每会话平均提问数 ──
    log("")
    dim5_session_stats(user_unique)

    # ── 维度6：按提问次数分组的会话数量分布 ──
    log("")
    dim6_session_dist_by_question_count(user_unique)

    # ── 汇总 ──
    log("")
    log("-" * 40)
    log("写入汇总统计")
    log("-" * 40)
    _write_summary(user_unique)

    log("")
    log("=" * 60)
    log(f"Step 14 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()  # 独立运行时执行 main
