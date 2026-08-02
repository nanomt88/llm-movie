# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文注释

"""
First Question Movie-ID Ratio Analysis
会话首个问题是否包含电影ID 的统计

对每个会话(session_id)，取其第一个用户提问(OP)，统计该提问中是否
包含电影ID(tt\d{7,9})，给出含ID与不含ID的占比。

说明：
  - 会话基础ID(session_id)由 conv_id 经 extract_session_base() 提取，
    例如 conv_id='t3_8upzwy_0/13' -> session_id='t3_8upzwy'。
  - 每个会话在 data_all.csv 中可能对应多条 conv_id(_0/N, _1/N, ...)，
    每个 conv_id 内 turn_order==0 且 is_seeker==True 的行即为原始提问(OP)，
    它们在同一会话内内容一致（作为上下文被重复）。
  - 本脚本对每个会话取一个 OP 代表，检查其 processed 文本中的 tt 电影ID。

Output: output/movie/first_question_movie_id/
  - first_question_movie_id_ratio.csv      (含/不含电影ID 占比统计)
  - first_question_movie_id_pie.png         (占比饼图)
  - first_question_examples.csv             (示例：含/不含电影ID的提问)
  - movie_id_count_distribution.csv         (含ID提问的电影ID数量分布+均值)
  - movie_id_count_distribution_bar.png     (电影ID数量分组柱状图)
"""

import os               # 操作系统接口，用于路径拼接和目录创建
import csv              # CSV 文件读写，用于保存结果
from collections import defaultdict  # 默认字典，用于按 session_id 分组

import matplotlib       # 绘图库
matplotlib.use('Agg')   # 使用 Agg 后端（无 GUI），适用于服务器环境
import matplotlib.pyplot as plt  # pyplot 模块，用于生成饼图

from movie.config import FULL_YEAR_CSV, OUTPUT_DIR, setup_matplotlib, log  # 共享配置
from movie.data_loader import load_conversations, extract_imdb_ids         # 数据加载与ID提取

setup_matplotlib()  # 初始化 matplotlib（Agg 后端 + 中文字体），必须在绘图代码之前

# 输出目录：output/movie/first_question_movie_id/
STEP_OUT = os.path.join(OUTPUT_DIR, 'first_question_movie_id')
os.makedirs(STEP_OUT, exist_ok=True)  # 自动创建输出目录


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 取单个会话的第一个用户提问(OP)
#  辅助函数：返回某个会话中最早的原始提问及其电影ID列表
# ═══════════════════════════════════════════════════════════════════════
def _first_question_of_session(session_rows: list[dict]) -> tuple[dict, list[str]] | None:
    """
    Given all rows of a session, return the first user question (OP) and its movie IDs.
    给定一个会话的所有行，返回第一个用户提问(OP)行及其包含的电影ID列表。

    选择策略：
      1. 优先取 turn_order==0 且 is_seeker==True 的行（OP 标记），
         按 utc_time 升序取最早一条（同一会话内 OP 内容一致，去重取一即可）；
      2. 若无 turn_order==0 的提问行（数据缺失的兜底），
         则在所有 is_seeker==True 行中取 utc_time 最早的一条。
    Args:
        session_rows: 某会话的全部数据行
    Returns:
        (first_question_row, movie_ids) 或 None（该会话无任何用户提问时）
    """
    seekers = [r for r in session_rows if r['is_seeker']]  # 筛选用户提问行
    if not seekers:                                          # 无提问行则跳过
        return None

    # 优先取 OP 标记行（turn_order==0），按时间升序
    op_rows = [r for r in seekers if r['turn_order'] == 0]
    if op_rows:
        op_rows.sort(key=lambda r: r['utc_time'])
        first = op_rows[0]
    else:
        # 兜底：取所有提问中时间最早的一条
        seekers.sort(key=lambda r: r['utc_time'])
        first = seekers[0]

    # 电影ID：优先用 data_loader 预提取的 imdb_ids，缺失时从 proc_text 重新提取
    ids = first.get('imdb_ids') or extract_imdb_ids(first.get('proc_text', ''))
    return first, ids


# ═══════════════════════════════════════════════════════════════════════
#  Main entry
#  主入口
# ═══════════════════════════════════════════════════════════════════════
def main(filepath: str = None):
    """
    Load conversations, group by session, find each session's first question,
    count whether it contains movie IDs, output CSV + pie chart + examples.
    加载会话数据，按会话分组，取每会话首个提问，统计含电影ID的占比。
    Args:
        filepath: CSV 文件路径，默认为全年会话数据 data_all.csv
    """
    if filepath is None:
        filepath = FULL_YEAR_CSV

    log("=" * 50)
    log("First-Question Movie-ID Ratio Analysis")
    log("=" * 50)

    # 加载全量会话数据（不经 load_all 的日期过滤，保留全部会话结构）
    rows = load_conversations(filepath)
    log(f"Loaded {len(rows)} rows, seekers={sum(1 for r in rows if r['is_seeker'])}")

    # 按 session_id 分组（session_id 在 load_conversations 中已提取）
    by_session = defaultdict(list)  # session_id -> 该会话的所有行
    for r in rows:
        by_session[r['session_id']].append(r)
    log(f"Unique sessions: {len(by_session)}")

    total = 0              # 有效会话数（有至少一条用户提问）
    with_id = 0            # 第一个问题含电影ID的会话数
    without_id = 0         # 第一个问题不含电影ID的会话数
    id_counts = []         # 含ID会话中，每个会话第一个问题包含的电影ID数量列表
    examples = []          # 示例记录：(session_id, category, text, num_ids, sample_ids)

    for sid, sess_rows in by_session.items():
        result = _first_question_of_session(sess_rows)
        if result is None:                 # 该会话无用户提问，跳过
            continue
        first, ids = result
        total += 1
        text = first.get('proc_text', '')
        n_ids = len(ids)
        if ids:
            with_id += 1
            id_counts.append(n_ids)        # 记录该会话首个问题含的电影ID数
            examples.append((sid, 'with_id', text[:150], n_ids, ','.join(ids[:5])))
        else:
            without_id += 1
            examples.append((sid, 'without_id', text[:150], 0, ''))

    ratio = with_id / total * 100 if total else 0.0  # 含电影ID的占比(百分比)

    # 日志输出
    log(f"Sessions with first question: {total}")
    log(f"  First question WITH movie ID:    {with_id} ({ratio:.2f}%)")
    log(f"  First question WITHOUT movie ID: {without_id} ({100 - ratio:.2f}%)")

    # ── CSV：统计结果 ──
    csv_path = os.path.join(STEP_OUT, 'first_question_movie_id_ratio.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['total_sessions', total])
        w.writerow(['first_question_with_movie_id', with_id])
        w.writerow(['first_question_without_movie_id', without_id])
        w.writerow(['ratio_percent', f'{ratio:.2f}'])
    log(f"Saved: {csv_path}")

    # ── CSV：示例（前 200 条，含/不含各取部分） ──
    ex_path = os.path.join(STEP_OUT, 'first_question_examples.csv')
    with open(ex_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['session_id', 'category', 'first_question_text', 'num_ids', 'sample_ids'])
        for sid, cat, text, nids, sample in examples[:200]:
            w.writerow([sid, cat, text, nids, sample])
    log(f"Saved: {ex_path}")

    # ── 饼图：含/不含电影ID 占比 ──
    fig, ax = plt.subplots(figsize=(8, 8))
    sizes = [with_id, without_id]
    labels = [
        f'With Movie ID\n{with_id} ({ratio:.2f}%)',
        f'Without Movie ID\n{without_id} ({100 - ratio:.2f}%)',
    ]
    colors = ['#ff6b6b', '#74b9ff']  # 红=含ID，蓝=不含ID（与项目配色一致）
    # 过滤掉为0的切片，避免空标签
    plot_sizes = [s for s in sizes if s > 0]
    plot_labels = [l for s, l in zip(sizes, labels) if s > 0]
    plot_colors = [c for s, c in zip(sizes, colors) if s > 0]
    ax.pie(plot_sizes, labels=plot_labels, colors=plot_colors,
           autopct='%1.2f%%', startangle=90,
           textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    ax.set_title('Does the First Question Contain a Movie ID? (by session)', fontsize=13)
    fig.tight_layout()
    pie_path = os.path.join(STEP_OUT, 'first_question_movie_id_pie.png')
    fig.savefig(pie_path)
    plt.close(fig)
    log(f"Saved: {pie_path}")

    # ═══════════════════════════════════════════════════════════════════════
    #  B: 含电影ID提问的 电影ID数量分布
    #     - 平均每个含ID提问包含多少个电影ID
    #     - 按ID数量分组统计会话数，了解用户提及电影的规模分布
    # ═══════════════════════════════════════════════════════════════════════
    log("=" * 50)
    log("B: Movie-ID Count Distribution (among first questions with IDs)")

    # 平均电影ID数（仅含ID的提问）
    if id_counts:
        avg_ids = sum(id_counts) / len(id_counts)
        max_ids = max(id_counts)
        min_ids = min(id_counts)
    else:
        avg_ids = 0.0
        max_ids = 0
        min_ids = 0
    log(f"  Avg movie IDs per (with-ID) first question: {avg_ids:.2f}")
    log(f"  Min: {min_ids}, Max: {max_ids}, Sessions: {len(id_counts)}")

    # 分组桶：1, 2, 3, 4, 5, 6, 7+
    buckets = [1, 2, 3, 4, 5, 6]
    bucket_labels = ['1', '2', '3', '4', '5', '6', '7+']
    dist_counts = {label: 0 for label in bucket_labels}
    for n in id_counts:
        if n in buckets:
            dist_counts[str(n)] += 1
        else:  # n >= 7
            dist_counts['7+'] += 1

    for label in bucket_labels:
        cnt = dist_counts[label]
        pct = cnt / len(id_counts) * 100 if id_counts else 0.0
        log(f"  {label} movie ID(s): {cnt} sessions ({pct:.2f}%)")

    # ── CSV：电影ID数量分布 ──
    dist_csv = os.path.join(STEP_OUT, 'movie_id_count_distribution.csv')
    with open(dist_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['sessions_with_movie_id', len(id_counts)])
        w.writerow(['avg_ids_per_with_id_question', f'{avg_ids:.2f}'])
        w.writerow(['min_ids', min_ids])
        w.writerow(['max_ids', max_ids])
        w.writerow(['---', '---'])
        w.writerow(['num_ids_group', 'session_count', 'percent'])
        for label in bucket_labels:
            cnt = dist_counts[label]
            pct = cnt / len(id_counts) * 100 if id_counts else 0.0
            w.writerow([label, cnt, f'{pct:.2f}'])
    log(f"Saved: {dist_csv}")

    # ── 柱状图：电影ID数量分组分布 ──
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    x = range(len(bucket_labels))
    vals = [dist_counts[label] for label in bucket_labels]
    bars = ax2.bar(x, vals, color='#377eb8', alpha=0.85,
                   edgecolor='white', linewidth=0.8, width=0.6)
    # 柱顶标注数量与占比
    for bar, label in zip(bars, bucket_labels):
        cnt = dist_counts[label]
        pct = cnt / len(id_counts) * 100 if id_counts else 0.0
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(bucket_labels)
    ax2.set_xlabel('Number of Movie IDs in First Question')
    ax2.set_ylabel('Number of Sessions')
    ax2.set_title(
        f'Movie-ID Count Distribution (sessions with IDs, n={len(id_counts)}, '
        f'avg={avg_ids:.2f})', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    dist_png = os.path.join(STEP_OUT, 'movie_id_count_distribution_bar.png')
    fig2.savefig(dist_png)
    plt.close(fig2)
    log(f"Saved: {dist_png}")

    log("Done.")


if __name__ == '__main__':
    main()
