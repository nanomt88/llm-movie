# -*- coding: utf-8 -*-
"""
Step 9: Co-occurrence Network & Sentiment Analysis
步骤 9：共现网络与情感分析

Analysis:
  - Movie-movie co-occurrence matrix from user seeker conversations
  - Co-occurrence network visualization (overall)
  - Sentiment-weighted co-occurrence edges (upvotes as proxy)
  - Holiday vs non-holiday co-occurrence comparison
  - Per-holiday co-occurrence pattern analysis
  - Genre co-occurrence network

Input: user seeker records with IMDB IDs
Output: output/movie/step9/*.png + CSV + GEXF (for Gephi)
"""

import os           # 文件路径操作
import csv          # CSV 读写
from collections import defaultdict, Counter   # 默认字典与计数器
from itertools import combinations              # 组合生成（用于电影对）

import numpy as np          # 数值计算
import networkx as nx       # 复杂网络分析库

import matplotlib
matplotlib.use('Agg')       # 非交互式后端（服务器环境）
import matplotlib.pyplot as plt

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log
from movie.utils.text import build_conv_system, get_system_movie_ids

# ── 初始化 ──────────────────────────────────────────────────────────
setup_matplotlib()
STEP_OUT = STEP_DIRS[9]                 # 输出目录：output/movie/step9/
os.makedirs(STEP_OUT, exist_ok=True)

# ── 网络可视化参数 ──────────────────────────────────────────────────
MIN_COOCCURRENCE = 2        # 最小共现次数，低于此值不保留边
TOP_N_NODES = 50            # 网络图中显示的最大节点数
MAX_EDGE_WIDTH = 6          # 边最大宽度
NODE_MIN_SIZE = 200         # 节点最小面积
NODE_MAX_SIZE = 2000        # 节点最大面积

# 各电影类型的配色方案
GENRE_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#16a085',
    '#c0392b', '#2980b9', '#27ae60', '#d35400', '#8e44ad',
    '#2c3e50', '#d4ac0d', '#7f8c8d',
]


# ═══════════════════════════════════════════════════════════════════════
#  共现计算
# ═══════════════════════════════════════════════════════════════════════

def _extract_movie_pairs(seekers: list[dict]) -> list[tuple[str, str]]:
    """Extract movie-movie co-occurrence pairs from seeker records.
        从用户提问记录中提取电影-电影共现对。
    Each seeker's system-replied movie IDs within the same record form a co-occurrence pair.
    同一记录中系统回复提及的多个电影 ID 构成共现对（规则8）。"""
    pairs = []
    for r in seekers:
        ids = list(r.get('system_movie_ids', set()))  # 从系统回复获取的电影ID
        if len(ids) >= 2:
            sorted_ids = sorted(set(ids))       # 去重排序，避免 (A,B) vs (B,A)
            for a, b in combinations(sorted_ids, 2):
                pairs.append((a, b))
    return pairs


def _build_cooccurrence_graph(
    seekers: list[dict],
    movie_info: dict,
    date_set: set = None,
) -> nx.Graph:
    """Build movie co-occurrence graph from seekers, optionally filtered by date set.
       从用户提问记录构建电影共现图，可按日期集合过滤。
    Args:
        seekers: 用户提问记录列表
        movie_info: 电影信息字典 {imdb_id: {title, year, genres, ...}}
        date_set: 可选日期集合，仅包含该集合中的日期
    Returns:
        NetworkX 无向图，节点属性含 mentions/title/year/genres，边权重为共现次数
    """
    mention_counter: Counter = Counter()    # 统计每部电影出现在多少条记录中
    pair_counter: Counter = Counter()       # 统计每对电影共现次数

    for r in seekers:
        if date_set is not None and r['date'] not in date_set:
            continue                        # 按日期过滤
        ids = list(r.get('system_movie_ids', set()))  # 从系统回复获取的电影ID
        if not ids:
            continue
        # 统计提及次数（同一记录中每部电影只计一次）
        for mid in set(ids):
            mention_counter[mid] += 1
        # 统计共现次数
        if len(set(ids)) >= 2:
            for a, b in combinations(sorted(set(ids)), 2):
                pair_counter[(a, b)] += 1

    # 构建图
    G = nx.Graph()

    # 添加节点，附带电影元信息
    for mid, count in mention_counter.items():
        info = movie_info.get(mid, {})
        title = info.get('original_title', '') if isinstance(info, dict) else ''
        year = info.get('year', '') if isinstance(info, dict) else ''
        genres = info.get('genres', []) if isinstance(info, dict) else []
        G.add_node(mid, mentions=count, title=title, year=year,
                   genres=','.join(genres) if genres else '')

    # 添加边（仅保留共现次数 >= MIN_COOCCURRENCE 的边）
    for (a, b), count in pair_counter.items():
        if count >= MIN_COOCCURRENCE:
            G.add_edge(a, b, weight=count)

    return G


def _get_genre_color(genres_str: str, genre_color_map: dict) -> str:
    """Assign a color to a node based on its primary genre.
       根据主要类型为节点分配颜色。"""
    if not genres_str:
        return '#95a5a6'  # 无类型信息时用灰色
    genre_list = [g.strip() for g in genres_str.split(',')]
    for g in genre_list:
        if g in genre_color_map:
            return genre_color_map[g]
    return '#95a5a6'


def _plot_network(
    G: nx.Graph,
    title: str, filename: str,
    max_nodes: int = TOP_N_NODES,
):
    """Plot a co-occurrence network graph.
       绘制共现网络图。
    Args:
        G: 网络图
        title: 图表标题
        filename: 输出文件名
        max_nodes: 最多显示的节点数（按度排序取 top）"""
    if G.number_of_nodes() == 0:
        log(f"  Empty graph for {filename}")
        return

    # 取度最高的子图
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
    H = G.subgraph(top_nodes).copy()

    if H.number_of_nodes() < 3:
        log(f"  Too few nodes ({H.number_of_nodes()}) for {filename}")
        return

    # 根据提及次数映射节点大小（线性缩放）
    mentions = [H.nodes[n].get('mentions', 1) for n in H.nodes()]
    node_sizes = [
        NODE_MIN_SIZE + (m - min(mentions)) / max(max(mentions) - min(mentions), 1)
        * (NODE_MAX_SIZE - NODE_MIN_SIZE)
        for m in mentions
    ]

    # 根据权重映射边粗细
    edge_weights = [H.edges[e].get('weight', 1) for e in H.edges()]
    edge_widths = [
        0.5 + (w - min(edge_weights)) / max(max(edge_weights) - min(edge_weights), 1)
        * MAX_EDGE_WIDTH
        for w in edge_weights
    ]

    # 根据类型分配节点颜色
    genre_color_map = {}
    color_idx = 0
    node_colors = []
    for n in H.nodes():
        g = H.nodes[n].get('genres', '')
        if g:
            primary = g.split(',')[0].strip()       # 取第一个类型为主要类型
            if primary not in genre_color_map:
                genre_color_map[primary] = GENRE_COLORS[color_idx % len(GENRE_COLORS)]
                color_idx += 1
            node_colors.append(genre_color_map[primary])
        else:
            node_colors.append('#95a5a6')

    # 布局：Spring layout（力导向布局）
    pos = nx.spring_layout(H, k=3 / max(H.number_of_nodes()**0.5, 0.5),
                           iterations=50, seed=42)

    fig, ax = plt.subplots(figsize=(14, 10))

    # 绘制边
    nx.draw_networkx_edges(
        H, pos, ax=ax, alpha=0.3,
        width=edge_widths, edge_color='#888888',
    )

    # 绘制节点
    nx.draw_networkx_nodes(
        H, pos, ax=ax, node_size=node_sizes,
        node_color=node_colors, alpha=0.85, edgecolors='#333333',
        linewidths=0.5,
    )

    # 为提及次数最高的 15 个节点添加标签
    top_mention_nodes = sorted(H.nodes(), key=lambda n: H.nodes[n].get('mentions', 0),
                               reverse=True)[:15]
    labels = {}
    for n in top_mention_nodes:
        title_text = H.nodes[n].get('title', '')
        year = H.nodes[n].get('year', '')
        labels[n] = f'{title_text}\n({year})' if title_text and year else n

    nx.draw_networkx_labels(
        H, pos, ax=ax, labels=labels, font_size=7,
        font_color='#222222',
    )

    ax.set_title(title, fontsize=13, pad=16)
    ax.axis('off')
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


def _save_gexf(G: nx.Graph, filename: str):
    """Save graph in GEXF format for Gephi.
       以 GEXF 格式保存图，可用 Gephi 做进一步可视化。"""
    if G.number_of_nodes() < 2:
        return
    path = os.path.join(STEP_OUT, filename)
    nx.write_gexf(G, path)
    log(f"Saved: {path}")


def _save_edge_csv(G: nx.Graph, filename: str):
    """Save edge list to CSV.
       将边列表保存到 CSV。"""
    path = os.path.join(STEP_OUT, filename)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', 'target', 'weight', 'source_title', 'target_title'])
        for src, tgt, data in G.edges(data=True):
            src_title = G.nodes[src].get('title', src)
            tgt_title = G.nodes[tgt].get('title', tgt)
            w.writerow([src, tgt, data.get('weight', 1), src_title, tgt_title])
    log(f"Saved: {path}")


def _save_node_csv(G: nx.Graph, filename: str):
    """Save node list to CSV.
       将节点列表保存到 CSV。"""
    path = os.path.join(STEP_OUT, filename)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['movie_id', 'title', 'year', 'mentions', 'degree', 'genres'])
        for n, data in G.nodes(data=True):
            w.writerow([
                n,
                data.get('title', ''),
                data.get('year', ''),
                data.get('mentions', 0),
                G.degree(n),
                data.get('genres', ''),
            ])
    log(f"Saved: {filename}")


def _network_stats(G: nx.Graph, label: str = "Network"):
    """Print basic network statistics.
       打印网络基本统计量：节点数、边数、密度、连通分量、度分布等。"""
    if G.number_of_nodes() == 0:
        log(f"  {label}: empty graph")
        return

    log(f"  {label}:")
    log(f"    Nodes: {G.number_of_nodes()}")          # 节点数
    log(f"    Edges: {G.number_of_edges()}")          # 边数
    log(f"    Density: {nx.density(G):.6f}")          # 网络密度

    # 连通分量
    components = list(nx.connected_components(G))
    log(f"    Components: {len(components)}")         # 连通分量数
    if components:
        largest = max(components, key=len)
        log(f"    Largest component: {len(largest)} nodes ({100*len(largest)/G.number_of_nodes():.1f}%)")

    # 度统计
    degrees = [d for _, d in G.degree()]
    if degrees:
        log(f"    Avg degree: {np.mean(degrees):.2f}")
        log(f"    Max degree: {max(degrees)}")

    # 度最高的 top 5 节点
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:5]
    for n in top_nodes:
        title = G.nodes[n].get('title', n)
        log(f"    Top node: {title} (deg={G.degree(n)}, mentions={G.nodes[n].get('mentions',0)})")

# ═══════════════════════════════════════════════════════════════════════
#  N1: 全局电影共现网络 (Network Graph)
#  N1: Overall Movie Co-occurrence Network
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】NetworkX 图 + Gephi 导出
# 【统计口径】
#   共现定义: 同一用户在同一次提问中提及的多部电影
#   _build_cooccurrence_graph(seekers, movie_info) 构建无向图
#   节点 = 电影(title+year+genre), 边 = 共现关系(权重=共现次数)
# 【输出文件】GEXF: n1_overall_network.gexf (Gephi 可用)
# 【特殊说明】返回 networkx.Graph 供 N4 中心度分析使用
# ═══════════════════════════════════════════════════════════════════════

def dim_n1_overall_network(seekers: list[dict], movie_info: dict):
    """Overall movie co-occurrence network.
        全局电影共现网络。"""
    log("=" * 50)
    log("N1: Overall Co-occurrence Network")

    # 构建全量共现图
    G = _build_cooccurrence_graph(seekers, movie_info)
    _network_stats(G, "Overall")

    # 输出：可视化图 + GEXF + CSV
    _plot_network(G, 'Movie Co-occurrence Network (Overall)',
                  'n1_overall_network.png')
    _save_gexf(G, 'n1_overall_network.gexf')
    _save_edge_csv(G, 'n1_overall_network_edges.csv')
    _save_node_csv(G, 'n1_overall_network_nodes.csv')

    return G


# ═══════════════════════════════════════════════════════════════════════
#  N2: 节假日 VS 非节假日 共现网络 (GEXF)
#  N2: Holiday vs Non-Holiday Co-occurrence Network
# ═══════════════════════════════════════════════════════════════════════
# 【统计口径】分别构建假日/非假日的共现网络
# 【输出文件】GEXF: n2_holiday_network.gexf, n2_nonholiday_network.gexf
# ═══════════════════════════════════════════════════════════════════════

def dim_n2_holiday_vs_nonholiday_network(
    seekers: list[dict], movie_info: dict,
):
    """Holiday vs non-holiday co-occurrence network comparison.
       节假日 vs 非节假日共现网络对比。"""
    log("=" * 50)
    log("N2: Holiday vs Non-Holiday Co-occurrence Network")

    # 按日期分别构建节假日和非节假日子图
    h_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')
    nh_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')

    G_h = _build_cooccurrence_graph(seekers, movie_info, date_set=h_dates)
    G_nh = _build_cooccurrence_graph(seekers, movie_info, date_set=nh_dates)

    _network_stats(G_h, "Holiday")
    _network_stats(G_nh, "Non-Holiday")

    _plot_network(G_h, 'Movie Co-occurrence Network (Holiday)',
                  'n2_holiday_network.png')
    _plot_network(G_nh, 'Movie Co-occurrence Network (Non-Holiday)',
                  'n2_nonholiday_network.png')

    _save_gexf(G_h, 'n2_holiday_network.gexf')
    _save_gexf(G_nh, 'n2_nonholiday_network.gexf')


# ═══════════════════════════════════════════════════════════════════════
#  N3: 最高共现电影对 (Table)
#  N3: Top Co-occurring Movie Pairs
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】表格/日志输出
# 【统计口径】规则8：从系统回复中提取电影ID，统计所有电影对(pair)的共现次数，取 TOP_N
# 【输出文件】CSV: n3_top_cooccurrences.csv
#   含: 电影A(title+year+genre), 电影B(title+year+genre), 共现次数
# ═══════════════════════════════════════════════════════════════════════

def dim_n3_top_cooccurrences(seekers: list[dict], movie_info: dict):
    """Top co-occurring movie pairs and their shared genres.
        最高共现的电影对及其共享类型。"""
    log("=" * 50)
    log("N3: Top Co-occurring Movie Pairs")

    # 统计所有共现对
    pair_counter: Counter = Counter()
    for r in seekers:
        ids = r.get('system_movie_ids', set())  # 规则8：从系统回复获取的电影ID
        if len(set(ids)) >= 2:
            for a, b in combinations(sorted(set(ids)), 2):
                pair_counter[(a, b)] += 1

    # 取 top 30 并打印 top 10
    top_pairs = pair_counter.most_common(30)
    log(f"  Top 10 co-occurring pairs:")
    for (a, b), cnt in top_pairs[:10]:
        info_a = movie_info.get(a, {})
        info_b = movie_info.get(b, {})
        title_a = info_a.get('original_title', a) if isinstance(info_a, dict) else a
        title_b = info_b.get('original_title', b) if isinstance(info_b, dict) else b
        year_a = info_a.get('year', '') if isinstance(info_a, dict) else ''
        year_b = info_b.get('year', '') if isinstance(info_b, dict) else ''
        try:
            log(f"    {title_a} ({year_a}) <-> {title_b} ({year_b}): {cnt}")
        except UnicodeEncodeError:
            log(f"    [unicode title] <-> [unicode title]: {cnt}")

    # 保存到 CSV
    csv_path = os.path.join(STEP_OUT, 'n3_top_cooccurrences.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['movie_a', 'movie_a_title', 'movie_a_year',
                     'movie_b', 'movie_b_title', 'movie_b_year',
                     'cooccurrence_count'])
        for (a, b), cnt in top_pairs:
            info_a = movie_info.get(a, {})
            info_b = movie_info.get(b, {})
            title_a = info_a.get('original_title', a) if isinstance(info_a, dict) else a
            title_b = info_b.get('original_title', b) if isinstance(info_b, dict) else b
            year_a = info_a.get('year', '') if isinstance(info_a, dict) else ''
            year_b = info_b.get('year', '') if isinstance(info_b, dict) else ''
            w.writerow([a, title_a, year_a, b, title_b, year_b, cnt])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  N4: 网络中心度分析 (Table)
#  N4: Network Centrality Analysis
# ═══════════════════════════════════════════════════════════════════════
# 【分析维度】
#   degree_centrality: 度中心度（共现广度）
#   betweenness_centrality: 中介中心度（桥梁作用）
#   closeness_centrality: 接近中心度（连接效率）
# 【输出文件】CSV: n4_network_centrality.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_n4_network_centrality(G: nx.Graph):
    """Network centrality analysis (degree, betweenness, closeness).
        网络中心度分析（度中心度、中介中心度、接近中心度）。"""
    log("=" * 50)
    log("N4: Network Centrality Analysis")

    if G.number_of_nodes() < 3:
        log("  Graph too small for centrality analysis")
        return

    # ── 度中心度（Degree Centrality）──
    # 度量节点直接连接的数量，反映节点在网络中的活跃程度
    deg_cent = nx.degree_centrality(G)
    top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]

    log("  Top 10 by degree centrality:")
    for node, val in top_deg:
        title = G.nodes[node].get('title', node)
        try:
            log(f"    {title}: {val:.4f}")
        except UnicodeEncodeError:
            log(f"    [unicode title]: {val:.4f}")

    # ── 中介中心度（Betweenness Centrality）──
    # 度量节点在最短路径中的重要性，反映节点在信息流中的桥梁作用
    # 为提高速度，仅计算最大连通分量
    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()
    if H.number_of_nodes() < 3:
        return

    bet_cent = nx.betweenness_centrality(H, k=min(20, H.number_of_nodes()),
                                          seed=42)
    top_bet = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    log("  Top 10 by betweenness centrality:")
    for node, val in top_bet:
        title = H.nodes[node].get('title', node)
        try:
            log(f"    {title}: {val:.4f}")
        except UnicodeEncodeError:
            log(f"    [unicode title]: {val:.4f}")

    # 保存中心度到 CSV
    csv_path = os.path.join(STEP_OUT, 'n4_network_centrality.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['movie_id', 'title', 'degree', 'degree_centrality',
                     'betweenness_centrality'])
        for node in G.nodes():
            bc = bet_cent.get(node, 0) if node in H else 0
            w.writerow([
                node,
                G.nodes[node].get('title', ''),
                G.degree(node),
                f'{deg_cent.get(node, 0):.4f}',
                f'{bc:.4f}',
            ])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  N5: 题材级共现网络 (Network Graph)
#  N5: Genre-Level Co-occurrence Network
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】NetworkX 图
# 【统计口径】规则8：从系统回复提取电影ID，再从电影共现关系聚合到类型层面
#   节点 = 电影类型, 边 = 两类型出现在同一系统回复中的次数(求和)
# 【输出文件】PNG: n5_genre_cooccurrence.png, GEXF: n5_*.gexf
# ═══════════════════════════════════════════════════════════════════════

def dim_n5_genre_cooccurrence(seekers: list[dict], movie_info: dict):
    """Genre-level co-occurrence network.
        题材级共现网络分析：从电影共现关系中提取类型之间的关系。"""
    log("=" * 50)
    log("N5: Genre Co-occurrence Network")

    # 对每条记录，提取提及电影的类型，统计跨电影的类型共现
    genre_pairs = []
    for r in seekers:
        ids = r.get('system_movie_ids', set())  # 规则8：从系统回复获取的电影ID
        if len(set(ids)) < 2:
            continue
        genres_per_movie = []
        for mid in set(ids):
            info = movie_info.get(mid, {})
            if isinstance(info, dict):
                gs = info.get('genres', [])
                genres_per_movie.append([g for g in gs if g])
        # 跨电影生成类型对（同一记录中不同电影的类型之间）
        genre_sets = [set(gs) for gs in genres_per_movie if gs]
        for i in range(len(genre_sets)):
            for j in range(i + 1, len(genre_sets)):
                for g1 in genre_sets[i]:
                    for g2 in genre_sets[j]:
                        if g1 < g2:
                            genre_pairs.append((g1, g2))
                        else:
                            genre_pairs.append((g2, g1))

    pair_counter = Counter(genre_pairs)
    top_pairs = pair_counter.most_common(20)

    log("  Top genre co-occurrences:")
    for (g1, g2), cnt in top_pairs[:10]:
        log(f"    {g1} <-> {g2}: {cnt}")

    # 构建类型共现图
    G = nx.Graph()
    for (g1, g2), cnt in pair_counter.items():
        if cnt >= MIN_COOCCURRENCE:
            G.add_edge(g1, g2, weight=cnt)

    if G.number_of_nodes() >= 3:
        _plot_network(G, 'Genre Co-occurrence Network',
                      'n5_genre_cooccurrence.png')

    # 保存 CSV
    csv_path = os.path.join(STEP_OUT, 'n5_genre_cooccurrence.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['genre_a', 'genre_b', 'cooccurrence_count'])
        for (g1, g2), cnt in pair_counter.most_common(100):
            w.writerow([g1, g2, cnt])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None):
    log("=" * 60)
    log("Step 9: Co-occurrence Network & Sentiment Analysis")
    log("=" * 60)

    # 加载数据
    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']
    movie_info = data['movie_info']
    rows = data.get('rows', [])
    log(f"Loaded {len(seekers)} seeker records")

    # 规则8：从系统回复中提取电影ID，附加到每条 seeker 记录
    conv_system = build_conv_system(rows)
    for r in seekers:
        r['system_movie_ids'] = get_system_movie_ids(r.get('conv_id', ''), conv_system)
    log(f"Built conv_system: {len(conv_system)} turn-level entries")

    # N1: 全局共现网络
    G = dim_n1_overall_network(seekers, movie_info)
    log("")

    # N2: 节假日 vs 非节假日网络对比
    dim_n2_holiday_vs_nonholiday_network(seekers, movie_info)
    log("")

    # N3: Top 共现电影对
    dim_n3_top_cooccurrences(seekers, movie_info)
    log("")

    # N4: 网络中心度分析
    if G and G.number_of_nodes() > 0:
        dim_n4_network_centrality(G)
        log("")

    # N5: 类型共现网络
    dim_n5_genre_cooccurrence(seekers, movie_info)

    log("")
    log("=" * 60)
    log(f"Step 9 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
