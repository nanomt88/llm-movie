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

import os
import csv
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import networkx as nx

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log
from movie.utils.genre_map import to_en

setup_matplotlib()
STEP_OUT = STEP_DIRS[9]
os.makedirs(STEP_OUT, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
MIN_COOCCURRENCE = 2        # Minimum times two movies co-occur to keep edge
TOP_N_NODES = 50            # Max nodes in network viz
MAX_EDGE_WIDTH = 6          # Max edge width in visualization
NODE_MIN_SIZE = 200         # Min node size in visualization
NODE_MAX_SIZE = 2000        # Max node size in visualization

GENRE_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#16a085',
    '#c0392b', '#2980b9', '#27ae60', '#d35400', '#8e44ad',
    '#2c3e50', '#d4ac0d', '#7f8c8d',
]


# ═══════════════════════════════════════════════════════════════════════
#  Co-occurrence computation
# ═══════════════════════════════════════════════════════════════════════

def _extract_movie_pairs(seekers: list[dict]) -> list[tuple[str, str]]:
    """Extract movie-movie co-occurrence pairs from seeker records.
       从用户提问记录中提取电影-电影共现对。
    Each seeker's IMDB IDs within the same record form a co-occurrence pair.
    同一记录中的多个 IMDB ID 构成共现对。"""
    pairs = []
    for r in seekers:
        ids = r.get('imdb_ids', [])
        if len(ids) >= 2:
            # Sort to avoid (A,B) vs (B,A) duplicates
            sorted_ids = sorted(set(ids))
            for a, b in combinations(sorted_ids, 2):
                pairs.append((a, b))
    return pairs


def _build_cooccurrence_graph(
    seekers: list[dict],
    movie_info: dict,
    date_set: set = None,
) -> nx.Graph:
    """Build movie co-occurrence graph from seekers, optionally filtered by date set.
       从用户提问记录构建电影共现图，可按日期集合过滤。"""
    # Count individual movie mentions
    mention_counter: Counter = Counter()
    pair_counter: Counter = Counter()

    for r in seekers:
        if date_set is not None and r['date'] not in date_set:
            continue
        ids = r.get('imdb_ids', [])
        if not ids:
            continue
        # Count mentions
        for mid in set(ids):
            mention_counter[mid] += 1
        # Count co-occurrences
        if len(set(ids)) >= 2:
            for a, b in combinations(sorted(set(ids)), 2):
                pair_counter[(a, b)] += 1

    # Build graph
    G = nx.Graph()

    # Add nodes with mention count as weight
    for mid, count in mention_counter.items():
        info = movie_info.get(mid, {})
        title = info.get('original_title', '') if isinstance(info, dict) else ''
        year = info.get('year', '') if isinstance(info, dict) else ''
        genres = info.get('genres', []) if isinstance(info, dict) else []
        G.add_node(mid, mentions=count, title=title, year=year,
                   genres=','.join(genres) if genres else '')

    # Add edges
    for (a, b), count in pair_counter.items():
        if count >= MIN_COOCCURRENCE:
            G.add_edge(a, b, weight=count)

    return G


def _get_genre_color(genres_str: str, genre_color_map: dict) -> str:
    """Assign a color to a node based on its primary genre.
       根据主要类型为节点分配颜色。"""
    if not genres_str:
        return '#95a5a6'  # Gray for unknown
    genre_list = [g.strip() for g in genres_str.split(',')]
    for g in genre_list:
        eng = to_en(g)
        if eng in genre_color_map:
            return genre_color_map[eng]
    return '#95a5a6'


def _plot_network(
    G: nx.Graph,
    title: str, filename: str,
    max_nodes: int = TOP_N_NODES,
):
    """Plot a co-occurrence network graph.
       绘制共现网络图。"""
    if G.number_of_nodes() == 0:
        log(f"  Empty graph for {filename}")
        return

    # Subgraph: keep top nodes by degree
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
    H = G.subgraph(top_nodes).copy()

    if H.number_of_nodes() < 3:
        log(f"  Too few nodes ({H.number_of_nodes()}) for {filename}")
        return

    # Node sizes by mentions
    mentions = [H.nodes[n].get('mentions', 1) for n in H.nodes()]
    node_sizes = [
        NODE_MIN_SIZE + (m - min(mentions)) / max(max(mentions) - min(mentions), 1)
        * (NODE_MAX_SIZE - NODE_MIN_SIZE)
        for m in mentions
    ]

    # Edge widths by weight
    edge_weights = [H.edges[e].get('weight', 1) for e in H.edges()]
    edge_widths = [
        0.5 + (w - min(edge_weights)) / max(max(edge_weights) - min(edge_weights), 1)
        * MAX_EDGE_WIDTH
        for w in edge_weights
    ]

    # Assign colors by genre
    genre_color_map = {}
    color_idx = 0
    node_colors = []
    for n in H.nodes():
        g = H.nodes[n].get('genres', '')
        if g:
            primary = g.split(',')[0].strip()
            eng = to_en(primary)
            if eng not in genre_color_map:
                genre_color_map[eng] = GENRE_COLORS[color_idx % len(GENRE_COLORS)]
                color_idx += 1
            node_colors.append(genre_color_map[eng])
        else:
            node_colors.append('#95a5a6')

    # Layout
    pos = nx.spring_layout(H, k=3 / max(H.number_of_nodes()**0.5, 0.5),
                           iterations=50, seed=42)

    fig, ax = plt.subplots(figsize=(14, 10))

    nx.draw_networkx_edges(
        H, pos, ax=ax, alpha=0.3,
        width=edge_widths, edge_color='#888888',
    )

    nx.draw_networkx_nodes(
        H, pos, ax=ax, node_size=node_sizes,
        node_color=node_colors, alpha=0.85, edgecolors='#333333',
        linewidths=0.5,
    )

    # Labels for top-mention nodes
    top_mention_nodes = sorted(H.nodes(), key=lambda n: H.nodes[n].get('mentions', 0),
                               reverse=True)[:15]
    labels = {}
    for n in top_mention_nodes:
        title = H.nodes[n].get('title', '')
        year = H.nodes[n].get('year', '')
        labels[n] = f'{title}\n({year})' if title and year else n

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
       以 GEXF 格式保存图，可用 Gephi 可视化。"""
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
       打印网络基本统计量。"""
    if G.number_of_nodes() == 0:
        log(f"  {label}: empty graph")
        return

    log(f"  {label}:")
    log(f"    Nodes: {G.number_of_nodes()}")
    log(f"    Edges: {G.number_of_edges()}")
    log(f"    Density: {nx.density(G):.6f}")

    # Components
    components = list(nx.connected_components(G))
    log(f"    Components: {len(components)}")
    if components:
        largest = max(components, key=len)
        log(f"    Largest component: {len(largest)} nodes ({100*len(largest)/G.number_of_nodes():.1f}%)")

    # Degree stats
    degrees = [d for _, d in G.degree()]
    if degrees:
        log(f"    Avg degree: {np.mean(degrees):.2f}")
        log(f"    Max degree: {max(degrees)}")

    # Centrality
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:5]
    for n in top_nodes:
        title = G.nodes[n].get('title', n)
        log(f"    Top node: {title} (deg={G.degree(n)}, mentions={G.nodes[n].get('mentions',0)})")


# ═══════════════════════════════════════════════════════════════════════
#  Analysis dimensions
# ═══════════════════════════════════════════════════════════════════════

def dim_n1_overall_network(seekers: list[dict], movie_info: dict):
    """Overall movie co-occurrence network.
       全局电影共现网络。"""
    log("=" * 50)
    log("N1: Overall Co-occurrence Network")

    G = _build_cooccurrence_graph(seekers, movie_info)
    _network_stats(G, "Overall")

    _plot_network(G, 'Movie Co-occurrence Network (Overall)',
                  'n1_overall_network.png')
    _save_gexf(G, 'n1_overall_network.gexf')
    _save_edge_csv(G, 'n1_overall_network_edges.csv')
    _save_node_csv(G, 'n1_overall_network_nodes.csv')

    return G


def dim_n2_holiday_vs_nonholiday_network(
    seekers: list[dict], movie_info: dict,
):
    """Holiday vs non-holiday co-occurrence network comparison.
       节假日 vs 非节假日共现网络对比。"""
    log("=" * 50)
    log("N2: Holiday vs Non-Holiday Co-occurrence Network")

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


def dim_n3_top_cooccurrences(seekers: list[dict], movie_info: dict):
    """Top co-occurring movie pairs and their shared genres.
       最高共现的电影对及其共享类型。"""
    log("=" * 50)
    log("N3: Top Co-occurring Movie Pairs")

    pair_counter: Counter = Counter()
    for r in seekers:
        ids = r.get('imdb_ids', [])
        if len(set(ids)) >= 2:
            for a, b in combinations(sorted(set(ids)), 2):
                pair_counter[(a, b)] += 1

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

    # Save CSV
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


def dim_n4_network_centrality(G: nx.Graph):
    """Network centrality analysis (degree, betweenness, closeness).
       网络中心度分析（度中心度、中介中心度、接近中心度）。"""
    log("=" * 50)
    log("N4: Network Centrality Analysis")

    if G.number_of_nodes() < 3:
        log("  Graph too small for centrality analysis")
        return

    # Degree centrality
    deg_cent = nx.degree_centrality(G)
    top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]

    log("  Top 10 by degree centrality:")
    for node, val in top_deg:
        title = G.nodes[node].get('title', node)
        try:
            log(f"    {title}: {val:.4f}")
        except UnicodeEncodeError:
            log(f"    [unicode title]: {val:.4f}")

    # Betweenness centrality (compute on largest component for speed)
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

    # Save centrality CSV
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


def dim_n5_genre_cooccurrence(seekers: list[dict], movie_info: dict):
    """Genre-level co-occurrence network.
       题材级共现网络分析。"""
    log("=" * 50)
    log("N5: Genre Co-occurrence Network")

    # For each seeker, extract genres of mentioned movies
    genre_pairs = []
    for r in seekers:
        ids = r.get('imdb_ids', [])
        if len(set(ids)) < 2:
            continue
        genres_per_movie = []
        for mid in set(ids):
            info = movie_info.get(mid, {})
            if isinstance(info, dict):
                gs = info.get('genres', [])
                genres_per_movie.append([to_en(g) for g in gs if g])
        # Cross-movie genre pairs
        genre_sets = [set(gs) for gs in genres_per_movie if gs]
        for i in range(len(genre_sets)):
            for j in range(i + 1, len(genre_sets)):
                for g1 in genre_sets[i]:
                    for g2 in genre_sets[j]:
                        if g1 < g2:  # alphabetical order
                            genre_pairs.append((g1, g2))
                        else:
                            genre_pairs.append((g2, g1))

    pair_counter = Counter(genre_pairs)
    top_pairs = pair_counter.most_common(20)

    log("  Top genre co-occurrences:")
    for (g1, g2), cnt in top_pairs[:10]:
        log(f"    {g1} <-> {g2}: {cnt}")

    # Build genre graph
    G = nx.Graph()
    for (g1, g2), cnt in pair_counter.items():
        if cnt >= MIN_COOCCURRENCE:
            G.add_edge(g1, g2, weight=cnt)

    if G.number_of_nodes() >= 3:
        _plot_network(G, 'Genre Co-occurrence Network',
                      'n5_genre_cooccurrence.png')

    # Save CSV
    csv_path = os.path.join(STEP_OUT, 'n5_genre_cooccurrence.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['genre_a', 'genre_b', 'cooccurrence_count'])
        for (g1, g2), cnt in pair_counter.most_common(100):
            w.writerow([g1, g2, cnt])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None):
    log("=" * 60)
    log("Step 9: Co-occurrence Network & Sentiment Analysis")
    log("=" * 60)

    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']
    movie_info = data['movie_info']
    log(f"Loaded {len(seekers)} seeker records")

    # N1: Overall network
    G = dim_n1_overall_network(seekers, movie_info)
    log("")

    # N2: Holiday vs non-holiday
    dim_n2_holiday_vs_nonholiday_network(seekers, movie_info)
    log("")

    # N3: Top co-occurrences
    dim_n3_top_cooccurrences(seekers, movie_info)
    log("")

    # N4: Centrality analysis
    if G and G.number_of_nodes() > 0:
        dim_n4_network_centrality(G)
        log("")

    # N5: Genre co-occurrence
    dim_n5_genre_cooccurrence(seekers, movie_info)

    log("")
    log("=" * 60)
    log(f"Step 9 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
