# -*- coding: utf-8 -*-
"""
Graph builder for the movie knowledge graph.
电影知识图谱构建器。

Builds a MultiDiGraph from two data sources:
1. movie_info.json  -> structured nodes + edges (Movie, Genre, Person, Country, Year)
2. data_all.csv     -> conversation-derived edges (CO_RECOMMENDED, RECOMMENDED_FOR)

从两个数据源构建多重有向图:
1. movie_info.json  -> 结构化节点与边 (电影/类型/人物/国家/年份)
2. data_all.csv     -> 对话衍生边 (共现推荐/推荐模式)
"""

import json
import os
import re
from collections import defaultdict, Counter
from itertools import combinations

import networkx as nx

from movie.config import MOVIE_INFO_PATH, FULL_YEAR_CSV, log
from movie.data_loader import extract_imdb_ids, load_conversations, load_movie_info
from movie.utils.genre_map import to_en
from movie.kg.schema import (
    NT_MOVIE, NT_GENRE, NT_PERSON, NT_COUNTRY, NT_YEAR,
    ET_HAS_GENRE, ET_DIRECTED_BY, ET_STARS_IN, ET_FROM_COUNTRY, ET_RELEASED_IN,
    ET_CO_RECOMMENDED, ET_RECOMMENDED_FOR,
    EA_WEIGHT, EA_UPVOTES, EA_COUNT, EA_SOURCE,
    NA_TITLE, NA_ORIGINAL_TITLE, NA_YEAR, NA_COUNTRY, NA_RUNTIME,
    NA_GENRES, NA_CAST, NA_RATING, NA_VOTE_COUNT, NA_OVERVIEW, NA_POSTER,
    NA_GENRE_CN, NA_GENRE_EN, NA_NODE_TYPE,
    NA_DIRECTOR,
)

TT_PATTERN = re.compile(r"tt\d{7,9}")


def build_graph(movie_info: dict = None, conv_rows: list = None,
                movie_info_path: str = None, conv_csv_path: str = None) -> nx.MultiDiGraph:
    """Build the movie knowledge graph.

    构建电影知识图谱。可传入预加载数据或文件路径。

    Args:
        movie_info: 预加载的电影信息字典 {imdb_id: {...}}；为 None 则从路径加载
        conv_rows:  预加载的会话行列表；为 None 则从路径加载
        movie_info_path: movie_info.json 路径（movie_info 为 None 时使用）
        conv_csv_path:  会话 CSV 路径（conv_rows 为 None 时使用）

    Returns:
        networkx.MultiDiGraph 知识图谱
    """
    if movie_info is None:
        if movie_info_path is None:
            movie_info_path = MOVIE_INFO_PATH
        with open(movie_info_path, "r", encoding="utf-8") as f:
            movie_info = json.load(f)
        log(f"Loaded {len(movie_info)} movies from {movie_info_path}", "KG-Builder")

    if conv_rows is None:
        if conv_csv_path is None:
            conv_csv_path = FULL_YEAR_CSV
        conv_rows = load_conversations(conv_csv_path)
        log(f"Loaded {len(conv_rows)} conversation rows from {conv_csv_path}", "KG-Builder")

    G = nx.MultiDiGraph()

    _add_structured_nodes_and_edges(G, movie_info)
    _add_conversation_edges(G, conv_rows)

    n_movies = sum(1 for n, d in G.nodes(data=True) if d.get(NA_NODE_TYPE) == NT_MOVIE)
    log(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"({n_movies} movie nodes)", "KG-Builder")
    return G


def _add_structured_nodes_and_edges(G: nx.MultiDiGraph, movie_info: dict):
    """Add Movie/Genre/Person/Country/Year nodes and structured edges.
       添加电影/类型/人物/国家/年份节点及结构化边。"""
    for imdb_id, info in movie_info.items():
        # --- Movie node (电影节点) ---
        G.add_node(imdb_id, **{
            NA_NODE_TYPE: NT_MOVIE,
            NA_TITLE: info.get("title", ""),
            NA_ORIGINAL_TITLE: info.get("original_title", ""),
            NA_YEAR: info.get("year", ""),
            NA_COUNTRY: info.get("country", ""),
            NA_RUNTIME: info.get("runtime_minutes", 0),
            NA_DIRECTOR: info.get("director", ""),
            NA_GENRES: info.get("genres", []),
            NA_CAST: info.get("cast", []),
            NA_RATING: info.get("rating", 0.0),
            NA_VOTE_COUNT: info.get("vote_count", 0),
            NA_OVERVIEW: info.get("overview", ""),
            NA_POSTER: info.get("poster_url", ""),
        })

        # --- Genre nodes + edges (类型节点与边) ---
        for genre_cn in info.get("genres", []):
            genre_en = to_en(genre_cn)
            if genre_en not in G:
                G.add_node(genre_en, **{
                    NA_NODE_TYPE: NT_GENRE,
                    NA_GENRE_CN: genre_cn,
                    NA_GENRE_EN: genre_en,
                })
            G.add_edge(imdb_id, genre_en, relation=ET_HAS_GENRE,
                       **{EA_SOURCE: "structured"})

        # --- Director node + edge (导演节点与边) ---
        director = info.get("director", "")
        if director:
            if director not in G:
                G.add_node(director, **{NA_NODE_TYPE: NT_PERSON})
            G.add_edge(imdb_id, director, relation=ET_DIRECTED_BY,
                       **{EA_SOURCE: "structured"})

        # --- Cast nodes + edges (演员节点与边) ---
        for actor in info.get("cast", []):
            if actor not in G:
                G.add_node(actor, **{NA_NODE_TYPE: NT_PERSON})
            G.add_edge(imdb_id, actor, relation=ET_STARS_IN,
                       **{EA_SOURCE: "structured"})

        # --- Country node + edge (国家节点与边) ---
        country = info.get("country", "")
        if country:
            if country not in G:
                G.add_node(country, **{NA_NODE_TYPE: NT_COUNTRY})
            G.add_edge(imdb_id, country, relation=ET_FROM_COUNTRY,
                       **{EA_SOURCE: "structured"})

        # --- Year node + edge (年份节点与边) ---
        year = info.get("year", "")
        if year:
            if year not in G:
                G.add_node(year, **{NA_NODE_TYPE: NT_YEAR})
            G.add_edge(imdb_id, year, relation=ET_RELEASED_IN,
                       **{EA_SOURCE: "structured"})

    log(f"Structured: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", "KG-Builder")


def _add_conversation_edges(G: nx.MultiDiGraph, conv_rows: list):
    """Add CO_RECOMMENDED and RECOMMENDED_FOR edges from conversation data.
       从对话数据添加共现推荐边与推荐模式边。

    For each conversation session (grouped by session_id):
      - CO_RECOMMENDED: movies mentioned together in the same session
      - RECOMMENDED_FOR: user-liked movie -> system-recommended movie
    """
    # Group rows by session (按会话分组)
    sessions = defaultdict(lambda: {"user_movies": set(), "system_movies": set(),
                                      "upvotes": 0.0})

    for row in conv_rows:
        session_id = row.get("session_id", "")
        if not session_id:
            continue
        ids = set(row.get("imdb_ids", []))
        if not ids:
            continue

        sess = sessions[session_id]
        if row.get("is_seeker"):
            sess["user_movies"].update(ids)
        else:
            sess["system_movies"].update(ids)
            sess["upvotes"] += float(row.get("upvotes", 0) or 0)

    # Build co-occurrence and recommendation edges
    # 构建共现与推荐边
    co_rec_counter = Counter()       # (movie_a, movie_b) -> count
    rec_for_counter = Counter()       # (liked, recommended) -> count
    rec_upvotes = defaultdict(float)  # (liked, recommended) -> total upvotes

    for sess in sessions.values():
        all_movies = sess["user_movies"] | sess["system_movies"]
        # Only count movies that exist in the graph (只统计图中存在的电影)
        all_movies = {m for m in all_movies if G.has_node(m)}
        if len(all_movies) < 2:
            continue

        # CO_RECOMMENDED: all pairs in the same session
        for a, b in combinations(sorted(all_movies), 2):
            co_rec_counter[(a, b)] += 1
            co_rec_counter[(b, a)] += 1

        # RECOMMENDED_FOR: user-liked -> system-recommended
        for liked in sess["user_movies"]:
            if not G.has_node(liked):
                continue
            for rec in sess["system_movies"]:
                if rec == liked or not G.has_node(rec):
                    continue
                rec_for_counter[(liked, rec)] += 1
                rec_upvotes[(liked, rec)] += sess["upvotes"]

    # Add edges to graph (将边添加到图中)
    for (a, b), cnt in co_rec_counter.items():
        if G.has_edge(a, b):
            # Check if CO_RECOMMENDED edge already exists
            existing = [k for k, d in G.get_edge_data(a, b).items()
                        if d.get("relation") == ET_CO_RECOMMENDED]
            if existing:
                G[a][b][existing[0]][EA_WEIGHT] = cnt
                G[a][b][existing[0]][EA_COUNT] = cnt
                continue
        G.add_edge(a, b, relation=ET_CO_RECOMMENDED,
                   **{EA_WEIGHT: cnt, EA_COUNT: cnt, EA_SOURCE: "conversation"})

    for (liked, rec), cnt in rec_for_counter.items():
        uv = rec_upvotes.get((liked, rec), 0.0)
        if G.has_edge(liked, rec):
            existing = [k for k, d in G.get_edge_data(liked, rec).items()
                        if d.get("relation") == ET_RECOMMENDED_FOR]
            if existing:
                G[liked][rec][existing[0]][EA_WEIGHT] = cnt
                G[liked][rec][existing[0]][EA_UPVOTES] = uv
                continue
        G.add_edge(liked, rec, relation=ET_RECOMMENDED_FOR,
                   **{EA_WEIGHT: cnt, EA_UPVOTES: uv, EA_SOURCE: "conversation"})

    log(f"Conversation edges: {len(co_rec_counter)} CO_RECOMMENDED, "
        f"{len(rec_for_counter)} RECOMMENDED_FOR", "KG-Builder")