# -*- coding: utf-8 -*-
"""
Query and retrieval interface for the movie knowledge graph.
电影知识图谱的查询与检索接口。

MovieKnowledgeGraph wraps a networkx MultiDiGraph and provides high-level
query methods for movie recommendation, including structured lookups,
composite similarity scoring, keyword search, and LLM-friendly context
generation with anti-hallucination guarantees.
MovieKnowledgeGraph 封装 networkx 多重有向图，提供高层查询方法用于电影推荐，
包括结构化查询、组合相似度评分、关键词搜索、以及面向 LLM 的上下文生成与防幻觉保障。
"""

import re
from collections import defaultdict
from typing import Optional

import networkx as nx

from movie.kg.schema import (
    NT_MOVIE, NT_GENRE, NT_PERSON,
    ET_HAS_GENRE, ET_DIRECTED_BY, ET_STARS_IN, ET_FROM_COUNTRY, ET_RELEASED_IN,
    ET_CO_RECOMMENDED, ET_RECOMMENDED_FOR,
    EA_WEIGHT, EA_UPVOTES, EA_SOURCE,
    SIM_WEIGHTS,
    NA_TITLE, NA_ORIGINAL_TITLE, NA_YEAR, NA_COUNTRY, NA_RUNTIME,
    NA_GENRES, NA_CAST, NA_RATING, NA_VOTE_COUNT, NA_OVERVIEW, NA_POSTER,
    NA_NODE_TYPE, NA_GENRE_EN, NA_GENRE_CN, NA_DIRECTOR,
)

TT_PATTERN = re.compile(r"tt\d{7,9}")


class MovieKnowledgeGraph:
    """High-level query interface over the movie knowledge graph.
       电影知识图谱的高层查询接口。

    Wraps a networkx.MultiDiGraph built by movie.kg.builder.build_graph().
    All recommendation outputs are constrained to movie nodes that exist
    in the graph, providing anti-hallucination guarantees for LLM use.
    封装由 build_graph() 构建的知识图谱。所有推荐输出均限定于图中存在的电影节点，
    为 LLM 提供防幻觉保障。
    """

    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
        self._movie_nodes = {n for n, d in G.nodes(data=True)
                             if d.get(NA_NODE_TYPE) == NT_MOVIE}

    # ═══════════════════════════════════════════════════════════════════
    #  Anti-hallucination: ID validation (防幻觉: ID 校验)
    # ═══════════════════════════════════════════════════════════════════

    def validate_movie_id(self, imdb_id: str) -> bool:
        """Check if a movie ID exists in the graph (anti-hallucination gate).
           检查电影 ID 是否在图中存在 (防幻觉关卡)。
        """
        return imdb_id in self._movie_nodes

    def validate_movie_ids(self, ids: list[str]) -> list[str]:
        """Filter a list of IDs, keeping only valid ones.
           过滤 ID 列表，仅保留图中存在的有效 ID。
        """
        return [i for i in ids if i in self._movie_nodes]

    def extract_known_ids(self, text: str) -> list[str]:
        """Extract tt IDs from text, keeping only those in the graph.
           从文本中提取 tt ID，仅保留图中存在的。
        """
        all_ids = TT_PATTERN.findall(text)
        return [i for i in all_ids if i in self._movie_nodes]

    # ═══════════════════════════════════════════════════════════════════
    #  Structured lookups (结构化查询)
    # ═══════════════════════════════════════════════════════════════════

    def get_movie_info(self, imdb_id: str) -> Optional[dict]:
        """Return all attributes of a movie as a dict, or None if not found.
           返回电影的所有属性字典，不存在则返回 None。
        """
        if not self.validate_movie_id(imdb_id):
            return None
        return dict(self.G.nodes[imdb_id])

    def get_genres(self, imdb_id: str) -> list[str]:
        """Return the list of genres (English) for a movie.
           返回电影的类型列表 (英文)。
        """
        if not self.validate_movie_id(imdb_id):
            return []
        return list(self.G.nodes[imdb_id].get(NA_GENRES, []))

    def get_cast(self, imdb_id: str) -> list[str]:
        """Return the cast list for a movie.
           返回电影的演员列表。
        """
        if not self.validate_movie_id(imdb_id):
            return []
        return list(self.G.nodes[imdb_id].get(NA_CAST, []))

    def get_director(self, imdb_id: str) -> str:
        """Return the director name for a movie.
           返回电影的导演名称。
        """
        if not self.validate_movie_id(imdb_id):
            return ""
        return self.G.nodes[imdb_id].get(NA_DIRECTOR, "")

    def get_movies_by_genre(self, genre: str, limit: int = 20) -> list[str]:
        """Return movie IDs for a given genre (Chinese or English name).
           按类型查询电影 ID (支持中文名或英文名)。
        """
        # Normalize: accept both CN and EN genre names
        genre_en = genre
        for n, d in self.G.nodes(data=True):
            if d.get(NA_NODE_TYPE) == NT_GENRE:
                if d.get(NA_GENRE_CN) == genre or d.get(NA_GENRE_EN) == genre:
                    genre_en = d.get(NA_GENRE_EN, genre)
                    break

        movies = []
        for _, tgt, d in self.G.edges(data=True):
            if d.get("relation") == ET_HAS_GENRE and tgt == genre_en:
                movies.append(_)
            if len(movies) >= limit:
                break
        # Sort by rating desc
        movies.sort(key=lambda m: self.G.nodes[m].get(NA_RATING, 0), reverse=True)
        return movies[:limit]

    def get_movies_by_actor(self, actor: str, limit: int = 20) -> list[str]:
        movies = []
        if actor in self.G.pred:
            for src, edges in self.G.pred[actor].items():
                for key, data in edges.items():
                    if data.get("relation") == ET_STARS_IN:
                        movies.append(src)
        movies.sort(key=lambda m: self.G.nodes[m].get(NA_RATING, 0), reverse=True)
        return movies[:limit]

    def get_movies_by_director(self, director: str, limit: int = 20) -> list[str]:
        movies = []
        if director in self.G.pred:
            for src, edges in self.G.pred[director].items():
                for key, data in edges.items():
                    if data.get("relation") == ET_DIRECTED_BY:
                        movies.append(src)
        movies.sort(key=lambda m: self.G.nodes[m].get(NA_RATING, 0), reverse=True)
        return movies[:limit]

    # ═══════════════════════════════════════════════════════════════════
    #  Composite similarity (组合相似度)
    # ═══════════════════════════════════════════════════════════════════

    def find_similar(self, imdb_id: str, k: int = 10) -> list[tuple[str, float]]:
        """Find top-k similar movies using composite scoring.
           使用组合评分查找最相似的 k 部电影。

        Scoring signals (评分信号):
          - genre_overlap  (Jaccard of genre sets)
          - shared_actors  (overlap coefficient)
          - shared_director (binary)
          - co_recommended (conversation co-occurrence weight)
          - recommended_for (recommendation pattern weight)

        Returns:
            List of (imdb_id, score) tuples, sorted by score desc.
            (imdb_id, 分数) 元组列表，按分数降序排列。
        """
        if not self.validate_movie_id(imdb_id):
            return []

        ref_genres = set(self.G.nodes[imdb_id].get(NA_GENRES, []))
        ref_cast = set(self.G.nodes[imdb_id].get(NA_CAST, []))
        ref_director = self.G.nodes[imdb_id].get(NA_DIRECTOR, "")

        # Pre-compute conversation edges from this movie
        co_rec_map = self._get_edge_targets(imdb_id, ET_CO_RECOMMENDED)
        rec_for_map = self._get_edge_targets(imdb_id, ET_RECOMMENDED_FOR)

        scores = defaultdict(float)

        for other in self._movie_nodes:
            if other == imdb_id:
                continue

            # genre overlap (Jaccard)
            other_genres = set(self.G.nodes[other].get(NA_GENRES, []))
            if ref_genres and other_genres:
                jaccard = len(ref_genres & other_genres) / len(ref_genres | other_genres)
            else:
                jaccard = 0.0
            scores[other] += SIM_WEIGHTS["genre_overlap"] * jaccard

            # shared actors (overlap coefficient)
            other_cast = set(self.G.nodes[other].get(NA_CAST, []))
            if ref_cast and other_cast:
                shared = len(ref_cast & other_cast)
                overlap = shared / min(len(ref_cast), len(other_cast))
            else:
                overlap = 0.0
            scores[other] += SIM_WEIGHTS["shared_actors"] * overlap

            # shared director (binary)
            other_director = self.G.nodes[other].get(NA_DIRECTOR, "")
            if ref_director and other_director and ref_director == other_director:
                scores[other] += SIM_WEIGHTS["shared_director"]

            # co-recommended weight
            co_w = co_rec_map.get(other, 0.0)
            if co_w > 0:
                # Normalize by max co-rec weight for this movie
                max_co = max(co_rec_map.values()) if co_rec_map else 1.0
                scores[other] += SIM_WEIGHTS["co_recommended"] * (co_w / max_co)

            # recommended-for weight
            rf_w = rec_for_map.get(other, 0.0)
            if rf_w > 0:
                max_rf = max(rec_for_map.values()) if rec_for_map else 1.0
                scores[other] += SIM_WEIGHTS["recommended_for"] * (rf_w / max_rf)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(mid, s) for mid, s in ranked[:k] if s > 0]

    def get_co_recommended(self, imdb_id: str, k: int = 10) -> list[tuple[str, float]]:
        """Return movies co-occurring with this one in conversations.
           返回在对话中与本电影共现的其他电影。
        """
        co_map = self._get_edge_targets(imdb_id, ET_CO_RECOMMENDED)
        ranked = sorted(co_map.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    # ═══════════════════════════════════════════════════════════════════
    #  Keyword search (关键词搜索)
    # ═══════════════════════════════════════════════════════════════════

    def search_movies(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """Search movies by title/overview keyword match.
           按标题/简介关键词搜索电影。

        Returns:
            List of (imdb_id, relevance_score) tuples.
        """
        keywords = [w.lower() for w in query.split() if len(w) >= 2]
        if not keywords:
            return []

        results = []
        for mid in self._movie_nodes:
            node = self.G.nodes[mid]
            title = (node.get(NA_TITLE, "") + " " + node.get(NA_ORIGINAL_TITLE, "")).lower()
            overview = node.get(NA_OVERVIEW, "").lower()
            text = title + " " + overview

            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                results.append((mid, score))

        results.sort(key=lambda x: (x[1],
                                    self.G.nodes[x[0]].get(NA_RATING, 0)),
                     reverse=True)
        return results[:k]

    # ═══════════════════════════════════════════════════════════════════
    #  LLM context generation (LLM 上下文生成)
    # ═══════════════════════════════════════════════════════════════════

    def get_llm_context(self, user_query: str, k: int = 10) -> str:
        """Build a structured context string for an LLM prompt.
           为 LLM prompt 构建结构化上下文字符串。

        Extracts movie IDs from the query, finds similar and co-recommended
        movies, and formats their structured attributes as a context block.
        This is the core anti-hallucination mechanism: the LLM receives
        verified facts from the graph and is instructed to only recommend
        movies from this set.
        从用户查询中提取电影 ID，查找相似和共现推荐电影，
        将其结构化属性格式化为上下文块。这是核心防幻觉机制:
        LLM 接收图中已验证的事实，并被指示仅从该集合中推荐电影。
        """
        known_ids = self.extract_known_ids(user_query)

        # Collect candidate movies
        candidates = set(known_ids)
        reasoning_parts = []

        for mid in known_ids:
            info = self.get_movie_info(mid)
            if info:
                reasoning_parts.append(f"User mentioned: {mid} - {info.get(NA_TITLE, '')}")
            for sid, _ in self.find_similar(mid, k=k):
                candidates.add(sid)
            for sid, _ in self.get_co_recommended(mid, k=k):
                candidates.add(sid)

        # Also do keyword search if no IDs found
        if not known_ids:
            for sid, _ in self.search_movies(user_query, k=k):
                candidates.add(sid)

        # If still empty, return a note
        if not candidates:
            return "No matching movies found in the knowledge graph."

        # Format context
        lines = ["=== Knowledge Graph Context ==="]
        lines.append(f"User query: {user_query}")
        lines.append(f"Referenced movies: {known_ids}")
        lines.append("")

        for mid in sorted(candidates, key=lambda m: self.G.nodes[m].get(NA_RATING, 0), reverse=True)[:k * 2]:
            info = self.get_movie_info(mid)
            if not info:
                continue
            title = info.get(NA_TITLE, "")
            orig = info.get(NA_ORIGINAL_TITLE, "")
            year = info.get(NA_YEAR, "")
            genres = info.get(NA_GENRES, [])
            director = info.get(NA_DIRECTOR, "")
            cast = info.get(NA_CAST, [])[:5]
            rating = info.get(NA_RATING, 0)
            overview = info.get(NA_OVERVIEW, "")[:200]
            lines.append(f"[{mid}] {title} / {orig} ({year})")
            lines.append(f"  Genres: {genres}")
            lines.append(f"  Director: {director}")
            lines.append(f"  Cast: {cast}")
            lines.append(f"  Rating: {rating}")
            if overview:
                lines.append(f"  Overview: {overview}")
            lines.append("")

        lines.append(f"=== Total candidate movies: {len(candidates)} ===")
        lines.append("IMPORTANT: Only recommend movies from the above list. "
                       "Do not invent movie IDs or titles.")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════
    #  Internal helpers (内部辅助)
    # ═══════════════════════════════════════════════════════════════════

    def _get_edge_targets(self, source: str, relation: str) -> dict[str, float]:
        """Get all targets of a specific relation from source, with weights.
           获取从源节点出发的指定关系类型的所有目标节点及权重。
        """
        result = {}
        if not self.G.has_node(source):
            return result
        for _, tgt, d in self.G.edges(source, data=True):
            if d.get("relation") == relation:
                result[tgt] = d.get(EA_WEIGHT, 1.0)
        return result

    # ═══════════════════════════════════════════════════════════════════
    #  Stats (统计信息)
    # ═══════════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        """Return graph statistics.
           返回图谱统计信息。
        """
        node_types = defaultdict(int)
        for _, d in self.G.nodes(data=True):
            node_types[d.get(NA_NODE_TYPE, "Unknown")] += 1

        edge_types = defaultdict(int)
        for _, _, d in self.G.edges(data=True):
            edge_types[d.get("relation", "Unknown")] += 1

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "movie_count": len(self._movie_nodes),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
        }