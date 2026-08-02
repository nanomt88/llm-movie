# -*- coding: utf-8 -*-
"""
Intent parser for natural-language movie queries without movie IDs.
无电影 ID 的自然语言查询意图解析器。

When a user query contains no tt IDs (e.g. "想看紧张刺激的犯罪片，类似吴宇森"),
this module extracts structured intent (genres, persons, keywords) from the
text and maps them to graph queries.
当用户查询不含 tt ID 时（如"想看紧张刺激的犯罪片，类似吴宇森"），
本模块从文本中提取结构化意图（类型、人物、关键词），并映射到图谱查询。

Two strategies:
  - Rule-based (规则匹配): match genre names and person names against the query text
  - LLM-assisted (LLM辅助): use LLM to parse natural language into structured intent
"""

import re
from collections import defaultdict
from typing import Optional

from movie.kg.graph import MovieKnowledgeGraph
from movie.kg.schema import (
    NT_GENRE, NT_PERSON,
    NA_TITLE, NA_ORIGINAL_TITLE, NA_GENRES, NA_RATING, NA_OVERVIEW,
    NA_GENRE_CN, NA_GENRE_EN, NA_NODE_TYPE,
)


# ═══════════════════════════════════════════════════════════════════════
#  Intent data structure (意图数据结构)
# ═══════════════════════════════════════════════════════════════════════

class Intent:
    """Parsed user intent from a natural-language query.
       从自然语言查询中解析出的用户意图。"""

    def __init__(self, raw: str = ""):
        self.raw = raw            # 原始查询文本
        self.genres: list[str] = []     # 匹配到的类型 (英文名)
        self.persons: list[str] = []    # 匹配到的人物名
        self.keywords: list[str] = []   # 剩余关键词
        self.year_from: str = ""        # 年份范围-起
        self.year_to: str = ""          # 年份范围-止

    def has_any_signal(self) -> bool:
        return bool(self.genres or self.persons or self.keywords)

    def __str__(self) -> str:
        parts = [f"raw={self.raw!r}"]
        if self.genres:
            parts.append(f"genres={self.genres}")
        if self.persons:
            parts.append(f"persons={self.persons}")
        if self.keywords:
            parts.append(f"keywords={self.keywords}")
        return f"Intent({', '.join(parts)})"


# ═══════════════════════════════════════════════════════════════════════
#  Rule-based intent parser (规则意图解析器)
# ═══════════════════════════════════════════════════════════════════════

class IntentParser:
    """Parse natural-language queries into structured intent.
       将自然语言查询解析为结构化意图。

    Pre-builds indices at init time for fast query-time matching:
    初始化时预建索引，查询时快速匹配:
      - Genre name lookup (26 genres, CN + EN)
      - Person name set (205K names, substring match)
      - Title bigram inverted index (for keyword fallback)
    """

    def __init__(self, kg: MovieKnowledgeGraph):
        self.kg = kg
        self._build_genre_index()
        self._build_person_index()
        self._build_title_index()

    def _build_genre_index(self):
        """Build {genre_name_lower: genre_en} for all 26 genres.
           为所有类型构建 {类型名小写: 英文名} 查找表。"""
        self._genre_lookup = {}
        for node, data in self.kg.G.nodes(data=True):
            if data.get(NA_NODE_TYPE) != NT_GENRE:
                continue
            cn = data.get(NA_GENRE_CN, "")
            en = data.get(NA_GENRE_EN, "")
            if cn:
                self._genre_lookup[cn] = en
            if en:
                self._genre_lookup[en.lower()] = en

    def _build_person_index(self):
        """Build a set of all person names for substring matching.
           构建所有人物名的集合，用于子串匹配。"""
        self._person_names = set()
        for node, data in self.kg.G.nodes(data=True):
            if data.get(NA_NODE_TYPE) == NT_PERSON:
                self._person_names.add(node)

    def _build_title_index(self):
        """Build bigram inverted index: {2-char substring: set(movie_ids)}.
           构建二元组倒排索引: {2字符子串: 电影ID集合}。

        Only indexes movie titles (not overviews) for speed.
        Speeds up Chinese keyword matching dramatically.
        """
        self._title_bigrams = defaultdict(set)
        for mid in self.kg._movie_nodes:
            node = self.kg.G.nodes[mid]
            title = node.get(NA_TITLE, "") + node.get(NA_ORIGINAL_TITLE, "")
            for i in range(len(title) - 1):
                bigram = title[i:i + 2]
                if bigram.strip():
                    self._title_bigrams[bigram].add(mid)

    def parse(self, query: str) -> Intent:
        """Extract genres, persons, and keywords from a natural-language query.
           从自然语言查询中提取类型、人物和关键词。

        Steps (步骤):
          1. Match genre names (CN + EN) against query text
          2. Match person names (directors/actors) against query text
          3. Extract year range if mentioned (e.g. "2010年以后")
          4. Remaining text becomes keywords for title/overview search
        """
        intent = Intent(raw=query)
        query_lower = query.lower()

        # --- 1. Genre matching (类型匹配) ---
        for genre_name, genre_en in self._genre_lookup.items():
            if genre_name in query or genre_name in query_lower:
                if genre_en not in intent.genres:
                    intent.genres.append(genre_en)

        # --- 2. Person matching (人物匹配) ---
        # Check if any person name appears as substring in the query.
        # 205K names, each check is a fast C-level string search.
        for name in self._person_names:
            if len(name) >= 2 and name in query:
                intent.persons.append(name)

        # --- 3. Year range extraction (年份范围提取) ---
        year_match = re.search(r"(20\d{2}|19\d{2}).*?(?:以后|之后|after)", query)
        if year_match:
            intent.year_from = year_match.group(1)
        year_match2 = re.search(r"(20\d{2}|19\d{2}).*?(?:以前|之前|before)", query)
        if year_match2:
            intent.year_to = year_match2.group(1)

        # --- 4. Keyword extraction (关键词提取) ---
        # Remove matched genres and persons from query, extract remaining keywords.
        remaining = query
        for g in intent.genres:
            cn_name = next((k for k, v in self._genre_lookup.items()
                           if v == g and len(k) > 1), "")
            if cn_name:
                remaining = remaining.replace(cn_name, " ")
            if g:
                remaining = remaining.replace(g, " ")
        for p in intent.persons:
            remaining = remaining.replace(p, " ")

        # Extract 2-3 char Chinese substrings as keywords
        # (Chinese has no spaces, so we use sliding window)
        cleaned = re.sub(r"[\s,，。、！？!?「」\"']", "", remaining)
        keywords = set()
        for i in range(len(cleaned) - 1):
            bigram = cleaned[i:i + 2]
            if len(bigram) >= 2:
                keywords.add(bigram)
        intent.keywords = list(keywords)

        return intent


# ═══════════════════════════════════════════════════════════════════════
#  Intent-based search (基于意图的搜索)
# ═══════════════════════════════════════════════════════════════════════

def search_by_intent(intent: Intent, kg: MovieKnowledgeGraph,
                     k: int = 20) -> list[tuple[str, float]]:
    """Search the graph based on parsed intent.
       基于解析出的意图查询图谱。

    Scoring (评分):
      - Genre match: +2.0 per matching genre
      - Person match: +3.0 per matching person (stronger signal)
      - Keyword match: +0.5 per matching title bigram
      - Year filter: 0 (filter, not scored)
      - Rating bonus: added as tiebreaker

    Returns:
        List of (imdb_id, score) tuples, sorted by score desc.
    """
    candidates = defaultdict(float)

    # --- Genre matches (类型匹配) ---
    for genre in intent.genres:
        for mid in kg.get_movies_by_genre(genre, limit=k * 3):
            candidates[mid] += 2.0

    # --- Person matches (人物匹配) ---
    for person in intent.persons:
        for mid in kg.get_movies_by_actor(person, limit=k * 3):
            candidates[mid] += 3.0
        for mid in kg.get_movies_by_director(person, limit=k * 3):
            candidates[mid] += 3.0

    # --- Keyword matches via title bigram index (关键词匹配) ---
    parser = _get_parser(kg)
    for kw in intent.keywords:
        for mid in parser._title_bigrams.get(kw, set()):
            candidates[mid] += 0.5

    # --- Year filter (年份过滤) ---
    if intent.year_from or intent.year_to:
        filtered = {}
        for mid, score in candidates.items():
            year = kg.G.nodes[mid].get("year", "")
            if intent.year_from and year and year >= intent.year_from:
                filtered[mid] = score
            elif intent.year_to and year and year <= intent.year_to:
                filtered[mid] = score
            elif not intent.year_from and not intent.year_to:
                filtered[mid] = score
        candidates = defaultdict(float, filtered)

    # --- Rank: score desc, then rating desc ---
    ranked = sorted(candidates.items(),
                     key=lambda x: (x[1],
                                    kg.G.nodes[x[0]].get(NA_RATING, 0)),
                     reverse=True)
    return ranked[:k]


# ═══════════════════════════════════════════════════════════════════════
#  LLM-assisted intent parsing (LLM 辅助意图解析)
# ═══════════════════════════════════════════════════════════════════════

LLM_INTENT_PROMPT = """You are a movie query intent parser. Parse the user's movie 
recommendation request into structured JSON.

Extract these fields (提取以下字段):
- genres: list of movie genres mentioned (in English, e.g. ["Crime", "Thriller"])
- persons: list of actor or director names mentioned
- keywords: list of mood/style keywords (e.g. ["slow-paced", "dark", "tense"])
- year_from: earliest year (string or null)
- year_to: latest year (string or null)

If a Chinese genre name is used, translate to English.
Map mood words to closest genre if possible (e.g. "紧张刺激" -> "Thriller").

Respond ONLY with JSON, no explanation.
Example input: "想看紧张刺激的犯罪片，类似吴宇森的风格"
Example output: {"genres": ["Crime", "Thriller"], "persons": ["吴宇森"], "keywords": ["tense", "stylish"], "year_from": null, "year_to": null}
"""


def llm_parse_intent(query: str, llm) -> Intent:
    """Use an LLM to parse natural language into structured intent.
       使用 LLM 将自然语言解析为结构化意图。

    More accurate than rule-based parsing for complex queries:
    - Understands semantic mappings ("紧张刺激" -> Thriller)
    - Handles paraphrases and implicit intent
    对于复杂查询比规则解析更准确:
    - 理解语义映射 ("紧张刺激" -> 惊悚)
    - 处理转述和隐含意图
    """
    response = llm.chat(LLM_INTENT_PROMPT, f"User query: {query}")

    # Extract JSON from response
    json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
    if not json_match:
        return Intent(raw=query)

    import json
    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return Intent(raw=query)

    intent = Intent(raw=query)
    intent.genres = data.get("genres", [])
    intent.persons = data.get("persons", [])
    intent.keywords = data.get("keywords", [])
    intent.year_from = data.get("year_from") or ""
    intent.year_to = data.get("year_to") or ""
    return intent


# ═══════════════════════════════════════════════════════════════════════
#  Intent-based LLM context (基于意图的 LLM 上下文)
# ═══════════════════════════════════════════════════════════════════════

def get_intent_llm_context(intent: Intent, kg: MovieKnowledgeGraph,
                           k: int = 10) -> str:
    """Build LLM context from intent-based search results.
       从意图搜索结果构建 LLM 上下文。

    Used when the user query has no tt IDs. Searches the graph
    using the parsed intent and formats results as context.
    当用户查询不含 tt ID 时使用。用解析出的意图搜索图谱，
    将结果格式化为上下文。
    """
    results = search_by_intent(intent, kg, k=k * 2)

    if not results:
        return "No matching movies found in the knowledge graph."

    lines = ["=== Knowledge Graph Context (Intent-Based) ==="]
    lines.append(f"User query: {intent.raw}")
    lines.append(f"Parsed intent: genres={intent.genres}, "
                   f"persons={intent.persons}, keywords={intent.keywords}")
    lines.append("")

    for mid, score in results[:k * 2]:
        node = kg.G.nodes[mid]
        title = node.get(NA_TITLE, "")
        orig = node.get(NA_ORIGINAL_TITLE, "")
        year = node.get("year", "")
        genres = node.get(NA_GENRES, [])
        director = node.get("director", "")
        cast = node.get("cast", [])[:5]
        rating = node.get(NA_RATING, 0)
        overview = node.get(NA_OVERVIEW, "")[:200]
        lines.append(f"[{mid}] {title} / {orig} ({year})")
        lines.append(f"  Genres: {genres}")
        lines.append(f"  Director: {director}")
        lines.append(f"  Cast: {cast}")
        lines.append(f"  Rating: {rating}")
        lines.append(f"  Match Score: {score:.2f}")
        if overview:
            lines.append(f"  Overview: {overview}")
        lines.append("")

    lines.append(f"=== Total candidate movies: {len(results)} ===")
    lines.append("IMPORTANT: Only recommend movies from the above list. "
                   "Do not invent movie IDs or titles.")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Internal: parser cache (解析器缓存)
# ═══════════════════════════════════════════════════════════════════════

_parser_cache: dict[int, IntentParser] = {}


def _get_parser(kg: MovieKnowledgeGraph) -> IntentParser:
    """Get or create an IntentParser for a given KG (cached).
       获取或创建图谱的 IntentParser（带缓存）。"""
    kg_id = id(kg)
    if kg_id not in _parser_cache:
        _parser_cache[kg_id] = IntentParser(kg)
    return _parser_cache[kg_id]