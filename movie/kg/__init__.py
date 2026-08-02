# -*- coding: utf-8 -*-
"""
Movie Knowledge Graph package for KG+LLM movie recommendation.
电影知识图谱包, 用于知识图谱结合 LLM 的电影推荐。

Usage (用法):
    from movie.kg import build_graph, MovieKnowledgeGraph, KGRecommender

    G = build_graph()                          # Build from default data paths
    kg = MovieKnowledgeGraph(G)                # Wrap with query interface
    recs = kg.find_similar("tt0163676")        # Find similar movies
    ctx = kg.get_llm_context(user_query)       # Generate LLM context

Anti-hallucination guarantees (防幻觉保障):
    - All recommended movie IDs are validated against the graph
    - LLM context contains only verified facts from movie_info.json
    - Conversation-derived edges provide real recommendation patterns
"""

from movie.kg.builder import build_graph
from movie.kg.graph import MovieKnowledgeGraph
from movie.kg.recommender import (
    KGRecommender,
    Recommendation,
    RecommendationResult,
    LLMClient,
    SYSTEM_PROMPT,
)
from movie.kg.intent import (
    Intent,
    IntentParser,
    search_by_intent,
    llm_parse_intent,
    get_intent_llm_context,
    LLM_INTENT_PROMPT,
)
from movie.kg.schema import (
    NT_MOVIE, NT_GENRE, NT_PERSON, NT_COUNTRY, NT_YEAR,
    ET_HAS_GENRE, ET_DIRECTED_BY, ET_STARS_IN,
    ET_CO_RECOMMENDED, ET_RECOMMENDED_FOR, ET_SIMILAR_TO,
    SIM_WEIGHTS,
)

__all__ = [
    "build_graph",
    "MovieKnowledgeGraph",
    "KGRecommender",
    "Recommendation",
    "RecommendationResult",
    "LLMClient",
    "SYSTEM_PROMPT",
    "Intent",
    "IntentParser",
    "search_by_intent",
    "llm_parse_intent",
    "get_intent_llm_context",
    "LLM_INTENT_PROMPT",
    "NT_MOVIE", "NT_GENRE", "NT_PERSON", "NT_COUNTRY", "NT_YEAR",
    "ET_HAS_GENRE", "ET_DIRECTED_BY", "ET_STARS_IN",
    "ET_CO_RECOMMENDED", "ET_RECOMMENDED_FOR", "ET_SIMILAR_TO",
    "SIM_WEIGHTS",
]