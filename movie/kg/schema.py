# -*- coding: utf-8 -*-
"""
Schema definitions for the movie knowledge graph.
电影知识图谱的节点/边类型与关系常量定义。

This module centralizes all string constants for node types, edge types,
and relation names so that builder and query code reference symbols
instead of magic strings.
本模块集中定义所有节点类型、边类型、关系名称的字符串常量，
使构建器与查询代码引用符号而非魔法字符串。
"""

# Node types (节点类型)
NT_MOVIE = "Movie"
NT_GENRE = "Genre"
NT_PERSON = "Person"
NT_COUNTRY = "Country"
NT_YEAR = "Year"

# Edge types -- structured (from movie_info.json)
# 边类型 -- 结构化关系 (来自 movie_info.json)
ET_HAS_GENRE = "HAS_GENRE"
ET_DIRECTED_BY = "DIRECTED_BY"
ET_STARS_IN = "STARS_IN"
ET_FROM_COUNTRY = "FROM_COUNTRY"
ET_RELEASED_IN = "RELEASED_IN"

# Edge types -- conversation-derived (from data_all.csv)
# 边类型 -- 对话衍生关系 (来自 data_all.csv)
ET_CO_RECOMMENDED = "CO_RECOMMENDED"
ET_RECOMMENDED_FOR = "RECOMMENDED_FOR"
ET_SIMILAR_TO = "SIMILAR_TO"

# Edge attribute keys (边属性键)
EA_WEIGHT = "weight"
EA_UPVOTES = "upvotes"
EA_COUNT = "count"
EA_SOURCE = "source"

# Scoring weights for composite similarity (组合相似度权重)
SIM_WEIGHTS = {
    "genre_overlap":   0.35,
    "shared_actors":   0.15,
    "shared_director": 0.15,
    "co_recommended":  0.25,
    "recommended_for": 0.10,
}

# Node attribute keys (节点属性键)
NA_TITLE = "title"
NA_ORIGINAL_TITLE = "original_title"
NA_YEAR = "year"
NA_COUNTRY = "country"
NA_RUNTIME = "runtime_minutes"
NA_DIRECTOR = "director"
NA_GENRES = "genres"
NA_CAST = "cast"
NA_RATING = "rating"
NA_VOTE_COUNT = "vote_count"
NA_OVERVIEW = "overview"
NA_POSTER = "poster_url"
NA_GENRE_CN = "genre_cn"
NA_GENRE_EN = "genre_en"
NA_NODE_TYPE = "node_type"