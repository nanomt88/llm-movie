# -*- coding: utf-8 -*-
"""
LLM-powered movie recommender with knowledge graph grounding.
基于知识图谱防幻觉的电影推荐器。

Provides KGRecommender, which combines the MovieKnowledgeGraph with an
optional LLM client to produce recommendations grounded in verified graph
facts. Four anti-hallucination layers ensure recommendations are accurate:
提供 KGRecommender，将知识图谱与可选的 LLM 客户端结合，
生成基于已验证图谱事实的推荐。四层防幻觉机制保障推荐准确性:

1. Context injection (上下文注入): LLM 接收图中已验证的结构化事实
2. Output constraint (输出约束): 指示 LLM 仅从候选集合中推荐
3. Post-validation (后校验): 从 LLM 输出提取 tt ID, 校验是否在图中
4. Evidence trace (证据追溯): 每个推荐附带图中的关系路径作为依据
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from movie.kg.graph import MovieKnowledgeGraph
from movie.kg.schema import (
    NA_TITLE, NA_ORIGINAL_TITLE, NA_YEAR, NA_GENRES,
    NA_DIRECTOR, NA_CAST, NA_RATING, NA_OVERVIEW,
)

TT_PATTERN = re.compile(r"tt\d{7,9}")


# ═══════════════════════════════════════════════════════════════════════
#  LLM client interface (LLM 客户端接口)
# ═══════════════════════════════════════════════════════════════════════

class LLMClient(Protocol):
    """Protocol for pluggable LLM clients (OpenAI, local, etc.).
       可插拔 LLM 客户端协议 (OpenAI, 本地模型等)。
    """
    def chat(self, system_prompt: str, user_message: str) -> str:
        """Send a chat request, return the assistant response text.
           发送聊天请求, 返回助手响应文本。
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
#  Recommendation result (推荐结果)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Recommendation:
    """A single movie recommendation with evidence.
       带证据的单个电影推荐。
    """
    imdb_id: str
    title: str
    original_title: str
    year: str
    genres: list[str]
    director: str
    rating: float
    overview: str
    similarity_score: float = 0.0
    co_occurrence: float = 0.0
    reason: str = ""
    source: str = "graph"  # "graph" | "llm"

    def __str__(self) -> str:
        title_str = self.title or self.original_title
        return f"[{self.imdb_id}] {title_str} ({self.year}) - Rating: {self.rating} - {self.reason}"


@dataclass
class RecommendationResult:
    """Complete recommendation result with metadata.
       完整的推荐结果及元数据。
    """
    query: str
    recommendations: list[Recommendation] = field(default_factory=list)
    context: str = ""
    llm_response: str = ""
    validated: bool = True
    rejected_ids: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"=== Recommendations for: {self.query} ==="]
        for rec in self.recommendations:
            lines.append(f"  {rec}")
        if self.rejected_ids:
            lines.append(f"  [REJECTED - not in graph] {self.rejected_ids}")
        if self.llm_response:
            lines.append(f"\n--- LLM Response ---\n{self.llm_response}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  System prompt template (系统提示词模板)
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a movie recommendation expert. You will receive a knowledge graph context with verified movie information and a user query.

Rules (必须遵守的规则):
1. ONLY recommend movies that appear in the "Knowledge Graph Context" section below.
   仅推荐下方"知识图谱上下文"中出现过的电影。
2. Do NOT invent movie IDs or titles. Use the exact tt IDs from the context.
   不要编造电影 ID 或标题。使用上下文中的确切 tt ID。
3. For each recommendation, explain WHY it matches the user request, referencing
   shared genres, actors, director, or conversation co-occurrence from the context.
   对于每个推荐, 解释为何匹配用户需求, 引用上下文中的共享类型/演员/导演/对话共现。
4. Return recommendations as: tt_id | title | reason (one per line).
   以如下格式返回推荐: tt_id | title | reason (每行一个)。
5. If no movies in the context match the request, say "No matching movies found."
   如果上下文中没有匹配的电影, 回答 "No matching movies found."
"""


# ═══════════════════════════════════════════════════════════════════════
#  KGRecommender (知识图谱推荐器)
# ═══════════════════════════════════════════════════════════════════════

class KGRecommender:
    """Movie recommender grounded in the knowledge graph.
       基于知识图谱的电影推荐器。

    Can operate in two modes:
      - Graph-only (纯图谱模式): No LLM; returns structured recommendations from graph queries.
      - KG+LLM (图谱+LLM模式): Uses LLM with graph context for natural language recommendations.
    可在两种模式下运行:
      - 纯图谱模式: 无 LLM; 返回图谱查询的结构化推荐。
      - 图谱+LLM模式: 使用 LLM 和图谱上下文生成自然语言推荐。
    """

    def __init__(self, kg: MovieKnowledgeGraph, llm: Optional[LLMClient] = None):
        self.kg = kg
        self.llm = llm

    def recommend(self, user_query: str, k: int = 5) -> RecommendationResult:
        """Generate recommendations for a user query.
           为用户查询生成推荐。

        Args:
            user_query: 用户的自然语言查询, 可能包含 tt ID 或电影描述
            k: 返回推荐数量

        Returns:
            RecommendationResult with validated recommendations.
        """
        # Step 1: Build graph context (构建图谱上下文)
        context = self.kg.get_llm_context(user_query, k=k)

        # Step 2a: If LLM available, use it (如果有 LLM, 使用它)
        if self.llm is not None:
            return self._recommend_with_llm(user_query, context, k)
        # Step 2b: Graph-only mode (纯图谱模式)
        return self._recommend_graph_only(user_query, k)

    def _recommend_graph_only(self, query: str, k: int) -> RecommendationResult:
        """Generate recommendations using graph queries only (no LLM).
           仅使用图谱查询生成推荐 (无 LLM)。
        """
        known_ids = self.kg.extract_known_ids(query)
        candidates: dict[str, dict] = {}  # imdb_id -> {score, co_rec, reason}

        if known_ids:
            for mid in known_ids:
                for sid, score in self.kg.find_similar(mid, k=k * 2):
                    if sid not in candidates:
                        candidates[sid] = {"score": score, "co_rec": 0.0,
                                           "reason": f"Similar to {mid}"}
                    else:
                        candidates[sid]["score"] += score
                        candidates[sid]["reason"] += f", similar to {mid}"
                for sid, co_w in self.kg.get_co_recommended(mid, k=k * 2):
                    if sid not in candidates:
                        candidates[sid] = {"score": 0.0, "co_rec": co_w,
                                           "reason": f"Co-mentioned with {mid}"}
                    else:
                        candidates[sid]["co_rec"] += co_w
                        candidates[sid]["reason"] += f", co-mentioned with {mid}"
        else:
            # Keyword search fallback
            for sid, score in self.kg.search_movies(query, k=k * 2):
                candidates[sid] = {"score": score, "co_rec": 0.0,
                                   "reason": "Keyword match"}

        # Rank and build recommendations
        ranked = sorted(candidates.items(),
                        key=lambda x: x[1]["score"] + x[1]["co_rec"] * 0.1,
                        reverse=True)

        recs = []
        for mid, info in ranked[:k]:
            mi = self.kg.get_movie_info(mid)
            if not mi:
                continue
            recs.append(Recommendation(
                imdb_id=mid,
                title=mi.get(NA_TITLE, ""),
                original_title=mi.get(NA_ORIGINAL_TITLE, ""),
                year=mi.get(NA_YEAR, ""),
                genres=mi.get(NA_GENRES, []),
                director=mi.get(NA_DIRECTOR, ""),
                rating=mi.get(NA_RATING, 0.0),
                overview=mi.get(NA_OVERVIEW, "")[:300],
                similarity_score=info["score"],
                co_occurrence=info["co_rec"],
                reason=info["reason"],
                source="graph",
            ))

        return RecommendationResult(query=query, recommendations=recs,
                                     context=self.kg.get_llm_context(query, k=k))

    def _recommend_with_llm(self, query: str, context: str, k: int) -> RecommendationResult:
        """Generate recommendations using LLM with graph context.
           使用 LLM 和图谱上下文生成推荐。

        Anti-hallucination pipeline (防幻觉流水线):
          1. Inject verified graph facts as context
          2. Constrain LLM to only use movies from the context
          3. Extract tt IDs from LLM output
          4. Validate each ID against the graph (reject hallucinations)
        """
        user_msg = f"""User Query: {query}

=== Knowledge Graph Context ===
{context}

Please recommend {k} movies from the context above. Format: tt_id | title | reason
"""

        llm_response = self.llm.chat(SYSTEM_PROMPT, user_msg)

        # Post-validation: extract tt IDs and validate (后校验: 提取并验证 tt ID)
        raw_ids = TT_PATTERN.findall(llm_response)
        valid_ids = self.kg.validate_movie_ids(raw_ids)
        rejected = [i for i in raw_ids if i not in self.kg._movie_nodes]

        # Build recommendation objects from validated IDs
        recs = []
        seen = set()
        for mid in valid_ids:
            if mid in seen:
                continue
            seen.add(mid)
            mi = self.kg.get_movie_info(mid)
            if not mi:
                continue
            # Extract reason from LLM response if possible
            reason = _extract_reason(llm_response, mid)
            recs.append(Recommendation(
                imdb_id=mid,
                title=mi.get(NA_TITLE, ""),
                original_title=mi.get(NA_ORIGINAL_TITLE, ""),
                year=mi.get(NA_YEAR, ""),
                genres=mi.get(NA_GENRES, []),
                director=mi.get(NA_DIRECTOR, ""),
                rating=mi.get(NA_RATING, 0.0),
                overview=mi.get(NA_OVERVIEW, "")[:300],
                reason=reason,
                source="llm",
            ))
            if len(recs) >= k:
                break

        return RecommendationResult(
            query=query, recommendations=recs,
            context=context, llm_response=llm_response,
            validated=(len(rejected) == 0),
            rejected_ids=rejected,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Helper functions (辅助函数)
# ═══════════════════════════════════════════════════════════════════════

def _extract_reason(llm_response: str, imdb_id: str) -> str:
    """Try to extract the reason text after a tt_id in LLM output.
       尝试从 LLM 输出中提取 tt_id 后面的推荐理由文本。
    """
    pattern = re.compile(
        re.escape(imdb_id) + r"\s*\|?\s*[^\|]*\|?\s*(.+?)(?:\n|$)"
    )
    m = pattern.search(llm_response)
    if m:
        return m.group(1).strip().strip("|").strip()
    return ""