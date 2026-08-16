# -*- coding: utf-8 -*-
"""
Test intent parsing for queries without movie IDs.
测试无电影 ID 查询的意图解析。
"""

from movie.config import log
from movie.data_loader import load_conversations, load_movie_info
from movie.kg import build_graph, MovieKnowledgeGraph
from movie.kg.intent import IntentParser, search_by_intent, get_intent_llm_context

DEMO_MAX_CONV_ROWS = 50000

def main():
    log("Building graph...", "IntentTest")
    movie_info = load_movie_info()
    conv_rows = load_conversations(
        "D:\\workspaces\\python\\llm-movie\\data\\conv\\data_all.csv",
        max_rows=DEMO_MAX_CONV_ROWS
    )
    G = build_graph(movie_info=movie_info, conv_rows=conv_rows)
    kg = MovieKnowledgeGraph(G)

    log("Building intent parser (pre-building indices)...", "IntentTest")
    parser = IntentParser(kg)
    log("Indices ready.", "IntentTest")

    # === Test 1: Chinese query with genre + person ===
    query1 = "想看紧张刺激的犯罪片，类似吴宇森的风格"
    print("\n" + "=" * 60)
    print(f"Test 1: {query1}")
    print("=" * 60)
    intent1 = parser.parse(query1)
    print(f"  Parsed: {intent1}")
    results1 = search_by_intent(intent1, kg, k=5)
    print(f"  Found {len(results1)} movies:")
    for mid, score in results1:
        node = kg.G.nodes[mid]
        title = node.get("title", "") or node.get("original_title", "")
        print(f"    [{mid}] {title} ({node.get('year', '')}) "
              f"Genres={node.get('genres', [])} Score={score:.2f}")

    # === Test 2: English query ===
    query2 = "I want a dark fantasy movie"
    print("\n" + "=" * 60)
    print(f"Test 2: {query2}")
    print("=" * 60)
    intent2 = parser.parse(query2)
    print(f"  Parsed: {intent2}")
    results2 = search_by_intent(intent2, kg, k=5)
    print(f"  Found {len(results2)} movies:")
    for mid, score in results2:
        node = kg.G.nodes[mid]
        title = node.get("title", "") or node.get("original_title", "")
        print(f"    [{mid}] {title} ({node.get('year', '')}) "
              f"Genres={node.get('genres', [])} Score={score:.2f}")

    # === Test 3: Pure mood description ===
    query3 = "想看轻松的喜剧"
    print("\n" + "=" * 60)
    print(f"Test 3: {query3}")
    print("=" * 60)
    intent3 = parser.parse(query3)
    print(f"  Parsed: {intent3}")
    results3 = search_by_intent(intent3, kg, k=5)
    print(f"  Found {len(results3)} movies:")
    for mid, score in results3:
        node = kg.G.nodes[mid]
        title = node.get("title", "") or node.get("original_title", "")
        print(f"    [{mid}] {title} ({node.get('year', '')}) "
              f"Genres={node.get('genres', [])} Score={score:.2f}")

    # === Test 4: Genre only ===
    query4 = "科幻片"
    print("\n" + "=" * 60)
    print(f"Test 4: {query4}")
    print("=" * 60)
    intent4 = parser.parse(query4)
    print(f"  Parsed: {intent4}")
    results4 = search_by_intent(intent4, kg, k=5)
    print(f"  Found {len(results4)} movies:")
    for mid, score in results4:
        node = kg.G.nodes[mid]
        title = node.get("title", "") or node.get("original_title", "")
        print(f"    [{mid}] {title} ({node.get('year', '')}) "
              f"Genres={node.get('genres', [])} Score={score:.2f}")

    # === Test 5: Generate LLM context from intent ===
    print("\n" + "=" * 60)
    print("Test 5: LLM Context from Intent")
    print("=" * 60)
    ctx = get_intent_llm_context(intent1, kg, k=5)
    print(ctx[:2500])
    if len(ctx) > 2500:
        print(f"... ({len(ctx)} chars total)")

    print("\n" + "=" * 60)
    print("Intent parsing test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()