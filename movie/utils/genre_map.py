# -*- coding: utf-8 -*-
"""
Chinese-to-English genre mapping for movie_info.json genres.
电影类型中文→英文映射工具。

The genres in movie_info.json are in Chinese (UTF-8). This module maps them
to English equivalents for downstream text analysis, chart labels, and ABSA.
movie_info.json 中的电影类型为中文字段，此模块将其映射为英文，
用于下游文本分析、图表标签和 ABSA（基于方面的情感分析）。
"""

# ═══════════════════════════════════════════════════════════════════════
#  Chinese → English genre mapping
#  中文 → 英文 电影类型映射表
# ═══════════════════════════════════════════════════════════════════════
# Keys are the actual Chinese UTF-8 strings as stored in movie_info.json.
# 键为 movie_info.json 中实际存储的中文字符串。
GENRE_CN_TO_EN = {
    '剧情': 'Drama',
    '喜剧': 'Comedy',
    '动作': 'Action',
    '爱情': 'Romance',
    '科幻': 'Sci-Fi',
    '恐怖': 'Horror',
    '悬疑': 'Mystery',
    '犯罪': 'Crime',
    '惊悚': 'Thriller',
    '奇幻': 'Fantasy',
    '冒险': 'Adventure',
    '战争': 'War',
    '动画': 'Animation',
    '纪录片': 'Documentary',
    '历史': 'History',
    '音乐': 'Music',
    '家庭': 'Family',
    '西部': 'Western',
    '电视电影': 'TV Movie',
    '动作冒险': 'Action Adventure',
    '真人秀': 'Reality TV',
    '儿童': 'Children',
    '肥皂剧': 'Soap Opera',
    '新闻': 'News',
    # English entries that already exist in the data (pass-through)
    'Sci-Fi & Fantasy': 'Sci-Fi & Fantasy',
    'War & Politics': 'War & Politics',
}

# Reverse mapping: English → Chinese (for lookup by English name)
# 反向映射：英文 → 中文
GENRE_EN_TO_CN = {v: k for k, v in GENRE_CN_TO_EN.items()}


def to_en(genre_cn: str) -> str:
    """Convert a Chinese genre name to English.
       将中文电影类型名称转换为英文。

    Args:
        genre_cn: Chinese genre string (e.g., '剧情', '喜剧')
                  中文类型字符串（如 '剧情', '喜剧'）

    Returns:
        English genre name, or the original string if not found in mapping.
        英文类型名称，如果未找到映射则返回原字符串。
    """
    return GENRE_CN_TO_EN.get(genre_cn, genre_cn)


def to_cn(genre_en: str) -> str:
    """Convert an English genre name back to Chinese.
       将英文电影类型名称转换回中文。

    Args:
        genre_en: English genre string (e.g., 'Drama', 'Comedy')
                  英文类型字符串（如 'Drama', 'Comedy'）

    Returns:
        Chinese genre name, or the original string if not found.
        中文类型名称，如果未找到映射则返回原字符串。
    """
    return GENRE_EN_TO_CN.get(genre_en, genre_en)


def translate_genre_set(genres: set) -> set:
    """Translate a set of genres (possibly containing mixed CN/EN) to English.
       将一组电影类型名称（可能混合中英文）全部转换为英文。

    Args:
        genres: Set of genre strings (e.g., {'剧情', '喜剧', 'Sci-Fi & Fantasy'})
                类型字符串集合

    Returns:
        Set of English genre strings.
        英文类型字符串集合
    """
    return {to_en(g) for g in genres}


def get_all_english_genres() -> list[str]:
    """Return sorted list of all known English genre names.
       返回所有已知英文类型名称的排序列表。"""
    return sorted(set(GENRE_CN_TO_EN.values()))
