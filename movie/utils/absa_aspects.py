# -*- coding: utf-8 -*-
"""
ABSA aspect detection utilities (Phase 1).
方面检测工具：升级版关键词词典 + 去噪（请求模式黑名单、POS 消歧、词边界、最长匹配、方面去重）。

Phase 1: candidate generation + de-noising (no NLI yet — NLI 在 Phase 2 引入)。
Phase 2 会复用 ASPECTS_V2 的 description/seed_examples 作为 NLI 假设。
"""

import re
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════
#  方面原型定义
# ═══════════════════════════════════════════════════════════════════════
# keywords: Phase 1 生效；description + seed_examples: Phase 2 供 NLI
# 跨方面歧义词（funny/scary/boring/like/best/role/scene/content）已清理：
#   - 双关归一方面（funny→emotion，不再归 genre）
#   - 单词歧义大的改为多词短语（like→similar to，best→one of the best）

ASPECTS_V2 = {
    'genre': {
        'keywords': [
            'type of movie', 'kind of movie',
            'genre', 'genres', 'comedy', 'horror', 'action', 'drama',
            'thriller', 'sci-fi', 'science fiction', 'fantasy', 'romance',
            'romantic', 'musical', 'animation', 'animated', 'documentary',
            'western', 'crime', 'mystery',
            'dark comedy', 'dark and gritty', 'lighthearted', 'light-hearted',
        ],
        'label_cn': '类型',
        'label_en': 'Genre/Style',
        'description': 'The text evaluates the genre, style, or type of a movie.',
        'seed_examples': [
            'It is a great dark comedy.',
            'The blend of sci-fi and horror works well.',
            'This is a slow-burn thriller.',
        ],
    },
    'plot': {
        'keywords': [
            'plot', 'story', 'storyline', 'narrative', 'twist', 'ending',
            'predictable', 'unpredictable', 'engaging',
            'confusing', 'complex plot', 'simple story', 'well-written',
            'screenplay', 'script', 'plotline',
            'storytelling', 'story telling',
        ],
        'label_cn': '剧情',
        'label_en': 'Plot/Story',
        'description': 'The text evaluates the plot, story, or screenplay of a movie.',
        'seed_examples': [
            'The plot was full of twists.',
            'A well-written screenplay with a satisfying ending.',
            'The story dragged in the second act.',
        ],
    },
    'cast': {
        'keywords': [
            'cast', 'actor', 'actress', 'performance', 'performances',
            'acting', 'ensemble cast', 'supporting cast',
            'voice acting', 'voice actor',
            'portrayal', 'played by', 'in the role of',
            'lead role', 'lead performance', 'title role',
            # 加回：过去时，"she starred in X" 是 cast 评价
            'starring', 'starred',
        ],
        'label_cn': '演员',
        'label_en': 'Cast/Acting',
        'description': 'The text evaluates acting performances, actors, or cast quality.',
        'seed_examples': [
            'The acting was phenomenal.',
            'Heath Ledger delivered a great performance.',
            'The ensemble cast had great chemistry.',
        ],
    },
    'visual': {
        'keywords': [
            'cinematography', 'cgi', 'special effects', 'visual effects', 'vfx',
            'art direction', 'production design',
            'visually stunning', 'beautifully shot', 'stunning visuals',
            'camera work', 'lighting', 'color palette', 'beautiful visuals',
            # 加回：复数形式，"the visuals were stunning" 是真视效评价
            'visuals',
        ],
        'label_cn': '视效',
        'label_en': 'Visual/Effects',
        'description': 'The text evaluates cinematography, visual effects, or visuals.',
        'seed_examples': [
            'The cinematography was breathtaking.',
            'The CGI looked dated.',
            'Beautifully shot with great lighting.',
        ],
    },
    'audio': {
        'keywords': [
            'soundtrack', 'score', 'sound design', 'sound effects',
            'background music', 'theme music', 'ost',
            'composer', 'dubbing',
            # 单词加回：电影评论里歧义低，请求句由黑名单兜底
            'music', 'song', 'songs', 'audio', 'dialogue',
        ],
        'label_cn': '音效',
        'label_en': 'Audio/Music',
        'description': 'The text evaluates soundtrack, score, or sound design.',
        'seed_examples': [
            'The soundtrack was memorable.',
            'Great score by Zimmer.',
            'The sound design was immersive.',
        ],
    },
    'direction': {
        'keywords': [
            'director', 'directing', 'directed by', 'direction',
            'filmmaker', 'auteur',
            'pacing', 'atmosphere', 'mood',
            'creative vision', 'artistic vision', 'visionary',
        ],
        'label_cn': '导演',
        'label_en': 'Direction',
        'description': 'The text evaluates directing, pacing, or directorial vision.',
        'seed_examples': [
            'The director nailed the pacing.',
            'Brilliant direction and atmosphere.',
            'The film suffers from slow pacing.',
        ],
    },
    'emotion': {
        'keywords': [
            'heartwarming', 'emotional', 'moving', 'touching', 'inspiring',
            'depressing', 'hilarious', 'frightening', 'creepy', 'chilling',
            'suspenseful', 'intense', 'thrilling',
            'feel-good', 'uplifting', 'powerful',
            'tearjerker', 'tear-jerker',
            # 双关词统一归 emotion（不再归 genre/plot）
            'funny', 'scary', 'sad', 'exciting',
            'boring', 'dull', 'enjoyable', 'entertaining',
        ],
        'label_cn': '情感',
        'label_en': 'Emotion/Tone',
        'description': 'The text evaluates the emotional impact or tone of a movie.',
        'seed_examples': [
            'A deeply moving and emotional film.',
            'Hilarious from start to finish.',
            'Boring and dull despite the hype.',
        ],
    },
    'recommendation': {
        'keywords': [
            'must-watch', 'must see', 'must-see',
            'worth watching', 'worth a watch',
            'highly recommend', 'highly recommended',
            'should watch', 'should see',
            'underrated', 'overrated', 'underappreciated',
            'masterpiece', 'classic',
            'favorite film', 'favourite film',
            'i recommend', 'i suggest', 'would recommend',
            'one of the best', 'best film', 'best movie', 'greatest film',
        ],
        'label_cn': '推荐',
        'label_en': 'Recommendation',
        'description': 'The text gives a recommendation or overall endorsement of a movie.',
        'seed_examples': [
            'A must-watch masterpiece.',
            'Highly recommended, one of the best films of the year.',
            'Underrated and worth watching.',
        ],
    },
    'comparison': {
        'keywords': [
            'reminds me of', 'remind you of',
            'same vibe', 'same feel',
            'similar to', 'alike',
            'if you like', 'fans of',
            'better than', 'worse than',
            'cross between', 'blend of', 'mix of',
        ],
        'label_cn': '比较',
        'label_en': 'Comparison',
        'description': 'The text compares one movie to another evaluatively.',
        'seed_examples': [
            "It reminds me of early Tarantino.",
            'A cross between Fargo and Heat.',
            'Better than the original in every way.',
        ],
    },
    'content': {
        'keywords': [
            'violent', 'gore', 'gory', 'blood', 'bloody',
            'mature content', 'adult content',
            'nudity', 'sexual content', 'explicit',
            'disturbing', 'graphic content',
            'family-friendly', 'kid-friendly', 'children-friendly',
            'age-appropriate', 'pg-rated', 'r-rated',
            'mature themes',
        ],
        'label_cn': '内容',
        'label_en': 'Content/Warnings',
        'description': 'The text evaluates content suitability, rating, or sensitive content.',
        'seed_examples': [
            'Very violent but not gratuitous.',
            'Family-friendly and appropriate for kids.',
            'Contains graphic content and nudity.',
        ],
    },
}

ASPECT_NAMES = list(ASPECTS_V2.keys())

# ═══════════════════════════════════════════════════════════════════════
#  词边界正则缓存 + 最长匹配排序
# ═══════════════════════════════════════════════════════════════════════

_KW_REGEX_CACHE: dict[str, re.Pattern] = {}


def _kw_pattern(kw: str) -> re.Pattern:
    """Compile word-boundary regex for a keyword (cached)."""
    if kw not in _KW_REGEX_CACHE:
        _KW_REGEX_CACHE[kw] = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
    return _KW_REGEX_CACHE[kw]


# 每个方面的关键词按长度降序（最长优先匹配）
_SORTED_KW: dict[str, list[str]] = {
    a: sorted(c['keywords'], key=len, reverse=True)
    for a, c in ASPECTS_V2.items()
}

# ═══════════════════════════════════════════════════════════════════════
#  请求模式黑名单（征询句式，非评价，整句丢弃）
# ═══════════════════════════════════════════════════════════════════════

_REQUEST_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'looking for .{0,40}?(movies?|shows?|films?|recommendations?)',
        r'(any|some)\s+(good|great|nice).{0,30}?\?',
        r'(best|worst|top|greatest)\s.{0,30}?\?',
        r'movies?\s+(like|similar to)\s',
        r'shows?\s+(like|similar to)\s',
        r'i\s+(want|need|am looking|would love|i.d like to see|am searching)',
        r'can you\s+(recommend|suggest)',
        r'(recommend|suggest).{0,20}?\?',
        r'^\s*(request|suggestion)\s*[:\[]',
        r'(looking|search|need|want)\s.{0,30}?(comedy|drama|horror|thriller|action)',
    ]
]


def is_request_sentence(sent: str) -> bool:
    """Return True if a sentence is a seeking/request pattern (not an evaluation).
       判断句子是否为征询/请求句式（非评价，应丢弃）。"""
    s = sent.strip()
    if not s:
        return True
    for pat in _REQUEST_PATTERNS:
        if pat.search(s):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════
#  POS 消歧（针对高频歧义词，nltk.pos_tag）
# ═══════════════════════════════════════════════════════════════════════

_POS_TAGGER = None


def _get_pos_tagger():
    """Lazy-load nltk.pos_tag. Returns False if unavailable."""
    global _POS_TAGGER
    if _POS_TAGGER is None:
        try:
            import nltk
            # 触发 punkt 加载（pos_tag 依赖）
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', quiet=True)
            _POS_TAGGER = nltk.pos_tag
        except Exception:
            _POS_TAGGER = False
    return _POS_TAGGER


# 只有这些单词歧义词需要 POS 消歧；其他关键词请求模式黑名单已过滤，
# 多词短语本身足够具体。短路避免对每个关键词匹配都调 nltk.pos_tag（性能）。
_AMBIGUOUS_KW_NEEDING_POS = {'like'}


def _is_request_like_usage(kw: str, sentence: str) -> bool:
    """POS-based disambiguation: True if keyword usage is request-like (drop).
       基于 POS 的消歧：若关键词用法是请求句式，则丢弃。
       保守规则，只处理高频歧义词：'like' 作介词(IN) → 丢弃。"""
    if kw.lower() not in _AMBIGUOUS_KW_NEEDING_POS:
        return False  # 非歧义关键词，无需 POS（请求模式黑名单已兜底）
    tagger = _get_pos_tagger()
    if not tagger:
        return False
    try:
        tokens = sentence.split()
        tagged = tagger(tokens)
    except Exception:
        return False
    kw_lower = kw.lower()
    for word, tag in tagged:
        if word.lower() != kw_lower:
            continue
        # like 作介词 → "movies like X" 征询句式，丢弃
        if kw_lower == 'like' and tag == 'IN':
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════
#  断句
# ═══════════════════════════════════════════════════════════════════════


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (handles . ! ? and newlines)."""
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


# ═══════════════════════════════════════════════════════════════════════
#  主检测函数
# ═══════════════════════════════════════════════════════════════════════


def detect_aspects_v2(text: str) -> list[dict]:
    """Detect aspect mentions in text with de-noising.
       带去噪的方面检测（Phase 1）。
    Returns:
        list of {aspect, snippet, keyword, confidence}
        Phase 1 confidence 固定 1.0；Phase 2 由 NLI entailment 提供
    """
    if not text or len(text.strip()) < 3:
        return []

    sentences = split_sentences(text)
    # 丢弃请求句（整句不提取方面）
    eval_sentences = [s for s in sentences if not is_request_sentence(s)]
    if not eval_sentences:
        return []

    detected: dict[str, dict] = {}  # aspect -> first match（方面去重：一方面一片段）

    for sent in eval_sentences:
        sent_lower = sent.lower()
        for aspect, kws in _SORTED_KW.items():
            if aspect in detected:
                continue
            for kw in kws:
                pat = _kw_pattern(kw)
                if pat.search(sent_lower):
                    if _is_request_like_usage(kw, sent):
                        continue
                    detected[aspect] = {
                        'aspect': aspect,
                        'snippet': sent,
                        'keyword': kw,
                        'confidence': 1.0,
                    }
                    break  # 该方面已命中，下一个方面
    return list(detected.values())
