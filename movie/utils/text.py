"""
Shared text processing utilities for the movie analysis pipeline.
文本处理工具：分词、去重、会话ID解析等。
"""

import re

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
    for pkg in ['wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)
    _HAS_NLTK = True
except ImportError:
    _lemmatizer = None
    _HAS_NLTK = False


from movie.config import log

# 停用词规范化缓存（避免每次调用都重新去撇号）
_stopword_cache = {}


def _get_normalized_stopwords(stopwords):
    """缓存停用词集的撇号规范化结果。"""
    if stopwords is None:
        return None
    key = id(stopwords)
    if key not in _stopword_cache:
        _stopword_cache[key] = {w.replace("'", "") for w in stopwords}
    return _stopword_cache[key]


def tokenize(text: str, min_len: int = 3, stopwords: set = None) -> list[str]:
    """Simple English tokenizer with POS-aware lemmatization.
    简单英文分词器（动词+名词双步词形还原）。
    Args:
        text:      input text to tokenize
        min_len:   minimum word length (default 3)
        stopwords: set of stopwords to filter out (default None = no filter)
    Returns:
        list of cleaned tokens
    """
    if not text:
        return []
    text = text.lower()
    tokens = re.split(r"[^a-z']+", text)
    # 统一去撇号：don't → dont，it's → its
    _stopwords = _get_normalized_stopwords(stopwords)
    result = []
    for t in tokens:
        t = t.strip("'").replace("'", "")   # 移除所有撇号
        if len(t) < min_len:
            continue
        if t.isnumeric():
            continue
        if _HAS_NLTK:
            t = _lemmatizer.lemmatize(t, pos='v')  # 先按动词还原：watched → watch
            t = _lemmatizer.lemmatize(t, pos='n')  # 再按名词还原：movies → movie
        if _stopwords and t in _stopwords:
            continue
        result.append(t)
    return result

def deduplicate_seekers(seekers: list[dict]) -> list[dict]:
    """Deduplicate seeker records by (session_id, text).
    按 (会话ID, 文本内容) 去重用户提问记录。

    规则9：在同一轮次会话中，用户提问相同时需要排重。
    不同会话中相同文本的提问不应被去除。
    """
    seen = set()
    deduped = []
    for r in seekers:
        text = r.get('proc_text', '')
        if not text:
            text = r.get('raw_text', '')
        sid = r.get('session_id', '')
        key = (sid, (text or '').strip().lower())
        if not key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    n_removed = len(seekers) - len(deduped)
    if n_removed > 0:
        log(f"  Deduplication: removed {n_removed} duplicates (left {len(deduped)})")
    return deduped


def parse_conv_turn(conv_id: str) -> tuple[str, str]:
    """Parse (session_id, turn_number) from conv_id.
    从 conv_id 中解析出 (会话ID, 轮次编号)。

    conv_id format: {session_id}_{current_turn}/{total_turns}
    Example: t3_rt7enj_1/14 -> ('t3_rt7enj', '1')
    """
    if '_' not in conv_id:
        return (conv_id, '')
    session_id = conv_id.rsplit('_', 1)[0]
    turn_part = conv_id.rsplit('_', 1)[1]
    turn_num = turn_part.split('/')[0] if '/' in turn_part else turn_part
    return (session_id, turn_num)


# ── 系统回复提取公共函数（规则8）─────────────────────────────────────

# IMDB ID 正则：匹配 tt + 7~9 位数字
_TT_PATTERN = re.compile(r'\b(tt\d{7,9})\b')


def build_conv_system(all_rows: list[dict]) -> dict[str, list[str]]:
    """Build conv_id -> list of system reply texts.

    构建系统回复映射表：按完整 conv_id 分组系统回复的 processed 文本。

    规则8：通过用户提问的 conv_id 获取同一轮次的系统回复内容。
    使用完整 conv_id 作为键，严格遵循"conv_id相同"的匹配规则。
    仅收集 is_seeker=False（系统回复）的行。

    Args:
        all_rows: 全部会话数据行（含用户提问和系统回复）
    Returns:
        dict[conv_id] -> list[str]（系统回复 processed 文本列表）
    """
    conv_system: dict[str, list[str]] = {}
    for row in all_rows:
        if row.get('is_seeker', False):
            continue  # 只处理系统回复
        processed = row.get('processed_raw', row.get('processed', ''))
        if not processed:
            continue
        conv_id = row.get('conv_id', '')
        if not conv_id:
            continue
        if conv_id not in conv_system:
            conv_system[conv_id] = []
        conv_system[conv_id].append(processed)
    return conv_system


def get_system_movie_ids(
    conv_id: str, conv_system: dict[str, list[str]],
) -> set[str]:
    """Extract movie IDs from system replies for a given conv_id.

    从指定 conv_id 对应的系统回复中提取电影 ID。

    规则8：从系统内容中提取电影 id（tt 格式）。
    使用完整 conv_id 精确匹配，避免不同总轮次的同轮次号误匹配。

    Args:
        conv_id:     用户提问的会话轮次 ID（如 't3_rt7fry_4/7'）
        conv_system: build_conv_system() 返回的映射表
    Returns:
        电影 ID 集合（如 {'tt1375666', 'tt0111161'}）
    """
    system_msgs = conv_system.get(conv_id, [])
    movie_ids = set()
    for msg in system_msgs:
        movie_ids.update(_TT_PATTERN.findall(str(msg)))
    return movie_ids
