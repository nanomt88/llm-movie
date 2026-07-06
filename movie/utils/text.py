"""
Shared text processing utilities for the movie analysis pipeline.
文本处理工具：分词、去重、会话ID解析等。
"""

import re

from movie.config import log


def tokenize(text: str, min_len: int = 3, stopwords: set = None) -> list[str]:
    """Simple English tokenizer.
    简单英文分词器。
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
    result = []
    for t in tokens:
        t = t.strip("'")
        if len(t) < min_len:
            continue
        if t.isnumeric():
            continue
        if stopwords and t in stopwords:
            continue
        result.append(t)
    return result


def deduplicate_seekers(seekers: list[dict]) -> list[dict]:
    """Deduplicate seeker records by text content.
    去除文本内容重复的用户提问记录。
    Keeps the first occurrence of each unique text.
    """
    seen = set()
    deduped = []
    for r in seekers:
        text = r.get('proc_text', '')
        if not text:
            text = r.get('raw_text', '')
        key = (text or '').strip().lower()
        if not key or key in seen:
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
