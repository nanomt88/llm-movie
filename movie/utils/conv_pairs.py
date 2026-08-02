# -*- coding: utf-8 -*-
"""
Conversation regrouping + (system→user) pairing utilities.
会话重组 + (系统回复→用户回应) 配对工具。

规则：
  - 按 session_id 重组所有行（含系统回复）
  - 组内按 (turn_order, utc_time) 排序
  - 每个 system reply 配对其后紧随的 user message
  - 时段/节假日属性从该 session 首条 seeker 行继承
  - 跨日会话（首末 seeker 日期不同）标记 cross_day
"""

from collections import defaultdict
from typing import Optional

from movie.utils.text import parse_conv_turn
from movie.config import log


def regroup_sessions(rows: list[dict]) -> dict[str, list[dict]]:
    """Group all rows by session_id, sort by turn_order then utc_time.
       按 session_id 分组所有行，组内按 turn_order 排序。"""
    sessions: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        sid = r.get('session_id') or parse_conv_turn(r.get('conv_id', ''))[0]
        if sid:
            sessions[sid].append(r)
    for sid, turns in sessions.items():
        turns.sort(key=lambda t: (t.get('turn_order', 0), t.get('utc_time', 0)))
    return dict(sessions)


def _session_first_seeker(turns: list[dict]) -> Optional[dict]:
    """Return the first seeker row in a session (for period inheritance)."""
    for t in turns:
        if t.get('is_seeker'):
            return t
    return None


def _session_last_seeker(turns: list[dict]) -> Optional[dict]:
    """Return the last seeker row in a session (for cross_day detection)."""
    last = None
    for t in turns:
        if t.get('is_seeker'):
            last = t
    return last


def is_cross_day(turns: list[dict]) -> bool:
    """Check if a session spans multiple dates (rule 13).
       检查会话是否跨日（规则13）。"""
    first = _session_first_seeker(turns)
    last = _session_last_seeker(turns)
    if not first or not last:
        return False
    d1 = first.get('date', '')
    d2 = last.get('date', '')
    return bool(d1 and d2 and d1 != d2)


def emit_pairs(turns: list[dict]) -> list[dict]:
    """Emit (system_reply → next_user_message) pairs within a session.
       在会话内产出 (系统回复 → 下条用户消息) 配对。
       末条系统回复若无人回应，作为 solo 系统回复保留（user_text=''）。
    Returns:
        list of {session_id, pair_id, system_turn_order, user_turn_order,
                 system_text, user_text, is_solo_system, utc_time, cross_day,
                 date, period, is_holiday, holiday_name, holiday_type}
    """
    if not turns:
        return []
    sid = turns[0].get('session_id', '') or parse_conv_turn(turns[0].get('conv_id', ''))[0]
    first_seeker = _session_first_seeker(turns)
    cross_day = is_cross_day(turns)

    # 继承自首条 seeker 的时段属性
    inherit_keys = ['date', 'period', 'is_holiday', 'holiday_name', 'holiday_type']
    inherited = {k: (first_seeker.get(k) if first_seeker else '') for k in inherit_keys}

    pairs = []
    pair_idx = 0
    for i, turn in enumerate(turns):
        if turn.get('is_seeker'):
            continue  # 只从系统回复起配对
        if not turn.get('proc_text'):
            continue
        # 找紧随其后的 user message；中间若夹另一条系统回复则本条 solo
        next_user = None
        for j in range(i + 1, len(turns)):
            if turns[j].get('is_seeker'):
                next_user = turns[j]
                break
            if not turns[j].get('is_seeker') and turns[j].get('proc_text'):
                break
        user_text = next_user.get('proc_text', '') if next_user else ''
        pairs.append({
            'session_id': sid,
            'pair_id': f"{sid}_p{pair_idx}",
            'system_turn_order': turn.get('turn_order', 0),
            'user_turn_order': next_user.get('turn_order') if next_user else None,
            'system_text': turn.get('proc_text', ''),
            'user_text': user_text,
            'is_solo_system': next_user is None,
            'utc_time': turn.get('utc_time', 0),
            'cross_day': cross_day,
            **inherited,
        })
        pair_idx += 1
    return pairs


def build_pairs_from_rows(rows: list[dict]) -> list[dict]:
    """Top-level: regroup sessions, emit pairs, inherit period.
       顶层入口：重组会话 → 配对 → 继承时段。"""
    sessions = regroup_sessions(rows)
    all_pairs = []
    for sid, turns in sessions.items():
        all_pairs.extend(emit_pairs(turns))
    n_sessions = len(sessions)
    n_cross = sum(1 for t in sessions.values() if is_cross_day(t))
    log(f"  Sessions: {n_sessions} | Pairs: {len(all_pairs)} | Cross-day: {n_cross}", "ConvPairs")
    return all_pairs
