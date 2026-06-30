# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文等非 ASCII 字符

"""
Data Loader Module
数据加载模块

Loads all data sources, parses fields, tags holiday/workday/weekend,
loads movie info and user profiles/age segments.
加载所有数据源，解析字段，标记节假日/工作日/周末，加载电影信息和用户画像/年龄段。

Can run standalone to verify data integrity.
可独立运行以验证数据完整性。

Output: tagged rows ready for downstream analysis steps.
输出：已标记的数据行，供下游分析步骤使用。
"""

import csv            # CSV 文件读写
import json           # JSON 文件读写
import os             # 操作系统接口，路径操作
import re             # 正则表达式，用于文本解析
import ast            # 抽象语法树，用于解析 Python 字面量
from collections import defaultdict, Counter  # 默认字典和计数器
from datetime import datetime, timezone, timedelta  # 日期时间处理

from movie.config import (                      # 从配置模块导入路径和常量
    DATA_DIR, FULL_YEAR_CSV, HOLIDAY_CSV, HOLIDAY_WORKDAY_CSV,
    MOVIE_INFO_PATH, ENTITY2ID_PATH, USER_AGE_SEG_PATH,
    MIN_DATA_ROWS, log,
)


# ═══════════════════════════════════════════════════════════════════════
#  1. Holiday definitions (with workday/weekend adjustment)
#  1. 节假日定义（含工作日/周末调休调整）
# ═══════════════════════════════════════════════════════════════════════

def load_holiday_definitions() -> dict[str, dict]:
    """
    Load holiday.csv, return dict: {date_str: {description, type}}.
    加载 holiday.csv，返回字典：{日期字符串: {描述, 类型}}。
    Returns:
        键为日期字符串（YYYY-MM-DD），值为包含 description 和 type 的字典
    """
    holidays = {}                                            # 临时字典，存储所有行
    with open(HOLIDAY_CSV, 'r', encoding='utf-8') as f:      # 打开节假日 CSV 文件
        reader = csv.DictReader(f)                            # 创建 CSV 字典读取器
        for row in reader:                                   # 遍历每一行
            d = row['date'].strip()                          # 获取日期字符串，去除空白
            holidays[d] = {                                  # 存入字典
                'description': row['description'].strip(),   # 节假日名称/描述
                'type': row['type'].strip(),                 # 节假日类型
            }
    # Merge duplicate dates（合并重复日期的节假日描述）
    merged = {}
    for d, info in holidays.items():
        if d in merged:                                      # 如果日期已存在
            merged[d]['description'] = f"{merged[d]['description']}&{info['description']}"
        else:                                                # 新日期
            merged[d] = dict(info)

    years = sorted(set(d[:4] for d in merged))               # 提取所有年份并排序
    log(f"Loaded {len(merged)} holiday dates across years: {', '.join(years)}")
    return merged


def load_holiday_workday_adjustments() -> dict[str, str]:
    """
    Load holiday-workday.csv for 补班/补休 adjustments.
    加载 holiday-workday.csv 中的补班/补休调休信息。

    Returns:
        dict[date_str] -> 'workday' (补班: weekend→workday)
                        or 'weekend' (补休: workday→rest day)
        返回字典：日期 -> 'workday'（补班，周末变工作日）或 'weekend'（补休，工作日变休息日）
    """
    adjustments = {}                                         # 调休调整字典
    if not os.path.exists(HOLIDAY_WORKDAY_CSV):              # 如果文件不存在
        log(f"holiday-workday.csv not found, skipping adjustments", "DataLoader")
        return adjustments                                   # 返回空字典

    with open(HOLIDAY_WORKDAY_CSV, 'r', encoding='utf-8') as f:  # 打开调休 CSV
        reader = csv.DictReader(f)
        for row in reader:
            d = row['date'].strip()                          # 日期字符串
            t = row['type'].strip()                          # 调休类型
            if 'substitute workday' in t or '补班' in t:     # 补班：周末改为工作日
                adjustments[d] = 'workday'
            elif 'substitute wenkend' in t or '补休' in t:   # 补休：工作日改为休息日
                adjustments[d] = 'weekend'
    log(f"Loaded {len(adjustments)} workday/weekend adjustments", "DataLoader")
    return adjustments


# ═══════════════════════════════════════════════════════════════════════
#  2. Conversation data loading & parsing
#  2. 会话数据加载与解析
# ═══════════════════════════════════════════════════════════════════════

def parse_processed_text(row: dict) -> str:
    """
    Extract USER text from 'raw' field.
    从 'raw' 字段中提取用户文本。
    Format（格式）: "['USER', 'text here']"
    Args:
        row: 数据行字典，包含 'raw' 键
    Returns:
        提取出的用户文本字符串，无内容则返回空字符串
    """
    raw = row.get('raw', '')                                 # 获取原始字段值
    if not raw:                                              # 空值直接返回
        return ''
    # 使用正则匹配 "['USER', 'text']" 格式，提取 text 部分
    m = re.search(r"\[\s*'USER'\s*,\s*'(.*)'\s*\]", raw, re.DOTALL)
    if m:
        return m.group(1)                                    # 返回匹配到的文本
    # 正则匹配失败时的兜底处理：手动去除前缀和后缀
    cleaned = re.sub(r"^\s*\[\s*'USER'\s*,\s*", '', raw)    # 去掉开头的 "['USER', "
    cleaned = re.sub(r"\s*\]\s*$", '', cleaned)              # 去掉结尾的 "]"
    return cleaned.strip().strip("'\"")                      # 去除空白和引号


def parse_processed_content(row: dict) -> str:
    """
    Extract text from 'processed' field (same format, may have tt IDs).
    从 'processed' 字段提取文本（格式相同，可能包含 tt 电影 ID）。
    Args:
        row: 数据行字典，包含 'processed' 键
    Returns:
        提取出的文本字符串
    """
    processed = row.get('processed', '')                     # 获取 processed 字段
    if not processed:                                        # 空值直接返回
        return ''
    # 匹配 "['USER', 'text']" 或 "['SYSTEM', 'text']" 格式
    m = re.search(r"\[\s*'(?:USER|SYSTEM)'\s*,\s*'(.*)'\s*\]", processed, re.DOTALL)
    if m:
        return m.group(1)                                    # 返回提取的文本
    # 兜底处理：手动去除前缀和后缀
    cleaned = re.sub(r"^\s*\[\s*'(?:USER|SYSTEM)'\s*,\s*", '', processed)
    cleaned = re.sub(r"\s*\]\s*$", '', cleaned)
    return cleaned.strip().strip("'\"")                      # 去除空白和引号


def extract_imdb_ids(text: str) -> list[str]:
    """Extract all tt... IDs from text.
       从文本中提取所有 tt... 格式的 IMDB ID。
    Args:
        text: 待搜索的文本
    Returns:
        IMDB ID 字符串列表（如 ["tt1375666", "tt0111161"]）
    """
    if not text:                                             # 空文本返回空列表
        return []
    # 匹配 tt 后跟 7-9 位数字的 IMDB 标准 ID 格式
    return re.findall(r'tt\d{7,9}', text)


def extract_session_base(conv_id: str) -> str:
    """
    Extract base session ID from conv_id.
    从 conv_id 中提取会话基础 ID。

    conv_id format（格式）: {session_id}_{current_turn}/{total_turns}
    e.g., 't3_rt7fry_4/7' -> session base = 't3_rt7fry'

    We rsplit on '_' once to extract the base.
    从右侧第一个下划线处分割，提取基础会话 ID。
    Args:
        conv_id: 会话 ID 字符串
    Returns:
        基础会话 ID
    """
    # 从右侧以 '_' 分割一次，取第一部分作为基础会话 ID
    base = conv_id.rsplit('_', 1)[0] if '_' in conv_id else conv_id
    return base


def load_conversations(filepath: str, max_rows: int = None) -> list[dict]:
    """
    Load a conversation CSV, parse raw/processed fields.
    加载会话 CSV，解析 raw/processed 字段。
    Args:
        filepath: CSV 文件路径
        max_rows: 最大加载行数（测试用，None 表示全部加载）
    Returns:
        已解析的字典列表，每个字典包含会话的一条记录
    """
    rows = []                                                # 存储解析后的数据行
    with open(filepath, 'r', encoding='utf-8-sig') as f:     # 使用 BOM 编码打开
        reader = csv.DictReader(f)                           # CSV 字典读取器
        for i, row in enumerate(reader):                     # 枚举行
            if max_rows and i >= max_rows:                   # 达到最大行数时停止
                break

            # Parse utc_time（解析 UTC 时间戳）
            utc_val = row.get('utc_time', '0')               # 获取时间戳字符串
            try:
                utc_time = int(float(utc_val))               # 转换为整数时间戳
            except (ValueError, TypeError):
                utc_time = 0                                 # 解析失败则设为 0

            # 构建解析后的数据字典
            parsed = {
                'conv_id': row.get('conv_id', '').strip(),           # 会话 ID
                'turn_id': row.get('turn_id', '').strip(),           # 轮次 ID
                'user_id': row.get('user_id', '').strip(),           # 用户 ID
                'turn_order': int(float(row.get('turn_order', 0))),  # 轮次顺序
                'is_seeker': row.get('is_seeker', '').strip().lower() == 'true',  # 是否是提问者
                'utc_time': utc_time,                                 # UTC 时间戳
                'upvotes': float(row.get('upvotes', 0) or 0),        # 点赞数
                'raw_text': parse_processed_text(row),               # 解析后的原始文本
                'proc_text': parse_processed_content(row),           # 解析后的处理文本
                'processed_raw': row.get('processed', ''),           # 原始 processed 字段（保留备用）
            }

            # Extract IMDB IDs from processed field（从 processed 字段提取 IMDB ID）
            parsed['imdb_ids'] = extract_imdb_ids(row.get('processed', ''))

            # Session base ID（提取会话基础 ID）
            parsed['session_id'] = extract_session_base(parsed['conv_id'])

            rows.append(parsed)                              # 添加到结果列表

    log(f"Loaded {len(rows)} rows from {filepath}", "DataLoader")
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  3. Holiday/Workday/Weekend tagging
#  3. 节假日/工作日/周末标记
# ═══════════════════════════════════════════════════════════════════════

def tag_period(rows: list[dict], holiday_map: dict,
               adjustments: dict[str, str] = None) -> list[dict]:
    """
    Add period classification to each row.
    为每一行添加时段分类标记。

    Tags added（添加的标记）:
      - date: YYYY-MM-DD string（日期字符串）
      - hour: hour of day (0-23)（小时）
      - weekday: 0=Mon, 6=Sun（星期几）
      - is_holiday: bool（是否节假日）
      - holiday_name: str（节假日名称，非节假日为空）
      - holiday_type: str（节假日类型，非节假日为空）
      - period: 'holiday' | 'workday' | 'weekend'（时段分类）

    Period logic（分类逻辑）:
      1. If date in holiday.csv -> holiday（在节假日列表中 -> 节假日）
      2. Else if date is 补班 -> workday（补班 -> 工作日）
      3. Else if date is 补休 -> weekend（补休 -> 周末）
      4. Else if weekday (Mon-Fri) -> workday（普通周一至周五 -> 工作日）
      5. Else (Sat-Sun) -> weekend（普通周六周日 -> 周末）
    Args:
        rows:       会话数据行列表
        holiday_map: 节假日定义字典
        adjustments: 调休调整字典
    Returns:
        添加了时段标记后的数据行列表
    """
    if adjustments is None:
        adjustments = {}

    holiday_date_set = set(holiday_map.keys())                # 节假日日期集合，用于 O(1) 查找
    tagged = []                                               # 存储标记后的行
    stats = {'holiday': 0, 'workday': 0, 'weekend': 0}       # 统计各时段数量

    for row in rows:                                         # 遍历每一行
        # 从 UTC 时间戳转换为日期时间对象（UTC 时区）
        dt = datetime.fromtimestamp(row['utc_time'], tz=timezone.utc)
        date_str = dt.strftime('%Y-%m-%d')                   # 格式化为 YYYY-MM-DD
        row['date'] = date_str                               # 添加日期字段
        row['hour'] = dt.hour                                # 添加小时字段
        row['weekday'] = dt.weekday()                        # 添加星期字段

        # Determine period（判断时段分类）
        if date_str in holiday_date_set:                     # 在节假日列表中的日期
            row['is_holiday'] = True
            row['holiday_name'] = holiday_map[date_str]['description']  # 节假日名称
            row['holiday_type'] = holiday_map[date_str]['type']         # 节假日类型
            row['period'] = 'holiday'                                      # 标记为节假日
            stats['holiday'] += 1
        elif date_str in adjustments:                        # 调休日期
            adj = adjustments[date_str]
            row['is_holiday'] = False
            row['holiday_name'] = ''
            row['holiday_type'] = ''
            row['period'] = adj   # 'workday' for 补班, 'weekend' for 补休
            stats[adj] += 1
        else:                                                # 普通日期
            row['is_holiday'] = False
            row['holiday_name'] = ''
            row['holiday_type'] = ''
            if dt.weekday() < 5:                             # 周一至周五（weekday 0-4）
                row['period'] = 'workday'                    # 工作日
                stats['workday'] += 1
            else:                                            # 周六周日（weekday 5-6）
                row['period'] = 'weekend'                    # 周末
                stats['weekend'] += 1

        tagged.append(row)

    log(f"Tagged: holiday={stats['holiday']}, workday={stats['workday']}, "
        f"weekend={stats['weekend']}", "DataLoader")
    return tagged


# ═══════════════════════════════════════════════════════════════════════
#  4. Movie info
#  4. 电影信息
# ═══════════════════════════════════════════════════════════════════════

def load_movie_info() -> dict:
    """Load movie_info.json into a dict keyed by IMDB ID.
       加载 movie_info.json，返回以 IMDB ID 为键的字典。
    Returns:
        字典 {IMDB_ID: {电影信息}}，文件不存在则返回空字典
    """
    if not os.path.exists(MOVIE_INFO_PATH):                  # 文件不存在检查
        log(f"movie_info.json not found at {MOVIE_INFO_PATH}", "DataLoader")
        return {}
    with open(MOVIE_INFO_PATH, 'r', encoding='utf-8') as f:  # 打开 JSON 文件
        data = json.load(f)                                  # 解析 JSON
    log(f"Loaded {len(data)} movies from movie_info.json", "DataLoader")
    return data


def lookup_genre_counts(imdb_ids: list[str], movie_info: dict) -> dict[str, int]:
    """Count genre occurrences for a list of IMDB IDs.
       统计一组 IMDB ID 对应的电影类型频次。
    Args:
        imdb_ids:  IMDB ID 列表
        movie_info: 电影信息字典（通过 load_movie_info 加载）
    Returns:
        字典 {电影类型: 出现次数}
    """
    counts = defaultdict(int)                                # 类型计数默认初始化为 0
    for tid in imdb_ids:                                    # 遍历每个 IMDB ID
        info = movie_info.get(tid)                           # 获取电影信息
        if info and 'genres' in info:                        # 如果信息存在且有 genres 字段
            for g in info['genres']:                         # 遍历该电影的类型列表
                counts[g] += 1                               # 类型计数 +1
    return dict(counts)


# ═══════════════════════════════════════════════════════════════════════
#  5. User age segments
#  5. 用户年龄段
# ═══════════════════════════════════════════════════════════════════════

def load_user_age_segments() -> dict[str, str]:
    """
    Load user age segments from totle_user_seg_v3.json.
    从 totle_user_seg_v3.json 加载用户年龄段信息。

    Returns:
        dict[user_id] -> age_segment (e.g., '18-25', '36-50', 'unknown')
        返回字典：用户 ID -> 年龄段
    """
    if not os.path.exists(USER_AGE_SEG_PATH):                # 文件不存在检查
        log(f"User age segment file not found: {USER_AGE_SEG_PATH}", "DataLoader")
        return {}
    with open(USER_AGE_SEG_PATH, 'r', encoding='utf-8') as f:  # 打开 JSON 文件
        data = json.load(f)                                  # 解析 JSON
    age_all = data.get('age_all', {})                        # 获取 age_all 子字典
    log(f"Loaded {len(age_all)} user age segments", "DataLoader")
    return age_all


# ═══════════════════════════════════════════════════════════════════════
#  6. Utility: get system replies for a given session
#  6. 工具函数：获取指定会话的系统回复
# ═══════════════════════════════════════════════════════════════════════

def get_system_replies_for_session(session_id: str,
                                   all_rows: list[dict]) -> list[dict]:
    """Get all system replies (is_seeker=False) for a given session.
       获取指定会话中所有系统回复行。
    Args:
        session_id: 会话基础 ID
        all_rows:   所有数据行
    Returns:
        该会话中 is_seeker=False 的行（即系统回复）列表
    """
    # 列表推导：筛选出同一会话且不是提问者的行
    return [r for r in all_rows
            if r['session_id'] == session_id and not r['is_seeker']]


# ═══════════════════════════════════════════════════════════════════════
#  7. Validation & summary
#  7. 验证与摘要
# ═══════════════════════════════════════════════════════════════════════

def validate_data(rows: list[dict]):
    """Print data validation summary.
       打印数据验证摘要信息。
    Args:
        rows: 已标记的完整数据行列表
    """
    total = len(rows)                                        # 总行数
    seekers = sum(1 for r in rows if r['is_seeker'])         # 提问者（用户）行数
    non_seekers = total - seekers                            # 非提问者（系统）行数
    unique_sessions = len(set(r['session_id'] for r in rows))  # 唯一会话数
    unique_users = len(set(r['user_id'] for r in rows if r['is_seeker']))  # 唯一用户数

    period_stats = Counter(r['period'] for r in rows)        # 各时段分布统计
    holiday_dates = sorted(set(r['date'] for r in rows if r['is_holiday']))  # 节假日日期列表

    log("=== Data Validation ===", "DataLoader")
    log(f"Total rows: {total}", "DataLoader")                # 总数据行数
    log(f"  User questions: {seekers}", "DataLoader")        # 用户提问数
    log(f"  System replies: {non_seekers}", "DataLoader")    # 系统回复数
    log(f"  Unique sessions: {unique_sessions}", "DataLoader")  # 唯一会话数
    log(f"  Unique users: {unique_users}", "DataLoader")     # 唯一用户数
    log(f"Period distribution: {dict(period_stats)}", "DataLoader")  # 时段分布

    dates = sorted(set(r['date'] for r in rows))             # 所有日期排序
    log(f"Date range: {dates[0]} ~ {dates[-1]} ({len(dates)} days)", "DataLoader")
    log(f"Holiday dates: {len(holiday_dates)}", "DataLoader")  # 节假日天数


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
#  主要入口点
# ═══════════════════════════════════════════════════════════════════════

def load_all(max_rows: int = None) -> dict:
    """
    Load and tag all data.
    加载并标记所有数据。

    Args:
        max_rows: 限制加载的行数（测试用，None 表示全部）

    Returns:
        dict with keys（返回字典，包含以下键）:
          - 'rows': tagged conversation rows（已标记的会话行）
          - 'seekers': filtered user questions（筛选出的用户提问行）
          - 'holiday_map': holiday definitions（节假日定义）
          - 'adjustments': workday/weekend adjustments（调休调整）
          - 'movie_info': movie info dict（电影信息）
          - 'user_ages': user age segment dict（用户年龄段）
    """
    log("=" * 50, "DataLoader")
    log("Loading all data...", "DataLoader")
    log("=" * 50, "DataLoader")

    # 1. Holiday definitions（加载节假日定义）
    holiday_map = load_holiday_definitions()

    # 2. Workday/weekend adjustments（加载调休调整）
    adjustments = load_holiday_workday_adjustments()

    # 3. Full year conversation data（加载全年会话数据并标记时段）
    rows = load_conversations(FULL_YEAR_CSV, max_rows=max_rows)
    rows = tag_period(rows, holiday_map, adjustments)
    rows = [r for r in rows if not r['date'].startswith('2018')]  # 过滤 2018 年数据，只保留 2019-2022
    seekers = [r for r in rows if r['is_seeker']]            # 筛选出用户提问行

    # 4. Movie info（加载电影信息）
    movie_info = load_movie_info()

    # 5. User age segments（加载用户年龄段）
    user_ages = load_user_age_segments()

    # 6. Validate（验证数据完整性）
    validate_data(rows)

    return {
        'rows': rows,
        'seekers': seekers,
        'holiday_map': holiday_map,
        'adjustments': adjustments,
        'movie_info': movie_info,
        'user_ages': user_ages,
    }


def main():
    """Standalone run for verification.
       独立运行模式，用于验证数据加载。"""
    data = load_all()
    log("Data loading complete!", "DataLoader")

    # Print top holiday names（打印出现频次最高的节假日名称）
    holiday_names = Counter()
    for r in data['seekers']:
        if r['is_holiday']:
            holiday_names[r['holiday_name']] += 1
    log(f"Unique holiday names: {len(holiday_names)}", "DataLoader")
    for name, cnt in holiday_names.most_common(20):          # 显示前20个
        log(f"  {name[:6]}: {cnt} questions", "DataLoader")


if __name__ == '__main__':
    main()  # 独立运行时执行 main 函数
