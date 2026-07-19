"""
fetch_user_profiles.py
======================
从 Reddit (via Pushshift) 获取用户公开数据并构建用户画像。

数据流：
  Phase 1A ── 扫描 data_all.csv → 提取目标用户的帖子关系映射
  Phase 1B ── Pushshift 按帖子 ID 查作者 → 建立 t2_... → 用户名映射
  Phase 2  ── Pushshift 按用户名采集全部帖子+评论 (≤2022-12-31)
  Phase 3  ── 离线分析 → 构建结构化用户画像

用法:
  python fetch_user_profiles.py                # 全流程
  python fetch_user_profiles.py --phase 1      # Phase 1A + 1B
  python fetch_user_profiles.py --phase 2      # Phase 2 only
  python fetch_user_profiles.py --phase 3      # Phase 3 only
  python fetch_user_profiles.py --resume       # 从检查点恢复
  python fetch_user_profiles.py --phase 2 --skip-existing  # 跳过已处理
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from math import floor

import concurrent.futures
import threading

import requests

# ── 修复 Windows 控制台日志乱码 ──
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif sys.stdout.encoding and sys.stdout.encoding.upper() != 'UTF-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ═══════════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CONV_DIR = os.path.join(DATA_DIR, 'conv')

# ── 输入文件 ──
USER_ID_LIST = os.path.join(CONV_DIR, 'all_user_id_list.txt')
DATA_ALL_CSV = os.path.join(CONV_DIR, 'data_all.csv')

# ── Phase 1 中间文件 ──
USER_POST_MAPPING = os.path.join(DATA_DIR, 'user_post_mapping.json')
USER_ID_TO_NAME = os.path.join(DATA_DIR, 'user_id_to_name.json')
USER_POST_NOT_FOUND = os.path.join(DATA_DIR, 'user_post_not_found.json')

# ── Phase 2 输出 ──
PUSHSHIFT_RAW = os.path.join(DATA_DIR, 'user_pushshift_raw.jsonl')  # JSONL 格式

# ── Phase 3 输出 ──
USER_PROFILES = os.path.join(DATA_DIR, 'user_profiles.json')
PUSHSHIFT_NOT_FOUND = os.path.join(DATA_DIR, 'user_pushshift_not_found.json')

# ── 检查点文件 ──
CP_SCAN = os.path.join(DATA_DIR, '_cp_scan_rows.txt')       # Phase 1A 已扫描行数
CP_POSTS = os.path.join(DATA_DIR, '_cp_resolved_posts.txt')  # Phase 1B 已解析帖子数
CP_USERS = os.path.join(DATA_DIR, '_cp_fetched_users.txt')   # Phase 2 已抓取用户数
CP_COMMENTS = os.path.join(DATA_DIR, '_cp_comment_posts.txt')  # Phase 1C 已查帖子数

# ── 全局限制 ──
CUTOFF_TIMESTAMP = 1672531199  # 2022-12-31 23:59:59 UTC
PUSHSHIFT_INTERVAL = 2.5       # 请求间隔（秒）：频率不影响配额，2.5s 折中吞吐与礼貌
PUSHSHIFT_TIMEOUT = 30         # 单次请求超时秒数
PUSHSHIFT_MAX_RETRIES = 2      # 最大重试次数（仅对网络错误有效，429 不重试）
PUSHSHIFT_PAGE_SIZE = 50       # 每页条目数
PUSHSHIFT_MAX_PAGES = 1        # 每类最大页数（50 条已足够画像分析）
PUSHSHIFT_CONCURRENT = False   # 是否启用并发模式（Phase 2 内动态切换）
CSV_CHECKPOINT_ROWS = 200000   # CSV 扫描每 N 行保存一次
POST_BATCH_SIZE = 50           # 查帖子时每次带 N 个 ID
USER_CHECKPOINT = 25           # Phase 2 每 N 用户保存一次

# ── Pushshift 保留字段（白名单） ──
SUBMISSION_FIELDS = {
    'id', 'author', 'author_fullname', 'author_created_utc', 'author_premium',
    'created_utc', 'retrieved_utc',
    'subreddit', 'subreddit_id', 'subreddit_subscribers',
    'title', 'selftext', 'url', 'permalink', 'domain',
    'score', 'upvote_ratio', 'num_comments',
    'over_18', 'spoiler', 'stickied', 'pinned',
    'link_flair_text', 'link_flair_css_class',
    'is_self', 'is_video', 'is_original_content',
    'thumbnail', 'distinguished', 'removed_by_category',
}

COMMENT_FIELDS = {
    'id', 'author', 'author_fullname', 'author_created_utc', 'author_premium',
    'created_utc', 'retrieved_utc',
    'subreddit', 'subreddit_id',
    'body', 'permalink', 'link_id', 'parent_id',
    'score', 'controversiality',
    'over_18', 'stickied', 'distinguished',
    'depth', 'top_comment',
}


# ═══════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def fmt_ts(ts):
    if ts:
        return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
    return 'N/A'


def get_system_proxy():
    """从 Windows 注册表读取系统代理。"""
    import platform
    if platform.system() != 'Windows':
        return None
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r'Software\Microsoft\Windows\CurrentVersion\Internet Settings'
        ) as key:
            enabled, _ = winreg.QueryValueEx(key, 'ProxyEnable')
            server, _ = winreg.QueryValueEx(key, 'ProxyServer')
        if enabled and server:
            server = server.strip()
            if not server.startswith('http://'):
                server = 'http://' + server
            return {'http': server, 'https': server}
    except Exception:
        pass
    return None


PROXIES = get_system_proxy()
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                  ' (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip',
    'Connection': 'close',  # 不复用连接（Pushshift/代理不支持 keep-alive）
}


def http_get(url, params, timeout):
    """封装 requests.get，保持配置一致但每次用新连接。"""
    return requests.get(url, headers=HEADERS, params=params,
                        proxies=PROXIES, timeout=timeout)

# ── Pushshift 限速器（线程安全） ──
class RateLimiter:
    def __init__(self, interval):
        self.interval = interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            elapsed = time.time() - self._last
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last = time.time()


ratelimit = RateLimiter(PUSHSHIFT_INTERVAL)
_last_429_time = 0.0           # 上次 429 的时间戳
_consecutive_429 = 0           # 连续 429 次数（跨调用）
_deep_cool_count = 0           # 深度冷却次数（递增等待）
_quota_req_count = 0           # 当前配额周期内已发起请求数
_QUOTA_LIMIT = 280             # 预判上限：无 429 则持续大步增加
_QUOTA_LIMIT_STEP = 30         # 大步进快速逼近真实限流阈值
_had_429_in_cycle = False      # 本轮是否触发过 429（阻止周期末增额）
_QUOTA_PAUSE_MIN = 600          # 配额暂停最短时间（10 分钟）
_QUOTA_PAUSE_MAX = 3600         # 配额暂停最长时间（60 分钟）
_rate_limit_lock = threading.Lock()  # 保护全局 429 状态


def pushshift_get(endpoint, params, max_retries=None, retry_delay=3):
    """调用 Pushshift API，返回 JSON 或 None。

    限流策略（基于观察：429 来自配额制，与请求频率无关）：
      - 固定间隔 PUSHSHIFT_INTERVAL（不搞自适应，因为频率无影响）
      - 429 不重试（省配额），直接深度冷却 60+ 分钟
      - 请求计数，达到预判上限时主动暂停，彻底避免 429
      - 配额重置窗口约 60 分钟（观测：30/45 分钟冷却后仍 429）
    """
    global _last_429_time, _consecutive_429, _deep_cool_count, PUSHSHIFT_CONCURRENT
    global _quota_req_count, _QUOTA_LIMIT, _had_429_in_cycle
    if max_retries is None:
        max_retries = PUSHSHIFT_MAX_RETRIES
    url = f'https://api.pullpush.io/reddit/{endpoint}'

    with _rate_limit_lock:
        # ── 预判性暂停：配额接近上限时主动休息 ──
        if _quota_req_count >= _QUOTA_LIMIT:
            pause_secs = random.randint(_QUOTA_PAUSE_MIN, _QUOTA_PAUSE_MAX)
            log(f'  配额使用 {_quota_req_count}/{_QUOTA_LIMIT}，主动暂停 {pause_secs // 60} 分钟...')
            _quota_req_count = 0
            if PUSHSHIFT_CONCURRENT:
                PUSHSHIFT_CONCURRENT = False
                log('  配额暂停 → 退化为串行模式')
            time.sleep(pause_secs)
            # ── 自适应提升配额：本轮无 429 则大步增加 ──
            if not _had_429_in_cycle:
                old_limit = _QUOTA_LIMIT
                _QUOTA_LIMIT += _QUOTA_LIMIT_STEP
                log(f'  本轮无 429，大步提升配额: {old_limit} → {_QUOTA_LIMIT}')
            else:
                log(f'  本轮有 429（上次冷却已处理），维持配额 {_QUOTA_LIMIT}')
                _had_429_in_cycle = False
            log('  配额暂停结束，继续采集')

        # ── 检测 429 → 深度冷却（不降配额，标记本轮有 429） ──
        if _consecutive_429 >= 1:
            _had_429_in_cycle = True
            _deep_cool_count += 1
            wait = random.randint(600, 3600)
            log(f'  检测到 429（冷却 #{_deep_cool_count}），深度冷却 {wait // 60} 分钟... '
                f'维持限额 {_QUOTA_LIMIT}')
            if PUSHSHIFT_CONCURRENT:
                PUSHSHIFT_CONCURRENT = False
                log('  429 触发 → 退化为串行模式')
            _quota_req_count = 0  # 配额重置
            _consecutive_429 = 0
            _last_429_time = 0
            time.sleep(wait)
            log('  深度冷却结束，继续采集')

    for attempt in range(1, max_retries + 1):
        ratelimit.wait()
        with _rate_limit_lock:
            _quota_req_count += 1
        try:
            resp = http_get(url, params, PUSHSHIFT_TIMEOUT)
            if resp.status_code == 200:
                with _rate_limit_lock:
                    _consecutive_429 = 0
                    _deep_cool_count = 0
                return resp.json()
            elif resp.status_code == 429:
                with _rate_limit_lock:
                    _consecutive_429 += 1
                    _last_429_time = time.time()
                # 429 不重试：直接跳到冷却（避免浪费配额）
                log(f'  429 限速(#{_consecutive_429})，放弃重试，等待冷却...')
                break
            else:
                log(f'  API 返回 {resp.status_code} (attempt {attempt}/{max_retries})')
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
        except requests.RequestException as e:
            log(f'  网络错误: {e} (attempt {attempt}/{max_retries})')
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
    return None


def read_list(filepath):
    """读取纯文本列表（每行一个 ID）。"""
    if not os.path.exists(filepath):
        log(f'错误: 文件不存在 {filepath}')
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def save_json(data, path):
    """写入 JSON，原子替换。"""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path, default=None):
    """读取 JSON，失败返回 default。"""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            pass
    return default if default is not None else {}


def save_jsonl_append(entries, path):
    """追加写入 JSONL（每行一个 JSON 对象）。"""
    with open(path, 'a', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def read_jsonl_keys(path):
    """读取 JSONL 文件，返回所有行的 key（如 t2_xxx）集合。"""
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if 'fullname' in obj:
                        keys.add(obj['fullname'])
                except json.JSONDecodeError:
                    pass
    return keys


def read_checkpoint(path):
    """读取检查点数值。"""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return int(f.read().strip())
        except:
            pass
    return 0


def write_checkpoint(path, value):
    """写入检查点数值。"""
    with open(path, 'w') as f:
        f.write(str(value) + '\n')


def clean_pushshift_data(raw_list, allowed_fields):
    """过滤 Pushshift 响应，只保留白名单字段。"""
    cleaned = []
    for item in raw_list:
        if isinstance(item, dict):
            cleaned.append({k: item[k] for k in allowed_fields if k in item})
    return cleaned


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1A — 扫描 CSV 提取用户-帖子映射
# ═══════════════════════════════════════════════════════════════════════

def phase1a_scan_csv(target_users):
    """
    扫描 data_all.csv，提取目标用户的 conv_id → t3_post_id 映射。
    支持检查点续扫。
    """
    log('=' * 55)
    log('Phase 1A: 扫描 CSV 提取用户帖子关系')
    log(f'  目标用户: {len(target_users)}')
    log(f'  源文件:   {DATA_ALL_CSV}')

    # 读取已有映射（用于 resume）
    existing = {}
    if os.path.exists(USER_POST_MAPPING):
        existing = load_json(USER_POST_MAPPING)
        # 移除 _meta 等非用户键
        existing = {k: v for k, v in existing.items()
                    if k.startswith('t2_')}

    start_row = read_checkpoint(CP_SCAN)
    if start_row > 0:
        log(f'  从第 {start_row} 行继续扫描（已有 {len(existing)} 个用户映射）')

    user_posts = defaultdict(set)
    # 加载已有数据
    for uid, posts in existing.items():
        user_posts[uid] = set(posts)

    rows_processed = start_row
    total_target = len(target_users)
    found_users = len(user_posts)

    with open(DATA_ALL_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < start_row:
                continue

            uid = row['user_id'].strip()
            if uid not in target_users:
                continue

            cid = row['conv_id'].strip()
            # conv_id 格式: t3_mdd5qq_0/7 → 提取 t3_mdd5qq
            m = re.match(r'(t3_[a-z0-9]+)', cid)
            if m:
                user_posts[uid].add(m.group(1))

            rows_processed = i + 1

            # 定期保存检查点
            if rows_processed % CSV_CHECKPOINT_ROWS == 0:
                # 转为普通 dict + list 存储
                out = {uid: sorted(list(posts))
                       for uid, posts in user_posts.items()}
                out['_meta'] = {'scan_progress': rows_processed}
                save_json(out, USER_POST_MAPPING)
                write_checkpoint(CP_SCAN, rows_processed)
                found = len([u for u in user_posts if user_posts[u]])
                log(f'  扫描 {rows_processed}/{1669720} 行, '
                    f'找到 {found}/{total_target} 用户')

    # 最终保存
    out = {uid: sorted(list(posts))
           for uid, posts in user_posts.items()}
    out['_meta'] = {'scan_progress': rows_processed,
                    'users_with_posts': len([u for u in user_posts if user_posts[u]]),
                    'total_users_found': len(user_posts)}
    save_json(out, USER_POST_MAPPING)
    # 清理检查点
    if os.path.exists(CP_SCAN):
        os.remove(CP_SCAN)

    with_posts = len([u for u in user_posts if user_posts[u]])
    total_posts = len(set().union(*user_posts.values())) if user_posts else 0
    log(f'  Phase 1A 完成: {with_posts}/{total_target} 用户有帖子, '
        f'共 {total_posts} 个不重复帖子')


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1B — Pushshift 查帖子 → 建立 t2_... → 用户名映射
# ═══════════════════════════════════════════════════════════════════════

def phase1b_resolve_usernames():
    """
    从 user_post_mapping.json 建立反向索引（帖子→用户列表），
    按用户覆盖率降序排列帖子，分批查询 Pushshift。
    优先查覆盖用户最多的帖子，以最小 API 调用量覆盖最多用户。
    """
    log('=' * 55)
    log('Phase 1B: 通过 Pushshift 解析帖子作者')

    mapping = load_json(USER_POST_MAPPING)
    if not mapping or '_meta' not in mapping:
        log('错误: 请先运行 Phase 1A')
        return

    # 加载已有映射（用于 resume）
    name_map = load_json(USER_ID_TO_NAME, {})
    if not isinstance(name_map, dict):
        name_map = {}
    already_resolved = {k for k in name_map if k.startswith('t2_')}
    log(f'  现有: {len(already_resolved)} 个用户已解析')

    # 需要解析的用户 = 所有 t2_ 用户 - 已解析的
    all_users = sorted({k for k in mapping if k.startswith('t2_')})
    pending_users = [u for u in all_users if u not in already_resolved]
    log(f'  待解析: {len(pending_users)} 个用户')

    if not pending_users:
        log('  所有用户已解析完毕')
        return

    # 建立反向索引：post_id → [覆盖的用户列表]（仅看还没解析的用户）
    post_to_users = defaultdict(list)
    for uid in pending_users:
        posts = mapping.get(uid, [])
        for pid in posts:
            # 去掉 t3_ 前缀
            base = pid.replace('t3_', '') if pid.startswith('t3_') else pid
            if base:
                post_to_users[base].append(uid)

    # 按覆盖用户数降序排列帖子
    ranked_posts = sorted(post_to_users.items(),
                          key=lambda x: len(x[1]),
                          reverse=True)

    log(f'  不重复帖子: {len(ranked_posts)} (覆盖 {len(pending_users)} 用户)')

    not_found_bases = set()
    api_calls = 0
    newly_resolved = 0
    recently_new = 0  # 最近一批新增用户数

    for batch_start in range(0, len(ranked_posts), POST_BATCH_SIZE):
        batch = ranked_posts[batch_start:batch_start + POST_BATCH_SIZE]
        batch_bases = [b for b, _ in batch]

        # 跳过已经很明显的未找到帖子（之前查过但不属于已找到用户的帖子）
        batch_bases = [b for b in batch_bases if b not in not_found_bases]
        if not batch_bases:
            continue

        ids_param = ','.join(batch_bases)
        result = pushshift_get('submission/search',
                               {'ids': ids_param, 'size': POST_BATCH_SIZE})

        api_calls += 1
        covered_users = set()

        if result and 'data' in result:
            for item in result['data']:
                an = item.get('author_fullname', '')
                author = item.get('author', '')
                if an and author and an.startswith('t2_'):
                    if an not in name_map:
                        name_map[an] = author
                        newly_resolved += 1
                    # 这个帖子找到了 -> 标记其覆盖的用户为已覆盖
                    pid_base = item.get('id', '')
                    if pid_base in post_to_users:
                        covered_users.update(post_to_users[pid_base])
            # 记录未在结果中返回的帖子（没查到）
            returned_ids = {item.get('id', '') for item in result['data']}
            missing = [b for b in batch_bases if b not in returned_ids]
            for b in missing:
                not_found_bases.add(b)

        # 即使 API 返回 None（网络错误），也继续下一批
        recently_new += newly_resolved

        # 定期保存 + 报告进度
        if api_calls % 10 == 0 or newly_resolved > 0:
            resolved_count = len(name_map)
            pct = round(resolved_count / len(all_users) * 100, 1) if all_users else 0
            log(f'  查询 {api_calls} 次, 已解析 {resolved_count}/{len(all_users)} '
                f'({pct}%), 未找到帖子 {len(not_found_bases)}')
            save_json(name_map, USER_ID_TO_NAME)
            recently_new = 0
            write_checkpoint(CP_POSTS, api_calls)

        # 所有用户已全部找到 → 提前退出
        if len(name_map) >= len(all_users):
            log(f'  所有用户已解析完毕！')
            break

    # 最终保存
    save_json(name_map, USER_ID_TO_NAME)
    if os.path.exists(CP_POSTS):
        os.remove(CP_POSTS)

    # 统计
    total_users = len(all_users)
    resolved_users = len([k for k in name_map if k.startswith('t2_')])
    unresolved = total_users - resolved_users
    log(f'  Phase 1B 完成: {resolved_users}/{total_users} 用户已解析'
        f'{" (" + str(unresolved) + " 个未找到)" if unresolved else ""}')
    log(f'  API 调用: {api_calls} 次')


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1C — Pushshift 查评论区 → 解析评论者用户名
# ═══════════════════════════════════════════════════════════════════════

def phase1c_resolve_commenters():
    """
    对于 Phase 1B 无法解析的用户（非 OP 的评论者），通过查询 Pushshift
    comment/search 按 post_id 拉取评论区数据，从 comments 中匹配 author_fullname。
    
    策略：按帖子覆盖的未解析用户数降序排列，优先查覆盖最广的帖子。
    """
    log('=' * 55)
    log('Phase 1C: 通过 Pushshift 评论数据解析评论者用户名')

    mapping = load_json(USER_POST_MAPPING)
    if not mapping or '_meta' not in mapping:
        log('错误: 请先运行 Phase 1A')
        return

    name_map = load_json(USER_ID_TO_NAME, {})
    if not isinstance(name_map, dict):
        name_map = {}
    already_resolved = {k for k in name_map if k.startswith('t2_')}
    log(f'  现有: {len(already_resolved)} 个用户已解析')

    all_users = sorted({k for k in mapping if k.startswith('t2_')})
    pending_users = [u for u in all_users if u not in already_resolved]
    log(f'  待解析: {len(pending_users)} 个用户')

    if not pending_users:
        log('  所有用户已解析完毕')
        return

    # 建立反向索引：post_id → [覆盖的未解析用户列表]
    post_to_users = defaultdict(list)
    for uid in pending_users:
        posts = mapping.get(uid, [])
        for pid in posts:
            base = pid.replace('t3_', '') if pid.startswith('t3_') else pid
            post_to_users[base].append(uid)

    # 按覆盖用户数降序排列帖子
    ranked_posts = sorted(post_to_users.items(),
                          key=lambda x: len(x[1]),
                          reverse=True)

    log(f'  不重复帖子: {len(ranked_posts)}, 覆盖 {len(pending_users)} 未解析用户')

    api_calls = 0
    newly_found = 0
    total_new = 0

    # 检查点恢复
    start_idx = read_checkpoint(CP_COMMENTS)
    if start_idx > 0:
        log(f'  从帖子索引 {start_idx} 继续')

    for batch_start in range(start_idx, len(ranked_posts), 1):  # 一次查一个帖子（评论区）
        post_id, covered_users = ranked_posts[batch_start]

        # 这个帖子的用户全都已解析？跳过
        remaining = [u for u in covered_users if u not in name_map]
        if not remaining:
            continue

        # 查评论区（支持分页，最多查 5 页 = 2500 条评论）
        found_in_this_post = 0
        all_comments = []
        after = None
        for page in range(5):
            params = {'link_id': f't3_{post_id}', 'size': 500}
            if after:
                params['after'] = after
            result = pushshift_get('comment/search', params)
            api_calls += 1

            if not result or 'data' not in result:
                break

            comments = result['data']
            if not comments:
                break

            all_comments.extend(comments)

            for comment in comments:
                an = comment.get('author_fullname', '')
                author = comment.get('author', '')
                if an and author and an.startswith('t2_') and an not in name_map:
                    name_map[an] = author
                    found_in_this_post += 1
                    newly_found += 1
                    total_new += 1

            # 下一页：用最后一个 comment 的 created_utc 作为游标
            last_ts = comments[-1].get('created_utc', 0)
            if last_ts:
                after = last_ts - 1
            if len(comments) < 500:
                break  # 没有更多了

        if found_in_this_post > 0:
            log(f'  帖子 {post_id}: 找到 {found_in_this_post} 个新用户 ({len(all_comments)} 条评论扫描)')

        # 每查一个帖子就更新检查点
        write_checkpoint(CP_COMMENTS, batch_start + 1)

        # 定期保存
        if api_calls % 5 == 0:
            resolved_count = len([k for k in name_map if k.startswith('t2_')])
            pct = round(resolved_count / len(all_users) * 100, 1)
            log(f'  查询 {api_calls} 次, 已解析 {resolved_count}/{len(all_users)} '
                f'({pct}%), 本轮新增 {total_new}')
            save_json(name_map, USER_ID_TO_NAME)

        # 所有人已解析？
        if len([k for k in name_map if k.startswith('t2_')]) >= len(all_users):
            log(f'  所有用户已解析完毕！')
            break

    # 最终保存
    save_json(name_map, USER_ID_TO_NAME)
    if os.path.exists(CP_COMMENTS):
        os.remove(CP_COMMENTS)
    resolved_count = len([k for k in name_map if k.startswith('t2_')])
    log(f'  Phase 1C 完成: {resolved_count}/{len(all_users)} 用户已解析')
    log(f'  API 调用: {api_calls} 次, 本轮新增: {total_new}')


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2 — Pushshift 按用户名采集数据
# ═══════════════════════════════════════════════════════════════════════

def phase2_fetch_user_data(skip_existing=False):
    """
    按用户名从 Pushshift 采集 submissions + comments（≤ 2022-12-31）。
    结果追加写入 JSONL 文件。
    """
    log('=' * 55)
    log('Phase 2: 按用户名采集 Pushshift 数据')
    log(f'  截止时间: {fmt_ts(CUTOFF_TIMESTAMP)}')

    name_map = load_json(USER_ID_TO_NAME)
    if not name_map:
        log('错误: 请先运行 Phase 1B')
        return

    # 收集所有 t2_ → username 映射
    user_list = [(k, v) for k, v in name_map.items()
                 if k.startswith('t2_') and isinstance(v, str) and v]
    log(f'  待抓取用户: {len(user_list)}')

    # 已抓取的用户（从 JSONL 读取 key）
    done_keys = set()
    if skip_existing and os.path.exists(PUSHSHIFT_RAW):
        done_keys = read_jsonl_keys(PUSHSHIFT_RAW)
        log(f'  已存在 {len(done_keys)} 个用户数据，跳过')

    # 检查点
    start_idx = read_checkpoint(CP_USERS)
    log(f'  从 index {start_idx} 继续')

    to_fetch = []
    for i, (fullname, username) in enumerate(user_list):
        if i < start_idx:
            continue
        if fullname in done_keys:
            continue
        to_fetch.append((fullname, username))

    log(f'  实际需要抓取: {len(to_fetch)} 个用户')

    if not to_fetch:
        log('  所有用户已抓取完毕')
        return

    # 以并发模式启动；遇到 429 会自动切换回串行
    global PUSHSHIFT_CONCURRENT
    PUSHSHIFT_CONCURRENT = True
    log('  并发模式: 同时拉取 submissions + comments（外层 2 用户并行）')

    # 单用户处理函数：每个用户每种数据只请求一次，没有分页循环
    def _process_single_user(fullname, username, abs_idx, total):
        """处理单个用户，返回 (submissions, comments)"""
        log(f'  [{abs_idx}/{total}] {username} ({fullname})')

        def _fetch_subs(uname):
            """拉取用户发帖（一次请求，有就有，没有就跳过）"""
            params = {
                'author': uname,
                'size': PUSHSHIFT_PAGE_SIZE,
                'sort': 'desc',
                'order': 'created_utc',
                'before': CUTOFF_TIMESTAMP,
            }
            result = pushshift_get('submission/search', params)
            if not result or 'data' not in result:
                return []
            batch = result['data']
            if not batch:
                return []
            # 客户端过滤（第二道保险：确保没有漏网之鱼）
            def _ts_ok(item):
                ts = item.get('created_utc')
                if isinstance(ts, (int, float)):
                    return ts <= CUTOFF_TIMESTAMP
                if isinstance(ts, str):
                    try:
                        return int(float(ts)) <= CUTOFF_TIMESTAMP
                    except (ValueError, TypeError):
                        return True
                return True  # 无时间戳 → 保留
            batch = [i for i in batch if _ts_ok(i)]
            if not batch:
                return []
            return clean_pushshift_data(batch, SUBMISSION_FIELDS)

        def _fetch_coms(uname):
            """拉取用户评论（一次请求，有就有，没有就跳过）"""
            params = {
                'author': uname,
                'size': PUSHSHIFT_PAGE_SIZE,
                'sort': 'desc',
                'order': 'created_utc',
                'before': CUTOFF_TIMESTAMP,
            }
            result = pushshift_get('comment/search', params)
            if not result or 'data' not in result:
                return []
            batch = result['data']
            if not batch:
                return []
            # 客户端过滤（第二道保险：确保没有漏网之鱼）
            def _ts_ok(item):
                ts = item.get('created_utc')
                if isinstance(ts, (int, float)):
                    return ts <= CUTOFF_TIMESTAMP
                if isinstance(ts, str):
                    try:
                        return int(float(ts)) <= CUTOFF_TIMESTAMP
                    except (ValueError, TypeError):
                        return True
                return True  # 无时间戳 → 保留
            batch = [i for i in batch if _ts_ok(i)]
            if not batch:
                return []
            return clean_pushshift_data(batch, COMMENT_FIELDS)

        if PUSHSHIFT_CONCURRENT:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                sf = executor.submit(_fetch_subs, username)
                cf = executor.submit(_fetch_coms, username)
                return sf.result(), cf.result()
        else:
            return _fetch_subs(username), _fetch_coms(username)

    stats = {'ok': 0, 'not_found': 0, 'error': 0}
    not_found_users = load_json(PUSHSHIFT_NOT_FOUND, {})

    # 外层线程池：同时处理 2 个用户（"增加一倍并发"）
    outer_workers = 2
    # 持久化 JSONL 文件句柄（避免每用户 open/close 开销）
    with open(PUSHSHIFT_RAW, 'a', encoding='utf-8') as jsonl_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=outer_workers) as outer_exec:
            futures_ordered = []
            for idx, (fullname, username) in enumerate(to_fetch):
                abs_idx = start_idx + idx + 1
                fut = outer_exec.submit(_process_single_user, fullname, username, abs_idx, len(user_list))
                futures_ordered.append((fut, fullname, username, abs_idx))

            # 按原始顺序收集结果（保存逻辑在主线程串行执行，避免文件 I/O 竞争）
            for fut, fullname, username, abs_idx in futures_ordered:
                try:
                    submissions, comments = fut.result()
                except Exception as e:
                    log(f'    Error: {e}')
                    traceback.print_exc()
                    stats['error'] += 1
                    write_checkpoint(CP_USERS, abs_idx)
                    if abs_idx % USER_CHECKPOINT == 0:
                        save_json(not_found_users, PUSHSHIFT_NOT_FOUND)
                        log(f'  [检查点: {abs_idx}/{len(user_list)}]')
                    continue

                # ── 保存 ──
                if submissions or comments:
                    entry = {
                        'fullname': fullname,
                        'username': username,
                        'total_submissions': len(submissions),
                        'total_comments': len(comments),
                        'submissions': submissions,
                        'comments': comments,
                        'fetched_at': time.time(),
                    }
                    jsonl_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    jsonl_f.flush()
                    stats['ok'] += 1
                    subs = len(submissions)
                    comms = len(comments)
                    log(f'    OK: {subs} submissions, {comms} comments')
                else:
                    not_found_users[fullname] = username
                    stats['not_found'] += 1
                    log(f'    无数据')

                # 检查点
                write_checkpoint(CP_USERS, abs_idx)
                if abs_idx % USER_CHECKPOINT == 0:
                    save_json(not_found_users, PUSHSHIFT_NOT_FOUND)
                    log(f'  [检查点: {abs_idx}/{len(user_list)}]')

    # 最终保存
    save_json(not_found_users, PUSHSHIFT_NOT_FOUND)
    if os.path.exists(CP_USERS):
        os.remove(CP_USERS)

    ok = stats['ok']
    nf = stats['not_found']
    err = stats['error']
    total = ok + nf + err
    log(f'  Phase 2 完成: OK={ok}, 无数据={nf}, 错误={err}, 总计={total}')


# ═══════════════════════════════════════════════════════════════════════
#  Phase 3 — 构建用户画像
# ═══════════════════════════════════════════════════════════════════════

def build_profile_from_raw(raw_entry):
    """从单个用户的原始 Pushshift 数据构建结构化画像。"""
    username = raw_entry.get('username', '')
    fullname = raw_entry.get('fullname', '')
    submissions = raw_entry.get('submissions', [])
    comments = raw_entry.get('comments', [])

    profile = {
        'fullname': fullname,
        'username': username,
    }

    # ── 账号信息 ──
    account_created = None
    for s in submissions:
        ac = s.get('author_created_utc')
        if ac:
            account_created = ac
            break
    if not account_created and comments:
        account_created = comments[0].get('author_created_utc')
    profile['account_created'] = account_created
    profile['account_created_date'] = fmt_ts(account_created)
    profile['is_premium'] = False
    for s in submissions:
        if s.get('author_premium'):
            profile['is_premium'] = True
            break

    # ── 数量统计 ──
    profile['total_submissions'] = len(submissions)
    profile['total_comments'] = len(comments)
    profile['total_activity'] = len(submissions) + len(comments)

    # ── Subreddit 活动分布 ──
    sub_stats = defaultdict(lambda: {'posts': 0, 'comments': 0, 'karma': 0, 'last_active': 0})
    for s in submissions:
        sub = s.get('subreddit', 'unknown')
        sub_stats[sub]['posts'] += 1
        sub_stats[sub]['karma'] += s.get('score', 0) or 0
        ts = s.get('created_utc', 0)
        if isinstance(ts, (int, float)):
            sub_stats[sub]['last_active'] = max(sub_stats[sub]['last_active'], ts)
    for c in comments:
        sub = c.get('subreddit', 'unknown')
        sub_stats[sub]['comments'] += 1
        sub_stats[sub]['karma'] += c.get('score', 0) or 0
        ts = c.get('created_utc', 0)
        if isinstance(ts, (int, float)):
            sub_stats[sub]['last_active'] = max(sub_stats[sub]['last_active'], ts)

    # 排序：按活动总量
    sorted_subs = sorted(
        sub_stats.items(),
        key=lambda x: x[1]['posts'] + x[1]['comments'],
        reverse=True,
    )
    profile['top_subreddits'] = [
        {
            'name': sub,
            'posts': stats['posts'],
            'comments': stats['comments'],
            'total': stats['posts'] + stats['comments'],
            'karma': stats['karma'],
            'last_active': fmt_ts(stats.get('last_active', 0)),
        }
        for sub, stats in sorted_subs[:20]
    ]

    # ── Karma ──
    total_karma = sum(s.get('score', 0) or 0 for s in submissions)
    total_karma += sum(c.get('score', 0) or 0 for c in comments)
    profile['karma_total_estimate'] = total_karma
    profile['karma_avg_per_post'] = round(
        total_karma / max(len(submissions), 1), 1)
    profile['karma_avg_per_comment'] = round(
        total_karma / max(len(comments), 1), 1) if comments else 0

    # ── 时间活跃模式 ──
    hours = {'morning(6-12)': 0, 'afternoon(12-18)': 0,
             'evening(18-24)': 0, 'night(0-6)': 0}
    weekdays = Counter()
    months = Counter()

    all_items = submissions + comments
    for item in all_items:
        ts = item.get('created_utc', 0)
        if isinstance(ts, (int, float)) and ts:
            dt = datetime.utcfromtimestamp(ts)
            h = dt.hour
            if 6 <= h < 12:
                hours['morning(6-12)'] += 1
            elif 12 <= h < 18:
                hours['afternoon(12-18)'] += 1
            elif 18 <= h < 24:
                hours['evening(18-24)'] += 1
            else:
                hours['night(0-6)'] += 1
            weekdays[dt.strftime('%a')] += 1
            months[dt.strftime('%Y-%m')] += 1

    profile['activity_hour_distribution'] = hours
    profile['active_weekdays'] = dict(weekdays.most_common())
    # 只保留最近 24 个月的月度分布
    profile['activity_monthly'] = dict(sorted(months.items())[-24:])

    # ── 内容特征 ──
    title_lengths = []
    selftext_lengths = []
    body_lengths = []
    for s in submissions:
        title = s.get('title', '') or ''
        selftext = s.get('selftext', '') or ''
        title_lengths.append(len(title))
        selftext_lengths.append(len(selftext))
    for c in comments:
        body = c.get('body', '') or ''
        body_lengths.append(len(body))

    profile['avg_title_length'] = round(
        sum(title_lengths) / max(len(title_lengths), 1), 1)
    profile['avg_selftext_length'] = round(
        sum(selftext_lengths) / max(len(selftext_lengths), 1), 1)
    profile['avg_comment_length'] = round(
        sum(body_lengths) / max(len(body_lengths), 1), 1)
    profile['avg_overall_length'] = round(
        (sum(title_lengths) + sum(selftext_lengths) + sum(body_lengths))
        / max(len(all_items), 1), 1)

    # ── NSFW ──
    has_nsfw = any(s.get('over_18') for s in submissions)
    has_nsfw = has_nsfw or any(c.get('over_18') for c in comments)
    profile['has_nsfw_content'] = has_nsfw

    # ── 时间跨度 ──
    all_timestamps = [s['created_utc'] for s in submissions
                      if isinstance(s.get('created_utc'), (int, float))]
    all_timestamps += [c['created_utc'] for c in comments
                       if isinstance(c.get('created_utc'), (int, float))]
    if all_timestamps:
        profile['first_active'] = fmt_ts(min(all_timestamps))
        profile['last_active'] = fmt_ts(max(all_timestamps))
        profile['active_days_span'] = (
            max(all_timestamps) - min(all_timestamps)) // 86400
    else:
        profile['first_active'] = None
        profile['last_active'] = None
        profile['active_days_span'] = 0

    # ── 最活跃 subreddit 分类 ──
    top_subs = sorted_subs[:5]
    profile['topics'] = [s[0] for s in top_subs]

    return profile


def phase3_build_profiles():
    """
    读取 JSONL 原始数据，逐个构建用户画像，写入 user_profiles.json。
    """
    log('=' * 55)
    log('Phase 3: 构建用户画像')

    if not os.path.exists(PUSHSHIFT_RAW):
        log('错误: Pushshift 原始数据不存在，请先运行 Phase 2')
        return

    profiles = {}
    line_count = 0

    with open(PUSHSHIFT_RAW, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_entry = json.loads(line)
                profile = build_profile_from_raw(raw_entry)
                profiles[profile['fullname']] = profile
                line_count += 1
                if line_count % 500 == 0:
                    log(f'  已处理 {line_count} 个用户')
            except Exception as e:
                log(f'  处理用户时出错: {e}')
                traceback.print_exc()

    save_json(profiles, USER_PROFILES)
    log(f'  Phase 3 完成: {line_count} 个用户画像已保存到 {USER_PROFILES}')


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='从 Pushshift 获取 Reddit 用户数据并构建画像')
    parser.add_argument('--phase', type=str, choices=['1', '1b', '1c', '2', '3'],
                        default=None, help='运行指定阶段 (1=1A+1B, 1b=1B only, 1c=1C only)')
    parser.add_argument('--resume', action='store_true',
                        help='从中断处继续')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已处理的用户（Phase 2）')
    args = parser.parse_args()

    if PROXIES:
        log(f'系统代理已启用: {list(PROXIES.values())[0]}')
    else:
        log('未检测到系统代理，将直连')

    phase = args.phase

    # Phase 1 系列
    if phase is None or phase == '1':
        target_users = read_list(USER_ID_LIST)
        if not target_users:
            log(f'错误: 无法读取用户 ID 列表 {USER_ID_LIST}')
            return
        log(f'目标用户总数: {len(target_users)}')
        phase1a_scan_csv(target_users)
        phase1b_resolve_usernames()
        log('Phase 1 完成!\n')
    elif phase == '1b':
        phase1b_resolve_usernames()
        log('Phase 1B 完成!\n')
    elif phase == '1c':
        phase1c_resolve_commenters()
        log('Phase 1C 完成!\n')

    # Phase 2: 采集 Pushshift 数据
    if phase is None or phase == '2':
        phase2_fetch_user_data(skip_existing=args.skip_existing)
        log('Phase 2 完成!\n')

    # Phase 3: 构建画像
    if phase is None or phase == '3':
        phase3_build_profiles()
        log('Phase 3 完成!\n')

    # 概要报告
    if phase is None:
        report_summary()

    log('全部完成!')


def report_summary():
    """打印各阶段结果摘要。"""
    print()
    print('=' * 55)
    print('  完成报告')
    print('=' * 55)

    # Phase 1A
    mapping = load_json(USER_POST_MAPPING, {})
    if mapping and '_meta' in mapping:
        meta = mapping['_meta']
        print(f'  Phase 1A:')
        print(f'    扫描进度: {meta.get("scan_progress", "?")}')
        print(f'    有帖子的用户: {meta.get("users_with_posts", "?")}')

    # Phase 1B
    name_map = load_json(USER_ID_TO_NAME, {})
    resolved = len([k for k in name_map if k.startswith('t2_')])
    print(f'  Phase 1B:')
    print(f'    已解析用户名: {resolved}')

    # Phase 2
    if os.path.exists(PUSHSHIFT_RAW):
        user_count = len(read_jsonl_keys(PUSHSHIFT_RAW))
        not_found = load_json(PUSHSHIFT_NOT_FOUND, {})
        print(f'  Phase 2:')
        print(f'    已抓取用户: {user_count}')
        print(f'    无数据的用户: {len(not_found)}')

    # Phase 3
    profiles = load_json(USER_PROFILES, {})
    if profiles:
        print(f'  Phase 3:')
        print(f'    用户画像: {len(profiles)} 个')


if __name__ == '__main__':
    main()
