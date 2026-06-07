"""
movie_info.py
从 TMDB (The Movie Database) API 获取电影类型和主要元素信息（基于 IMDb ID）。

输入:
  - data/entity2id.json  : {电影名称(含年份): IMDb ID}
  - data/my-test-data.csv     : processed 列中包含 ttID 的对话数据
输出:
  - data/movie_info.json : {IMDb ID: {imdb_id, tmdb_id, title, genres, director, cast, ...}}
  - data/movie_not_found.json : 未在 TMDB 找到的电影

TMDB API:
  Find by IMDb ID: GET /3/find/{imdb_id}?external_source=imdb_id
  Movie details:   GET /3/movie/{tmdb_id}
  Movie credits:   GET /3/movie/{tmdb_id}/credits
"""

import json
import csv
import re
import time
import os
import sys
import platform
from datetime import datetime

import requests

# ── 控制台编码 ─────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.upper() in ("GBK", "GB2312"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── TMDB 配置 ─────────────────────────────────────────────────────────
TMDB_API_KEY = "a3aa1505f3cc56c3433b437ba3738435"
TMDB_ACCESS_TOKEN = (
    "eyJhbGciOiJIUzI1NiJ9"
    ".eyJhdWQiOiJhM2FhMTUwNWYzY2M1NmMzNDMzYjQzN2JhMzczODQzNSIsIm5iZiI6MTc4MDA2NjcwNy43MTcsInN1YiI6IjZhMTlhOTkzY2I2NjUyMGM1Y2JhYjdmMSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ"
    ".bfzSxYNcJYPzOGy84bAOObUOav-AIDVKAR1zMvnMukM"
)
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# ── 文件路径 ──────────────────────────────────────────────────────────
ENTITY2ID_PATH = "../data/entity2id.json"
CSV_PATH = "../data/my-test-data.csv"
OUTPUT_PATH = "../data/movie_info.json"
NOT_FOUND_PATH = "../data/movie_not_found.json"

REQUEST_INTERVAL = 0.3   # TMDB 没有严格限制，但保持礼貌
MAX_RETRIES = 3
RETRY_DELAY = 3

# ── 系统代理 ──────────────────────────────────────────────────────────
def _get_system_proxy():
    """从 Windows 注册表读取系统代理设置。"""
    if platform.system() != "Windows":
        return None
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        ) as key:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            server, _ = winreg.QueryValueEx(key, "ProxyServer")
        if enabled and server:
            server = server.strip()
            if not server.startswith("http://"):
                server = "http://" + server
            return {"http": server, "https": server}
    except Exception:
        pass
    return None


PROXIES = _get_system_proxy()

HEADERS = {
    "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}",
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

# ── 日志 ──────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── 限速器 ────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, interval):
        self.interval = interval
        self._last = 0.0

    def wait(self):
        elapsed = time.time() - self._last
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last = time.time()


ratelimit = RateLimiter(REQUEST_INTERVAL)


# ── 1. 收集所有唯一的 IMDb ID ────────────────────────────────────────
def collect_all_ids():
    """从 entity2id.json 和 my-test-data.csv 中收集所有唯一的 IMDb ID。"""
    log("收集所有 IMDb ID ...")

    # entity2id.json
    with open(ENTITY2ID_PATH, "r", encoding="utf-8") as f:
        name_to_id = json.load(f)
    id_to_name = {v: k for k, v in name_to_id.items()}
    entity_ids = set(name_to_id.values())
    log(f"  entity2id.json: {len(entity_ids)} 个 ID")

    # my-test-data.csv
    csv_ids = set()
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed = row.get("processed", "")
            if processed:
                found = re.findall(r"tt\d+", processed)
                csv_ids.update(found)
    log(f"  my-test-data.csv:   {len(csv_ids)} 个 ID")

    all_ids = entity_ids | csv_ids
    log(f"  去重后共:      {len(all_ids)} 个 ID\n")

    return all_ids, id_to_name


# ── 2. TMDB API 调用 ─────────────────────────────────────────────────
def tmdb_get(path, params=None):
    """向 TMDB API 发送 GET 请求，返回 JSON 响应。"""
    ratelimit.wait()
    url = TMDB_BASE_URL + path
    p = dict(params) if params else {}
    p.setdefault("language", "zh-CN")  # 优先中文
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=p, proxies=PROXIES, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log(f"  ⚠ TMDB 请求失败 (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return None


def find_by_imdb_id(imdb_id):
    """通过 IMDb ID 查找 TMDB 电影 ID。返回 (tmdb_id, media_type) 或 None。"""
    data = tmdb_get(f"/find/{imdb_id}", {"external_source": "imdb_id"})
    if not data:
        return None
    # 优先找 movie 结果
    for media_type in ("movie_results", "tv_results"):
        results = data.get(media_type, [])
        if results:
            return (results[0]["id"], media_type.replace("_results", ""))
    return None


def get_movie_details(tmdb_id, media_type="movie"):
    """获取电影/剧集详情。"""
    return tmdb_get(f"/{media_type}/{tmdb_id}")


def get_movie_credits(tmdb_id, media_type="movie"):
    """获取电影/剧集演职员信息。"""
    return tmdb_get(f"/{media_type}/{tmdb_id}/credits")


def search_by_title(title):
    """通过电影名称搜索 TMDB（降级方案）。返回第一个匹配的 TMDB ID 或 None。"""
    data = tmdb_get("/search/movie", {"query": title, "page": 1})
    if not data:
        return None
    results = data.get("results", [])
    if results:
        return results[0]["id"]
    return None


# ── 3. 获取电影信息 ──────────────────────────────────────────────────
def fetch_movie_info(imdb_id, known_name=None):
    """
    获取单个电影的信息。
    返回 dict 或 None（如果完全找不到）。
    """
    # Step 1: 通过 IMDb ID 查找 TMDB
    found = find_by_imdb_id(imdb_id)
    tmdb_id = None
    media_type = "movie"

    if found:
        tmdb_id, media_type = found
        log(f"  TMDB ID: {tmdb_id} ({media_type})")
    elif known_name:
        # Step 2: 降级 - 用电影名称搜索
        name_clean = re.sub(r"\s*\(\d{4}\)\s*", "", known_name).strip()
        if name_clean:
            tmdb_id = search_by_title(name_clean)

    if not tmdb_id:
        return None

    # Step 3: 获取详情
    details = get_movie_details(tmdb_id, media_type)
    if not details:
        return None

    # Step 4: 获取演职员
    credits = get_movie_credits(tmdb_id, media_type)

    # ── 解析 ──────────────────────────────────────────────────────
    # 导演
    director = ""
    if credits:
        for person in credits.get("crew", []):
            if person.get("job") == "Director":
                director = person.get("name", "")
                break

    # 主演（取前 10）
    cast = []
    if credits:
        for person in credits.get("cast", [])[:10]:
            cast.append(person.get("name", ""))

    # 类型
    genres = [g["name"] for g in details.get("genres", [])]

    # 制片国家/地区
    countries = [c["name"] for c in details.get("production_countries", [])]

    # 发行年份
    release_date = details.get("release_date", "")
    year = release_date[:4] if release_date else ""

    # 时长（分钟）
    runtime = details.get("runtime") or 0

    # 海报
    poster_path = details.get("poster_path")
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""

    info = {
        "imdb_id": imdb_id,
        "tmdb_id": tmdb_id,
        "media_type": media_type,
        "title": details.get("title") or details.get("name", ""),
        "original_title": details.get("original_title") or details.get("original_name", ""),
        "year": year,
        "country": " / ".join(countries),
        "genres": genres,
        "runtime_minutes": runtime,
        "director": director,
        "cast": cast,
        "rating": details.get("vote_average", 0),
        "vote_count": details.get("vote_count", 0),
        "poster_url": poster_url,
        "overview": details.get("overview", ""),
    }

    return info


# ── 4. 主流程 ────────────────────────────────────────────────────────
def load_existing(path):
    """加载已存在的 JSON 文件。"""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            log(f"  ⚠ 无法读取 {path}，跳过")
    return {}


def save_json(data, path):
    """安全地保存 JSON 文件（先写 tmp 再替换）。"""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def needs_refresh(data):
    """
    检查现有数据是否是旧格式（Douban 格式）。
    如果是，需要清空重新抓取。
    """
    for v in data.values():
        if "douban_id" in v:
            return True
    return False


def main():
    all_ids, id_to_name = collect_all_ids()

    results = load_existing(OUTPUT_PATH)
    not_found = load_existing(NOT_FOUND_PATH)

    # 如果数据是旧版 Douban 格式，清空重新抓取
    if needs_refresh(results):
        log("  ℹ 检测到旧版豆瓣数据格式，清空缓存重新抓取 ...")
        results = {}
        not_found = {}

    log(f"已有缓存: {len(results)} 个电影")
    log(f"未找到记录: {len(not_found)} 个\n")

    # 确定需要抓取的 ID（排除已成功和已记录为未找到的）
    done_ids = set(results.keys()) | set(not_found.keys())
    todo = sorted(all_ids - done_ids)
    log(f"需要抓取: {len(todo)} 个电影\n")

    if not todo:
        log("全部已完成!")
        return

    stats = {"ok": 0, "skip": 0, "fail": 0}
    last_save_count = 0

    for idx, imdb_id in enumerate(todo, 1):
        known_name = id_to_name.get(imdb_id)
        label = known_name or imdb_id
        log(f"[{idx}/{len(todo)}] {label} ({imdb_id})")

        info = None
        for attempt in range(1, MAX_RETRIES + 1):
            info = fetch_movie_info(imdb_id, known_name)
            if info is not None:
                break
            if attempt < MAX_RETRIES:
                log(f"  重试 {attempt}/{MAX_RETRIES} ...")
                time.sleep(RETRY_DELAY * attempt)

        if info:
            results[imdb_id] = info
            stats["ok"] += 1
            genre_str = " / ".join(info.get("genres", [])) or "N/A"
            log(f'  \u2713 {info["title"]}  [{genre_str}]')
        else:
            placeholder = {"imdb_id": imdb_id, "title": known_name or ""}
            not_found[imdb_id] = placeholder
            stats["fail"] += 1
            log(f"  \u2717 未在 TMDB 找到")

        # 每 10 个保存一次
        current_total = len(results) + len(not_found)
        if (idx % 10 == 0 or idx == len(todo)) and last_save_count < current_total:
            save_json(results, OUTPUT_PATH)
            save_json(not_found, NOT_FOUND_PATH)
            last_save_count = current_total

    # 最后保存一次
    save_json(results, OUTPUT_PATH)
    save_json(not_found, NOT_FOUND_PATH)

    # 统计
    total_fetched = stats["ok"] + stats["fail"]
    hit_rate = stats["ok"] / total_fetched * 100 if total_fetched > 0 else 0
    log("\n" + "=" * 55)
    log("完成!")
    log(f"  成功: {stats['ok']}  |  失败: {stats['fail']}  |  命中率: {hit_rate:.1f}%")
    log(f"  总缓存: {len(results)} 个电影")
    log(f"  结果已保存到: {OUTPUT_PATH}")
    log(f"  未找到记录:  {NOT_FOUND_PATH}")


if __name__ == "__main__":
    main()
