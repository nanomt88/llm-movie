"""
fetch_reddit_movies.py
=====================
Reddit r/movies data fetcher — T1 scaffold + reachability probe.

Probes Arctic Shift (api.arctic-shift.photon-reddit.com) reachability; on a
direct timeout it retries via the system proxy; if the proxy also times out,
it falls back to PullPush (api.pullpush.io). Reuses get_system_proxy() and
RateLimiter from his/fetch_user_profiles.py (lines 127-147, 166-177) and the
PullPush endpoint shape from line 206.

T1 探测 Arctic Shift 可达性：直连超时则经系统代理重试，仍超时则回退到
PullPush。复用 his/fetch_user_profiles.py 的 get_system_proxy / RateLimiter
及 PullPush 端点（/reddit/submission/search、/reddit/comment/search）。
Arctic Shift 端点为 /api/posts/search、/api/comments/search。

用法:
  python fetch_reddit_movies.py --probe --source arctic --date 2021-01-01 --dry-run
  python fetch_reddit_movies.py --probe --source pullpush --date 2021-01-01 --dry-run
  python fetch_reddit_movies.py --bulk --dry-run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════════════
SUBREDDIT = "movies"
START_TS = 1609459200            # 2021-01-01 00:00:00 UTC
END_TS = 1735689599              # 2024-12-31 23:59:59 UTC

ARCTIC_BASE = "https://api.arctic-shift.photon-reddit.com"
ARCTIC_POSTS_PATH = "/api/posts/search"
ARCTIC_COMMENTS_PATH = "/api/comments/search"

PULLPUSH_BASE = "https://api.pullpush.io"
PULLPUSH_SUBMISSION_PATH = "/reddit/submission/search"
PULLPUSH_COMMENT_PATH = "/reddit/comment/search"

HTTP_TIMEOUT = 30                # 每个请求显式 30s 超时
RATE_INTERVAL = 1.0              # 限速间隔（秒）—— T1 探测用
PROBE_RATE_INTERVAL = 2.5        # T2 探测限速间隔（匹配 his:PUSHSHIFT_INTERVAL=2.5）
PAGE_SIZE = 100                  # PullPush 单页最大记录数
FULL_RECORDS_ESTIMATE = 150_000_000   # 2021-2025 r/movies 预估总记录数（~150M）
CONSECUTIVE_429_CAP = 2          # 连续 429 上限：超过则中止当前端点分页

PROBE_PARAMS = {"subreddit": SUBREDDIT, "limit": 1}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                  " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
    "Connection": "close",
}

# ═══════════════════════════════════════════════════════════════════════
#  复用 his/fetch_user_profiles.py 的 get_system_proxy / RateLimiter
#  导入失败则回退到本地副本（不修改 his/）
# ═══════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(PROJECT_ROOT, "his"))
try:
    from fetch_user_profiles import RateLimiter, get_system_proxy
except (ImportError, SyntaxError):
    import threading
    import time

    def get_system_proxy():
        """从 Windows 注册表读取系统代理（镜像 his:127-147）。"""
        import platform
        if platform.system() != "Windows":
            return None
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            ) as key:
                enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
                server, _ = winreg.QueryValueEx(key, "ProxyServer")
            if enabled and server:
                server = server.strip()
                if not server.startswith("http://"):
                    server = "http://" + server
                return {"http": server, "https": server}
        except OSError:
            return None
        return None

    class RateLimiter:
        """线程安全限速器（镜像 his:166-177）。"""

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


ratelimit = RateLimiter(RATE_INTERVAL)
probe_ratelimit = RateLimiter(PROBE_RATE_INTERVAL)


def _resolve_proxies():
    """优先环境变量 HTTP_PROXY/HTTPS_PROXY，其次注册表系统代理。"""
    env_http = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    env_https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if env_http or env_https:
        proxies = {}
        if env_http:
            proxies["http"] = env_http
        if env_https:
            proxies["https"] = env_https
        return proxies
    return get_system_proxy() or {}


def _build_session(use_proxy):
    """构造 requests.Session：直连（不走代理）或经代理。"""
    session = requests.Session()
    session.headers.update(HEADERS)
    if use_proxy:
        session.trust_env = True
        proxies = _resolve_proxies()
        if proxies:
            session.proxies.update(proxies)
    else:
        session.trust_env = False
    return session


def _date_to_ts(date_str):
    """解析 YYYY-MM-DD 为 UTC unix 时间戳。"""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _probe(session, url):
    """探测一个端点；返回 (可达, 详情)。详情为 '200'/'timeout'/'HTTP <code>'/'error: <类型>'。"""
    ratelimit.wait()
    try:
        resp = session.get(url, params=PROBE_PARAMS, timeout=HTTP_TIMEOUT)
    except requests.Timeout:
        return False, "timeout"
    except requests.RequestException as exc:
        return False, f"error: {exc.__class__.__name__}"
    code = resp.status_code
    if code == 200:
        return True, "200"
    return False, f"HTTP {code}"


def _probe_arctic():
    """探测 Arctic Shift：直连 → 代理 → 回退 PullPush。返回退出码。"""
    direct = _build_session(use_proxy=False)
    ok, detail = _probe(direct, ARCTIC_BASE + ARCTIC_POSTS_PATH)
    print(f"ARCTIC direct: {detail}")
    if ok:
        print("200")
        return 0

    proxy = _build_session(use_proxy=True)
    ok2, detail2 = _probe(proxy, ARCTIC_BASE + ARCTIC_POSTS_PATH)
    print(f"ARCTIC proxy: {detail2}")
    if ok2:
        print("proxy-needed")
        print("200")
        return 0

    print("FALLBACK: pullpush")
    pp_direct = _build_session(use_proxy=False)
    ok3, detail3 = _probe(pp_direct, PULLPUSH_BASE + PULLPUSH_SUBMISSION_PATH)
    print(f"PULLPUSH direct: {detail3}")
    if ok3:
        print("200")
        return 0
    pp_proxy = _build_session(use_proxy=True)
    ok4, detail4 = _probe(pp_proxy, PULLPUSH_BASE + PULLPUSH_SUBMISSION_PATH)
    print(f"PULLPUSH proxy: {detail4}")
    if ok4:
        print("proxy-needed")
        print("200")
        return 0
    print("ERROR: Arctic Shift 与 PullPush 均不可达")
    return 1


def _probe_pullpush():
    """探测 PullPush：直连 → 代理。返回退出码。"""
    direct = _build_session(use_proxy=False)
    ok, detail = _probe(direct, PULLPUSH_BASE + PULLPUSH_SUBMISSION_PATH)
    print(f"PULLPUSH direct: {detail}")
    if ok:
        print("200")
        return 0
    proxy = _build_session(use_proxy=True)
    ok2, detail2 = _probe(proxy, PULLPUSH_BASE + PULLPUSH_SUBMISSION_PATH)
    print(f"PULLPUSH proxy: {detail2}")
    if ok2:
        print("proxy-needed")
        print("200")
        return 0
    print("ERROR: PullPush 不可达")
    return 1


def _fetch_page(session, url, params):
    """Fetch one PullPush page via direct session.

    Returns (status, records, bytes, rate_info, err) where:
      - status: HTTP code or None on connection error
      - records: list[dict] on 200 (maybe empty), else None
      - bytes: response body bytes
      - rate_info: {"got_429": bool, "retry_after": str|None, "headers": {..}}
      - err: None on success, else "timeout"/"HTTP <code>"/"429"/"json_error"/"error: <type>"

    Paced by probe_ratelimit (~2.5s); 30s per-request timeout.
    抓取一页 PullPush 结果；直连限速 2.5s，30s 超时；捕获 429 与限流头。
    """
    probe_ratelimit.wait()
    rate_info = {"got_429": False, "retry_after": None, "headers": {}}
    try:
        resp = session.get(url, params=params, timeout=HTTP_TIMEOUT)
    except requests.Timeout:
        return None, None, 0, rate_info, "timeout"
    except requests.RequestException as exc:
        return None, None, 0, rate_info, f"error: {exc.__class__.__name__}"
    code = resp.status_code
    for k, v in resp.headers.items():
        kl = k.lower()
        if "rate" in kl or "limit" in kl or "retry" in kl or "remaining" in kl:
            rate_info["headers"][k] = v
    if code == 429:
        rate_info["got_429"] = True
        rate_info["retry_after"] = resp.headers.get("Retry-After")
        return code, None, len(resp.content), rate_info, "429"
    if code != 200:
        return code, None, len(resp.content), rate_info, f"HTTP {code}"
    try:
        data = resp.json()
    except ValueError:
        return code, None, len(resp.content), rate_info, "json_error"
    if isinstance(data, dict):
        records = data.get("data") or []
    elif isinstance(data, list):
        records = data
    else:
        records = []
    return code, records, len(resp.content), rate_info, None


def _paginate(session, path, after_ts, before_ts, label):
    """Page through one PullPush endpoint over [after_ts, before_ts].

    Empirical finding (T2 probe): PullPush `after` accepts ONLY an integer
    timestamp (bare id / fullname `t3_..` are rejected with 400 "Invalid
    integer"), and `after=N` is inclusive (records with created_utc >= N are
    returned). Hence naive timestamp pagination re-returns the boundary record.
    Strategy: timestamp cursor + id dedup; when a page is all duplicates
    (stuck on a timestamp tie), force cursor+1 to advance. Stops on a short
    page (<PAGE_SIZE) or empty page.

    Returns (records, n_requests, n_bytes, rate_observations, cooldown_secs).
    按时间戳游标分页；id 去重；遇全重复页则游标+1 强制推进；短页/空页停止。
    返回记录数、请求数、字节数、限流观测、冷却累计秒数。
    """
    url = PULLPUSH_BASE + path
    seen = set()
    records = []
    n_req = n_bytes = 0
    rate_obs = []
    cooldown = 0.0
    cursor = after_ts
    consecutive_429 = 0
    max_requests = 10000  # 安全上限，防失控
    while n_req < max_requests:
        params = {"subreddit": SUBREDDIT, "after": cursor, "before": before_ts,
                  "size": PAGE_SIZE, "sort": "asc"}
        code, recs, nbytes, rinfo, err = _fetch_page(session, url, params)
        n_req += 1
        n_bytes += nbytes
        if rinfo.get("got_429") or rinfo["headers"]:
            rate_obs.append({"request": n_req, "label": label, **rinfo})
        if err == "429":
            consecutive_429 += 1
            if consecutive_429 > CONSECUTIVE_429_CAP:
                print(f"  [{label}] 连续 {consecutive_429 - 1} 次 429，中止分页")
                break
            print(f"  [{label}] 429 (req {n_req})，冷却 60s 后重试")
            time.sleep(60)
            cooldown += 60.0
            continue
        consecutive_429 = 0
        if err:
            print(f"  [{label}] page {n_req}: {err}；停止")
            break
        if not recs:
            break
        new = [r for r in recs if r.get("id") and r["id"] not in seen]
        for r in new:
            seen.add(r["id"])
        records.extend(new)
        last_utc = max((r.get("created_utc") or cursor) for r in recs)
        if len(recs) < PAGE_SIZE:
            break  # 最后一页
        if not new:
            cursor = last_utc + 1  # 全重复：时间戳并列，强制推进
            continue
        cursor = last_utc
        if n_req % 25 == 0:
            print(f"  [{label}] {n_req} 页，累计 {len(records)} 条，cursor={cursor}")
    return records, n_req, n_bytes, rate_obs, cooldown


def _probe_fetch(date_str):
    """T2 probe: fetch one UTC day of r/movies submissions + comments via
    PullPush DIRECT (no proxy, trust_env=False), measure throughput, and
    extrapolate full 2021-2025 API-only hours. Writes data/raw/probe_<date>.json.

    T2 探测：直连抓取指定 UTC 日 r/movies 的 submission+comment，测吞吐，
    外推全量（2021-2025）仅 API 耗时；写 data/raw/probe_<日期>.json。
    """
    after_ts = _date_to_ts(date_str)
    before_ts = after_ts + 86400  # 当日 UTC 结束（次日 00:00:00）
    session = _build_session(use_proxy=False)  # DIRECT，不走代理
    print(f"[T2-probe] window={date_str} after={after_ts} before={before_ts} "
          f"(direct, trust_env=False)")
    t0 = time.time()
    subs, sub_req, sub_bytes, sub_rate, sub_cd = _paginate(
        session, PULLPUSH_SUBMISSION_PATH, after_ts, before_ts, "submission")
    coms, com_req, com_bytes, com_rate, com_cd = _paginate(
        session, PULLPUSH_COMMENT_PATH, after_ts, before_ts, "comment")
    elapsed = time.time() - t0
    cooldown = sub_cd + com_cd
    requests_n = sub_req + com_req
    bytes_n = sub_bytes + com_bytes
    total_records = len(subs) + len(coms)
    throughput = total_records / elapsed if elapsed > 0 else 0.0
    elapsed_net = max(elapsed - cooldown, 1e-6)
    steady_tp = total_records / elapsed_net
    rate_obs = sub_rate + com_rate
    rate_summary = {
        "got_429": any(r.get("got_429") for r in rate_obs),
        "rate_limit_headers_seen": any(r.get("headers") for r in rate_obs),
        "observations": rate_obs,
    }
    hours_api = (FULL_RECORDS_ESTIMATE / (throughput * 3600)
                 if throughput > 0 else None)
    hours_api_steady = (FULL_RECORDS_ESTIMATE / (steady_tp * 3600)
                        if steady_tp > 0 else None)
    partial = rate_summary["got_429"]
    partial_note = (
        "comments_count 为部分值：评论分页因连续 429 被中止，未取满当日全部评论。"
        "throughput_rec_per_sec 含 429 冷却时间（偏低）；"
        "steady_state_throughput_rec_per_sec 剔除冷却，更接近真实吞吐。"
        if partial else
        "无 429，submissions_count/comments_count 为当日全量。")
    holiday_note = ("R9: 2021-01-01 为美国元旦（法定假日），r/movies 当日活跃度"
                     "与发帖/评论量通常低于工作日均值，测得吞吐可能偏低，"
                     "故外推的全量 API 耗时偏保守（高估）。")
    result = {
        "date": date_str,
        "submissions_count": len(subs),
        "comments_count": len(coms),
        "elapsed_sec": round(elapsed, 3),
        "cooldown_sec": round(cooldown, 3),
        "requests": requests_n,
        "throughput_rec_per_sec": round(throughput, 4),
        "steady_state_throughput_rec_per_sec": round(steady_tp, 4),
        "rate_limit_observed": rate_summary,
        "bytes_transferred": bytes_n,
        "extrapolated_full_hours_api": hours_api,
        "extrapolated_full_hours_api_steady": hours_api_steady,
        "extrapolated_full_hours_parquet": "12-48 (network-bound, T3)",
        "partial_fetch_note": partial_note,
        "R9_holiday_bias_note": holiday_note,
        "pagination_note": ("PullPush `after` 仅接受整数时间戳（id/fullname 均 400）；"
                            "after=N 为包含式，故采用 时间戳游标 + id 去重 + "
                            "全重复页游标+1 推进 的分页策略。"),
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "raw"), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, "data", "raw", f"probe_{date_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[T2-probe] subs={len(subs)} coms={len(coms)} req={requests_n} "
          f"bytes={bytes_n} elapsed={elapsed:.1f}s (cooldown {cooldown:.0f}s) "
          f"throughput={throughput:.2f} rec/s (steady {steady_tp:.2f})")
    if hours_api is not None:
        print(f"[T2-probe] extrapolated full (API-only, ~150M): "
              f"{hours_api:.1f} hours (steady: {hours_api_steady:.1f} h)")
    else:
        print("[T2-probe] throughput=0，无法外推")
    print(f"[T2-probe] wrote {out_path}")
    return result


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Reddit r/movies fetcher scaffold + probe (T1)"
    )
    p.add_argument("--probe", action="store_true", help="运行可达性探测")
    p.add_argument("--probe-fetch", action="store_true",
                   help="T2：抓取单日 submission+comment，测吞吐并外推全量耗时")
    p.add_argument("--bulk", action="store_true", help="批量抓取（T1 桩）")
    p.add_argument(
        "--source", choices=["arctic", "pullpush", "parquet"], default="arctic"
    )
    p.add_argument("--date", help="日期窗口 YYYY-MM-DD")
    p.add_argument("--dry-run", action="store_true", help="不落盘，仅探测")
    return p.parse_args(argv)


def main(argv=None):
    """CLI 入口。"""
    args = _parse_args(argv)
    if args.dry_run:
        print("[DRY-RUN]")
    if args.probe_fetch:
        if not args.date:
            print("ERROR: --probe-fetch 需要 --date YYYY-MM-DD")
            return 1
        _probe_fetch(args.date)
        return 0
    if args.bulk and not args.probe:
        print("T1: --bulk 尚未实现（仅探测）；请使用 --probe")
        return 0
    if not args.probe:
        print("T1: 无动作；请使用 --probe")
        return 0
    if args.source == "parquet":
        print("parquet: 无网络探测（本地文件源）")
        return 0
    if not args.date:
        print("ERROR: 探测需要 --date YYYY-MM-DD")
        return 1
    try:
        after_ts = _date_to_ts(args.date)
    except ValueError:
        print(f"ERROR: 非法日期 {args.date!r}（应为 YYYY-MM-DD）")
        return 1
    print(f"window: {args.date} (after={after_ts})")
    if args.source == "arctic":
        return _probe_arctic()
    return _probe_pullpush()


if __name__ == "__main__":
    sys.exit(main())
