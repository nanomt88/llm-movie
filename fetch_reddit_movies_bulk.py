"""Bulk-fetch r/movies posts from HuggingFace open-index/arctic Parquet.

Streams the comments and submissions tables of the open-index/arctic HuggingFace
dataset (12.1B items, 2005-12 to 2026-02) via DuckDB's hf:// protocol, filtered
to subreddit='movies' and a UTC created-utc range. Writes per-month NDJSON
shards under data/raw/ with per-month checkpoint resume. Measures predicate
pushdown efficacy in the setup phase.

中文说明：通过 DuckDB 的 hf:// 协议流式查询 HuggingFace 上 open-index/arctic
数据集的 comments 与 submissions 两张 Parquet 表，按 subreddit='movies' 与
UTC 时间区间过滤，按月分片写出 NDJSON 至 data/raw/，支持按月断点续传。
启动阶段测量谓词下推效率（query_metadata().scan_bytes 对比匹配行的估算字节数）。
仅依赖 duckdb + huggingface_hub，不依赖旧式 Reddit 抓取库。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import duckdb
from huggingface_hub import HfApi

REPO_ID = "open-index/arctic"
REPO_TYPE = "dataset"

DATA_RAW = Path("data/raw")
DUCKDB_TMP = DATA_RAW / "_duckdb_tmp"
EVIDENCE_DIR = Path(".omo/evidence")

# arctic 实际 schema 与计划假设不同：comments 无 author_fullname/permalink，
# submissions 亦无 author_fullname/permalink。故按“核心列必须存在、期望列按需
# 选取”的容错策略处理。
CORE_COLUMNS: dict[str, set[str]] = {
    "comments": {"id", "created_utc", "subreddit", "body"},
    "submissions": {"id", "created_utc", "subreddit", "title"},
}

# 期望选取的列（按顺序）；实际 SELECT 只取“期望 ∩ 实际存在”的部分。
DESIRED_COLUMNS: list[str] = [
    "id", "created_utc", "title", "selftext", "body", "author",
    "author_fullname", "parent_id", "link_id", "subreddit", "score",
    "permalink",
]


def _build_select(actual_cols: set[str]) -> str:
    """Build the SELECT clause from DESIRED_COLUMNS ∩ actual_cols. (EN)

    中文：按“期望列 ∩ 实际列”构造 SELECT 子句，跳过 arctic 不存在的列。
    """
    present = [c for c in DESIRED_COLUMNS if c in actual_cols]
    if not present:
        raise RuntimeError("no desired columns present in actual schema")
    return ", ".join(present)


def _connect() -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with the project-mandated config. (EN)

    中文：建立 DuckDB 连接，按计划要求设置 4GB 内存上限与临时目录。
    """
    DUCKDB_TMP.mkdir(parents=True, exist_ok=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB';")
    con.execute(f"SET temp_directory='{DUCKDB_TMP.as_posix()}';")
    con.execute("SET threads TO 4;")
    return con


def _discover_parquet_glob(api: HfApi, kind: str, local_dir: str | None = None) -> str:
    """Discover the parquet glob path for the comments or submissions table. (EN)

    If `local_dir` is set, returns a local filesystem glob for files whose name
    starts with `kind` under that directory (fallback when DuckDB hf:// is
    network-blocked). Otherwise walks the open-index/arctic repo tree to locate
    the directory holding the `<kind>` parquet shards and returns a DuckDB hf://
    glob that matches all shards of that table.

    中文：若 `local_dir` 已设置，返回该目录下以 `kind` 开头的本地 Parquet glob
    （当 DuckDB hf:// 被网络阻断时的回退路径）。否则遍历 open-index/arctic
    仓库目录树，定位 comments 或 submissions 的 Parquet 分片目录，返回匹配该表
    所有分片的 DuckDB hf:// glob 路径。
    """
    if local_dir:
        ldir = Path(local_dir)
        # arctic 布局：data/comments/YYYY/MM/000.parquet；本地平铺为 comments_*.parquet。
        glob_path = str(ldir / f"{kind}*.parquet")
        if not list(ldir.glob(f"{kind}*.parquet")):
            raise RuntimeError(
                f"no local parquet matching '{glob_path}' under {local_dir}"
            )
        return glob_path
    # 先看仓库根目录结构。
    root_entries = list(api.list_repo_tree(
        REPO_ID, repo_type=REPO_TYPE, path_in_repo="", recursive=False,
    ))
    kind_lower = kind.lower()
    candidate_dirs: list[str] = []
    for entry in root_entries:
        name = getattr(entry, "path", "") or getattr(entry, "name", "")
        if not name:
            continue
        base = name.rstrip("/").split("/")[-1].lower()
        if base == kind_lower or kind_lower in base:
            candidate_dirs.append(name.rstrip("/"))
    # 若根目录未直接命中，尝试常见 data/ 子目录。
    if not candidate_dirs:
        for entry in root_entries:
            name = getattr(entry, "path", "") or getattr(entry, "name", "")
            if name and name.rstrip("/").endswith("data"):
                sub_entries = list(api.list_repo_tree(
                    REPO_ID, repo_type=REPO_TYPE, path_in_repo=name.rstrip("/"),
                    recursive=False,
                ))
                for sub in sub_entries:
                    sname = getattr(sub, "path", "") or getattr(sub, "name", "")
                    if not sname:
                        continue
                    base = sname.rstrip("/").split("/")[-1].lower()
                    if base == kind_lower or kind_lower in base:
                        candidate_dirs.append(sname.rstrip("/"))
    if not candidate_dirs:
        raise RuntimeError(
            f"could not locate '{kind}' parquet directory in {REPO_ID}; "
            f"root entries: {[getattr(e, 'path', getattr(e, 'name', '?')) for e in root_entries]}"
        )
    table_dir = candidate_dirs[0]
    return f"hf://datasets/{REPO_ID}/{table_dir}/**/*.parquet"


def _table_glob_for_describe(glob: str) -> str:
    """Pick one concrete parquet file under the glob for a cheap DESCRIBE. (EN)

    For local globs (not hf://) the glob is returned as-is since DuckDB can
    DESCRIBE a local glob cheaply. For hf:// globs, huggingface_hub is used to
    pick a single concrete parquet file (avoids DuckDB listing every shard).

    中文：本地 glob（非 hf://）直接返回（DuckDB 对本地 glob 的 DESCRIBE 很便宜）。
    对 hf:// glob，用 huggingface_hub 取一个具体 Parquet 文件用于 DESCRIBE，
    避免 DuckDB 列出所有分片。
    """
    if not glob.startswith("hf://"):
        return glob
    # 从 glob 中解析出目录前缀。
    api = HfApi()
    # 从 glob 中解析出目录前缀，如 hf://datasets/open-index/arctic/data/comments/**/*.parquet
    # -> data/comments
    prefix = glob.replace(f"hf://datasets/{REPO_ID}/", "")
    prefix = prefix.split("/**/*.parquet")[0]
    files = list(api.list_repo_tree(
        REPO_ID, repo_type=REPO_TYPE, path_in_repo=prefix, recursive=True,
    ))
    first_file: str | None = None
    for entry in files:
        path = getattr(entry, "path", "") or getattr(entry, "name", "")
        if path and path.endswith(".parquet"):
            first_file = path
            break
    if first_file is None:
        raise RuntimeError(f"no parquet file found under {prefix} in {REPO_ID}")
    return f"hf://datasets/{REPO_ID}/{first_file}"


def _describe_table(con: duckdb.DuckDBPyConnection, source: str) -> list[tuple[str, str]]:
    """Run DESCRIBE on the parquet source and return [(col, type), ...]. (EN)

    中文：对 Parquet 源执行 DESCRIBE，返回 [(列名, 类型), ...]。
    """
    rows = con.execute(f"DESCRIBE SELECT * FROM '{source}'").fetchmany(1000)
    return [(str(r[0]), str(r[1])) for r in rows]


def _verify_columns(kind: str, columns: list[tuple[str, str]]) -> set[str]:
    """Fail fast if CORE columns are missing; return the actual column set. (EN)

    中文：若核心列缺失则立即失败并打印差异；返回实际列集合供后续 SELECT 构建。
    """
    actual = {name for name, _type in columns}
    core = CORE_COLUMNS[kind]
    missing_core = core - actual
    if missing_core:
        raise RuntimeError(
            f"[{kind}] missing CORE columns: {sorted(missing_core)}; "
            f"actual columns: {sorted(actual)}"
        )
    missing_desired = set(DESIRED_COLUMNS) - actual
    if missing_desired:
        print(f"[{kind}] note: arctic lacks these desired cols (skipped): "
              f"{sorted(missing_desired)}")
    return actual


def _measure_pushdown(
    con: duckdb.DuckDBPyConnection, glob: str, kind: str,
) -> dict[str, Any]:
    """Measure predicate-pushdown efficacy for a one-week window. (EN)

    Runs a small filtered query with DuckDB JSON profiling enabled on an
    ISOLATED connection (profiling-to-file is unreliable on a connection that
    has already run other queries in duckdb 1.5.5 on Windows), reads
    `total_bytes_read` from the profile (actual parquet bytes scanned after
    row-group + column pruning — the scan_bytes), then estimates matched-row
    bytes from the returned row count + text length. Reports
    pushdown_efficacy_ratio = scan_bytes / matched_row_bytes (lower is better;
    ~1.0 means DuckDB scanned only what matched). Also reports scan_bytes vs
    total file size as a pushdown-selectivity sanity check, plus a selectivity
    fallback (matched_rows / total_rows) computed from a metadata-only count.

    中文：在一个隔离连接上开启 DuckDB JSON profiling 运行一周窗口的过滤查询
    （在已运行过其他查询的连接上 profiling-to-file 在 duckdb 1.5.5/Windows
    下不可靠），从 profiling JSON 读取 total_bytes_read（经行组 + 列裁剪后
    实际扫描的 parquet 字节数，即 scan_bytes），按返回行数与文本长度估算
    匹配行字节数，计算 pushdown_efficacy_ratio = scan_bytes / matched_row_bytes
    （越低代表下推越好，约 1.0 表示 DuckDB 仅扫描了匹配行的字节）。同时给出
    scan_bytes 占文件总大小的比例，以及基于元数据计数的选择性回退指标
    matched_rows / total_rows。
    """
    import tempfile
    start_ts = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2021, 1, 8, tzinfo=timezone.utc).timestamp())
    # 按表选取文本列：comments 用 body，submissions 用 selftext 回退 title。
    text_expr = "coalesce(body, '')" if kind == "comments" \
        else "coalesce(selftext, title, '')"
    q = (
        f"SELECT count(*), sum(length({text_expr})) "
        f"FROM '{glob}' "
        f"WHERE subreddit='movies' AND created_utc BETWEEN {start_ts} AND {end_ts}"
    )
    profile_path = Path(tempfile.gettempdir()) / f"duck_profile_{kind}.json"
    # 隔离连接，避免主连接已执行的查询影响 profiling 输出。
    mcon = duckdb.connect()
    mcon.execute("PRAGMA enable_profiling='json';")
    mcon.execute(f"PRAGMA profiling_output='{profile_path}';")
    try:
        count_row = mcon.execute(q).fetchone()
    finally:
        mcon.execute("PRAGMA disable_profiling;")
        mcon.close()
    matched_rows = int(count_row[0]) if count_row and count_row[0] is not None else 0
    text_bytes = int(count_row[1]) if count_row and count_row[1] is not None else 0

    scan_bytes = 0
    rows_scanned = 0
    if profile_path.exists():
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            scan_bytes = int(data.get("total_bytes_read", 0) or 0)
            rows_scanned = int(data.get("cumulative_rows_scanned", 0) or 0)
        except (json.JSONDecodeError, ValueError):
            pass
    # 匹配行估算字节数：行数 * (平均文本字节 + 估算元数据开销 ~256B/行)。
    matched_row_bytes = matched_rows * (text_bytes / max(matched_rows, 1) + 256) \
        if matched_rows > 0 else 0
    ratio = (scan_bytes / matched_row_bytes) if matched_row_bytes > 0 else None
    # 全表总行数（元数据计数，仅读 row-group 统计，便宜）。
    total_row_count = con.execute(
        f"SELECT count(*) FROM '{glob}'"
    ).fetchone()[0]
    # 全表总字节数（文件系统），用于下推选择性参考。
    total_file_bytes = 0
    if not glob.startswith("hf://"):
        base = Path(glob)
        for p in base.parent.glob(base.name):
            total_file_bytes += p.stat().st_size
    return {
        "kind": kind,
        "matched_rows": matched_rows,
        "total_rows_in_file": int(total_row_count),
        "rows_scanned_full": rows_scanned,
        "scan_bytes": scan_bytes,
        "matched_row_bytes_est": int(matched_row_bytes),
        "pushdown_efficacy_ratio": round(ratio, 4) if ratio is not None else None,
        "total_file_bytes": total_file_bytes,
        "scan_bytes_to_file_ratio": (
            round(scan_bytes / total_file_bytes, 6) if total_file_bytes else None
        ),
        "selectivity_matched_over_total": (
            round(matched_rows / int(total_row_count), 6)
            if total_row_count else None
        ),
    }


def _shard_path(kind: str, year: int, month: int) -> Path:
    """Return the per-month NDJSON shard path for a table + month. (EN)

    中文：返回某表某月的 NDJSON 分片路径。
    """
    tag = "movies_comments_2021_2025" if kind == "comments" \
        else "movies_submissions_2021_2025"
    return DATA_RAW / f"{tag}_{year:04d}{month:02d}.jsonl"


def _checkpoint_path(kind: str, year: int, month: int) -> Path:
    """Per-month checkpoint file marking a shard as fully written. (EN)

    中文：标记某月分片已完整写入的检查点文件。
    """
    return DATA_RAW / f".{_shard_path(kind, year, month).name}.done"


def _month_iter(start: datetime, end: datetime) -> Iterator[tuple[int, int]]:
    """Yield (year, month) tuples for each month in [start, end). inclusive. (EN)

    中文：按月生成 [start, end) 区间内的 (年, 月) 元组。
    """
    y, m = start.year, start.month
    while (y, m) < (end.year, end.month) or (y == end.year and m == end.month and end.day >= 1):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _stream_table_to_shards(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    kind: str,
    start_dt: datetime,
    end_dt: datetime,
    resume: bool,
    actual_cols: set[str],
) -> dict[str, int]:
    """Stream a table filtered to r/movies over the date range into shards. (EN)

    Issues a single streaming SELECT with subreddit + created_utc predicates and
    iterates via fetchmany(10000), dispatching each row to its per-month NDJSON
    shard. Resumable: months already marked done are skipped when --resume.
    `actual_cols` selects only the columns present in arctic's real schema.

    中文：对一张表执行带 subreddit + created_utc 谓词的流式 SELECT，通过
    fetchmany(10000) 迭代，将每行分发到对应月份的 NDJSON 分片。支持 --resume：
    已标记完成的月份会被跳过。`actual_cols` 用于按 arctic 实际 schema 选取列。
    """
    start_ts = int(start_dt.replace(tzinfo=timezone.utc).timestamp()) \
        if start_dt.tzinfo is None else int(start_dt.timestamp())
    end_ts = int(end_dt.replace(tzinfo=timezone.utc).timestamp()) \
        if end_dt.tzinfo is None else int(end_dt.timestamp())

    # 打开各月分片句柄（仅对未完成且需写入的月份）。
    handles: dict[tuple[int, int], Any] = {}
    for year, month in _month_iter(start_dt, end_dt):
        if resume and _checkpoint_path(kind, year, month).exists():
            continue
        shard = _shard_path(kind, year, month)
        # 以追加模式打开（resume 场景续写），若无内容则从头开始。
        handles[(year, month)] = shard.open("a", encoding="utf-8")

    if not handles:
        return {"rows": 0, "months_written": 0}

    select_cols = _build_select(actual_cols)
    col_names = [c.strip() for c in select_cols.split(",")]
    created_idx = col_names.index("created_utc")
    q = (
        f"SELECT {select_cols} FROM '{glob}' "
        f"WHERE subreddit='movies' AND created_utc BETWEEN {start_ts} AND {end_ts}"
    )
    cur = con.execute(q)
    total_rows = 0
    months_written: set[tuple[int, int]] = set()
    try:
        while True:
            batch = cur.fetchmany(10000)
            if not batch:
                break
            for row in batch:
                created_utc = row[created_idx]
                if created_utc is None:
                    continue
                # created_utc 是 epoch 秒（int 或 float）。
                ts = int(created_utc) if not isinstance(created_utc, float) \
                    else int(created_utc)
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                key = (dt.year, dt.month)
                fh = handles.get(key)
                if fh is None:
                    continue
                record = {col_names[i]: row[i] for i in range(len(col_names))}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_rows += 1
                months_written.add(key)
    finally:
        for fh in handles.values():
            fh.close()

    for key in months_written:
        _checkpoint_path(kind, key[0], key[1]).touch()

    return {"rows": total_rows, "months_written": len(months_written)}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bulk-fetch r/movies from open-index/arctic via DuckDB hf://.",
    )
    p.add_argument("--source", default="parquet", choices=["parquet"],
                    help="Source backend (only parquet supported).")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (UTC).")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC, exclusive).")
    p.add_argument("--resume", action="store_true",
                   help="Skip per-month shards already marked done.")
    p.add_argument("--setup-only", action="store_true",
                   help="Run schema verify + pushdown measure only, no fetch.")
    p.add_argument("--local-dir", default=None,
                   help="Local dir of pre-downloaded arctic parquet shards "
                        "(fallback when DuckDB hf:// is network-blocked). "
                        "Expects files named comments*.parquet / submissions*.parquet.")
    p.add_argument("--hf-anon-attempt", action="store_true",
                   help="Attempt anonymous hf:// discovery before using --local-dir "
                        "(to record the anonymous-access verdict in evidence).")
    return p.parse_args(argv)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main(argv: list[str] | None = None) -> int:
    """CLI entry. (EN)

    中文：CLI 入口，编排 schema 校验、下推测量与流式抓取。
    """
    args = _parse_args(argv)
    start_dt = _parse_date(args.start)
    end_dt = _parse_date(args.end)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # 匿名访问优先：不设置任何 token，让 hf:// 与 HfApi 以匿名方式访问。
    print(f"[setup] repo={REPO_ID} anonymous_access=True "
          f"(HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'unset'})")

    anon_verdict = "not_attempted"
    api = HfApi()
    con = _connect()

    # 若指定 --hf-anon-attempt，先短暂尝试匿名 hf:// 发现；失败则记录结论。
    # 注意：DuckDB httpfs 连接 huggingface.co 在本网络环境下会挂起（libcurl
    # 无法连通，而 Windows Schannel 可），因此默认走 --local-dir 回退。
    if args.hf_anon_attempt and not args.local_dir:
        try:
            _discover_parquet_glob(api, "comments")
            anon_verdict = "ok"
        except Exception as exc:
            anon_verdict = f"failed: {type(exc).__name__}: {exc}"
            print(f"[anonymous] hf:// access verdict: {anon_verdict}")
            print("[anonymous] DuckDB httpfs cannot reach huggingface.co from "
                  "this network; use --local-dir with pre-downloaded shards.")
            return 2
    elif args.local_dir:
        anon_verdict = ("skipped (DuckDB httpfs network-blocked; see evidence); "
                        "using local parquet fallback")
        print(f"[anonymous] verdict: {anon_verdict}")

    # 步骤 0：发现表路径 + DESCRIBE + 列校验。
    print("[step0] discovering parquet globs + DESCRIBE ...")
    globs: dict[str, str] = {}
    schemas: dict[str, list[tuple[str, str]]] = {}
    col_sets: dict[str, set[str]] = {}
    for kind in ("comments", "submissions"):
        try:
            glob = _discover_parquet_glob(api, kind, local_dir=args.local_dir)
        except Exception as exc:
            print(f"[step0][{kind}] discovery FAILED: {exc}")
            print(f"[anonymous] if the error above is 401/403, a HF token is needed.")
            return 2
        globs[kind] = glob
        one_file = _table_glob_for_describe(glob)
        try:
            cols = _describe_table(con, one_file)
        except duckdb.Error as exc:
            print(f"[step0][{kind}] DESCRIBE FAILED: {exc}")
            print(f"[anonymous] if the error is an HTTP 401/403, a HF token is required "
                  f"to read {REPO_ID}.")
            return 2
        schemas[kind] = cols
        print(f"[step0][{kind}] glob={glob}")
        print(f"[step0][{kind}] columns: {[c[0] for c in cols]}")
        actual_cols = _verify_columns(kind, cols)
        col_sets[kind] = actual_cols
        print(f"[step0][{kind}] column check PASSED")

    # 步骤 0b：写出 schema 到 evidence。
    schema_dump: dict[str, list[dict[str, str]]] = {
        kind: [{"name": n, "type": t} for n, t in cols]
        for kind, cols in schemas.items()
    }
    (EVIDENCE_DIR / "arctic_schema.json").write_text(
        json.dumps(schema_dump, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # 步骤 1：测量谓词下推效率。
    print("[step1] measuring predicate pushdown efficacy ...")
    pushdown: dict[str, dict[str, Any]] = {}
    for kind in ("comments", "submissions"):
        try:
            pushdown[kind] = _measure_pushdown(con, globs[kind], kind)
            print(f"[step1][{kind}] {pushdown[kind]}")
        except Exception as exc:
            pushdown[kind] = {"error": str(exc)}
            print(f"[step1][{kind}] measurement FAILED: {exc}")

    if args.setup_only:
        print("[done] setup-only complete")
        return 0

    # 步骤 2：流式抓取并按月分片写 NDJSON。
    print(f"[step2] streaming r/movies {start_dt.date()}..{end_dt.date()} "
          f"resume={args.resume}")
    t0 = time.time()
    results: dict[str, dict[str, int]] = {}
    for kind in ("comments", "submissions"):
        try:
            res = _stream_table_to_shards(
                con, globs[kind], kind, start_dt, end_dt, args.resume,
                col_sets[kind],
            )
            results[kind] = res
            print(f"[step2][{kind}] {res}")
        except Exception as exc:
            results[kind] = {"error": str(exc)}
            print(f"[step2][{kind}] stream FAILED: {exc}")
    elapsed = time.time() - t0
    print(f"[step2] elapsed={elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
