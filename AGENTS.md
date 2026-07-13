# LLM-Movie — 代理指南

## 项目概述

分析 Reddit 电影讨论数据（2018–2022），按节假日/非节假日/工作日/周末分组对比。
共 12 个分析步骤，每步输出 PNG 图表 + CSV 到 `output/movie/stepN/`。

## 入口

```sh
# 顺序运行全部 12 步（耗时数小时，数据约 8.5 万行）
python -m movie.pipeline

# 只运行指定步骤
python -m movie.pipeline --steps 1 3 5

# 跳过某一步
python -m movie.pipeline --skip 4
```

每个 step 也可独立运行（`import` 后调用 `main(data)`），
但需要全量数据 dict 时需通过 `movie.data_loader.load_all()` 加载。

## 架构

### `movie/` — 主分析包（12 步流水线）

- `movie/config.py` — 路径、常量（`STEP_DIRS`、`AGE_SEGMENTS`）、
  `setup_matplotlib()`、`log()`。
- `movie/data_loader.py` — `load_all()` 加载全部数据并返回 dict。
- `movie/pipeline.py` — CLI 编排器，动态 import step 模块。
- `movie/stepN_*.py` — 每步导出 `main(data: dict = None)`。
  Step 6–12 在 `data is None` 时会内部调用 `load_all()` 作为 fallback。
- `movie/utils/plotting.py` — 共享配色常量（`COLOR_HOLIDAY` 等）和
  `annotate_heatmap()`。
- `movie/utils/text.py` — 共享工具：`tokenize()`、`deduplicate_seekers()`、
  `parse_conv_turn()`。
- `movie/utils/genre_map.py` — 电影类型中文→英文映射（`GENRE_CN_TO_EN`、
  `to_en()`）。

### `data_analyzer/` — 独立分析包

- 有自己的 `pipeline.py`、`config.py`、`step1–4` 模块。
- **`movie/step11_sentiment.py` 跨包依赖它**：
  `from data_analyzer.sentiment import analyze_batch`。
- `data_analyzer/sentiment.py` 依赖外部库 `vaderSentiment` 和 `afinn`
  （VADER + AFINN 混合情感分析）。

### 其他目录

- `src/` — 旧代码/实验脚本，非主流水线入口。
- 根目录散落脚本（`age_segment.py`、`fetch_movie_ids.py` 等）为独立工具脚本。

## 容易遗漏的约定

- **每个 step 文件必须在模块顶层调用 `setup_matplotlib()`**（在任何绘图代码
  之前）。它设置 Agg 后端 + 中文字体。
- **数据目录在 `data/`，输出目录在 `output/movie/`**（都在项目根目录）。
  输出子目录在 `config.py` import 时自动创建。
- **每步的 main() 接收共享的 `data` dict**，由 pipeline.py 一次性加载传入。
- **Step 6–12 内部会 import `load_all()`** 作为独立运行的保底。
- **仓库没有测试。** 验证靠实际运行步骤。
- **类型检查为 basic 模式**，多项诊断已关闭
  （`pyrightconfig.json` 中 `reportMissingImports`、
  `reportGeneralTypeIssues` 等为 `"none"`）。

## 数据模型

- `holiday.csv` — 日期、描述、类型。
- `holiday-workday.csv` — 日期、调休标记。
- `data_all.csv` / `all_holiday_records_v3.csv` — 对话行，含 `conv_id`、
  `is_seeker`、`raw_text`、`proc_text`、`date`、`utc_time` 等字段。
- `movie_info.json` — `{movie_id: {title, genres, year}}`。
  **genres 是中文**，需用 `genre_map.to_en()` 转英文。

## 依赖

- Python 3.10（项目目标版本，无 `pyproject.toml` / `requirements.txt`）。
- `numpy`、`matplotlib`、`wordcloud`、`gensim`、`scikit-learn`（从 step
  文件的 import 推得）。
- `vaderSentiment`、`afinn`（`data_analyzer/sentiment.py` 使用，Step 11 依赖）。

## 代码风格

- 双语文档字符串（英文概要 + 中文细节）。
- 私有辅助函数加 `_` 前缀。
- 图表密集型步骤：以过程式为主，每个分析独立函数。
- 新增共享工具请放在 `movie/utils/` 下（现有：`plotting.py`、`text.py`、
  `genre_map.py`）。
