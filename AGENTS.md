# LLM-Movie — 代理指南

## 项目概述

分析 Reddit 电影讨论数据（2018–2022），按节假日/非节假日/工作日/周末分组对比。
共 12 个分析步骤，每步输出 PNG 图表 + CSV 到 `output/movie/stepN/`。

## 语言
项目中注释使用中文，数据集中对话内容均是英文。

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

### 数据规则说明
数据规则说明：

1. 列表user_id为用户/系统ID，is_seeker=True表示用户，is_seeker=False表示系统回复
2. conv_id表示会话轮次，示例值：t3_rt7fry_4/7  ，前9位：t3_rt7fry表示会话id，最后:4/7表示一共7轮，当前第4轮 ； turn_id 表示会话轮次id ；
3. utc_time 为时间戳，10位，美国当地时间
4. processed 为用户提问/系统回复内容，格式为：['USER', '内容'] 或 ['SYSTEM', '内容'] ， USER/SYSTEM为当前角色；内容中电影名称已经处理过，格式类似：tt0108149 ； raw为用户提问/系统会话原文。
5. 节假日定义在 data/holiday.csv中，从2018-2022年所有的假日，csv内容为：日期、节日名称、节日类型
6. 周末和工作日的调休和补班参考：data/holiday-workday.csv中（文件中内容咱不完整，只有两个示例，后续补充），需要将对应的补班日算为工作日，补休日记为周末。
7. 电影信息在：data/movie_info.json中，里面有电影id对应的 电影基本信息，其中电影类型在genres字段中；电影id与电影名称对应关系文件：data/entity2id.json 
8. 用户观影类型获取方式为：通过用户提问中的会话id（conv_id），获取到会话conv_id相同的系统回复内容，从系统内容中提前电影id，并在 movie_info.json中找到该电影id的 电影类型 ；
9. 统计多轮会话（去重）时，在同一轮次会话中， 用户提问相同时需要排重，不同的用户提问才是为多轮会话；
10. 同一会话中提问平均间隔时间 定义为：同一轮会话中，用户两次提问时间可能相同，只有用户提问时间不相同的才为有效数据，进行统计（两次用户提问时间，均需在本周期内），计算本轮用户有效提问 - 时间间隔的累计时间差值（单位为秒），除以用户有效提问间隔的次数，为本轮提问时间间隔 ；最后对所有会话求平均；
11. 多轮会话平均持续时间定义为：同一会话中，用户第一次（用户第一次用户提问时间需在本周期内）和最后一次提问时间（最后一次用户提问时间无需在本周期内）的间隔，若两次提问时间均一致，则视为同一问题，不计入有效的多轮会话； 若两次提问不一致，计算持续时间（单位秒），并对所有有效会话求平均
12. 单日多会话定义： 单个会话（含对话中所有轮次）中所有提问时间均发生在同一天内，且同一天内出现两次及以上会话；
13. 跨日会话定义：在同一会话中，第一次用户提问和最后一次用户提问不在同一天；
14. 因同一个节假日每年都有，展示各个节假日维度时，需按照各个节假日名称分组，节假日名称只显示前6位即可；共计约20个节假日，图表长度和宽度要足够
15. 以下相似的对比指标，可以合并成一张图显示；不同的指标选取合适的图进行展示 

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
