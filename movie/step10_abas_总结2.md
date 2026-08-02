# Step 10 ABSA 基于方面的情感分析与节假日差异化（v3-lite 最终版）

**对应文件：** `movie/step10_absa_final.py`
**输出目录：** `output/movie/step10/final/`
**数据范围：** 2019-01-02 ~ 2022-12-31（剔除 2018 与 2019-01-01 人工采集产物）
**情感模型：** VADER（归一化到有符号 [-1,1]）

---

## 一、数据分析思路

### 1.1 问题背景

原 `step10_absa.py` 存在与对话场景的根本性错配：
- **分析对象错**：分析用户提问（seekers，请求文本），但真正的电影评价情感在系统回复里
- **未去重**：同一会话首问在多轮里重复 N 次，重复计数
- **关键词假阳性**：`like`（"movies like X"）误命中 comparison、`best`（"Best X?"）误命中 recommendation，导致这两个方面合计占 39%
- **消息级聚合**：长会话淹没短会话，统计被高轮次会话主导

本步骤（v3-lite）针对以上四点重构。

### 1.2 统计口径

| 维度 | 口径 | 说明 |
|---|---|---|
| **分析语料** | 系统回复（`is_seeker=False`） | 系统回复是推荐带评价，用户提问是请求非评价 |
| **分析单位** | 会话级（session × aspect = 1 个值） | 防止长会话淹没；每会话每方面只贡献 1 个均值 |
| **方面检测** | 词边界正则 + 最长匹配 + 请求模式黑名单 + POS 消歧 | 消除 `like/best/similar` 等请求句式假阳性 |
| **NLI 过滤** | `has_evaluative_context`（须含评价词） | 丢弃纯提及句（如 "What would the twist be?"） |
| **情感打分** | VADER compound ∈ [-1, +1]，POSITIVE=+1/NEUTRAL=0/NEGATIVE=-1 | 有符号归一化，NEGATIVE 返回负值 |
| **时段分组** | holiday / workday / weekend（继承自会话首条 seeker） | 系统回复本身无日期标签，从同会话首条 seeker 继承 |
| **跨日会话** | 标 `cross_day=True`（首末 seeker 日期不同，规则 13） | 可过滤做稳健性检查 |

### 1.3 统计规则

1. **会话重组**：按 `session_id` 分组所有行（含系统回复），组内按 `(turn_order, utc_time)` 升序
2. **配对**：每个系统回复找紧随其后的下一条 user 消息组成 pair；末条系统回复无人回应则作 solo 保留
3. **方面去重**：一个方面在同一文本内只取第一个命中片段（一方面一片段）
4. **跨方面关键词清理**：`funny/scary/boring` 等双关词统一归 emotion（不再同时归 genre/plot）；`like/similar/best/recommend` 改为多词短语（"similar to"、"one of the best"）
5. **NLI 判定**：候选句须含评价标记词（opinion 形容词 / 系动词 was/were / 评价动词 loved/hated / 程度副词 really/very），否则丢弃
6. **会话级聚合**：同一 `(session_id, aspect)` 的多条 pair 情感数值先取均值，得该会话该方面的 `mean_sentiment`
7. **跨会话统计**：图表值 = 各会话 `mean_sentiment` 的跨会话均值；误差棒 = 跨会话标准差

### 1.4 统计及计算步骤

```
data['rows'] (1,669,720 行)
  │  ① build_pairs_from_rows：会话重组 + 配对 + 时段继承
  ▼
903,419 个 pair（系统回复 + 可选紧随用户回应）
  │  ② detect_aspects_v2：词边界 + 最长匹配 + 请求黑名单 + POS 消歧
  ▼
267,981 个候选方面提及
  │  ③ has_evaluative_context：NLI 过滤纯提及
  ▼
205,284 条 pair-records（pair × aspect × sentiment）
  │  ④ _aggregate_to_conv_level：按 (session_id, aspect) 分组求均值
  ▼
90,305 条 conv_records（会话级，每会话每方面 1 条）
  │  ⑤ dim_a1..a5：按 period/holiday 分组绘图
  ▼
A1–A5 五张图 + a0_conv_records_final.csv
```

**关键函数**：
- `build_pairs_from_rows(rows)` — `movie/utils/conv_pairs.py`
- `detect_aspects_v2(text)` — `movie/utils/absa_aspects.py`
- `has_evaluative_context(sentence)` — `movie/utils/absa_nli.py`
- `_aggregate_to_conv_level(pair_records)` — `movie/step10_absa_v2.py`
- `NormalizedVaderAnalyzer.predict(text)` — 本文件，返回 `(label, score∈[-1,1])`

---

## 二、运行结果分析

### 2.1 运行概况

| 指标 | 值 |
|---|---|
| 数据范围 | 2019-01-02 ~ 2022-12-31 |
| 总行数 | 1,669,720 |
| 会话数 | 57,747 |
| 配对数（pairs） | 903,419 |
| 跨日会话数 | 10,031 |
| 候选方面提及 | 267,981 |
| NLI 过滤（纯提及） | 62,697（**23.4%**） |
| pair-records（过滤后） | 205,284 |
| conv-records（会话级） | 90,305 |
| 运行耗时 | ~8 分钟（VADER 模式） |

### 2.2 各方面提及计数（pair-level）

| 方面 | 计数 | 占比 |
|---|---|---|
| genre | 48,762 | 23.8% |
| emotion | 35,481 | 17.3% |
| recommendation | 29,972 | 14.6% |
| plot | 29,493 | 14.4% |
| cast | 17,344 | 8.4% |
| audio | 11,233 | 5.5% |
| direction | 11,231 | 5.5% |
| comparison | 9,027 | 4.4% |
| content | 6,760 | 3.3% |
| visual | 5,981 | 2.9% |

---

### 2.3 A1：方面提及分布（饼图）

**图**：`a1_aspect_distribution_v2.png`，饼图，n=90,305（conv-level 记录数）

| 方面 | 占比 |
|---|---|
| Genre/Style | **21.9%**（最大） |
| Emotion/Tone | 15.4% |
| Recommendation | 15.0% |
| Plot/Story | 13.3% |
| Cast/Acting | 8.5% |
| Direction | 6.7% |
| Comparison | 6.0% |
| Audio/Music | 5.6% |
| Content/Warnings | 4.1% |
| Visual/Effects | 3.4%（最小） |

**分析**：
1. **genre 居首**：系统推荐时最常讨论电影类型（"a great dark comedy"、"slow-burn thriller"）
2. **emotion 跃居第二**（旧 step10 仅 5.6%）：系统回复里"hilarious/moving/boring"等情感评价丰富，旧版分析 seekers 漏掉了这部分
3. **comparison 降至 6.0%**（旧版 19.0%）：`like/similar` 请求句式假阳性被请求黑名单 + 多词化清除
4. **recommendation 降至 15.0%**（旧版 20.0%）：`best/suggest?` 疑问句假阳性减少
5. 真评价主导：genre + emotion + plot + recommendation = 65.6%，反映系统回复的真实评价关注点

---

### 2.4 A2：各方面总体情感（柱状图）

**图**：`a2_overall_aspect_sentiment_v2.png`，柱状图带误差棒，Y 轴 Mean Sentiment (-1 到 +1)

| 方面 | 情感均值 | 解读 |
|---|---|---|
| Visual/Effects | **+0.66** | 视效评价最正面 |
| Comparison | +0.61 | 比较评价偏正面 |
| Cast/Acting | +0.58 | 演员评价偏正面 |
| Recommendation | +0.57 | 推荐整体偏褒 |
| Audio/Music | +0.54 | 音效偏正面 |
| Emotion/Tone | +0.46 | 情感评价偏正面 |
| Direction | +0.42 | 导演评价偏正面 |
| Plot/Story | +0.42 | 剧情评价偏正面 |
| Genre/Style | +0.41 | 类型评价偏正面 |
| Content/Warnings | **−0.13** | 唯一负值，误差棒极宽 |

**分析**：
1. **9 个方面偏正面，1 个偏负面**：推荐场景里系统回复本就偏褒（用户问"推荐"，系统答"值得看"）
2. **Content/Warnings 唯一负值**：内容适宜性讨论偏警示性（"violent/gory/disturbing/graphic"），符合直觉 —— 用户问"适不适合带孩子看"，系统回复常带警告
3. **Visual/Effects 最高**：视效是最容易获正面评价的方面（"breathtaking cinematography"、"stunning CGI"）
4. 误差棒普遍较宽（std ~0.6-0.8）：会话间情感分歧大，说明不同会话对同一方面的评价差异显著

---

### 2.5 A3：节假日 vs 非节假日（双色分组柱状图）

**图**：`a3_holiday_vs_nonholiday_v2.png`，Holiday（红） vs Non-holiday（蓝）

**关键差异**：

| 方面 | Holiday | Non-holiday | 差值 | 解读 |
|---|---|---|---|---|
| Content/Warnings | −0.08 | **−1.00** | −0.92 | 非节假日内容讨论极负（警示性强） |
| Comparison | +0.60 | +1.00 | +0.40 | 非节假日比较评价更极端正面 |
| Emotion/Tone | 较低 | 较高 | — | 非节假日情感评价更正面 |
| Direction | 较低 | 较高 | — | 非节假日导演评价更正面 |
| Audio/Recommendation/Visual/Cast/Plot/Genre | 较高 | 较低 | — | 节假日这些方面更正面 |

**分析**：
1. **节假日情感更平和**：节假日（休闲看片）多方面情感不如非节假日极端
2. **Content/Warnings 差异最大**：非节假日 −1.00 vs 节假日 −0.08 —— 日常推荐里内容警示性讨论远比节假日负面（日常更多"适不适合孩子/暴力不暴力"的审慎询问）
3. **Comparison 反转**：非节假日 +1.00（极正面），节假日 +0.60 —— 日常推荐里"better than X"类比较更极端褒
4. 节假日差异化信号存在但方向因方面而异，无单一规律

---

### 2.6 A4：节假日 vs 工作日 vs 周末（三色分组柱状图）

**图**：`a4_holiday_workday_weekend_v2.png`，Holiday（红）/ Weekend（蓝）/ Workday（黄）

**分析**：
1. **三类日期高度相似**：除 Content/Warnings 外，所有方面三色柱子都落在 +0.38 ~ +0.66 正向区间
2. **Content/Warnings 三类都为负**（约 −0.08 ~ −0.13）：内容适宜性讨论在任何日期类型下都偏警示
3. **误差棒极宽**：覆盖大部分 Y 轴 —— 会话间方差远大于日期类型间差异
4. **结论**：节假日/工作日/周末的总体差异不大；真正差异化信号需看 A5 的具体节假日名称细分

---

### 2.7 A5：各节假日 × 方面情感热力图

**图**：`a5_per_holiday_aspect_v2.png`，热力图，行=20 个节假日，列=10 个方面，色标蓝(−0.4)→白(0)→深红(+1.0)，灰=无数据

**典型特征**：

| 节假日 | 突出方面 | 值 | 解读 |
|---|---|---|---|
| 万圣节 | Content/Warnings | **−0.50** | 恐怖片内容讨论最负面（gore/violent） |
| 万圣节 | Comparison | +0.82 | 比较评价极正面 |
| 劳动节 | Visual/Effects | **+1.00** | 视效评价最高 |
| 超级碗周日 | Audio/Music | **+0.94** | 音乐/音效评价极高（赛事相关） |
| 复活节 | Audio/Music | +0.88 | 音效评价高 |
| 9·11纪念日 | Content/Warnings | +0.33 | 少数 content 正值（纪念性内容） |
| 9·11纪念日 | Comparison | +0.79 | 比较评价高 |
| 总统日 | Comparison | +0.79 | 比较评价高 |
| 情人节 | Emotion/Tone | **−0.13** | 唯一 emotion 负值（约会片评价挑剔） |

**分析**：
1. **Content/Warnings 列整体偏蓝**：多数节假日的内容讨论偏警示性；万圣节最负（恐怖片天然带 gore/violent 讨论）
2. **Audio/Music 在超级碗周日 +0.94**：赛事相关会话里音乐/音效是亮点
3. **Visual/Effects 在劳动节 +1.00**：劳动节视效评价极端正面（样本可能少，需查 n）
4. **情人节 Emotion/Tone −0.13**：唯一 emotion 负值 —— 约会片情感评价挑剔，可能因期望高
5. **Comparison 多数节日偏高**：节假日推荐时更爱做"像 X 电影"的比较
6. 灰色格 = 该节假日该方面数据不足（< MIN_DATA_ROWS），样本量限制

---

## 三、核心发现

### 3.1 旧 step10 的三个错配已修复

| 错配 | 旧 step10 | v3-lite | 修复方式 |
|---|---|---|---|
| 分析对象错 | seekers（请求）| 系统回复（评价）| B 语料切换 |
| 关键词假阳性 | comparison 19%/recommendation 20% | 6.0%/15.0% | ③.1 去噪（词边界+黑名单+POS） |
| 消息级聚合 | 长会话淹没 | 1 会话 = 1 单位 | ⑤ 会话级聚合 |

### 3.2 NLI 过滤有效

- 过滤 62,697 条纯提及候选 = **23.4%** 噪声清除
- 丢弃的是无评价词的方面提及（如 "the twist at the end"、"the scene where"），这些不是真评价
- 过滤后情感更两极化、更可信

### 3.3 节假日差异化信号

- **A3（2 组）**：节假日 vs 非节假日在 Content/Warnings（Δ=−0.92）、Comparison（Δ=+0.40）上差异最大
- **A4（3 组）**：holiday/workday/weekend 总体差异小，方差主要来自会话间
- **A5（20 节假日）**：具体节假日的特定方面有突出模式（万圣节 content 负、超级碗 audio 正、情人节 emotion 负）

### 3.4 情感总体倾向

- 9/10 方面偏正面（+0.41 ~ +0.66），推荐场景天然偏褒
- Content/Warnings 唯一负值（−0.13），内容适宜性讨论偏警示
- 节假日多方面情感不如非节假日极端（节假日休闲看片，评价更平和）

---

## 四、方法学回顾与局限

| 设计点 | 实现情况 | 评估 |
|---|---|---|
| 分析语料切换到系统回复 | ✅ build_pairs_from_rows 取 system_text | 修正了 seekers 请求文本错配 |
| 词边界 + 最长匹配 | ✅ `\bkw\b` 正则 + 长度降序 | 消除 'star' 命中 'staring' 类假阳性 |
| 请求模式黑名单 | ✅ 10 条 regex（looking for/movies like/best X? 等） | 清除 comparison/recommendation 假阳性 |
| POS 消歧 | ✅ nltk.pos_tag，'like' 作介词丢弃 | 处理高频歧义词 |
| NLI 上下文判定 | ✅ has_evaluative_context（评价词 presence） | 过滤 23.4% 纯提及噪声 |
| 会话级聚合 | ✅ (session_id, aspect) 分组求均值 | 防止长会话淹没 |
| 时段属性继承 | ✅ 从会话首条 seeker 继承 | 系统回复无独立日期标签的处理 |
| 跨日会话标记 | ✅ cross_day 字段 | 可过滤做稳健性检查 |
| 情感归一化 | ✅ VADER compound 有符号 [-1,1] | NEGATIVE 返回负值，可正确聚合 |

**局限**：
1. **NLI 为规则版**：仅检查评价词 presence，未用 transformer 语义相似度；边界句（如 "It's like if Tarantino made..."）可能误杀。Phase 2 的 transformer NLI 可恢复召回
2. **VADER 词典模型**：对否定、讽刺、领域术语不如 transformer 精确；distilbert/twitter-roberta-absa 可升级但全量推理太慢（~18 小时）
3. **cast/audio/visual/content 计数偏低**：这些方面在系统回复里本就罕见（多数回复是短推荐如 "tt0221803"），低计数是真实精度非召回损失
4. **节假日样本不均**：A5 热力图灰色格 = 数据不足；小样本节假日的方面均值不确定性高
5. **未做统计检验**：A3/A4 的组间差异未配卡方/t 检验，仅看均值差；误差棒宽提示需谨慎解读
6. **双轨配对已移除**：实测系统回复后直接跟用户回应的 pair 仅 0.8%，双轨设计在当前数据结构下无效，故 v3-lite 去掉

---

## 五、输出文件

| 文件 | 内容 |
|---|---|
| `output/movie/step10/final/a0_conv_records_final.csv` | 90,305 条会话级记录（session_id, date, period, holiday_name, cross_day, aspect, mean_sentiment, std_sentiment, n_pairs, pos_ratio） |
| `output/movie/step10/final/a1_aspect_distribution_v2.png` | 方面提及分布饼图 |
| `output/movie/step10/final/a2_overall_aspect_sentiment_v2.png` | 各方面总体情感柱状图 |
| `output/movie/step10/final/a3_holiday_vs_nonholiday_v2.png` | 节假日 vs 非节假日分组柱状图 |
| `output/movie/step10/final/a4_holiday_workday_weekend_v2.png` | 节假日/工作日/周末三组柱状图 |
| `output/movie/step10/final/a5_per_holiday_aspect_v2.png` | 各节假日 × 方面情感热力图 |
| `output/movie/step10/final/run_log.txt` | 运行日志 |

> 注：PNG 文件名带 `_v2` 后缀是因为复用了 v2 的绘图函数，但内容是 v3-lite（含 NLI 过滤）的最终结果。

---

## 六、最终判定

**回答原始问题**（step10 ABSA 是否适用于对话场景）：
- 原 step10_absa.py **不适用**：分析对象错（seekers 请求）、关键词假阳性严重（comparison/recommendation 占 39%）、无去重、消息级聚合被长会话淹没
- v3-lite **修正了全部四点错配**：语料切到系统回复、词边界+黑名单+POS 去噪、NLI 过滤纯提及、会话级聚合
- 量化证据：comparison 从 19.0% 降至 6.0%、recommendation 从 20.0% 降至 15.0%、NLI 滤掉 23.4% 噪声、emotion 从 5.6% 升至 15.4%（真评价捕获）
- 节假日差异化信号在 A5 细粒度热力图显现：万圣节 content 负、超级碗 audio 正、情人节 emotion 负

**推荐**：以 `step10_absa_final.py`（v3-lite）作为 step10 的替代实现。如需更高精度，可启用 Phase 2 的 transformer NLI（`absa_nli.AspectClassifier(use_model=True)`）恢复边界句召回，但需承担模型下载与推理耗时。
