# 方面级情感分析（ABSA Final v3-lite）说明文档

**脚本文件:** `movie/step10_absa_final.py`
**创建日期:** 2026/7/26
**输出目录:** `output/movie/step10/final/`

---

## 一、分析背景与问题

本步骤是 Step 10（基于方面的情感分析，ABSA）的**最终推荐版本**（v3-lite），在前序版本（v1 → v2）的基础上做了三项核心改造：

| 版本 | 脚本 | 分析语料 | 去噪 | 聚合粒度 | 情感分析器 |
|---|---|---|---|---|---|
| v1 | `step10_absa.py` | 用户提问（seekers） | 简单关键词匹配 | 记录级 | distilbert→VADER→规则 三级降级 |
| v2 | `step10_absa_v2.py` | 系统回复（system） | detect_aspects_v2 去噪 | 会话级 | 同 v1 |
| **v3-lite** | **`step10_absa_final.py`** | **系统回复** | **detect_aspects_v2 + NLI 过滤** | **会话级** | **归一化 VADER（有符号 [-1,1]）** |

**v3-lite 相比 v2 的增量改动：**

1. **新增 NLI 过滤**（`has_evaluative_context`）：丢弃纯提及句（无评价词的方面候选），提升精度
2. **去掉双轨配对**（v3 实验性功能）：实测覆盖率仅 0.8%，性价比低，移除
3. **分析器统一为归一化 VADER**：NEGATIVE 返回负值（有符号分数），便于直接计算均值

---

## 二、数据分析思路

### 2.1 统计口径

| 维度         | 口径                                                         | 说明                                          |
| ------------ | ------------------------------------------------------------ | --------------------------------------------- |
| **分析语料** | 系统回复（`is_seeker=False`）                                | 系统回复是推荐带评价，用户提问是请求非评价    |
| **分析单位** | 会话级（session × aspect = 1 个值）                          | 防止长会话淹没；每会话每方面只贡献 1 个均值   |
| **方面检测** | 词边界正则 + 最长匹配 + 请求模式黑名单 + POS 消歧            | 消除 `like/best/similar` 等请求句式假阳性     |
| **NLI 过滤** | `has_evaluative_context`（须含评价词）                       | 丢弃纯提及句（如 "What would the twist be?"） |
| **情感打分** | VADER compound ∈ [-1, +1]，POSITIVE=+1/NEUTRAL=0/NEGATIVE=-1 | 有符号归一化，NEGATIVE 返回负值               |
| **分组维度** | period ∈ {holiday, workday, weekend}；节假日名称（约 20 个） |                                               |
| **聚合粒度** | 1 会话 × 1 方面 = 1 条 conv 级记录                           | 取该会话内同方面多次检测的均值                |
| **跨日会话** | 标 `cross_day=True`（首末 seeker 日期不同）                  | 可过滤做稳健性检查                            |

### 2.2 统计规则

#### 10 个评价方面（ASPECTS_V2）

| 方面 | 英文标签 | 关键词示例 |
|---|---|---|
| genre | Genre/Style | comedy, horror, sci-fi, thriller, fantasy... |
| plot | Plot/Story | plot, story, twist, ending, screenplay, well-written... |
| cast | Cast/Acting | actor, performance, acting, starred, portrayal... |
| visual | Visual/Effects | cinematography, cgi, visual effects, beautifully shot... |
| audio | Audio/Music | soundtrack, score, sound design, composer, music... |
| direction | Direction | director, pacing, atmosphere, creative vision... |
| emotion | Emotion/Tone | funny, scary, moving, boring, thrilling, suspenseful... |
| recommendation | Recommendation | must-watch, worth watching, masterpiece, underrated... |
| comparison | Comparison | similar to, better than, cross between, reminds me of... |
| content | Content/Warnings | violent, gore, nudity, family-friendly, r-rated... |

#### ③.1 方面检测去噪（detect_aspects_v2）

1. **断句**：按 `. ! ?` 和换行切分句子
2. **请求句黑名单**：征询句式（如 "looking for movies", "can you recommend"）整句丢弃
3. **POS 消歧**：高频歧义词 `like` 作介词（IN）时丢弃（"movies like X"）
4. **词边界正则**：用 `\b` 匹配避免子串误命中
5. **最长匹配**：每个方面的关键词按长度降序排列，优先匹配多词短语
6. **方面去重**：一方面在一段文本中仅取首个命中片段（一方面一片段）

#### ③.2 NLI 评价过滤（has_evaluative_context）

在方面检测后，对每个候选片段进一步过滤——**丢弃不含评价标记的纯提及句**：

判定为"评价句"的条件（满足任一）：
- 句中含**评价形容词**（good, great, boring, stunning... 共 ~70 个）
- 句中含**系动词**（was, were, is, felt, seemed...）
- 句中含**评价动词**（loved, hated, enjoyed, delivered, nailed...）
- 句中含**程度副词**（really, very, incredibly...）
- 含**否定 + 评价形容词**组合（"not bad"）

示例：`"What would the twist be?"` → 含 "twist" 命中 plot 方面，但无评价词 → **NLI 过滤丢弃**

#### ⑤ 会话级聚合

对每个 `(session_id, aspect)` 组合：
- 将该会话内同方面多次检测的 `mean_sentiment` 取均值
- 记录：`mean_sentiment`、`std_sentiment`、`n_pairs`（配对次数）、`pos_ratio`（正面占比）

---

### 2.3 统计及计算步骤

#### 步骤 1：会话重组与配对

```python
pairs = build_pairs_from_rows(rows)
```

- 按 `session_id` 重组全部 162 万行数据
- 组内按 `(turn_order, utc_time)` 排序
- 每个系统回复配对其后紧随的用户消息
- 时段/节假日属性从该会话**首条 seeker 行**继承
- 标记跨日会话（首末 seeker 日期不同 → `cross_day=True`）

#### 步骤 2：方面检测 + NLI 过滤 + 情感分析

```python
for p in pairs:
    candidates = detect_aspects_v2(p['system_text'])     # ③.1 方面检测+去噪
    for c in candidates:
        if not has_evaluative_context(c['snippet']):     # ③.2 NLI 过滤
            continue
        lab, sc = analyzer.predict(c['snippet'])          # VADER 情感分析
        pair_records.append({...})
```

对每条配对的系统回复文本：
1. 断句 → 丢弃请求句 → 关键词匹配 → 方面去重（detect_aspects_v2）
2. NLI 过滤：候选片段不含评价标记 → 丢弃
3. VADER 情感分析：compound 分数 → POSITIVE/NEGATIVE/NEUTRAL + 有符号分数

#### 步骤 3：会话级聚合

```python
conv_records = v2._aggregate_to_conv_level(pair_records)
```

按 `(session_id, aspect)` 分组，对组内多条记录：
- `mean_sentiment` = 情感数值的均值
- `pos_ratio` = 正面记录占比

#### 步骤 4：A1–A5 可视化分析

复用 v2 的 dim 函数，输出到 `final/` 子目录：

| 维度 | 函数 | 图表类型 | 说明 |
|---|---|---|---|
| A1 | `dim_a1` | 饼图 | 方面提及分布（会话级，各 aspect 占比） |
| A2 | `dim_a2` | 柱状图 | 各方面总体情感均值（不分组） |
| A3 | `dim_a3` | 分组柱状图 | 节假日 vs 非节假日各方面情感 |
| A4 | `dim_a4` | 分组柱状图 | 节假日 vs 工作日 vs 周末 |
| A5 | `dim_a5` | 热力图 | 各节假日 × 各方面情感矩阵 |

---

## 三、运行结果分析

### 3.1 运行概况

| 指标 | 值 |
|---|---|
| 数据范围 | 2019–2022，全量数据 |
| 总会话数 | **57,747** |
| 总配对数（system→user） | **903,419** |
| 跨日会话数 | 10,031（17.7%） |
| NLI 过滤前候选记录 | 267,981 |
| NLI 过滤丢弃（纯提及） | **62,697（23.4%）** |
| 过滤后有效记录（pair 级） | **205,284** |
| 会话级聚合后记录（conv 级） | **90,305** |
| 情感分析器 | VADER（归一化，有符号 [-1,1]） |
| 运行耗时 | ~9 分钟（15:33:01 → 15:41:53） |

### 3.2 方面提及分布（A1）

| 排名 | 方面 | 候选数（pair 级） | 占比 |
|---|---|---|---|
| 1 | genre | 48,762 | 23.8% |
| 2 | emotion | 35,481 | 17.3% |
| 3 | recommendation | 29,972 | 14.6% |
| 4 | plot | 29,493 | 14.4% |
| 5 | cast | 17,344 | 8.5% |
| 6 | audio | 11,233 | 5.5% |
| 7 | direction | 11,231 | 5.5% |
| 8 | comparison | 9,027 | 4.4% |
| 9 | content | 6,760 | 3.3% |
| 10 | visual | 5,981 | 2.9% |

**观察**：genre 和 emotion 合计占 41%，是系统回复中最常被评价的方面；visual 最少（系统回复较少讨论视效细节）。

### 3.3 各方面总体情感（A2）

> 系统回复整体呈现**正面倾向**，10 个方面中 9 个均值为正，仅 content（内容警示）为负

| 方面 | 均值 | 标准差 | 样本数 | 正面占比 |
|---|---|---|---|---|
| **visual** | **+0.656** | 0.645 | 3,069 | 79.1% |
| comparison | +0.613 | 0.673 | 5,430 | 75.4% |
| cast | +0.579 | 0.687 | 7,682 | 74.7% |
| recommendation | +0.573 | 0.615 | 13,577 | 72.1% |
| audio | +0.557 | 0.677 | 5,100 | 72.6% |
| emotion | +0.457 | 0.762 | 13,931 | 68.9% |
| direction | +0.422 | 0.739 | 6,073 | 62.9% |
| plot | +0.419 | 0.748 | 12,005 | 65.1% |
| genre | +0.410 | 0.746 | 19,748 | 65.1% |
| **content** | **−0.127** | 0.873 | 3,690 | 35.8% |

**关键发现**：
1. **视效（visual）最正面**（+0.656，79% 正面）：系统推荐电影时倾向于正面评价画面质量
2. **内容警示（content）唯一为负**（−0.127）：系统在讨论暴力/分级等内容时更多使用负面语气（"very violent", "disturbing"）
3. **genre 情感偏低**（+0.410）：类型讨论中包含较多中性描述和负面评价
4. 所有方面的标准差都较大（0.6–0.9），说明情感分布分散，并非一边倒

### 3.4 节假日 vs 非节假日（A3）

> 节假日与非节假日的方面情感**差异极小**，Δ 均在 ±0.06 以内

| 方面 | 节假日 | 非节假日 | Δ | 方向 |
|---|---|---|---|---|
| content | −0.073 | −0.129 | **+0.057** | 节假日更正面 ↑ |
| direction | +0.457 | +0.420 | +0.038 | ↑ |
| recommendation | +0.597 | +0.571 | +0.026 | ↑ |
| plot | +0.443 | +0.418 | +0.025 | ↑ |
| audio | +0.580 | +0.555 | +0.025 | ↑ |
| visual | +0.654 | +0.656 | −0.002 | ≈ |
| comparison | +0.607 | +0.613 | −0.006 | ≈ |
| emotion | +0.439 | +0.458 | −0.018 | ↓ |
| genre | +0.379 | +0.412 | −0.032 | 节假日更负面 ↓ |
| cast | +0.525 | +0.582 | **−0.057** | ↓ |

**关键观察**：
1. **cast（演员）是节假日降幅最大的方面**（Δ=−0.057）：节假日系统对演员评价略低于平时
2. **content（内容警示）是节假日升幅最大的方面**（Δ=+0.057）：节假日讨论暴力/分级内容时语气略缓
3. 整体来看，节假日效应**非常微弱**，各方面 Δ 都在 ±0.06 以内，无统计检验意义
4. 节假日样本仅 4,802 条（占 5.3%），非节假日 85,503 条，样本量悬殊

### 3.5 节假日 vs 工作日 vs 周末（A4）

| 方面 | Holiday (n=4,802) | Workday (n=59,137) | Weekend (n=26,300) |
|---|---|---|---|
| visual | +0.654 | +0.660 | +0.650 |
| comparison | +0.607 | +0.605 | +0.631 |
| cast | +0.525 | +0.582 | +0.583 |
| recommendation | +0.597 | +0.566 | +0.584 |
| audio | +0.580 | +0.552 | +0.564 |
| emotion | +0.439 | +0.459 | +0.455 |
| direction | +0.457 | +0.417 | +0.425 |
| plot | +0.443 | +0.420 | +0.415 |
| genre | +0.379 | +0.410 | +0.415 |
| content | −0.073 | −0.131 | −0.123 |

**观察**：
- **周末的 comparison 略高**（+0.631 vs workday +0.605）：周末系统更多做电影间正面比较
- **节假日的 cast 略低**（+0.525 vs workday +0.582）：节假日演员评价略降
- 三组之间差异极小，系统回复的情感模式不随时段显著变化

### 3.6 各节假日方面情感热力图（A5）

20 个有足够数据的节假日（≥30 条记录），以下选取代表性节假日：

| 节假日 | genre | plot | cast | visual | audio | direction | emotion | recommend | comparison | content | 总记录 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 超级碗周日 | +0.574 | +0.631 | +0.680 | +0.750 | **+0.944** | +0.385 | +0.336 | +0.775 | +0.700 | +0.111 | 185 |
| 情人节 | +0.324 | +0.120 | +0.661 | +0.417 | +0.500 | **−0.134** | +0.522 | **+0.770** | +0.635 | +0.500 | 195 |
| 万圣节 | +0.161 | +0.383 | +0.440 | +0.385 | +0.750 | +0.485 | +0.294 | +0.485 | +0.824 | **−0.495** | 278 |
| 感恩节 | +0.418 | +0.247 | +0.528 | +0.812 | +0.232 | +0.395 | +0.417 | +0.628 | +0.583 | **−0.367** | 211 |
| 9·11纪念日 | +0.280 | +0.555 | +0.655 | **+0.875** | +0.667 | +0.450 | +0.454 | +0.486 | +0.792 | **+0.325** | 256 |
| 元旦 | +0.344 | +0.577 | +0.391 | +0.556 | +0.792 | **+0.821** | +0.490 | +0.745 | +0.500 | −0.075 | 186 |
| 圣诞节 | +0.421 | +0.329 | +0.622 | +0.608 | +0.646 | +0.294 | +0.418 | +0.685 | +0.600 | −0.090 | 315 |
| 独立日 | +0.308 | +0.451 | +0.493 | +0.778 | +0.655 | +0.490 | +0.259 | +0.527 | +0.588 | −0.075 | 225 |
| 劳动节 | +0.273 | +0.442 | +0.540 | +1.000 | +0.458 | **+0.852** | +0.348 | +0.617 | +0.462 | +0.250 | 247 |
| 圣帕特里克节 | +0.405 | +0.475 | +0.417 | +0.722 | +0.583 | +0.600 | **+0.699** | +0.582 | +0.750 | +0.476 | 186 |

**关键发现**：
1. **超级碗周日的 audio 极高**（+0.944）：超级碗以音乐/中场秀闻名，系统回复中音频方面评价极正面
2. **情人节的 direction 唯一为负**（−0.134）：情人节推荐的影片导演评价偏负面
3. **万圣节和感恩节的 content 最负**（−0.495, −0.367）：万圣节影片多含暴力/恐怖元素，感恩节家庭片含较多内容警示
4. **9·11纪念日 content 为正**（+0.325）：9·11相关影片讨论中内容警示更多是正面表述（"not gratuitous"）
5. **劳动节 direction 极高**（+0.852）但样本中 audio 偏低（+0.458），模式独特
6. **Visual/Effects 在劳动节 +1.00**：劳动节视效评价极端正面
7. **Comparison 多数节日偏高**：节假日推荐时更爱做"像 X 电影"的比较
8. 每个节假日样本量 185–315 条，个别方面单元格仅 10–30 条，不确定性较高

### 3.7 NLI 过滤效果

| 指标 | 值 |
|---|---|
| 过滤前候选总数 | 267,981 |
| NLI 丢弃（纯提及句） | **62,697** |
| 过滤率 | **23.4%** |
| 过滤后有效记录 | 205,284 |

NLI 过滤丢弃了近四分之一的候选，这些是"方面关键词命中但无评价词"的纯提及句（如 "What kind of horror movies do you have?"），有效提升了精度。



---

## 四、解释与结论

1. **系统回复整体正面**：推荐系统在回复中倾向于正面评价电影（9/10 方面均值为正），这是推荐系统的本质——它不会在回复中贬低推荐的电影。
2. **节假日效应极弱**：节假日与非节假日的方面情感差异在 ±0.06 以内，系统回复的情感模式不随节假日变化。这与 step7 的发现一致——推荐系统对日期不敏感。
3. **content 是唯一负面方面**：讨论暴力/分级/敏感内容时系统自然使用负面语气（"violent", "disturbing", "graphic"），但这些词在 NLI 过滤后仍被保留，因为它们是评价性表述。
4. **NLI 过滤有效**：23.4% 的候选被丢弃，这些是纯征询句（"any good horror movies?"），过滤后剩余的是真正的评价性表述。
5. **节假日特异性**：超级碗的 audio 极高、万圣节的 content 极低等模式符合直觉——不同节假日的电影主题差异反映在系统回复的情感分布上。

---

## 五、方法学回顾与局限

| 设计点 | 实现情况 | 评估 |
|---|---|---|
| 语料从用户提问切换为系统回复 | ✅ | 分析推荐系统自身的情感表达，而非用户情感 |
| 三级去噪（黑名单+POS+NLI） | ✅ | 23.4% 噪声被过滤，精度提升 |
| 会话级聚合 | ✅ | 避免同一会话内多次提及重复计数 |
| 有符号 VADER 分数 [-1,1] | ✅ | 均值可直接计算且方向明确 |
| 时段继承自首条 seeker | ✅ | 统一会话的时段归属 |
| 跨日会话标记 | ✅ | 10,031 条（17.7%）被标记 |

**局限**：
- NLI 过滤为规则版（评价词表），可能遗漏含蓄评价句或误过滤无评价词但语义为评价的句子
- VADER 对领域适应性有限（电影评论 vs 社交媒体）
- 节假日样本量小（185–315 条/方面），热力图单元格不确定性高
- 节假日 vs 非节假日差异经 Welch's t 检验确认均不显著（全部 p > 0.05，Cohen's d < 0.1），详见第七节
- 双轨配对（system→user 反馈配对）已移除，无法分析"用户对特定推荐的反馈情感"

---

## 六、输出文件清单

| 文件 | 说明 |
|---|---|
| `output/movie/step10/final/a0_conv_records_final.csv` | 会话级聚合记录（90,305 行，含 session_id/date/period/aspect/mean_sentiment 等） |
| `output/movie/step10/final/a1_aspect_distribution_v2.png` | A1 方面提及分布饼图 |
| `output/movie/step10/final/a2_overall_aspect_sentiment_v2.png` | A2 各方面总体情感柱状图 |
| `output/movie/step10/final/a3_holiday_vs_nonholiday_v2.png` | A3 节假日 vs 非节假日分组柱状图 |
| `output/movie/step10/final/a4_holiday_workday_weekend_v2.png` | A4 节假日/工作日/周末三组柱状图 |
| `output/movie/step10/final/a5_per_holiday_aspect_v2.png` | A5 各节假日 × 各方面情感热力图（20 节假日 × 10 方面） |
| `output/movie/step10/final/run_log.txt` | 运行日志（含会话数、配对数、NLI 过滤率等） |

---

## 七、统计检验结果（节假日效应显著性验证）

### 7.1 误差线说明：std vs SEM

原始 A2–A4 图表中的误差线使用的是**标准差（std）**，而非**标准误（SEM）**。两者含义完全不同：

| 指标 | 含义 | 数值范围 | 解读 |
|---|---|---|---|
| **std**（原始） | 个体会话情感分数的离散程度 | 0.59–0.89 | 单条会话情感从 −1 到 +1 波动剧烈 |
| **SEM** = std/√n | 均值估计的不确定性 | 0.005–0.064 | 均值本身估计可靠，不确定性远小于 std |

**原始图表的问题**：std 误差线是 SEM 的 13–33 倍，导致组间差异（Δ ≈ 0.02–0.06）被误差线视觉淹没，看起来“差异不显著”的视觉效果被进一步放大。

**已修复**：`_plot_bars` 函数已将误差线从 std 改为 **95% CI（=1.96×SEM）**，更新后误差线缩小到 0.01–0.13，组间差异视觉上更清晰。

### 7.2 A3 节假日 vs 非节假日：Welch's t 检验

> **核心结论：10 个方面全部不显著（p > 0.05），Cohen's d 全部 < 0.1（可忽略效应）**

| 方面 | 节假日均值 | 非节假日均值 | Δ | 节假日 n | 非节假日 n | SEM_节假日 | t 值 | p 值 | Cohen's d | 显著性 |
|---|---|---|---|---|---|---|---|---|---|---|
| content | −0.073 | −0.129 | +0.057 | 193 | 3,497 | 0.064 | 0.860 | 0.391 | 0.065 | ns |
| direction | +0.457 | +0.420 | +0.038 | 332 | 5,741 | 0.040 | 0.916 | 0.360 | 0.051 | ns |
| cast | +0.525 | +0.582 | **−0.057** | 400 | 7,282 | 0.037 | −1.521 | 0.129 | −0.084 | ns |
| genre | +0.379 | +0.412 | −0.032 | 1,047 | 18,701 | 0.023 | −1.344 | 0.179 | −0.043 | ns |
| recommendation | +0.597 | +0.571 | +0.026 | 745 | 12,832 | 0.022 | 1.162 | 0.246 | 0.042 | ns |
| audio | +0.580 | +0.555 | +0.025 | 261 | 4,839 | 0.042 | 0.569 | 0.570 | 0.036 | ns |
| plot | +0.443 | +0.418 | +0.025 | 603 | 11,402 | 0.030 | 0.811 | 0.417 | 0.033 | ns |
| emotion | +0.439 | +0.458 | −0.018 | 763 | 13,168 | 0.028 | −0.632 | 0.528 | −0.024 | ns |
| visual | +0.654 | +0.656 | −0.002 | 162 | 2,907 | 0.051 | −0.045 | 0.964 | −0.004 | ns |
| comparison | +0.607 | +0.613 | −0.006 | 296 | 5,134 | 0.039 | −0.149 | 0.882 | −0.009 | ns |

**解读**：
1. Δ 值（0.002–0.057）与节假日 SEM（0.022–0.064）处于同一量级，差异在统计噪声范围内
2. Cohen's d 全部 < 0.1，远低于 0.2 的“可忽略”门槛，即使样本量再大也不会有实际意义
3. cast 的 Δ 最大（−0.057），p=0.129 仍未达显著；且 Cohen's d=−0.084 仍属可忽略

### 7.3 逐节假日 vs 非节假日：显著差异

对 20 个节假日 × 10 个方面 = **170 组检验**（Welch's t 检验，仅 n≥10 的组参与）：

| 指标 | 值 |
|---|---|
| 总检验数 | 170 |
| p < 0.05 的 | 12（7.1%） |
| 随机假阳性期望（5%） | ~8.5 |
| Bonferroni 校正后仍显著 | **1** |

**Bonferroni 校正后仍显著的唯一结果**：

| 节假日 | 方面 | Δ | Cohen's d | p 值 | 效应等级 |
|---|---|---|---|---|---|
| 劳动节 | direction | +0.432 | 0.585 | 0.0001 *** | 中等效应 |

**未通过 Bonferroni 校正但 p < 0.05 的结果**（需谨慎解读，部分可能是假阳性）：

| 节假日 | 方面 | Δ | Cohen's d | p 值 | 解释 |
|---|---|---|---|---|---|
| 情人节 | direction | −0.554 | −0.748 | 0.025 * | 推荐影片导演评价极负面 |
| 复活节 | audio | +0.320 | 0.472 | 0.0045 ** | 音频评价偏高 |
| 哥伦布日 | genre | −0.308 | −0.413 | 0.0061 ** | 类型评价偏低 |
| 超级碗周日 | recommendation | +0.204 | 0.331 | 0.0098 ** | 推荐意愿更强 |
| 万圣节 | content | −0.366 | −0.420 | 0.045 * | 恐怖片内容警示更负面 |
| 万圣节 | genre | −0.250 | −0.336 | 0.016 * | 类型评价偏低 |
| 元旦 | direction | +0.402 | 0.544 | 0.016 * | 导演评价偏高 |
| 哥伦布日 | cast | +0.194 | 0.283 | 0.035 * | 演员评价偏高 |
| 情人节 | recommendation | +0.199 | 0.323 | 0.043 * | 推荐意愿更强 |
| 父亲节 | comparison | −0.435 | −0.645 | 0.042 * | 电影间比较偏少 |
| 退伍军人节 | recommendation | −0.264 | −0.427 | 0.016 * | 推荐意愿偏弱 |

### 7.4 效应量分布

| 效应量等级 | 数量 | 占比 |
|---|---|---|
| 可忽略（< 0.2） | 117 | 68.8% |
| 小效应（0.2–0.5） | 48 | 28.2% |
| 中等效应（0.5–0.8） | 5 | 2.9% |
| 大效应（> 0.8） | 0 | 0% |

### 7.5 结论

1. **聚合层面（节假日 vs 非节假日二元对比）无统计意义**：10 个方面全部 p > 0.05，Cohen's d < 0.1。这本身是一个有效的负面发现——推荐系统的情感表达不因节假日而全局调整。

2. **逐节假日层面有少量显著差异**：170 组检验中 12 组 p < 0.05（7.1%），仅略高于 5% 随机基线；经 Bonferroni 校正后仅劳动节 direction 存活。部分结果（如万圣节 content 偏负面、超级碗 recommendation 偏正面）有合理的业务解释。

3. **根本原因**：节假日本身是异质的——万圣节、情人节、超级碗的观影主题完全不同，笼统归为“holiday”再与非节假日对比会互相抵消。正确的分析维度是逐节假日（A5 热力图），而非二元对比（A3）。

4. **与 step7 的一致性**：step7 的情人节（爱情片份额不增反降）和独立日（爱国片无差异）分析已佐证推荐系统对日期不敏感，本节的统计检验进一步确认了这一点。







深度思考
```The user wants me to summarize the analysis steps in the step10_absa_final_v2.py file. Let me read this file first.

```
**已查看文件** *step10_absa_final_v2.py*
深度思考
```Now let me read the dependencies to understand the full picture - the key modules that this file imports from.

```
让我读取它依赖的关键模块，以便完整个步骤。理解每


**已查看文件** *absa_aspects.py*

**已查看文件** *conv_pairs.py*

**已查看文件** *absa_nli.py*
深度思考
```Now I need to read the v2 module to understand the dim functions that are reused (dim_a1 through dim_a5, _save_csv, _aggregate_to_conv_level).

```
**已查看文件** *step10_absa_v2.py*
深度思考
```Now I have a complete understanding of the code. Let me also read the step10_absa.py to understand the SentimentAnalyzer and _sentiment_to_numeric functions referenced.

```
现在我已经完整理解了整个代码及其依赖。以下是 `step10_absa_final_v2.py` 的详细分析步骤总结：







## 分析步骤详解

#### 步骤 0：初始化与输出目录重定向

**代码位置**：[第 22–25 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L22-L25)

- 调用 `setup_matplotlib()` 设置 Agg 后端 + 中文字体
- 将 v2 模块的 `STEP_OUT` 变量重定向到 `output/movie/step10/final_user/`
- 复用 v2 的全部 `dim_a1`–`dim_a5` 绘图函数和 `_save_csv`、`_aggregate_to_conv_level` 工具函数，**不重复实现**

**说明**：这是一个巧妙的设计——通过修改 v2 模块的全局变量 `STEP_OUT`，让 v2 的所有输出函数自动写入新目录，避免代码重复。

---

#### 步骤 1：归一化 VADER 情感分析器

**代码位置**：[第 32–52 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L32-L52) — `NormalizedVaderAnalyzer` 类

**功能**：仅使用 VADER 进行情感分析，分数归一化到 [-1, 1] 有符号区间。

**详细逻辑**：
1. 初始化时加载 `vaderSentiment` 的 `SentimentIntensityAnalyzer`
2. `predict(text)` 方法：
   - 空文本或长度 < 3 → 返回 `('NEUTRAL', 0.0)`，`last_model = 'none'`
   - 调用 `polarity_scores()` 取 `compound` 分数（范围 [-1, 1]）
   - `compound >= 0.05` → `('POSITIVE', compound)`，保留正值
   - `compound <= -0.05` → `('NEGATIVE', compound)`，**保留负值**（有符号）
   - 其余 → `('NEUTRAL', 0.0)`
3. `last_model` 固定为 `'vader'`（接口与 v2 的三级降级分析器兼容）

**与 v2 的 `SentimentAnalyzer` 对比**：
- v2 使用三级降级：distilbert 模型 → VADER → 词表朴素统计
- 本版本只用 VADER，但关键改进是 **NEGATIVE 返回负值**（v2 的 `_sentiment_to_numeric` 把 NEGATIVE 映射为 -1，但分数丢失了强度信息；本版本保留 compound 分值作为强度）

---

#### 步骤 2：会话重组与配对

**代码位置**：[第 63 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L63) 调用 `build_pairs_from_rows(rows)`，实现在 [conv_pairs.py](file://D:\workspaces\python\llm-movie\movie\utils\conv_pairs.py)

**功能**：将原始数据行重组为会话，并生成 (系统回复 → 用户提问) 配对。

**详细逻辑**：
1. **`regroup_sessions(rows)`**：按 `session_id` 分组所有行，组内按 `(turn_order, utc_time)` 排序
2. **`emit_pairs(turns)`**：在每个 `conv_id` 内部（**不跨 conv_id**）产出配对：
   - 找到系统回复行 → 在同一 `conv_id` 内找紧随其后的用户消息 → 组成一对
   - 末条系统回复若无人回应 → 标记为 `is_solo_system` 单独保留
   - 时段/节假日属性从该 session 的**首条 seeker 行**继承（session 级）
   - 跨日会话标记 `cross_day`（首末 seeker 日期不同）
3. **输出**：每个 pair 包含 `session_id`、`pair_id`、`system_text`、`user_text`、`date`、`period`、`is_holiday`、`holiday_name`、`cross_day` 等字段

**说明**：配对在 `conv_id` 内部进行是因为 Reddit 一个会话含多个并行 `conv_id`（每个是对原问题的一个独立回答链），跨 `conv_id` 配对会把无关子线程错配。

---

#### 步骤 3：方面检测（Phase 1 候选生成 + 去噪）

**代码位置**：[第 72 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L72) 调用 `detect_aspects_v2(text)`，实现在 [absa_aspects.py](file://D:\workspaces\python\llm-movie\movie\utils\absa_aspects.py)

**功能**：在用户提问文本中检测 10 个电影评价方面的提及，并做多级去噪。

**10 个方面**（[ASPECTS_V2](file://D:\workspaces\python\llm-movie\movie\utils\absa_aspects.py#L21-L205)）：

| 方面 key       | 英文标签         | 中文标签 | 关键词示例                                   |
| -------------- | ---------------- | -------- | -------------------------------------------- |
| genre          | Genre/Style      | 类型     | comedy, horror, sci-fi, romance...           |
| plot           | Plot/Story       | 剧情     | plot, story, twist, ending, screenplay...    |
| cast           | Cast/Acting      | 演员     | actor, performance, starring, portrayed...   |
| visual         | Visual/Effects   | 视效     | cinematography, CGI, visual effects...       |
| audio          | Audio/Music      | 音效     | soundtrack, score, background music...       |
| direction      | Direction        | 导演     | director, pacing, atmosphere, vision...      |
| emotion        | Emotion/Tone     | 情感     | moving, hilarious, boring, funny...          |
| recommendation | Recommendation   | 推荐     | must-watch, highly recommend, masterpiece... |
| comparison     | Comparison       | 比较     | reminds me of, better than, similar to...    |
| content        | Content/Warnings | 内容     | violent, family-friendly, R-rated...         |

**去噪机制**（4 层）：
1. **请求模式黑名单**（[`_REQUEST_PATTERNS`](file://D:\workspaces\python\llm-movie\movie\utils\absa_aspects.py#L233-L246)）：整句丢弃征询句式（如 "looking for movies..."、"can you recommend..."），这些不是评价
2. **POS 消歧**（[`_is_request_like_usage`](file://D:\workspaces\python\llm-movie\movie\utils\absa_aspects.py#L294-L315)）：用 `nltk.pos_tag` 判断 `like` 是介词（"movies like X" → 丢弃）还是动词（"I liked it" → 保留）
3. **词边界正则**（[`_kw_pattern`](file://D:\workspaces\python\llm-movie\movie\utils\absa_aspects.py#L216-L220)）：用 `\b` 边界匹配避免子串误匹配
4. **最长匹配 + 方面去重**（[第 352–371 行](file://D:\workspaces\python\llm-movie\movie\utils\absa_aspects.py#L352-L371)）：关键词按长度降序排列优先匹配长词；同一方面只取第一个命中片段（一方面一片段）

**输出**：候选列表 `[{aspect, snippet, keyword, confidence}]`，`confidence` 固定 1.0（Phase 2 NLI 提供）。

---

#### 步骤 4：NLI 评价上下文过滤（Phase 2 ③.2）

**代码位置**：[第 74–76 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L74-L76) 调用 `has_evaluative_context(c['snippet'])`，实现在 [absa_nli.py](file://D:\workspaces\python\llm-movie\movie\utils\absa_nli.py)

**功能**：过滤掉"纯提及但无评价"的候选句（如 "What would the twist be?" 含 `twist` 关键词但没有评价词）。

**规则版逻辑**（[`has_evaluative_context`](file://D:\workspaces\python\llm-movie\movie\utils\absa_nli.py#L62-L72)）：
- 句中是否出现以下任一评价标记词集：
  - `_OPINION_ADJ`：评价形容词（good, great, boring, amazing...）
  - `_COPULA`：系动词（was, were, felt, seemed...）
  - `_EVALUATIVE_VERB`：评价动词（loved, hated, delivered, nailed...）
  - `_DEGREE`：程度副词（really, very, incredibly...）
- 否定 + 评价形容词也算评价（"not bad"）

**可选升级**：`AspectClassifier` 类支持加载 `all-MiniLM-L6-v2` transformer 模型，用句向量余弦相似度判定，但默认关闭（规则版更快）。

**统计**：被过滤的候选数 `n_nli_filtered` 记入 stats，最终日志输出过滤比例。

---

#### 步骤 5：情感预测

**代码位置**：[第 78 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L78) 调用 `analyzer.predict(c['snippet'])`

**功能**：对通过 NLI 过滤的每个候选片段调用 VADER 情感分析器，得到 `(标签, 分数)`。

**说明**：步骤 4 和 5 在同一个循环中完成——先过滤，后预测，通过的候选组装成 `pair_record`，包含：
- 会话信息：`session_id`、`pair_id`、`date`、`period`、`is_holiday`、`holiday_name`、`cross_day`
- 方面信息：`aspect`、`aspect_label`、`keyword`、`snippet`（截取前 200 字符）
- 情感结果：`sentiment`（标签）、`score`（有符号分数）、`model_used`

---

#### 步骤 6：会话级聚合

**代码位置**：[第 101 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L101) 调用 `v2._aggregate_to_conv_level(pair_records)`，实现在 [step10_absa_v2.py 第 70–95 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_v2.py#L70-L95)

**功能**：将 pair 级记录聚合为会话级记录（1 session × 1 aspect = 1 row）。

**聚合逻辑**：
1. 按 `(session_id, aspect)` 分组所有 pair 记录
2. 对每组：
   - 将情感标签转为数值（POSITIVE→1, NEGATIVE→-1, NEUTRAL→0）通过 `_sentiment_to_numeric`
   - 计算组内 `mean_sentiment`（均值）、`std_sentiment`（标准差）、`n_pairs`（配对数）、`pos_ratio`（正面比例 = 正值数/总数）
3. 继承首条记录的时段/节假日属性

**说明**：这一步消除了同一会话内多次提及同一方面带来的重复计数问题，确保每个会话对每个方面只贡献一个聚合值。

---

#### 步骤 7：输出 CSV 数据

**代码位置**：[第 128 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L128) 调用 `v2._save_csv(conv_records, 'a0_conv_records_final_user.csv')`，实现在 [step10_absa_v2.py 第 172–184 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_v2.py#L172-L184)

**输出文件**：`output/movie/step10/final_user/a0_conv_records_final_user.csv`

**CSV 列**：`session_id`、`date`、`period`、`holiday_name`、`cross_day`、`aspect`、`aspect_label`、`mean_sentiment`、`std_sentiment`、`n_pairs`、`pos_ratio`

---

#### 步骤 8：A1 — 方面分布饼图

**代码位置**：[第 129 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L129) 调用 `v2.dim_a1(conv_records)`，实现在 [step10_absa_v2.py 第 191–208 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_v2.py#L191-L208)

**输出**：`a1_aspect_distribution_v2.png`

**内容**：统计各方面在会话级记录中的出现频次，绘制饼图（Set3 配色），显示各占比百分比。按频次降序排列。

---

#### 步骤 9：A2 — 整体方面情感柱状图

**代码位置**：[第 130 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L130) 调用 `v2.dim_a2(conv_records)`，实现在 [step10_absa_v2.py 第 211–217 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_v2.py#L211-L217)

**输出**：`a2_overall_aspect_sentiment_v2.png`

**内容**：合并所有时段（`_overall`），按方面分组绘制柱状图，显示每个方面的平均情感分数（-1 到 +1），带标准差误差棒，零线虚线参照。

---

#### 步骤 10：A3 — 节假日 vs 非节假日对比柱状图

**代码位置**：[第 131 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L131) 调用 `v2.dim_a3(conv_records)`，实现在 [step10_absa_v2.py 第 220–232 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_v2.py#L220-L232)

**输出**：`a3_holiday_vs_nonholiday_v2.png`

**内容**：
- 节假日组（`period == 'holiday'`）与非节假日组（`period != 'holiday'`）分别聚合
- 每个方面并排两根柱子（红色=Holiday / 蓝色=Non-holiday），带误差棒
- 可直观对比节假日是否影响各方面的情感倾向

---

#### 步骤 11：A4 — 节假日 vs 工作日 vs 周末三组对比柱状图

**代码位置**：[第 132 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L132) 调用 `v2.dim_a4(conv_records)`，实现在 [step10_absa_v2.py 第 235–246 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_v2.py#L235-L246)

**输出**：`a4_holiday_workday_weekend_v2.png`

**内容**：
- 按 `period` 分三组：Holiday / Workday / Weekend
- 每个方面并排三根柱子，带误差棒
- 比 A3 更细粒度，能看出工作日和周末是否有差异

---

#### 步骤 12：A5 — 各节假日 × 方面情感热力图

**代码位置**：[第 133 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_final_v2.py#L133) 调用 `v2.dim_a5(conv_records)`，实现在 [step10_absa_v2.py 第 249–282 行](file://D:\workspaces\python\llm-movie\movie\step10_absa_v2.py#L249-L282)

**输出**：`a5_per_holiday_aspect_v2.png`

**内容**：
- 只取节假日记录（`is_holiday == True`），按 `holiday_name`（取前 8 字符）分组
- 过滤数据量不足的节假日（< `MIN_DATA_ROWS` 行）
- 绘制 热力图：行=节假日名称，列=10 个方面，值=平均情感分数
- 红色=更正面，蓝色=更负面，灰色=无数据（NaN）
- `center=0` 以零为中心对称配色

---

### 整体流程图

```
原始数据 rows
    │
    ▼
① 会话重组 + 配对 (build_pairs_from_rows)
    │  → (系统回复 → 用户提问) 配对
    ▼
② 方面检测 (detect_aspects_v2)
    │  → 10 方面关键词匹配 + 4 层去噪
    ▼
③ NLI 过滤 (has_evaluative_context)
    │  → 丢弃纯提及、保留真评价
    ▼
④ 情感预测 (NormalizedVaderAnalyzer.predict)
    │  → VADER compound 分数 [-1, +1]
    ▼
⑤ 会话级聚合 (_aggregate_to_conv_level)
    │  → 1 session × 1 aspect = 1 row
    ▼
⑥ 输出 CSV + 5 张图表 (A1-A5)
    │  → output/movie/step10/final_user/
    ▼
完成
```


---

### 关键设计决策

| 决策                            | 原因                                          |
| ------------------------------- | --------------------------------------------- |
| 分析 user_text 而非 system_text | 用户提问更能反映用户对电影的真实评价          |
| 只用 VADER 不用 distilbert      | 简化依赖，VADER 足够稳定且快（全量约 8 分钟） |
| NEGATIVE 保留负值               | 有符号分数保留情感强度信息                    |
| NLI 规则版而非模型版            | 避免 transformer 模型下载，规则版足够精准     |
| 去掉双轨配对                    | 实测覆盖率仅 0.8%，性价比太低                 |
| 复用 v2 的 dim 函数             | 通过重定向 `STEP_OUT` 变量避免代码重复        |



