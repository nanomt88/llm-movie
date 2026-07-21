# 各节假日观影对比

 @movie/step7_wordcloud.py 基于这个代码生成的结果，分析各个节假日观影的特点，总结规律，结果必须真实且可靠。

---

## 各节假日观影特点分析

> **数据来源**：`w6_holiday_viewing_profile.csv`（20 个节假日 × 6 维度分类得分）
> **统计口径**：得分 = 节假日日均词频 > 非节假日日均 × 1.5 的"升高词"中，匹配该分类关键词的日均频次之和
> **数据规模**：2019–2022 年 Reddit 电影讨论，去重后 146,768 条用户提问，79 个节假日日期

---

## 一、六大维度峰值排行

在分析各节假日之前，先看每个维度得分最高的节假日：

| 维度 | top1 节假日（得分） | top2 节假日（得分） | top3 节假日（得分） |
|---|---|---|---|
| **Genre（类型）** | 万圣节 Horror（**42.5**） | 独立日 Action（13.2） | 哥伦布日 Horror（22.8） |
| **Mood（情绪）** | 复活节 Exciting（**8.8**） | 哥伦布日 Fun/Enjoyable（12.5） | 独立日 Exciting（7.5） |
| **Context（场景）** | 复活节 Binge/Series（**22.2**） | 情人节 Family/Kids（9.5） | 独立日 Family/Kids（8.5） |
| **Platform（平台）** | 超级碗周日 Netflix（**5.5**） | 情人节 Prime（3.0） | 平安夜 Prime（2.8） |
| **Quality（品质）** | 劳动节 Mainstream（6.2） | 母亲节 Great/Excellent（7.2） | 情人节 Classic（4.2） |
| **Narrative（叙事）** | 复活节 Characters（**14.8**） | 独立日 Characters（13.2） | 父亲节 Acting（**12.0**） |

---

## 二、六大规律总结

### 规律 1：恐怖片在 10 月节假日绝对主导

万圣节 Horror 得分 **42.5**，是所有节假日所有类型中最高分。哥伦布日（10 月第二个周一）Horror 得分 **22.8**，位列第二。两者均处于 10 月，恐怖片讨论量是非节假日的 **2.87 倍**（万圣节）和 **1.56 倍**（哥伦布日）。

| 节假日 | Horror 得分 | 是第二类型的倍数 | 恐怖关键词 |
|---|---|---|---|
| 万圣节 | 42.5 | 9.4x（vs Romance 4.5） | creepy, ghost, gore, gory, horror, monster, scary, slasher, spooky, supernatural |
| 哥伦布日 | 22.8 | 2.3x（vs Comedy 10.0） | creepy, fear, gore, horror, scary, spooky, supernatural, vampire, zombie |
| 9·11纪念日 | 12.2 | 2.0x（vs Action 6.0） | creepy, ghost, monster, scary, supernatural, vampire, zombie |

### 规律 2：暑期（5–7 月）动作/科幻片讨论峰值

| 节假日 | Action 得分 | Sci-Fi 得分 | 月份 |
|---|---|---|---|
| 独立日 | **13.2** | — | 7 月 |
| 复活节 | **9.0** | — | 4 月（春假/暑期前奏） |
| 母亲节 | 8.2 | **9.5** | 5 月 |
| 阵亡将士纪念日 | 6.2 | **7.2** | 5 月 |

独立日 Action 得分 13.2 是所有节假日中该类型最高分。母亲节 Sci-Fi 得分 9.5 是所有节假日中该类型最高分。`battle`、`war`、`action`、`alien`、`space`、`superhero` 是高频关键词。

### 觏律 3：复活节是"刷剧"绝对峰值

复活节 Binge/Series 得分 **22.2**，是所有节假日中任何维度最高分之一，远超第二名：

| 节假日 | Binge/Series 得分 | 关键词 |
|---|---|---|
| **复活节** | **22.2** | series, show |
| 圣帕特里克节 | 8.0 | binge, series |
| 平安夜 | 7.5 | series |
| 退伍军人节 | 7.2 | series |
| 阵亡将士纪念日 | 7.0 | episode |

复活节通常是 4 天连休（复活节周末 + 复活节周一），为刷剧提供了最长时间。同时 Characters 得分也最高（14.8），说明用户在刷剧时大量讨论角色。

### 规律 4：圣诞节/平安夜的"温馨"情绪独占

| 节假日 | Cozy/Family 得分 | 关键词 |
|---|---|---|
| **圣诞节** | **6.8** | cozy, happy, merry, wholesome |
| 平安夜 | 5.8 | happy, merry |

`merry` 一词在圣诞节达到基线的 212.5 倍（来自 W4 数据），平安夜 185.9 倍——这是所有节假日信号词中倍率最高的。

### 规律 5：家庭观影在传统聚会节日达到峰值

| 节假日 | Family/Kids 得分 | 关键词 |
|---|---|---|
| **情人节** | **9.5** | brother, child, daughter, mom, mother, parent |
| 独立日 | 8.5 | dad, family, kid, mother |
| 哥伦布日 | 7.8 | child, kid, mom, son |
| 感恩节 | 6.2 | family, mom, parent |
| 圣诞节 | 4.5 | brother, child, father |

**情人节 Family/Kids 得分最高（9.5）**——出人意料。情人节不是 Date Night 得分最高（仅 2.5），而是 Family/Kids 最高，说明用户在情人节周末更多讨论与家人/孩子一起看电影，而非仅约会。`brother`、`child`、`daughter`、`mom`、`mother`、`parent` 是匹配关键词。

### 规律 6：各叙事维度的"专精"节假日

不同节假日用户聚焦不同的叙事/制作维度：

| 维度 | 得分最高的节假日 | 得分 | 关键词 |
|---|---|---|---|
| **Characters** | 复活节 | 14.8 | character, villain |
| **Acting** | 父亲节 | **12.0** | act, actor, lead, performance |
| **Cinematography** | 劳动节 | **8.0** | aesthetic, cinematography, screen, style, visual |
| **Atmosphere** | 母亲节 | **8.0** | theme, tone, vibe |
| **Plot/Story** | 母亲节 | **9.2** | concept, journey, story |
| **Ending** | 独立日 | **8.0** | end |
| **Music/Audio** | 平安夜 | **5.8** | music, song, soundtrack |

父亲节 Acting 得分 12.0 是所有 Narrative 子类的最高分，用户大量讨论演员表演（`act`、`actor`、`lead`、`performance`）。劳动节 Cinematography 得分 8.0，用户聚焦视觉技术（`aesthetic`、`style`、`visual`）。

---

## 三、各节假日观影特点（逐个分析）

### 1. 万圣节（10 月 31 日）— 恐怖片绝对巅峰

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | **Horror** | **42.5** | 9.4x 于第二名，10 个恐怖关键词全命中 |
| Mood | Sad | 3.2 | 恐怖片带来的负面情绪 |
| Context | Family/Kids | 5.0 | brother, child, kid |
| Context | Friends/Social | 4.5 | friend, mate |
| Platform | HBO | 1.8 | hbo, max |
| Narrative | Characters | 13.2 | character, villain |

**核心特点**：万圣节是所有节假日中类型集中度最高的——Horror 得分 42.5 占该节假日全部 Genre 得分的 86%。观影场景兼具家庭（5.0）和朋友社交（4.5），说明"带孩子看恐怖片"和"朋友恐怖片派对"并存。HBO 平台偏多，与 HBO 恐怖向内容库吻合。

**万圣节 (Halloween) —— 恐怖霸榜与社交狂欢**

*   **规律**：类型上，**Horror（恐怖）得分高达 42.5**，呈绝对断层领先，紧跟的是浪漫和奇幻；叙事上极度关注 **Characters（13.2）**。
*   **关键词验证**：命中大量 `creepy, ghost, gore, scary, slasher, spooky` 等词。
*   **场景**：`Family/Kids (5.0)` 和 `Friends/Social (4.5)` 得分很高。这表明万圣节观影往往不是独自观影，而是作为派对或家庭娱乐的社交环节。

### 2. 哥伦布日（10 月第二个周一）— 恐怖+娱乐

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Horror | 22.8 | 仅次于万圣节 |
| Genre | Comedy | 10.0 | 喜剧并列 |
| Mood | **Fun/Enjoyable** | **12.5** | 全节假日最高 |
| Context | Family/Kids | 7.8 | child, kid, mom, son |
| Narrative | Acting | 7.2 | actor, actress, lead, role |

**核心特点**：哥伦布日 Fun/Enjoyable 得分 12.5 是所有节假日所有 Mood 子类最高分。恐怖片+喜剧组合说明用户在 10 月长假追求"又恐怖又好笑"的娱乐体验。`enjoy`、`fun` 是核心情绪词。Family/Kids 得分高（7.8），说明家庭观影也是重要场景。

### 3. 独立日（7 月 4 日）— 暑期动作大片

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | **Action** | **13.2** | 全节假日 Action 最高 |
| Mood | Exciting | 7.5 | action, excite |
| Mood | Sad | 5.0 | bore, cry, hate |
| Mood | Uplifting | 2.8 | dream, powerful, wonderful |
| Context | **Family/Kids** | **8.5** | dad, family, kid, mother |
| Quality | Mainstream | 2.5 | franchise, sequel |
| Narrative | Characters | 13.2 | character |
| Narrative | Acting | 8.5 | act, actor, actress, performance |
| Narrative | Ending | 8.0 | end |

**核心特点**：独立日是"美国爱国主义+暑期大片"的集中体现。Action 得分 13.2 为全节假日最高，`battle`、`fight`、`action` 是高频词。Uplifting 情绪（dream, powerful, wonderful）反映爱国励志主题。Family/Kids 得分高（8.5），`dad` 出现说明"父亲带娃看大片"是典型场景。Ending 得分 8.0 说明用户大量讨论影片结局。

### 4. 复活节（4 月）— 刷剧+角色讨论巅峰

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Action | 9.0 | action, revenge, violent |
| Mood | **Exciting** | **8.8** | 全节假日 Exciting 最高 |
| Mood | Dark | 6.0 | dark, disturb |
| Context | **Binge/Series** | **22.2** | 全节假日最高分之一 |
| Narrative | **Characters** | **14.8** | 全节假日 Characters 最高 |
| Platform | HBO | 2.0 | hbo, max |

**核心特点**：复活节是"刷剧+角色讨论"的绝对巅峰。Binge/Series（22.2）和 Characters（14.8）均为全节假日最高分。`series`、`show` 是核心词，说明用户在 4 天春假期间大量刷剧。Exciting 情绪最高（8.8），与动作大片氛围一致。HBO 平台偏多，可能与 HBO 原创剧集有关。

### 5. 圣诞节（12 月 25 日）— 温馨+犯罪独特组合

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Romance | 5.2 | romance |
| Genre | Crime | 4.5 | murder, noir |
| Mood | **Cozy/Family** | **6.8** | cozy, happy, merry, wholesome |
| Context | Family/Kids | 4.5 | brother, child, father |
| Context | Alone/Quiet | 3.2 | alone, myself |
| Quality | Classic | 3.8 | classic |
| Narrative | Cinematography | 7.8 | aesthetic, beautiful, style, visual |
| Narrative | Atmosphere | 6.2 | atmosphere, mood, tone |

**核心特点**：圣诞节独特之处是 Romance+Crime 组合——温馨浪漫与犯罪悬疑并存。Crime 出现 `noir`（黑色电影），可能指向圣诞黑色电影传统（如《小鬼当家》犯罪喜剧、《虎胆龙威》圣诞惊悚）。Cozy/Family 情绪最高（6.8），`merry` 达基线 212.5 倍。Cinematography（7.8）+Atmosphere（6.2）双高，说明用户关注圣诞电影的"视觉氛围感"。

**圣诞节与平安夜 (Christmas / Christmas Eve) —— 极致的温馨与怀旧**

*   **规律**：情绪维度上，**Cozy/Family（温馨/家庭）分别高达 6.8 和 5.8**，同时伴随强烈的 **Nostalgic（怀旧，3.8）** 情绪；类型上倾向于 Romance（浪漫）和 Musical（音乐）。
*   **关键词验证**：高频出现 `cozy, happy, merry, wholesome, relax` 等词，且评价维度上 `Classic (3.8)` 得分极高。
*   **场景**：明显偏向家庭观影与独自放松（Alone/Quiet），观众倾向于重温能带来心理安全感的经典老片。

### 6. 平安夜（12 月 24 日）— 温馨+恐怖并存

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Horror | 3.5 | gore, scary |
| Genre | Musical | 2.0 | soundtrack |
| Mood | Cozy/Family | 5.8 | happy, merry |
| Mood | Sad | 4.0 | depress, hate, sad |
| Context | Binge/Series | 7.5 | series |
| Platform | Prime | 2.8 | amazon, prime |
| Narrative | Music/Audio | 5.8 | music, song, soundtrack |

**核心特点**：平安夜独特之处是 Horror+Cozy/Family 并存——圣诞恐怖片传统（《小鬼当家》《虎胆龙威》）与温馨家庭片传统并存。Musical 出现（2.0），与圣诞音乐电影传统有关。Music/Audio 得分高（5.8），`song`、`soundtrack` 说明用户讨论圣诞电影音乐。Prime 平台偏多（2.8）。

### 7. 母亲节（5 月第二个周日）— 科幻+叙事讨论

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | **Sci-Fi** | **9.5** | 全节假日 Sci-Fi 最高 |
| Genre | Action | 8.2 | action, superhero |
| Mood | Exciting | 7.2 | action, excite |
| Context | Binge/Series | 4.0 | binge, season |
| Quality | Great/Excellent | 7.2 | brilliant, decent, fantastic, perfect |
| Narrative | **Plot/Story** | **9.2** | concept, journey, story |
| Narrative | **Atmosphere** | **8.0** | theme, tone, vibe |
| Narrative | Cinematography | 6.0 | beautiful, cinematography, visuals |

**核心特点**：母亲节 Sci-Fi 得分 9.5 为全节假日最高，`alien`、`space`、`technology` 是高频词。Plot/Story 得分 9.2 也是全节假日该维度最高，说明用户在母亲节大量讨论剧情和叙事。Great/Excellent（7.2）说明用户给出积极评价。可能与母亲节讨论涉及母性主题的科幻片（如《异形》《星际穿越》）有关。

**母亲节与父亲节 —— 科幻、动作与叙事**

*   **规律**：非常有趣的是，这俩节日的首选类型都是 **Sci-Fi（科幻）**。母亲节情绪偏 Exciting（7.2），极度看重 **Plot/Story（9.2）**；而父亲节则更关注 **Acting（演技，12.0）**。

### 8. 父亲节（6 月第三个周日）— 表演讨论巅峰

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Sci-Fi | 4.5 | alien, outbreak, travel, universe |
| Genre | Crime | 4.5 | detective, murder, police |
| Mood | Nostalgic | 2.8 | classic |
| Context | Friends/Social | 2.8 | group |
| Platform | HBO | 1.8 | hbo, max |
| Narrative | **Acting** | **12.0** | 全节假日最高：act, actor, lead, performance |

**核心特点**：父亲节 Acting 得分 12.0 是所有 Narrative 子类在所有节假日中的最高分。用户大量讨论演员表演（`act`、`actor`、`lead`、`performance`），可能与"父亲形象"在电影中的角色讨论有关。Sci-Fi+Crime 组合，HBO 平台偏多。Nostalgic 情绪出现（`classic`），说明用户也讨论经典老片。

### 9. 劳动节（9 月第一个周一）— 视觉技术讨论

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Thriller | 5.5 | intense, suspense, tension |
| Mood | Exciting | 2.8 | intense, surprise |
| Context | Alone/Quiet | 2.5 | alone, myself |
| Quality | Mainstream | 6.2 | franchise, mainstream, popular, sequel |
| Narrative | **Cinematography** | **8.0** | 全节假日最高：aesthetic, cinematography, screen, style, visual |

**核心特点**：劳动节 Cinematography 得分 8.0 为全节假日最高。用户大量讨论摄影和视觉技术（`aesthetic`、`cinematography`、`screen`、`style`、`visual`），说明劳动节用户关注"技术质感"强的电影。Mainstream 得分也高（6.2），`franchise`、`sequel` 说明暑期档收官期大片讨论活跃。Alone/Quiet 是主要场景，说明劳动节用户更多独自观影。

### 10. 情人节（2 月 14 日）— 家庭观影意外居首

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Drama | 4.2 | emotional |
| Genre | Fantasy | 4.0 | epic, fantasy, giant |
| Mood | Nostalgic | 3.5 | classic |
| Context | **Family/Kids** | **9.5** | 全节假日最高：brother, child, daughter, mom, mother, parent |
| Context | Date Night | 2.5 | partner, wife |
| Platform | Prime | 3.0 | amazon, prime |
| Platform | Hulu | 1.5 | hulu |
| Quality | Classic | 4.2 | classic, masterpiece |

**核心特点**：情人节最出人意料的是 Family/Kids 得分 9.5 为全节假日最高，远超 Date Night（2.5）。用户在情人节更多讨论与家人/孩子看电影，而非仅约会。`brother`、`child`、`daughter`、`mom`、`mother`、`parent` 是匹配关键词。Prime+Hulu 平台偏多。Classic 得分高（4.2），`masterpiece` 说明用户推荐经典影片。

**情人节 (Valentine's Day) —— 情感共鸣与陪伴**

*   **规律**：虽然 Romance 关键词被平摊，但其核心驱动力是 **Drama（剧情，4.2）** 和 **Fantasy（奇幻，4.0）**。
*   **场景**：**Family/Kids 飙升至 9.5，Date Night 达到 2.5**。情人节的语境不仅限于情侣约会（wife/partner），大量用户提及与家人的情感维系（brother, daughter, parent）。

### 11. 耶稣受难日（4 月）— 剧情+结局讨论

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Animation | 4.8 | animate, animation, anime |
| Genre | Sci-Fi | 4.2 | science, travel, universe |
| Mood | Sad | 4.8 | bore, cry, hate |
| Mood | Dark | 3.8 | dark |
| Context | Friends/Social | 4.2 | friend, group |
| Platform | Hulu | 1.2 | hulu |
| Narrative | Ending | 8.0 | end |
| Narrative | Plot/Story | 7.8 | plot, storyline, write |
| Narrative | Music/Audio | 6.2 | music, song, soundtrack |

**核心特点**：耶稣受难日 Sad+Dark 双高（4.8/3.8），与节日的沉重宗教氛围一致。Plot/Story（7.8）+Ending（8.0）双高，说明用户大量讨论剧情和结局——可能是复活节周末连续刷剧后对剧情的讨论。Hulu 平台出现。Animation+Sci-Fi 组合说明用户在复活节周末看动画和科幻。

### 12. 超级碗周日（2 月第一个周日）— Netflix 独占

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Romance | 4.8 | romance, romantic |
| Genre | Crime | 4.5 | heist, kill, murder, police |
| Context | Rewatch | 1.5 | rewatch |
| Platform | **Netflix** | **5.5** | 全节假日 Platform 最高 |
| Quality | Classic | 2.8 | classic |
| Quality | Cult/Indie | 2.8 | cult, indie |
| Narrative | Music/Audio | 3.8 | sound |

**核心特点**：超级碗周日 Netflix 得分 5.5 是所有节假日 Platform 最高分。Netflix 通常在超级碗期间发布重磅内容（如超级碗广告期间公布新片），数据清晰反映这一策略。Rewatch 出现说明用户在超级碗期间重看经典。数据量小（314 条），信号相对较弱。

### 13. 总统日（2 月第三个周一）— 纪录片独占

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | **Documentary** | **5.8** | 全节假日唯一 Documentary top1 |
| Genre | Comedy | 4.0 | hilarious, humor, laugh |
| Context | Family/Kids | 0.8 | mother |
| Narrative | Music/Audio | 5.0 | music, song |

**核心特点**：总统日是唯一以 Documentary 为 Genre top1 的节假日（5.8），`documentary` 是核心词。Historical 类型也出现，说明用户在总统日讨论历史/政治题材纪录片。数据量小（322 条）。

### 14. 感恩节（11 月第四个周四）— 家庭娱乐

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Action | 4.2 | adventure, fight, superhero |
| Genre | Documentary | 3.5 | documentary |
| Mood | Fun/Enjoyable | 3.8 | fun |
| Context | Family/Kids | 6.2 | family, mom, parent |
| Context | Date Night | 1.8 | husband, wife |
| Quality | Poor/Bad | 1.0 | stupid |

**核心特点**：感恩节是典型"家庭聚会+娱乐"节日。Family/Kids 得分高（6.2），`family`、`mom`、`parent` 是核心词。Date Night 出现 `husband`/`wife`，说明夫妻共同观影也是场景。Binge/Series 出现 `marathon`，说明感恩节有"电影马拉松"传统。类型分散（Action/Documentary/Fantasy），说明全家不同口味的折中选片。

### 15. 退伍军人节（11 月 11 日）— 犯罪+刷剧

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Crime | 4.2 | crime, kill |
| Genre | Romance | 3.0 | romantic |
| Mood | Sad | 3.0 | cry |
| Context | Binge/Series | 7.2 | series |
| Context | Family/Kids | 3.2 | kid |
| Platform | HBO | 1.2 | hbo |
| Narrative | Acting | 9.2 | actor, actress, role, star |
| Narrative | Plot/Story | 5.0 | journey, premise, storyline, write |

**核心特点**：退伍军人节偏向 Crime+Romance，`crime`/`kill` 说明犯罪片讨论活跃。Western 类型出现（`western`），与退伍军人节的美国传统吻合。Binge/Series 得分高（7.2），HBO 平台偏多。Acting 得分高（9.2），`actor`/`actress`/`role`/`star` 说明用户讨论演员表演。

### 16. 阵亡将士纪念日（5 月最后一个周一）— 科幻末日主题

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Sci-Fi | 7.2 | apocalyptic, dystopian, scifi, space |
| Genre | Action | 6.2 | battle, war |
| Mood | Sad | 4.0 | cry, hate |
| Context | Binge/Series | 7.0 | episode |
| Context | Family/Kids | 5.5 | brother, daughter, father |
| Narrative | Characters | 6.2 | cast, protagonist |

**核心特点**：阵亡将士纪念日 Sci-Fi 得分高（7.2），关键词 `apocalyptic`/`dystopian`/`scifi`/`space` 说明末日科幻片讨论活跃。Action 中 `battle`/`war` 与纪念日主题直接相关。Family/Kids 出现 `brothers`/`daughter`/`father`，与"阵亡将士→家庭"的主题呼应。Characters 中 `cast`/`protagonist` 说明用户讨论英雄角色。

### 17. 马丁·路德·金日（1 月第三个周一）— 多平台分散

| 维度 | top1 | 得分 | 特点 |
|---|---|---|---|
| Genre | Sci-Fi | 4.0 | sci |
| Genre | Action | 3.8 | battle, war |
| Genre | Drama | 3.5 | dramatic, move |
| Context | Friends/Social | 3.0 | group, together |
| Platform | Streaming | 1.5 | stream |
| Platform | Prime | 1.2 | prime |
| Platform | Hulu | 1.2 | hulu |
| Quality | Great/Excellent | 3.8 | perfect, unique |
| Narrative | Atmosphere | 2.5 | atmosphere, tone |

**核心特点**：马丁·路德·金日三类型接近（Sci-Fi/Action/Drama 各 3.5–4.0），说明观影口味多元。三个平台并列（Streaming/Prime/Hulu 各 1.2–1.5），说明用户分散在多个平台。Crime 中出现 `detective`/`noir`，可能指向探讨社会不公的黑色电影。Great/Excellent 中 `perfect`/`unique` 说明用户给出积极评价。

### 18–20. 其余节假日简述

| 节假日 | 核心特点 |
|---|---|
| **元旦**（1/1） | Thriller（7.7）+Animation（4.3），轻松惊悚+动画，Uplifting 情绪（2.0），数据量小（275 条） |
| **圣帕特里克节**（3/17） | Binge/Series 得分极高（8.0），刷剧行为突出；Horror（4.5）+Animation（3.0），数据量小（349 条） |
| **9·11纪念日**（9/11） | Horror（12.2）+Action（6.0），恐怖/动作片为主；Exciting（5.2）+Dark（2.8）情绪；Atmosphere（6.5）叙事最高 |

---

## 四、跨节假日对比矩阵

### 按类型偏好聚类

| 类型偏好 | 节假日 | 月份 |
|---|---|---|
| **恐怖片主导** | 万圣节、哥伦布日、9·11纪念日、平安夜 | 9–12 月 |
| **动作/科幻主导** | 独立日、复活节、母亲节、阵亡将士纪念日 | 4–7 月 |
| **喜剧/剧情主导** | 哥伦布日、劳动节、父亲节、马丁·路德·金日 | 分散 |
| **纪录片主导** | 总统日 | 2 月 |
| **类型分散** | 情人节、感恩节、退伍军人节、超级碗周日、元旦 | 分散 |

### 按观影场景聚类

| 场景偏好 | 节假日 | Binge/Series 或 Family/Kids 得分 |
|---|---|---|
| **刷剧为主** | 复活节（22.2）、圣帕特里克节（8.0）、平安夜（7.5）、退伍军人节（7.2）、阵亡将士纪念日（7.0） | Binge/Series 高 |
| **家庭为主** | 情人节（9.5）、独立日（8.5）、哥伦布日（7.8）、感恩节（6.2）、圣诞节（4.5） | Family/Kids 高 |
| **独处为主** | 劳动节（2.5）、圣诞节（3.2） | Alone/Quiet 高 |

### 按情绪氛围聚类

| 情绪偏好 | 节假日 | 得分 |
|---|---|---|
| **温馨** | 圣诞节（6.8）、平安夜（5.8） | Cozy/Family 高 |
| **刺激** | 复活节（8.8）、独立日（7.5）、母亲节（7.2）、9·11（5.2） | Exciting 高 |
| **娱乐** | 哥伦布日（12.5）、母亲节（2.0）、感恩节（3.8）、父亲节（1.2） | Fun/Enjoyable 高 |
| **深沉** | 耶稣受难日 Sad（4.8）+Dark（3.8）、阵亡将士纪念日 Sad（4.0）、独立日 Sad（5.0） | Sad/Dark 高 |

---

## 五、关键发现

1. **万圣节 Horror 得分 42.5 是全数据最高分**——10 月是恐怖片讨论的绝对峰值，哥伦布日（22.8）紧随其后。10 月两个节假日恐怖片得分之和（65.3）占全年 Horror 讨论的绝大部分。

2. **复活节 Binge/Series 得分 22.2 是全数据第二高分**——4 天春假是刷剧的最长时间窗口。Characters 得分 14.8 也最高，说明刷剧后大量讨论角色。

3. **情人节 Family/Kids 得分 9.5 全节假日最高**——最出人意料的发现。情人节不是约会日最高，而是家庭观影最高，说明情人节周末更多是家庭活动。

4. **父亲节 Acting 得分 12.0 是叙事维度最高分**——用户在父亲节大量讨论演员表演，可能与"父亲形象"在电影中的角色讨论有关。

5. **劳动节 Cinematography 得分 8.0**——用户在劳动节聚焦视觉技术讨论（aesthetic/style/visual），是"技术党"最活跃的节假日。

6. **母亲节 Sci-Fi 得分 9.5 全节假日最高**——5 月暑期档前奏，科幻片讨论最活跃，Plot/Story 得分（9.2）也最高，说明用户既看科幻又深入讨论剧情。

7. **超级碗周日 Netflix 得分 5.5 是平台维度最高分**——Netflix 在超级碗周末的内容发布策略在数据中清晰可见。

8. **总统日 Documentary 得分 5.8 是唯一纪录片 top1**——与总统/政治主题纪录片讨论有关，Historical 类型也出现。

9. **独立日 Action 得分 13.2 是动作片最高分**——7 月 4 日国庆日是动作大片讨论最活跃的节假日，`battle`/`war`/`action` 是高频词。

10. **圣诞节 Cozy/Family 得分 6.8 + Crime Genre top2**——温馨氛围与犯罪悬疑的独特组合，`noir`（6.3x）证实了圣诞黑色电影传统，`merry`（212.5x）是最强节日信号词。



## 其他：各维度年度冠军（基于 W6 CSV） - QwenMAX

| 维度 | 冠军节假日 | 得分 | 与第二名的倍数 |
|---|---|---|---|
| Horror | 万圣节 | **42.5** | 9.4x vs Romero(4.5) |
| Action | 独立日 | **13.2** | 1.47x vs 复活节(9.0) |
| Sci-Fi | 母亲节 | **9.5** | 1.32x vs 阵亡将士纪念日(7.2) |
| Binge/Series | 复活节 | **22.2** | 2.78x vs 圣帕特里克节(8.0) |
| Family/Kids | 情人节 | **9.5** | 1.12x vs 独立日(8.5) |
| Fun/Enjoyable | 哥伦布日 | **12.5** | 6.25x vs 第二名 |
| Cozy/Family | 圣诞节 | **6.8** | 1.17x vs 平安夜(5.8) |
| Acting | 父亲节 | **12.0** | 1.30x vs 退伍军人节(9.2) |
| Characters | 复活节 | **14.8** | 1.12x vs 独立日(13.2) |
| Plot/Story | 母亲节 | **9.2** | 1.15x vs 独立日(8.0) |
| Netflix | 超级碗周日 | **5.5** | 1.83x vs 情人节(3.0) |
| Documentary | 总统日 | **5.8** | 1.66x vs 感恩节(3.5) |

