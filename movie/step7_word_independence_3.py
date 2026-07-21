# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文等非 ASCII 字符
"""
Step 7.3: US Independence Day vs Patriotic Movies — Statistical Verification (v2)
步骤 7.3：美国独立日 vs 爱国主题电影 关联性统计验证（v2 增强版）

【v2 增强内容】
  1. 扩充爱国主题词：加入美国总统名（华盛顿/林肯/罗斯福/肯尼迪等）、
     美国象征（自由女神/星条旗/山姆大叔/白宫）、军事术语（marines/veteran/GI）、
     美国历史事件（独立战争/南北战争/珍珠港/9·11/越战/登月）等
  2. 新增定义 E：精选美国大众熟知的爱国电影清单（116 部，含《拯救大兵瑞恩》
     《林肯》《独立日》《壮志凌云》《生于七月四日》《美国狙击手》《血战钢锯岭》等）
  3. 新增分析：7/4 当天被推荐最多的 top-30 电影清单（含体裁/国家/年份/主题/各定义命中标记），
     用于人工核查其中是否有爱国主题电影

【5 层爱国分类器】
  - A：扩充关键词命中（title + original_title + overview 含中英爱国关键词）
  - B：体裁代用品（genres ∈ {战争, 历史}）
  - C：A ∪ B 联合（最大召回）
  - D：B AND country 含 "United States of America"
  - E：精选 116 部知名美国爱国电影 IMDB ID 集合（人工策展，最精准）

【方法】
  实验组：2019–2022 共 4 个独立日（4 天）
  对照组：同年 7 月 ±14 天内 period ∈ {workday, weekend} 的非节假日日期
  主指标：patriotic_share = records_with_patriotic / total_records_with_movie
  检验：2×2 列联表 Pearson χ² + Fisher 精确检验 + Cohen's h + RR

【输出】
  output/movie/step7/i_independence_vs_control_patriotic.png   5 子图柱状图
  output/movie/step7/i_independence_vs_control_patriotic.csv  逐年 × 5 定义 + 汇总检验
  output/movie/step7/i_top10_patriotic_movies_on_independence.csv  7/4 当天 top-10 美国爱国片（定义 D）
  output/movie/step7/i_top30_movies_on_independence.csv  7/4 当天推荐最多的 top-30 电影（含各定义命中标记）
"""

import os            # 文件路径操作
import re             # 正则表达式（用于从 proc_text 提取 tt ID）
import csv            # CSV 输出
from collections import defaultdict, Counter   # 带默认值的字典、计数器
from datetime import date, timedelta   # 日期算术

import numpy as np                       # 数值计算
import pandas as pd                     # 表格处理
import matplotlib                       # 绘图基础库
matplotlib.use('Agg')                    # 非交互式后端
import matplotlib.pyplot as plt          # pyplot 接口
from scipy import stats                 # 卡方 / Fisher 检验

from movie.config import STEP_DIRS, setup_matplotlib, log   # 配置：输出目录、字体、日志
from movie.utils.text import build_conv_system, get_system_movie_ids   # 规则8 公共函数

setup_matplotlib()                      # 模块顶层初始化 matplotlib（中文字体等）
STEP_OUT = STEP_DIRS[7]                 # 复用 step7 输出目录：output/movie/step7/
os.makedirs(STEP_OUT, exist_ok=True)    # 确保目录存在


# ── 实验组：独立日日期 ──
INDEPENDENCE_DATES = {
    '2019-07-04', '2020-07-04', '2021-07-04', '2022-07-04',
}

# 对照窗口半径（天）：每年独立日前后 ±14 天
CONTROL_WINDOW_DAYS = 14


# ═══════════════════════════════════════════════════════════════════════
#  爱国主题关键词清单（v2 扩充版）
# ═══════════════════════════════════════════════════════════════════════
PATRIOTIC_KEYWORDS_CN = [
    # ── 独立/解放 ──
    '独立', '独立日', '独立宣言', '独立战争',
    '解放', '自由', '解救', '摆脱',
    # ── 爱国/国家 ──
    '爱国', '国父', '星条旗', '国旗', '国歌',
    '美利坚', '合众国', '建国', '立国', '为国', '报国', '效忠',
    # ── 美国总统名（中文译名）──
    '华盛顿', '林肯', '杰斐逊', '罗斯福', '肯尼迪', '里根',
    '艾森豪威尔', '杜鲁门', '尼克松', '卡特', '克林顿',
    '布什', '奥巴马', '特朗普', '拜登', '亚当斯',
    '麦迪逊', '门罗', '威尔逊', '约翰逊',
    # ── 美国象征 ──
    '自由女神', '自由钟', '白宫', '五角大楼', '国会山', '国会',
    '参议院', '最高法院', '宪法', '权利法案',
    '山姆大叔', '白头鹰', '老鹰', '鹰',
    # ── 美国历史事件 ──
    '革命', '南北战争', '美国内战', '美国革命', '美国独立战争',
    '二战', '越战', '越南战争', '韩战', '朝鲜战争',
    '珍珠港', '9·11', '911', '海湾战争', '阿富汗', '伊拉克战争',
    '登月', '阿波罗', '太空竞赛',
    '淘金热', '西部拓荒', '西进运动', '大萧条',
    # ── 民族/祖国 ──
    '民族', '祖国', '家园', '同胞',
    # ── 美国身份 ──
    '美国梦', '机会之地', '大熔炉', '先驱', '先驱者',
    # ── 军事术语 ──
    '海军', '陆军', '空军', '海军陆战队', '老兵', '军人', '部队', '战士',
    '征兵', '退役', '阵亡', '战俘', '荣誉勋章',
]
PATRIOTIC_KEYWORDS_EN = [
    # ── 独立/解放 ──
    'independence', 'freedom', 'liberty', 'emancipation', 'revolution',
    # ── 爱国/国家 ──
    'patriot', 'patriotic', 'founding father', 'declaration of independence',
    # ── 美国总统名（英文姓氏，仅 distinctive 的）──
    'washington', 'lincoln', 'jefferson', 'roosevelt', 'kennedy',
    'reagan', 'eisenhower', 'truman', 'nixon', 'clinton',
    'obama', 'biden', 'madison', 'monroe',
    # ── 美国象征 ──
    'statue of liberty', 'liberty bell', 'white house', 'pentagon',
    'capitol', 'congress', 'senate', 'supreme court',
    'constitution', 'bill of rights', 'uncle sam', 'bald eagle',
    # ── 美国历史事件 ──
    'revolutionary war', 'civil war', 'world war', 'vietnam war',
    'korean war', 'pearl harbor', '9/11', 'gulf war',
    'afghanistan', 'iraq war', 'cold war', 'moon landing',
    'apollo', 'space race', 'gold rush', 'frontier',
    'westward expansion', 'great depression',
    # ── 美国身份 ──
    'american dream', 'land of opportunity', 'melting pot',
    'forefather', 'homeland', 'motherland',
    # ── 军事术语 ──
    'marines', 'navy', 'army', 'air force', 'veteran', 'serviceman',
    'soldier', ' GI ', 'draft', 'deployment', 'platoon', 'battalion',
    'boot camp', 'medal of honor', 'prisoner of war', 'pow',
]


# ═══════════════════════════════════════════════════════════════════════
#  定义 E：精选 116 部美国大众熟知的爱国电影 IMDB ID 清单
#  （已确认全部在 movie_info.json 中）
# ═══════════════════════════════════════════════════════════════════════
CURATED_PATRIOTIC_IDS = {
    # ── 独立日 / 外星入侵（7/4 经典）──
    'tt0116629',  # Independence Day (1996)
    'tt1628841',  # Independence Day: Resurgence (2016)
    # ── 军事航空 ──
    'tt0092099',  # Top Gun (1986)
    'tt1745960',  # Top Gun: Maverick (2022)
    # ── 越南战争 ──
    'tt0096969',  # Born on the Fourth of July (1989)
    'tt0078788',  # Apocalypse Now (1979)
    'tt0091763',  # Platoon (1986)
    'tt0093058',  # Full Metal Jacket (1987)
    'tt0077416',  # The Deer Hunter (1978)
    'tt0083944',  # First Blood (1982)
    'tt0089880',  # Rambo: First Blood Part II (1985)
    'tt0063035',  # The Green Berets (1968)
    'tt0077362',  # Coming Home (1978)
    'tt0093105',  # Good Morning, Vietnam (1987)
    'tt0277434',  # We Were Soldiers (2002)
    'tt0097027',  # Casualties of War (1989)
    'tt0107096',  # Heaven & Earth (1993)
    # ── 二战 ──
    'tt0120815',  # Saving Private Ryan (1998)
    'tt0066206',  # Patton (1970)
    'tt6924650',  # Midway (2019)
    'tt0074899',  # Midway (1976)
    'tt0066473',  # Tora! Tora! Tora! (1970)
    'tt0056197',  # The Longest Day (1962)
    'tt0057115',  # The Great Escape (1963)
    'tt0034167',  # Sergeant York (1941)
    'tt0213149',  # Pearl Harbor (2001)
    'tt0418689',  # Flags of Our Fathers (2006)
    'tt0498380',  # Letters from Iwo Jima (2006)
    'tt2119532',  # Hacksaw Ridge (2016)
    'tt0046359',  # Stalag 17 (1953)
    'tt0080437',  # The Big Red One (1980)
    'tt0037366',  # Thirty Seconds Over Tokyo (1944)
    'tt0043887',  # Operation Pacific (1951)
    'tt0038160',  # They Were Expendable (1945)
    # ── 独立战争 / 建国 ──
    'tt0187393',  # The Patriot (2000)
    'tt0068156',  # 1776 (1972)
    'tt8503618',  # Hamilton (2025)
    # ── 南北战争 ──
    'tt0443272',  # Lincoln (2012)
    'tt0097441',  # Glory (1989)
    'tt0107007',  # Gettysburg (1993)
    'tt0279111',  # Gods and Generals (2003)
    'tt1124037',  # Free State of Jones (2016)
    'tt0159365',  # Cold Mountain (2003)
    # ── 现代战争（伊拉克/阿富汗）──
    'tt2179136',  # American Sniper (2014)
    'tt0887912',  # The Hurt Locker (2008)
    'tt1790885',  # Zero Dark Thirty (2012)
    'tt1091191',  # Lone Survivor (2013)
    'tt1413492',  # 12 Strong (2018)
    'tt0891527',  # Lions for Lambs (2007)
    'tt0947810',  # Green Zone (2010)
    'tt0790712',  # The Messenger (2009)
    'tt0478134',  # In the Valley of Elah (2007)
    'tt0937237',  # Redacted (2007)
    'tt0977855',  # Fair Game (2010)
    # ── 冷战 ──
    'tt0087985',  # Red Dawn (1984)
    'tt1234719',  # Red Dawn (2012)
    'tt0099810',  # The Hunt for Red October (1990)
    'tt0349825',  # Miracle (2004)
    'tt0089927',  # Rocky IV (1985)
    # ── 9/11 与反恐 ──
    'tt0475276',  # United 93 (2006)
    'tt0469641',  # World Trade Center (2006)
    'tt4572514',  # Patriots Day (2016)
    'tt0490204',  # Reign Over Me (2007)
    'tt0443274',  # Vantage Point (2008)
    'tt8236336',  # The Report (2019)
    'tt6266538',  # Vice (2018)
    # ── 总统/政治 ──
    'tt0102138',  # JFK (1991)
    'tt0113987',  # Nixon (1995)
    'tt1175491',  # W. (2008)
    'tt0870111',  # Frost/Nixon (2008)
    'tt0112346',  # The American President (1995)
    'tt0120885',  # Wag the Dog (1997)
    'tt0119942',  # Primary Colors (1998)
    'tt1124035',  # The Ides of March (2011)
    'tt6294822',  # The Post (2017)
    'tt0031679',  # Mr. Smith Goes to Washington (1939)
    'tt0106673',  # Dave (1993)
    'tt0118571',  # Air Force One (1997)
    'tt2302755',  # Olympus Has Fallen (2013)
    'tt2334879',  # White House Down (2013)
    'tt4778988',  # LBJ (2017)
    'tt0037465',  # Wilson (1944)
    'tt0032155',  # Young Mr. Lincoln (1939)
    # ── 太空/NASA 爱国 ──
    'tt0112384',  # Apollo 13 (1995)
    'tt1213641',  # First Man (2018)
    'tt4846340',  # Hidden Figures (2016)
    'tt0086197',  # The Right Stuff (1983)
    'tt0132477',  # October Sky (1999)
    # ── 美国历史/文化 ──
    'tt0109830',  # Forrest Gump (1994)
    'tt0372588',  # Team America: World Police (2004)
    'tt0090633',  # An American Tail (1986)
    # ── 美国队长系列 ──
    'tt0458339',  # Captain America: The First Avenger (2011)
    'tt1843866',  # Captain America: The Winter Soldier (2014)
    'tt3498820',  # Captain America: Civil War (2016)
    # ── 西部拓荒 ──
    'tt0099348',  # Dances with Wolves (1990)
    'tt0049730',  # The Searchers (1956)
    'tt0031971',  # Stagecoach (1939)
    'tt0040724',  # Red River (1948)
    'tt0056217',  # The Man Who Shot Liberty Valance (1962)
    # ── 灾难片/美国拯救世界 ──
    'tt0120591',  # Armageddon (1998)
    'tt0120647',  # Deep Impact (1998)
}


# 5 个定义的标签
DEFINITIONS = ['A', 'B', 'C', 'D', 'E']
DEFINITION_NAMES = {
    'A': 'A: Keyword (expanded)',
    'B': 'B: War/History genre',
    'C': 'C: A ∪ B (Union)',
    'D': 'D: B & US country',
    'E': 'E: Curated 116 ID list',
}


# ═══════════════════════════════════════════════════════════════════════
#  用户提问侧定义（分析用户提问内容，非系统推荐内容）
#  F：用户提问文本（raw_text，英文原文）含爱国关键词
#  G：用户提问中提及任一精选 116 部爱国电影 IMDB ID
# ═══════════════════════════════════════════════════════════════════════
USER_DEFINITIONS = ['F', 'G']
USER_DEFINITION_NAMES = {
    'F': 'F: User question contains patriotic keyword (EN)',
    'G': 'G: User question mentions curated patriotic movie ID',
}

# 用于从 proc_text 提取 tt ID 的正则（与 movie/utils/text.py 的 _TT_PATTERN 一致）
_USER_TT_PATTERN = re.compile(r'\b(tt\d{7,9})\b')


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 预计算每部电影在 5 个定义下的命中情况
# ═══════════════════════════════════════════════════════════════════════
def _build_patriotic_index(movie_info: dict) -> dict:
    """Precompute, for each movie, which patriotic definitions it matches.
       预计算每部电影在 5 个定义下的命中情况。

    Returns dict[mid -> {'A':bool, 'B':bool, 'C':bool, 'D':bool, 'E':bool}].
    """
    index = {}
    for mid, info in movie_info.items():
        if not isinstance(info, dict):
            continue

        # 定义 A：关键词命中（title + original_title + overview）
        text = (
            (info.get('title', '') or '') + ' ' +
            (info.get('original_title', '') or '') + ' ' +
            (info.get('overview', '') or '')
        ).lower()
        match_a = any(kw.lower() in text
                      for kw in PATRIOTIC_KEYWORDS_CN + PATRIOTIC_KEYWORDS_EN)

        # 定义 B：体裁代用品（War or History）
        genres = set(info.get('genres', []) or [])
        match_b = bool(genres & {'战争', '历史'})

        # 定义 C：A ∪ B 联合
        match_c = match_a or match_b

        # 定义 D：B AND country 含 United States of America
        country = str(info.get('country', '') or '')
        match_d = match_b and ('United States of America' in country)

        # 定义 E：精选 116 部爱国电影 ID 集合
        match_e = mid in CURATED_PATRIOTIC_IDS

        index[mid] = {
            'A': match_a,
            'B': match_b,
            'C': match_c,
            'D': match_d,
            'E': match_e,
        }
    return index


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 把 seekers 聚合为带 5 个定义标记的记录列表
# ═══════════════════════════════════════════════════════════════════════
def _aggregate_records(seekers, conv_system, patriotic_index):
    """Aggregate seekers into records with year/date/group + 5 patriotic flags.
       把 seekers 聚合为带 year/date/group + 5 个定义布尔标记的记录列表。

    实验组：date ∈ INDEPENDENCE_DATES
    对照组：同年 7 月内、距 7/4 ±14 天、period ∈ {workday, weekend}
            （排除所有 period=='holiday' 的日期，含其他节假日）
    排除规则：系统回复无电影 ID 的记录（无法判定类型）。
    """
    records = []
    for r in seekers:
        date_str = r.get('date', '')
        if not date_str:
            continue
        try:
            y, m, d = map(int, date_str.split('-'))
        except Exception:
            continue
        if m != 7:                        # 实验+对照窗口都在 7 月
            continue

        # 分组判定
        if date_str in INDEPENDENCE_DATES:
            group = 'Independence'
        elif r.get('period') in ('workday', 'weekend'):
            v_date = date(y, 7, 4)
            cur_date = date(y, m, d)
            delta = (cur_date - v_date).days
            if abs(delta) > CONTROL_WINDOW_DAYS or delta == 0:
                continue                  # 超出对照窗口或落在独立日当天
            group = 'Control'
        else:
            continue                      # period == 'holiday' 的其他节假日排除

        # 取该记录系统回复中的电影 ID 集合
        mids = get_system_movie_ids(r.get('conv_id', ''), conv_system)
        if not mids:                      # 系统回复无电影 ID，无法判定
            continue

        # 对 5 个定义分别判定：该记录是否至少推荐一部命中电影
        has = {k: False for k in DEFINITIONS}
        for mid in mids:
            flags = patriotic_index.get(mid)
            if flags:
                for k in DEFINITIONS:
                    if flags[k]:
                        has[k] = True

        records.append({
            'year': y,
            'date': date_str,
            'group': group,
            'has_A': has['A'],
            'has_B': has['B'],
            'has_C': has['C'],
            'has_D': has['D'],
            'has_E': has['E'],
            'movie_ids': list(mids),      # 保留以供 top-N 分析
        })
    return records


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 逐年 × 分组 聚合统计（单定义）
# ═══════════════════════════════════════════════════════════════════════
def _per_year_stats(records, def_key: str) -> pd.DataFrame:
    """Per-year per-group: total records, patriotic records, patriotic share.
       逐年 × 分组聚合：总记录数、含爱国片记录数、爱国片份额。"""
    agg = defaultdict(lambda: defaultdict(int))   # (year, group) -> {total, patriotic}
    for r in records:
        key = (r['year'], r['group'])
        agg[key]['total'] += 1
        if r[def_key]:
            agg[key]['patriotic'] += 1
    rows = []
    for (y, g), c in agg.items():
        rows.append({
            'year': y,
            'group': g,
            'n_records': c['total'],
            'n_patriotic': c['patriotic'],
            'patriotic_share': c['patriotic'] / c['total'] if c['total'] else 0.0,
        })
    df = pd.DataFrame(rows).sort_values(['year', 'group']).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 汇总统计 + 显著性检验（单定义）
# ═══════════════════════════════════════════════════════════════════════
def _pooled_stats(records, def_key: str) -> dict:
    """Pool all years and run 2x2 chi-square + Fisher + Cohen's h + RR.
       汇总全部年份做 2×2 列联表检验 + 效应量。"""
    val = [r for r in records if r['group'] == 'Independence']
    ctrl = [r for r in records if r['group'] == 'Control']
    val_n = len(val)
    val_r = sum(1 for r in val if r[def_key])
    ctrl_n = len(ctrl)
    ctrl_r = sum(1 for r in ctrl if r[def_key])
    val_share = val_r / val_n if val_n else 0.0
    ctrl_share = ctrl_r / ctrl_n if ctrl_n else 0.0

    # 2×2 列联表
    table = np.array([
        [val_r,  val_n  - val_r],
        [ctrl_r, ctrl_n - ctrl_r],
    ])

    # Pearson 卡方检验（默认带 Yates 连续性修正）
    if val_n > 0 and ctrl_n > 0 and table.sum() > 0:
        chi2, p_chi2, dof, expected = stats.chi2_contingency(table)
        odds_ratio, p_fisher = stats.fisher_exact(table, alternative='two-sided')
    else:
        chi2 = p_chi2 = odds_ratio = p_fisher = float('nan')

    # Cohen's h 效应量
    def _phi(p):
        return 2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0)))
    h = _phi(val_share) - _phi(ctrl_share)

    # 相对风险 RR
    rr = val_share / ctrl_share if ctrl_share > 0 else float('inf')

    return {
        'independence_n': val_n,
        'independence_patriotic': val_r,
        'independence_share': val_share,
        'control_n': ctrl_n,
        'control_patriotic': ctrl_r,
        'control_share': ctrl_share,
        'delta_share': val_share - ctrl_share,
        'relative_risk': rr,
        'cohens_h': h,
        'chi2': chi2,
        'p_chi2': p_chi2,
        'odds_ratio': odds_ratio,
        'p_fisher': p_fisher,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 5 子图柱状图（2x3 布局，第6格空）
# ═══════════════════════════════════════════════════════════════════════
def _plot_all_definitions(per_year_by_def, pooled_by_def, path):
    """5-subplot bar chart (2x3 grid, last cell empty):
       per-year Independence vs Control + 95% CI."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for idx, dkey in enumerate(DEFINITIONS):
        ax = axes_flat[idx]
        per_year = per_year_by_def[dkey]
        pooled = pooled_by_def[dkey]

        years = sorted(per_year['year'].unique())
        ind_shares, ctrl_shares, ind_errs, ctrl_errs = [], [], [], []
        for y in years:
            i_row = per_year[(per_year['year'] == y) & (per_year['group'] == 'Independence')]
            c_row = per_year[(per_year['year'] == y) & (per_year['group'] == 'Control')]
            is_ = i_row['patriotic_share'].iloc[0] if len(i_row) else 0.0
            in_ = i_row['n_records'].iloc[0]    if len(i_row) else 0
            cs_ = c_row['patriotic_share'].iloc[0] if len(c_row) else 0.0
            cn_ = c_row['n_records'].iloc[0]    if len(c_row) else 0
            ind_shares.append(is_)
            ctrl_shares.append(cs_)
            ind_errs.append(1.96 * np.sqrt(is_ * (1 - is_) / max(in_, 1)) if in_ else 0.0)
            ctrl_errs.append(1.96 * np.sqrt(cs_ * (1 - cs_) / max(cn_, 1)) if cn_ else 0.0)

        x = np.arange(len(years))
        w = 0.35
        ax.bar(x - w / 2, ind_shares,  w, yerr=ind_errs,  label='Independence',
               color='#e74c3c', alpha=0.85, capsize=4)
        ax.bar(x + w / 2, ctrl_shares, w, yerr=ctrl_errs,
               label=f'Control (±{CONTROL_WINDOW_DAYS}d, non-holiday)',
               color='#3498db', alpha=0.85, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_ylabel('Patriotic Share')
        ax.set_title(
            f"{DEFINITION_NAMES[dkey]}\n"
            f"Δ={pooled['delta_share']:.4f}, RR={pooled['relative_risk']:.2f}, "
            f"p(χ²)={pooled['p_chi2']:.4f}, p(F)={pooled['p_fisher']:.4f}, "
            f"h={pooled['cohens_h']:.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ymax = max(ind_shares + ctrl_shares) if (ind_shares or ctrl_shares) else 0.0
        ax.set_ylim(0, ymax * 1.4 + 0.02)

    # 隐藏第 6 个空格子
    axes_flat[5].axis('off')

    fig.suptitle("US Independence Day vs Control — Patriotic Movie Share (5 Definitions)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Helper: CSV 输出（5 个定义 + 汇总检验）
# ═══════════════════════════════════════════════════════════════════════
def _save_csv_all(per_year_by_def, pooled_by_def, path):
    """Save per-year stats + pooled tests for all 5 definitions to CSV."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        for dkey in DEFINITIONS:
            w.writerow([])
            w.writerow([f'# Definition {DEFINITION_NAMES[dkey]}'])
            w.writerow(['year', 'group', 'n_records', 'n_patriotic', 'patriotic_share'])
            for _, row in per_year_by_def[dkey].iterrows():
                w.writerow([row['year'], row['group'], row['n_records'],
                            row['n_patriotic'], f"{row['patriotic_share']:.4f}"])
            p = pooled_by_def[dkey]
            w.writerow(['# Pooled'])
            w.writerow(['group', 'n_records', 'n_patriotic', 'patriotic_share'])
            w.writerow(['Independence', p['independence_n'], p['independence_patriotic'],
                        f"{p['independence_share']:.4f}"])
            w.writerow(['Control', p['control_n'], p['control_patriotic'],
                        f"{p['control_share']:.4f}"])
            w.writerow(['# Tests'])
            w.writerow(['delta_share',   f"{p['delta_share']:.4f}"])
            w.writerow(['relative_risk', f"{p['relative_risk']:.4f}"])
            w.writerow(['cohens_h',      f"{p['cohens_h']:.4f}"])
            w.writerow(['chi2',          f"{p['chi2']:.4f}"])
            w.writerow(['p_chi2',        f"{p['p_chi2']:.4f}"])
            w.writerow(['odds_ratio',    f"{p['odds_ratio']:.4f}"])
            w.writerow(['p_fisher',     f"{p['p_fisher']:.4f}"])
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 7/4 当天 top-10 美国爱国片（按定义 D）
# ═══════════════════════════════════════════════════════════════════════
def _save_top10_movies(records, patriotic_index, movie_info, path):
    """Top-10 patriotic movies (definition D) recommended on Independence Day."""
    mid_counter = Counter()
    for r in records:
        if r['group'] != 'Independence':
            continue
        for mid in r['movie_ids']:
            flags = patriotic_index.get(mid)
            if flags and flags['D']:
                mid_counter[mid] += 1

    top = mid_counter.most_common(10)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'movie_id', 'title_cn', 'original_title',
                    'genres', 'country', 'year', 'n_records_recommended'])
        for rank, (mid, n) in enumerate(top, 1):
            info = movie_info.get(mid, {})
            title = info.get('title', '?')
            ot = info.get('original_title', '?')
            g = info.get('genres', [])
            c = info.get('country', '?')
            yr = info.get('year', '?')
            w.writerow([rank, mid, title, ot, '|'.join(g), c, yr, n])
    log(f"Saved: {path} ({len(top)} unique D-movies recommended)")


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 7/4 当天推荐最多的 top-30 电影（含各定义命中标记）
# ═══════════════════════════════════════════════════════════════════════
def _save_top30_movies_on_independence(records, patriotic_index, movie_info, path):
    """Top-30 most recommended movies on Independence Day, with patriotic flags.
       7/4 当天推荐最多的 top-30 电影，含体裁/国家/年份 + 各定义命中标记。"""
    mid_counter = Counter()
    for r in records:
        if r['group'] != 'Independence':
            continue
        for mid in r['movie_ids']:
            mid_counter[mid] += 1

    top = mid_counter.most_common(30)
    n_patriotic_any = 0
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'movie_id', 'title_cn', 'original_title',
                    'genres', 'country', 'year',
                    'n_records_recommended',
                    'match_A_keyword', 'match_B_genre', 'match_D_US_wh',
                    'match_E_curated', 'any_patriotic'])
        for rank, (mid, n) in enumerate(top, 1):
            info = movie_info.get(mid, {})
            title = info.get('title', '?')
            ot = info.get('original_title', '?')
            g = info.get('genres', [])
            c = info.get('country', '?')
            yr = info.get('year', '?')
            flags = patriotic_index.get(mid, {})
            ma = 'Y' if flags.get('A') else 'N'
            mb = 'Y' if flags.get('B') else 'N'
            md = 'Y' if flags.get('D') else 'N'
            me = 'Y' if flags.get('E') else 'N'
            any_p = 'Y' if (flags.get('A') or flags.get('B') or
                            flags.get('D') or flags.get('E')) else 'N'
            if any_p == 'Y':
                n_patriotic_any += 1
            w.writerow([rank, mid, title, ot, '|'.join(g), c, yr, n,
                        ma, mb, md, me, any_p])
    log(f"Saved: {path} ({n_patriotic_any}/{len(top)} top movies match ≥1 patriotic def)")


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 聚合用户提问记录（含 F/G 标记 + raw_text）
# ═══════════════════════════════════════════════════════════════════════
def _aggregate_user_records(seekers):
    """Aggregate seekers with user-side patriotic flags (F/G) + raw_text.
       聚合用户提问记录，计算 F/G 两个用户侧定义的命中标记。

    与 _aggregate_records 的区别：
      - 不需要 conv_system（不分析系统回复）
      - 不排除"系统回复无电影"的记录（分析用户提问本身）
      - 额外保留 raw_text 用于关键词频率分析
    """
    records = []
    for r in seekers:
        date_str = r.get('date', '')
        if not date_str:
            continue
        try:
            y, m, d = map(int, date_str.split('-'))
        except Exception:
            continue
        if m != 7:                        # 实验+对照窗口都在 7 月
            continue

        # 分组判定（与 _aggregate_records 一致）
        if date_str in INDEPENDENCE_DATES:
            group = 'Independence'
        elif r.get('period') in ('workday', 'weekend'):
            v_date = date(y, 7, 4)
            cur_date = date(y, m, d)
            delta = (cur_date - v_date).days
            if abs(delta) > CONTROL_WINDOW_DAYS or delta == 0:
                continue
            group = 'Control'
        else:
            continue                      # period == 'holiday' 的其他节假日排除

        raw_text = r.get('raw_text', '') or ''
        proc_text = r.get('proc_text', '') or ''

        # 定义 F：用户提问文本（raw_text，英文原文）含爱国关键词
        text_lower = raw_text.lower()
        has_f = any(kw.lower() in text_lower
                    for kw in PATRIOTIC_KEYWORDS_EN)

        # 定义 G：用户提问中提及任一精选爱国电影 ID
        # 优先使用已解析的 imdb_ids 字段，fallback 用正则从 proc_text 提取
        user_imdb_ids = set(r.get('imdb_ids', []) or [])
        if not user_imdb_ids:
            user_imdb_ids = set(_USER_TT_PATTERN.findall(str(proc_text)))
        has_g = bool(user_imdb_ids & CURATED_PATRIOTIC_IDS)

        records.append({
            'year': y,
            'date': date_str,
            'group': group,
            'has_F': has_f,
            'has_G': has_g,
            'raw_text': raw_text,
        })
    return records


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 2 子图柱状图（用户侧定义 F/G）
# ═══════════════════════════════════════════════════════════════════════
def _plot_user_definitions(per_year_by_ndef, pooled_by_ndef, path):
    """2-subplot bar chart for user-side definitions F/G + 95% CI."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, dkey in enumerate(USER_DEFINITIONS):
        ax = axes[idx]
        per_year = per_year_by_ndef[dkey]
        pooled = pooled_by_ndef[dkey]

        years = sorted(per_year['year'].unique())
        ind_shares, ctrl_shares, ind_errs, ctrl_errs = [], [], [], []
        for y in years:
            i_row = per_year[(per_year['year'] == y) & (per_year['group'] == 'Independence')]
            c_row = per_year[(per_year['year'] == y) & (per_year['group'] == 'Control')]
            is_ = i_row['patriotic_share'].iloc[0] if len(i_row) else 0.0
            in_ = i_row['n_records'].iloc[0]    if len(i_row) else 0
            cs_ = c_row['patriotic_share'].iloc[0] if len(c_row) else 0.0
            cn_ = c_row['n_records'].iloc[0]    if len(c_row) else 0
            ind_shares.append(is_)
            ctrl_shares.append(cs_)
            ind_errs.append(1.96 * np.sqrt(is_ * (1 - is_) / max(in_, 1)) if in_ else 0.0)
            ctrl_errs.append(1.96 * np.sqrt(cs_ * (1 - cs_) / max(cn_, 1)) if cn_ else 0.0)

        x = np.arange(len(years))
        w = 0.35
        ax.bar(x - w / 2, ind_shares,  w, yerr=ind_errs,  label='Independence',
               color='#e74c3c', alpha=0.85, capsize=4)
        ax.bar(x + w / 2, ctrl_shares, w, yerr=ctrl_errs,
               label=f'Control (±{CONTROL_WINDOW_DAYS}d, non-holiday)',
               color='#3498db', alpha=0.85, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_ylabel('User-Question Patriotic Share')
        ax.set_title(
            f"{USER_DEFINITION_NAMES[dkey]}\n"
            f"Δ={pooled['delta_share']:.4f}, RR={pooled['relative_risk']:.2f}, "
            f"p(χ²)={pooled['p_chi2']:.4f}, p(F)={pooled['p_fisher']:.4f}, "
            f"h={pooled['cohens_h']:.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ymax = max(ind_shares + ctrl_shares) if (ind_shares or ctrl_shares) else 0.0
        ax.set_ylim(0, ymax * 1.4 + 0.02)

    fig.suptitle("User Question Content: Patriotic Interest on Independence Day vs Control",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 用户侧分析 CSV 输出（F/G 两个定义）
# ═══════════════════════════════════════════════════════════════════════
def _save_user_question_csv(per_year_by_ndef, pooled_by_ndef, path):
    """Save user-side per-year stats + pooled tests for F/G to CSV."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        for dkey in USER_DEFINITIONS:
            w.writerow([])
            w.writerow([f'# User Definition {USER_DEFINITION_NAMES[dkey]}'])
            w.writerow(['year', 'group', 'n_records', 'n_patriotic', 'patriotic_share'])
            for _, row in per_year_by_ndef[dkey].iterrows():
                w.writerow([row['year'], row['group'], row['n_records'],
                            row['n_patriotic'], f"{row['patriotic_share']:.4f}"])
            p = pooled_by_ndef[dkey]
            w.writerow(['# Pooled'])
            w.writerow(['group', 'n_records', 'n_patriotic', 'patriotic_share'])
            w.writerow(['Independence', p['independence_n'], p['independence_patriotic'],
                        f"{p['independence_share']:.4f}"])
            w.writerow(['Control', p['control_n'], p['control_patriotic'],
                        f"{p['control_share']:.4f}"])
            w.writerow(['# Tests'])
            w.writerow(['delta_share',   f"{p['delta_share']:.4f}"])
            w.writerow(['relative_risk', f"{p['relative_risk']:.4f}"])
            w.writerow(['cohens_h',      f"{p['cohens_h']:.4f}"])
            w.writerow(['chi2',          f"{p['chi2']:.4f}"])
            w.writerow(['p_chi2',        f"{p['p_chi2']:.4f}"])
            w.writerow(['odds_ratio',    f"{p['odds_ratio']:.4f}"])
            w.writerow(['p_fisher',     f"{p['p_fisher']:.4f}"])
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Helper: top-30 关键词频率对比（Independency vs Control 用户提问）
# ═══════════════════════════════════════════════════════════════════════
def _save_top_keywords_csv(user_records, path):
    """Top-30 patriotic keywords sorted by Independence/Control share ratio.
       按 Ind/Ctrl 份额比排序的 top-30 爱国关键词，看 7/4 用户提问中哪些词更频繁。"""
    ind_recs = [r for r in user_records if r['group'] == 'Independence']
    ctrl_recs = [r for r in user_records if r['group'] == 'Control']
    n_ind = len(ind_recs)
    n_ctrl = len(ctrl_recs)

    keyword_stats = []
    for kw in PATRIOTIC_KEYWORDS_EN:
        kw_lower = kw.lower()
        ind_count = sum(1 for r in ind_recs if kw_lower in (r['raw_text'] or '').lower())
        ctrl_count = sum(1 for r in ctrl_recs if kw_lower in (r['raw_text'] or '').lower())
        ind_share = ind_count / max(n_ind, 1)
        ctrl_share = ctrl_count / max(n_ctrl, 1)
        ratio = ind_share / max(ctrl_share, 1e-6)
        keyword_stats.append({
            'keyword': kw,
            'ind_count': ind_count,
            'ctrl_count': ctrl_count,
            'ind_share': ind_share,
            'ctrl_share': ctrl_share,
            'ratio': ratio,
            'delta': ind_share - ctrl_share,
        })

    # 按比率降序（7/4 更频繁的关键词排前面）
    keyword_stats.sort(key=lambda x: x['ratio'], reverse=True)

    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'keyword', 'ind_count', 'ctrl_count',
                    'ind_share', 'ctrl_share', 'ratio_ind_to_ctrl', 'delta_share'])
        for rank, s in enumerate(keyword_stats[:30], 1):
            w.writerow([rank, s['keyword'], s['ind_count'], s['ctrl_count'],
                        f"{s['ind_share']:.4f}", f"{s['ctrl_share']:.4f}",
                        f"{s['ratio']:.2f}", f"{s['delta']:.4f}"])
    log(f"Saved: {path} (top 30 of {len(keyword_stats)} EN keywords)")


# ═══════════════════════════════════════════════════════════════════════
#  Main  主函数入口
# ═══════════════════════════════════════════════════════════════════════
def main(data: dict = None):
    log("=" * 60)
    log("Step 7.3 (v2): US Independence Day vs Patriotic Movies — Verification")
    log("=" * 60)

    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']
    rows = data['rows']
    movie_info = data['movie_info']

    log(f"Building conv_system from {len(rows)} rows ...")
    conv_system = build_conv_system(rows)

    log("Building patriotic movie index (5 definitions, v2 expanded keywords + curated list) ...")
    patriotic_index = _build_patriotic_index(movie_info)
    db_stats = {k: sum(1 for f in patriotic_index.values() if f[k]) for k in DEFINITIONS}
    n_movies = len(movie_info)
    for k in DEFINITIONS:
        log(f"  Definition {k} ({DEFINITION_NAMES[k]}): "
            f"{db_stats[k]} movies ({100*db_stats[k]/n_movies:.2f}%)")

    log(f"Processing {len(seekers)} seekers ...")
    records = _aggregate_records(seekers, conv_system, patriotic_index)
    n_ind = sum(1 for r in records if r['group'] == 'Independence')
    n_ctrl = sum(1 for r in records if r['group'] == 'Control')
    log(f"Valid records: Independence={n_ind}, Control={n_ctrl}")

    # 对 5 个定义分别计算逐年 + 汇总
    per_year_by_def = {k: _per_year_stats(records, f'has_{k}') for k in DEFINITIONS}
    pooled_by_def = {k: _pooled_stats(records, f'has_{k}') for k in DEFINITIONS}

    # 逐年日志
    log("")
    log("Per-year stats (per definition):")
    for dkey in DEFINITIONS:
        log(f"  --- Definition {DEFINITION_NAMES[dkey]} ---")
        for _, row in per_year_by_def[dkey].iterrows():
            log(f"    {row['year']} {row['group']:>13}: n={row['n_records']:>5}, "
                f"patriotic={row['n_patriotic']:>4}, "
                f"share={row['patriotic_share']:.4f}")

    # 汇总 + 检验日志
    log("")
    log("Pooled test results:")
    for dkey in DEFINITIONS:
        p = pooled_by_def[dkey]
        log(f"  {DEFINITION_NAMES[dkey]}:")
        log(f"    Independence: n={p['independence_n']}, patriotic={p['independence_patriotic']}, "
            f"share={p['independence_share']:.4f}")
        log(f"    Control:      n={p['control_n']}, patriotic={p['control_patriotic']}, "
            f"share={p['control_share']:.4f}")
        log(f"    Δ={p['delta_share']:.4f}  RR={p['relative_risk']:.3f}  "
            f"Cohen's h={p['cohens_h']:.3f}")
        log(f"    χ²={p['chi2']:.3f}  p(χ²)={p['p_chi2']:.4f}  "
            f"OR={p['odds_ratio']:.3f}  p(Fisher)={p['p_fisher']:.4f}")

    # 综合判定
    log("")
    log("=== Overall Verdict ===")
    sig_results = []
    for dkey in DEFINITIONS:
        p = pooled_by_def[dkey]
        if not np.isnan(p['p_chi2']) and p['p_chi2'] < 0.05 and p['delta_share'] > 0:
            sig_results.append((dkey, '+'))
        elif not np.isnan(p['p_chi2']) and p['p_chi2'] < 0.05 and p['delta_share'] < 0:
            sig_results.append((dkey, '-'))
        else:
            sig_results.append((dkey, '0'))

    plus = sum(1 for _, s in sig_results if s == '+')
    minus = sum(1 for _, s in sig_results if s == '-')
    zero = sum(1 for _, s in sig_results if s == '0')

    for dkey, sign in sig_results:
        log(f"  Definition {dkey}: {sign}")

    if plus == len(DEFINITIONS):
        verdict = "Independence Day recommends SIGNIFICANTLY MORE patriotic movies (all defs, p<0.05, Δ>0)"
    elif minus == len(DEFINITIONS):
        verdict = "Independence Day recommends SIGNIFICANTLY FEWER patriotic movies (all defs, p<0.05, Δ<0)"
    elif zero == len(DEFINITIONS):
        verdict = "No significant association between Independence Day and patriotic movies (all defs p>0.05)"
    else:
        verdict = f"INCONSISTENT across definitions ({plus}↑, {minus}↓, {zero}=)"
    log(f"  Verdict: {verdict}")

    # 输出
    _plot_all_definitions(
        per_year_by_def, pooled_by_def,
        os.path.join(STEP_OUT, 'i_independence_vs_control_patriotic.png'),
    )
    _save_csv_all(
        per_year_by_def, pooled_by_def,
        os.path.join(STEP_OUT, 'i_independence_vs_control_patriotic.csv'),
    )
    _save_top10_movies(
        records, patriotic_index, movie_info,
        os.path.join(STEP_OUT, 'i_top10_patriotic_movies_on_independence.csv'),
    )
    _save_top30_movies_on_independence(
        records, patriotic_index, movie_info,
        os.path.join(STEP_OUT, 'i_top30_movies_on_independence.csv'),
    )

    # ── 用户提问侧分析（新增）：分析用户提问内容是否含爱国主题 ──
    log("")
    log("-" * 40)
    log("Section B: User Question Content Analysis (F/G definitions)")
    log("-" * 40)

    user_records = _aggregate_user_records(seekers)
    n_ind_u = sum(1 for r in user_records if r['group'] == 'Independence')
    n_ctrl_u = sum(1 for r in user_records if r['group'] == 'Control')
    log(f"User records: Independence={n_ind_u}, Control={n_ctrl_u}")

    per_year_by_ndef = {k: _per_year_stats(user_records, f'has_{k}') for k in USER_DEFINITIONS}
    pooled_by_ndef = {k: _pooled_stats(user_records, f'has_{k}') for k in USER_DEFINITIONS}

    log("")
    log("Per-year stats (user-side):")
    for dkey in USER_DEFINITIONS:
        log(f"  --- {USER_DEFINITION_NAMES[dkey]} ---")
        for _, row in per_year_by_ndef[dkey].iterrows():
            log(f"    {row['year']} {row['group']:>13}: n={row['n_records']:>5}, "
                f"patriotic={row['n_patriotic']:>4}, "
                f"share={row['patriotic_share']:.4f}")

    log("")
    log("Pooled test results (user-side):")
    for dkey in USER_DEFINITIONS:
        p = pooled_by_ndef[dkey]
        log(f"  {USER_DEFINITION_NAMES[dkey]}:")
        log(f"    Independence: n={p['independence_n']}, patriotic={p['independence_patriotic']}, "
            f"share={p['independence_share']:.4f}")
        log(f"    Control:      n={p['control_n']}, patriotic={p['control_patriotic']}, "
            f"share={p['control_share']:.4f}")
        log(f"    Δ={p['delta_share']:.4f}  RR={p['relative_risk']:.3f}  "
            f"Cohen's h={p['cohens_h']:.3f}")
        log(f"    χ²={p['chi2']:.3f}  p(χ²)={p['p_chi2']:.4f}  "
            f"OR={p['odds_ratio']:.3f}  p(Fisher)={p['p_fisher']:.4f}")

    # 用户侧综合判定
    log("")
    log("=== User-Side Verdict ===")
    for dkey in USER_DEFINITIONS:
        p = pooled_by_ndef[dkey]
        if not np.isnan(p['p_chi2']) and p['p_chi2'] < 0.05 and p['delta_share'] > 0:
            sign = '+ (significantly MORE patriotic interest on Independence Day)'
        elif not np.isnan(p['p_chi2']) and p['p_chi2'] < 0.05 and p['delta_share'] < 0:
            sign = '- (significantly LESS patriotic interest on Independence Day)'
        else:
            sign = '0 (no significant difference)'
        log(f"  User Definition {dkey}: {sign}")

    # 输出用户侧分析文件
    _plot_user_definitions(
        per_year_by_ndef, pooled_by_ndef,
        os.path.join(STEP_OUT, 'i_user_question_patriotic.png'),
    )
    _save_user_question_csv(
        per_year_by_ndef, pooled_by_ndef,
        os.path.join(STEP_OUT, 'i_user_question_patriotic.csv'),
    )
    _save_top_keywords_csv(
        user_records,
        os.path.join(STEP_OUT, 'i_top30_keywords_independence_vs_control.csv'),
    )

    log("")
    log("=" * 60)
    log(f"Step 7.3 (v2) complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
