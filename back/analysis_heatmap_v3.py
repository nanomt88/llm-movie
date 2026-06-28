# -*- coding: utf-8 -*-
"""
LLM-Movie 情感分析脚本 v3.0

功能：
  1. 加载 my-test-data.csv + holiday.csv
  2. 按年龄段分组用户 (复用 age_segment.py)
  3. 使用基于词典的情感分析方法(VADER + AFINN)进行情感分析
  4. 情感分为三类：正面 / 中性 / 负面
  5. 电影类型从 movie_info.json 中提取（按 IMDB ID 查找），不再依赖正则关键词猜测
  6. 对每组输出 3 张热图
     - 热图 1: 情感 × 节假日（按 description 分组）
     - 热图 2: 情感 × 影片类型
     - 热图 3: 节假日 × 影片类型

使用方式:
  python analysis_heatmap_v3.py

依赖:
  pip install vaderSentiment afinn nltk textblob pandas numpy matplotlib seaborn
"""

import ast
import json
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── 词典情感分析 ─────────────────────────────────────────────────────
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn

# ── 中文字体回退 ──────────────────────────────────────────────────
_CN_FONTS = [
    'Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei',
    'Noto Sans CJK SC', 'Source Han Sans SC',
]
_CN_FONT = None
for f in _CN_FONTS:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        _CN_FONT = f
        break
    except Exception:
        continue

if _CN_FONT:
    matplotlib.rcParams['font.sans-serif'] = [_CN_FONT] + matplotlib.rcParams['font.sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False


def _sanitize_filename(name: str) -> str:
    return name.replace('<', 'lt').replace('>', 'gt').replace('+', 'p').replace(' ', '_')


# ── 项目模块 ──────────────────────────────────────────────────────
from src import age_segment
from src.holiday_util import HolidayCalendar

# ══════════════════════════════════════════════════════════════════
#  0. 路径配置
# ══════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'my-train-data.csv')
HOLIDAY_CSV = os.path.join(DATA_DIR, 'holiday.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  1. 情感分析（基于词典）
# ══════════════════════════════════════════════════════════════════

SENTIMENT_CATEGORIES = ['正面', '中性', '负面']

# 初始化基于词典的分析器
_vader = SentimentIntensityAnalyzer()
_afinn = Afinn(language='en')

# 情感极性词/短语补充规则（领域自适应）
_POSITIVE_BOOST = re.compile(
    r'(?i)\b(great|amazing|awesome|fantastic|wonderful|excellent|love|perfect|'
    r'brilliant|incredible|must.?watch|highly.?recommend|best|masterpiece|'
    r'favorite|beautiful|outstanding|superb|terrific|magnificent|splendid)\b'
)
_NEGATIVE_BOOST = re.compile(
    r'(?i)\b(terrible|awful|horrible|boring|waste|disappointing|worst|'
    r'trash|garbage|dreadful|atrocious|pathetic|miserable|abysmal|'
    r'painful|annoying|frustrating|ridiculous|stupid)\b'
)


def classify_sentiment_vader(text: str) -> str:
    """使用 VADER 进行情感分析，返回 '正面'/'中性'/'负面'"""
    if not text or not isinstance(text, str):
        return '中性'
    scores = _vader.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return '正面'
    elif compound <= -0.05:
        return '负面'
    else:
        return '中性'


def classify_sentiment_afinn(text: str) -> str:
    """使用 AFINN 进行情感分析，返回 '正面'/'中性'/'负面'"""
    if not text or not isinstance(text, str):
        return '中性'
    score = _afinn.score(text)
    if score > 0:
        return '正面'
    elif score < 0:
        return '负面'
    else:
        return '中性'


def classify_sentiment_hybrid(text: str) -> str:
    """
    混合使用 VADER + AFINN + 领域关键词规则进行情感分析。

    策略：
      1. 先检查领域关键词规则（高置信度）
      2. VADER 和 AFINN 加权投票
      3. 若 VADER compound 分数绝对值 > 0.5，则信任 VADER
      4. 否则取两个分析器的平均值
    """
    if not text or not isinstance(text, str):
        return '中性'

    # 领域关键词规则（高置信度覆盖）
    has_positive = _POSITIVE_BOOST.search(text)
    has_negative = _NEGATIVE_BOOST.search(text)
    if has_positive and not has_negative:
        return '正面'
    if has_negative and not has_positive:
        return '负面'

    # VADER
    vader_scores = _vader.polarity_scores(text)
    vader_compound = vader_scores['compound']

    # AFINN
    afinn_score = _afinn.score(text)

    # 如果 VADER 置信度较高，直接使用 VADER
    if abs(vader_compound) > 0.5:
        if vader_compound >= 0.05:
            return '正面'
        elif vader_compound <= -0.05:
            return '负面'
        return '中性'

    # 加权融合：VADER compound 归一化到 [-1,1]，AFINN 归一化
    vader_norm = vader_compound
    afinn_norm = np.clip(afinn_score / 10.0, -1.0, 1.0)

    combined = 0.6 * vader_norm + 0.4 * afinn_norm

    if combined >= 0.1:
        return '正面'
    elif combined <= -0.1:
        return '负面'
    else:
        return '中性'


# 默认使用混合方法
classify_sentiment = classify_sentiment_hybrid

# ══════════════════════════════════════════════════════════════════
#  2. 电影类型分类（基于 movie_info.json）
# ══════════════════════════════════════════════════════════════════

GENRE_CATEGORIES = [
    '动作', '喜剧', '剧情', '恐怖', '爱情',
    '科幻', '惊悚', '动画', '奇幻', '悬疑/犯罪',
]

MOVIE_INFO_PATH = os.path.join(DATA_DIR, 'movie_info.json')
_movie_info_cache: dict | None = None


def _load_movie_info() -> dict:
    """加载 movie_info.json 到内存缓存"""
    global _movie_info_cache
    if _movie_info_cache is None:
        with open(MOVIE_INFO_PATH, 'r', encoding='utf-8') as f:
            _movie_info_cache = json.load(f)
    return _movie_info_cache


_IMDB_ID_RE = re.compile(r'tt\d+')


def extract_imdb_ids(text: str) -> list[str]:
    """
    从文本中提取所有 IMDB ID (ttXXXXXXX) 并去重。
    返回保持首次出现顺序的列表。
    """
    if not text:
        return []
    seen = set()
    result = []
    for m in _IMDB_ID_RE.finditer(text):
        tid = m.group()
        if tid not in seen:
            seen.add(tid)
            result.append(tid)
    return result


def get_genres_from_movie_info(imdb_id: str,
                                movie_info: dict,
                                not_found_log: list[str]) -> list[str] | None:
    """
    从 movie_info 中按 IMDB ID 查找电影类型。

    返回：
      - 类型列表（若找到）
      - None（若未找到，将 ID 记录到 not_found_log）
    """
    info = movie_info.get(imdb_id)
    if info is None:
        not_found_log.append(imdb_id)
        return None
    genres = info.get('genres', [])
    return genres


def classify_genre_from_ids(imdb_ids: list[str],
                            movie_info: dict,
                            not_found_log: list[str]) -> list[str]:
    """
    对一组 IMDB ID 查找电影类型，合并去重后返回。
    未找到的 ID 会记录到 not_found_log。
    """
    all_genres = set()
    for imdb_id in imdb_ids:
        genres = get_genres_from_movie_info(imdb_id, movie_info, not_found_log)
        if genres is not None:
            all_genres.update(genres)
    return list(all_genres) if all_genres else ['剧情']


# ══════════════════════════════════════════════════════════════════
#  3. 数据加载与解析
# ══════════════════════════════════════════════════════════════════

def parse_raw_field(value):
    """
      将 CSV 中的字符串字段安全地转换为 Python 对象（通常是列表）。
      例如：将 "['USER', '内容']" 转换为 ['USER', '内容']
      """
    if pd.isna(value) or not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def extract_user_text_from_context(context_raw) -> str | None:
    """
       从上下文字段中提取用户的实际说话内容。
       上下文通常是一个包含多个对话条目的列表。
       """
    ctx = parse_raw_field(context_raw)
    if not ctx or not isinstance(ctx, list):
        return None
    for entry in ctx:
        if isinstance(entry, list) and len(entry) >= 2 and entry[0] == 'USER':
            return entry[1]
    return None


def extract_movie_text_from_system(raw_value) -> str | None:
    """
      从系统推荐字段中提取电影相关的描述文本。
      通常用于获取 SYSTEM 角色推荐的电影名称或简介。
      """
    parsed = parse_raw_field(raw_value)
    if not parsed or not isinstance(parsed, list) or len(parsed) < 2:
        return None
    if parsed[0] == 'SYSTEM':
        return parsed[1]
    return None


def extract_user_text_direct(processed_value) -> str | None:
    """
    直接从处理后的字段中提取用户发言。
    与 extract_user_text_from_context 不同，这个函数假设字段里只有一条对话。
    """
    parsed = parse_raw_field(processed_value)
    if not parsed or not isinstance(parsed, list) or len(parsed) < 2:
        return None
    if parsed[0] == 'USER':
        return parsed[1]
    return None


# ══════════════════════════════════════════════════════════════════
#  4. 主分析流程
# ══════════════════════════════════════════════════════════════════

def build_analysis_data(df: pd.DataFrame, cal: HolidayCalendar,
                        user_seg: dict) -> list[dict]:
    """
      核心数据组装函数：遍历原始数据，提取特征并生成分析记录。
      :param df: 原始对话数据表 (DataFrame)
      :param cal: 节假日日历对象，用于判断日期属性
      :param user_seg: 用户年龄段映射字典 {user_id: age_segment}
      :return: 包含所有分析记录的列表
      """
    movie_info = _load_movie_info()
    not_found_ids: list[str] = []
    records = []

    for idx, row in df.iterrows():
        if row.get('is_seeker') != False:
            continue

        user_id = row.get('user_id')
        age_seg = user_seg.get(user_id, 'unknown')

        # ---- 情感分析（基于词典） ----
        user_text = extract_user_text_from_context(row.get('context_raw'))
        if not user_text:
            user_text = extract_user_text_from_context(row.get('context_processed'))
        sentiment = classify_sentiment(user_text or '')

        # ---- 电影类型提取（从 movie_info.json） ----
        sys_text = extract_movie_text_from_system(row.get('processed'))
        if not sys_text:
            continue

        imdb_ids = extract_imdb_ids(sys_text)
        if not imdb_ids:
            continue

        genres = classify_genre_from_ids(imdb_ids, movie_info, not_found_ids)
        if not genres:
            continue

        # ---- 节假日判断 ----
        utc_val = row.get('utc_time')
        holiday_name = '非节假日'
        try:
            dt = datetime.utcfromtimestamp(int(utc_val))
            if cal.is_holiday(dt):
                names = cal.get_holiday_names(dt)
                if names:
                    holiday_name = names[0]
        except (ValueError, TypeError, OverflowError):
            pass

        for genre in genres:
            records.append({
                'age_segment': age_seg,
                'holiday_name': holiday_name,
                'sentiment': sentiment,
                'genre': genre,
            })

    if not_found_ids:
        unique_not_found = sorted(set(not_found_ids))
        print(f"\n  [!] {len(unique_not_found)} 个 IMDB ID 在 movie_info.json 中未找到:")
        for tid in unique_not_found:
            print(f"      - {tid}")

    return records


def generate_heatmaps(records: list[dict]):
    """
      根据分析记录生成多维度的热力图。
      它会按年龄段分组，为每一组用户生成 3 张热图。
      """
    df_rec = pd.DataFrame(records)
    print(f"  总分析记录数: {len(df_rec)}")
    print(f"  情感分布:\n{df_rec['sentiment'].value_counts().to_string()}\n")

    age_groups = df_rec['age_segment'].unique()
    age_order = [s for s in age_segment.AGE_SEGMENTS if s in age_groups]
    age_order += sorted(set(age_groups) - set(age_segment.AGE_SEGMENTS))

    for age in age_order:
        print(f"\n{'='*55}")
        print(f"  年龄段: {age}")
        print(f"{'='*55}")

        sub = df_rec[df_rec['age_segment'] == age]
        if len(sub) < 3:
            print(f"    数据不足 ({len(sub)} 条)，跳过")
            continue

        safe_age = _sanitize_filename(age)

        # ── 热图 1: 情感 × 节假日（按 holiday.csv description 分组） ──
        # 横轴：节假日名称（按 description 分组去重）
        # 纵轴：情感类型（正面/中性/负面）
        # 颜色：观影次数（某组用户 × 某节假日 × 某种情感的记录总数）
        sub_holiday = sub[sub['holiday_name'] != '非节假日'].copy()
        if not sub_holiday.empty:
            # 构建透视表：纵轴 sentiment，横轴 holiday_name
            pivot = sub_holiday.pivot_table(
                index='sentiment',
                columns='holiday_name',
                aggfunc='size',
                fill_value=0,
            )
            # 确保情感行顺序固定：正面、中性、负面
            row_order = [s for s in SENTIMENT_CATEGORIES if s in pivot.index]
            pivot = pivot.loc[row_order]

            # 节假日列按总频次降序排列
            col_order = pivot.sum(axis=0).sort_values(ascending=False).index
            pivot = pivot[col_order]

            if pivot.shape[0] > 0 and pivot.shape[1] > 0:
                fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.2),
                                                max(5, pivot.shape[0] * 0.6)))
                sns.heatmap(
                    pivot, annot=True, fmt='d', cmap='YlOrRd',
                    linewidths=0.5, ax=ax, cbar_kws={'label': '观影次数'},
                )
                ax.set_title(f'{age} — 不同节假日用户情感倾向热图', fontsize=14, pad=16)
                ax.set_xlabel('节假日', fontsize=11)
                ax.set_ylabel('情感类型', fontsize=11)
                plt.xticks(rotation=30, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()

                out_path = os.path.join(OUTPUT_DIR,
                                        f'v3_heatmap1_sentiment_holiday_{safe_age}.png')
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    [OK] v3_heatmap1_sentiment_holiday_{safe_age}.png  "
                      f"({pivot.shape[0]}x{pivot.shape[1]})")
            else:
                print(f"    \u26a0 跳过热图1: 透视表为空")
        else:
            print(f"    \u26a0 跳过热图1: 无节假日数据")

        # ── 热图 2: 情感 × 影片类型 ──
        _plot_heatmap(
            data=sub,
            index='sentiment',
            columns='genre',
            title=f'{age} — 不同情感对用户观影类型影响热图（movie_info 类型）',
            filename=f'v3_heatmap2_sentiment_genre_{safe_age}.png',
        )

        # ── 热图 3: 节假日 × 影片类型 ──
        _plot_heatmap(
            data=sub,
            index='holiday_name',
            columns='genre',
            title=f'{age} — 不同假日对用户观影类型影响热图（movie_info 类型）',
            filename=f'v3_heatmap3_holiday_genre_{safe_age}.png',
        )


def _plot_heatmap(data: pd.DataFrame, index: str, columns: str,
                  title: str, filename: str):
    """
      通用的热力图绘制函数。
      :param data: 当前年龄段的数据集
      :param index: 热力图的纵轴字段名
      :param columns: 热力图的横轴字段名
      :param title: 图片标题
      :param filename: 保存的文件名
      """
    pivot = data.pivot_table(
        index=index,
        columns=columns,
        aggfunc='size',
        fill_value=0,
    )

    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    if pivot.empty or pivot.shape[0] == 0 or pivot.shape[1] == 0:
        print(f"    \u26a0 跳过 {filename}: 透视表为空")
        return

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.2),
                                    max(5, pivot.shape[0] * 0.6)))
    sns.heatmap(
        pivot, annot=True, fmt='d', cmap='YlOrRd',
        linewidths=0.5, ax=ax, cbar_kws={'label': '观影次数'},
    )
    ax.set_title(title, fontsize=14, pad=16)
    ax.set_xlabel(columns, fontsize=11)
    ax.set_ylabel(index, fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [OK] {filename}  ({pivot.shape[0]}x{pivot.shape[1]})")


# ══════════════════════════════════════════════════════════════════
#  5. 入口
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  LLM-Movie 情感分析脚本 v3.0")
    print("=" * 55)

    print("\n[1/4] 用户年龄分段...")
    user_seg_df = age_segment.segment_users(CSV_PATH)
    user_seg = dict(zip(user_seg_df['user_id'], user_seg_df['age_segment']))
    known = user_seg_df[user_seg_df['age_segment'] != 'unknown']
    print(f"  总用户: {len(user_seg_df)}, 可分段: {len(known)} ({len(known)/max(len(user_seg_df),1)*100:.1f}%)")

    print("\n[2/4] 加载节假日日历...")
    cal = HolidayCalendar(HOLIDAY_CSV)
    print(f"  共 {len(cal.all_holidays)} 个节假日")

    print("\n[3/4] 加载 movie_info.json...")
    movie_info = _load_movie_info()
    print(f"  共 {len(movie_info)} 部电影")

    print("\n[4/4] 加载对话数据并构建分析记录...")
    df = pd.read_csv(CSV_PATH)
    records = build_analysis_data(df, cal, user_seg)
    print(f"  原始行数: {len(df)}, 分析记录数: {len(records)}")

    print("\n[5/4] 生成热图...")
    generate_heatmaps(records)

    print(f"\n{'='*55}")
    print(f"  [OK] 完成！所有图片保存至: {OUTPUT_DIR}")
    print(f"{'='*55}")

    print("\n\n== 摘要统计 ==")
    print("-" * 55)
    df_rec = pd.DataFrame(records)
    for age in sorted(df_rec['age_segment'].unique()):
        sub = df_rec[df_rec['age_segment'] == age]
        print(f"\n  [{age}] 共 {len(sub)} 条记录")
        print(f"    情感 TOP3: {sub['sentiment'].value_counts().head(3).to_dict()}")
        print(f"    类型 TOP3: {sub['genre'].value_counts().head(3).to_dict()}")
        hcount = sub[sub['holiday_name'] != '非节假日']
        print(f"    节假日记录: {len(hcount)} 条")
        if len(hcount) > 0:
            print(f"    节假日 TOP3: {hcount['holiday_name'].value_counts().head(3).to_dict()}")


if __name__ == '__main__':
    main()
