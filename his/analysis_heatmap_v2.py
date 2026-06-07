# -*- coding: utf-8 -*-
"""
LLM-Movie 情感分析脚本 v2.1 (基于 movie_info.json 类型提取)

功能：
  1. 加载 test-min-data.csv + holiday.csv
  2. 按年龄段分组用户 (复用 age_segment.py)
  3. 使用基于词典的情感分析方法(VADER + AFINN)进行情感分析
  4. 情感分为三类：正面 / 中性 / 负面
  5. 电影类型从 movie_info.json 中提取（按 IMDB ID 查找），不再依赖正则关键词猜测
  6. 对每组输出 3 张热图

使用方式:
  python analysis_heatmap_v2.py

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
CSV_PATH = os.path.join(DATA_DIR, 'my-test-data.csv')
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
    #VADER 的 compound 分数是经过归一化处理的，但在实际语料库测试中，
    # 作者发现将“中性”的范围设定在 [-0.05, 0.05] 之间能最好地匹配人类的情感标注
    if abs(vader_compound) > 0.5:
        if vader_compound >= 0.05:
            return '正面'
        elif vader_compound <= -0.05:
            return '负面'
        return '中性'

    # 加权融合：VADER compound 归一化到 [-1,1]，AFINN 归一化
    vader_norm = vader_compound
    # 如果发现很多明显的正面评价被分到了“中性”，原因很可能是 AFINN 的归一化除数 10.0 太大了。
    # 暂时先使用5，不行改为10
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
#  替换原来的正则关键词匹配，改为从 movie_info.json 中按 IMDB ID 提取类型

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
        # ast.literal_eval 可以安全地执行字符串到 Python 字面量的转换
        # 相比 eval()，它不会执行任意代码，更加安全
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def extract_user_text_from_context(context_raw) -> str | None:
    """
        从上下文字段中提取用户的实际说话内容。
        上下文通常是一个包含多个对话条目的列表。
        """
    # 1. 先调用 parse_raw_field 将原始字符串转为python对象
    ctx = parse_raw_field(context_raw)
    if not ctx or not isinstance(ctx, list):
        return None
    for entry in ctx:
        # 4. 检查条目格式：必须是列表、长度至少为2、且第一个元素是 'USER'
        if isinstance(entry, list) and len(entry) >= 2 and entry[0] == 'USER':
            # 5. 返回第二个元素，即用户的具体说话内容
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

        # [提取ID] 获取当前行的用户唯一标识
        user_id = row.get('user_id')
        # [匹配年龄] 根据用户ID在字典中查找对应的年龄段；如果找不到，默认为 'unknown'
        age_seg = user_seg.get(user_id, 'unknown')

        # ---- 情感分析（基于词典） ----
        user_text = extract_user_text_from_context(row.get('context_raw'))
        if not user_text:
            user_text = extract_user_text_from_context(row.get('context_processed'))
        # [情感打分] 调用混合分析函数，判断这句话的情感（正面/中性/负面）
        sentiment = classify_sentiment(user_text or '')

        # ---- 电影类型提取（从 movie_info.json） ----
        # 用 processed 字段提取 IMDB ID（processed 中 SYSTEM 消息已将电影名替换为 ttID）
        sys_text = extract_movie_text_from_system(row.get('processed'))
        if not sys_text:
            continue

        imdb_ids = extract_imdb_ids(sys_text)
        if not imdb_ids:
            continue

        # [类型分类] 根据系统文本中的关键词，判断电影属于哪些类型（如['科幻', '动作']）
        genres = classify_genre_from_ids(imdb_ids, movie_info, not_found_ids)
        if not genres:
            continue

        # ---- 节假日判断 ----
        # [获取时间戳] 提取该条对话发生的 UTC 时间戳
        utc_val = row.get('utc_time')
        holiday_name = '非节假日'
        try:
            # [时间转换] 将整数型时间戳转换为日期时间对象 (datetime)
            dt = datetime.utcfromtimestamp(int(utc_val))
            if cal.is_holiday(dt):
                # [获取名称] 如果是假日，获取具体的节日名称列表（如['春节']）
                names = cal.get_holiday_names(dt)
                if names:
                    holiday_name = names[0]
        except (ValueError, TypeError, OverflowError):
            pass

        # [展开记录] 因为一部电影可能有多个类型标签，需要为每个标签生成一条独立记录
        for genre in genres:
            # [存入列表] 将这一行数据的所有关键特征打包成字典，加入总记录表
            records.append({
                'age_segment': age_seg,      # 用户年龄段
                'holiday_name': holiday_name, # 是否节假日及名称
                'sentiment': sentiment,       # 用户情感倾向
                'genre': genre,               # 电影具体类型
            })

    # 报告未找到的 IMDB ID
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

    # [排序逻辑] 按照预定义的 AGE_SEGMENTS 顺序排列年龄段（如 18-25, 26-35...）
    age_groups = df_rec['age_segment'].unique()
    # 这样生成的图片顺序是固定的，不会乱跳
    age_order = [s for s in age_segment.AGE_SEGMENTS if s in age_groups]
    # 把那些不在预定义列表里的“意外”年龄段（如 'unknown'）排在最后
    age_order += sorted(set(age_groups) - set(age_segment.AGE_SEGMENTS))

    for age in age_order:
        print(f"\n{'='*55}")
        print(f"  年龄段: {age}")
        print(f"{'='*55}")

        # [筛选] 提取出当前年龄段的所有数据
        sub = df_rec[df_rec['age_segment'] == age]
        # [过滤] 如果数据量太少（少于3条），画图没有意义，直接跳过
        if len(sub) < 3:
            print(f"    数据不足 ({len(sub)} 条)，跳过")
            continue

        # [文件名处理] 清理年龄段字符串中的特殊字符，防止作为文件名时报错
        safe_age = _sanitize_filename(age)

        # ── 热图 1: 节假日 × 情感 ──
        # 目的：观察在不同假日，用户的情感是更积极还是更消极
        _plot_heatmap(
            data=sub,
            index='holiday_name',  # 纵轴：节假日名称
            columns='sentiment',  # 横轴：情感分类
            title=f'{age} — 假日场景下用户情感分析热图（movie_info 类型）',
            filename=f'v2_heatmap1_holiday_sentiment_{safe_age}.png',
        )

        # ── 热图 2: 情感 × 影片类型 ──
        # 目的：观察用户在不同情感下（如开心时）更喜欢看什么类型的电影
        _plot_heatmap(
            data=sub,
            index='sentiment',      # 纵轴：情感分类
            columns='genre',        # 横轴：电影类型
            title=f'{age} — 不同情感对用户观影类型影响热图（movie_info 类型）',
            filename=f'v2_heatmap2_sentiment_genre_{safe_age}.png',
        )

        # ── 热图 3: 节假日 × 影片类型 ──
        _plot_heatmap(
            data=sub,
            index='holiday_name',   # 纵轴：节假日名称
            columns='genre',        # 横轴：电影类型
            title=f'{age} — 不同假日对用户观影类型影响热图（movie_info 类型）',
            filename=f'v2_heatmap3_holiday_genre_{safe_age}.png',
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
    # [透视表] 将原始数据转换成二维矩阵。
    # aggfunc='size' 表示统计出现的次数（即观影次数/对话次数）
    # fill_value=0 表示如果某个交叉点没数据，就填 0
    pivot = data.pivot_table(
        index=index,
        columns=columns,
        aggfunc='size',
        fill_value=0,
    )

    # [排序] 按照每一行的总和（出现频次）从高到低排序，让重要的数据排在前面
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    # [检查] 如果透视表是空的，或者没有行列，就不画了
    if pivot.empty or pivot.shape[0] == 0 or pivot.shape[1] == 0:
        print(f"    \u26a0 跳过 {filename}: 透视表为空")
        return
    # [创建画布] 根据数据的行列数动态调整图片大小，防止文字挤在一起
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.2),
                                    max(5, pivot.shape[0] * 0.6)))
    # [绘制热力图]
    # annot=True: 在格子里显示数字
    # fmt='d': 数字格式为整数
    # cmap='YlOrRd': 颜色方案为黄-橙-红（越红代表次数越多）
    sns.heatmap(
        pivot, annot=True, fmt='d', cmap='YlOrRd',
        linewidths=0.5, ax=ax, cbar_kws={'label': '观影次数'},
    )
    # [美化] 设置标题、坐标轴标签和字体大小
    ax.set_title(title, fontsize=14, pad=16)
    ax.set_xlabel(columns, fontsize=11)
    ax.set_ylabel(index, fontsize=11)
    # [旋转] 横轴标签旋转30度，防止重叠；纵轴保持水平
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    # [布局] 自动调整边距，确保标签不会被切掉
    plt.tight_layout()

    # [保存] 拼接输出路径，将图片保存到 output 文件夹
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    # [清理] 关闭画布，释放内存（这对批量画图非常重要）
    plt.close(fig)
    print(f"    [OK] {filename}  ({pivot.shape[0]}x{pivot.shape[1]})")


# ══════════════════════════════════════════════════════════════════
#  5. 入口
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  LLM-Movie 情感分析脚本 v2.1（movie_info 类型提取）")
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
