# -*- coding: utf-8 -*-
"""
==============================
LLM-Movie 数据分析脚本 v1.0
==============================

功能：
  1. 加载 test-min-data.csv + holiday.csv
  2. 按年龄段分组用户 (复用 age_segment.py)
  3. 对每组输出 3 张热图：
      ① 假日 × 情感       (颜色=观影次数)
      ② 情感 × 影片类型    (颜色=观影次数)
      ③ 假日 × 影片类型    (颜色=观影次数)

使用方式:
  python analysis_heatmap.py

依赖:
  pip install pandas numpy matplotlib seaborn
"""

import ast
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # 无头环境也可用
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    """将可能含特殊字符的字符串转为安全的文件名片段。"""
    return name.replace('<', 'lt').replace('>', 'gt').replace('+', 'p').replace(' ', '_')


# ── 项目模块 ──────────────────────────────────────────────────────
import age_segment
from holiday_util import HolidayCalendar

# ══════════════════════════════════════════════════════════════════
#  0. 路径配置
# ══════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'test-min-data.csv')
HOLIDAY_CSV = os.path.join(DATA_DIR, 'holiday.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  1. 情感分类（基于关键词）
# ══════════════════════════════════════════════════════════════════

EMOTION_CATEGORIES = [
    '快乐/轻松', '悲伤/抑郁', '恐惧/惊吓',
    '浪漫/温情', '兴奋/刺激', '怀旧/经典',
    '存在主义/沉思', '普通/综合',
]

_EMOTION_RULES = [
    # 快乐/轻松
    (r'(?i)\b(feel\s*good|happy|uplifting|fun\b|cheerful|joy|comedy|comic|lighthearted'
     r'|pick\s*me\s*up|feel\s*better|good\s*time|funny|hilarious|heartwarming)', '快乐/轻松'),
    (r'(?i)\b(laugh|smile|good\s*mood|enjoyable|upbeat|wholesome)', '快乐/轻松'),

    # 悲伤/抑郁
    (r'(?i)\b(sad|depress|depression|lonely|loneliness|heartbreaking|soul.?crushing'
     r'|cry\b|sobbing|tear|ugly\s*cry|dark\s*mood|emotional.?damage|melancholy)', '悲伤/抑郁'),
    (r'(?i)\b(gloomy|bleak|despair|hopeless|grief|mourning|tragic|break.?heart)', '悲伤/抑郁'),

    # 恐惧/惊吓
    (r'(?i)\b(scary|horror|jump.?scare|blood\b|gory|terrifying|creepy|spooky'
     r'|nightmare|frightening|haunt|chilling|disturbing)', '恐惧/惊吓'),
    (r'(?i)\b(horror\s*movie|scare\s*me|afraid|terrify)', '恐惧/惊吓'),

    # 浪漫/温情
    (r'(?i)\b(romance|romantic|rom.?com|love\s*story|date\s*night|date\b'
     r'|chick\s*flick|romcom|romantic\s*comedy|couple)', '浪漫/温情'),
    (r'(?i)\b(heartfelt|sweet|tender|love\b|relationship|valentine)', '浪漫/温情'),

    # 兴奋/刺激
    (r'(?i)\b(action|thriller|exciting|intense|suspense|edge.?of.?seat'
     r'|adrenaline|kick.?ass|bad.?ass|explosion|martial\s*arts|fight)', '兴奋/刺激'),
    (r'(?i)\b(blockbuster|high.?octane|pulse.?pounding|gripping)', '兴奋/刺激'),

    # 怀旧/经典
    (r'(?i)\b(nostalgia|nostalgic|classic\b|old\s*movie|80s|90s|childhood'
     r'|grew\s*up|retro|vintage|throwback)', '怀旧/经典'),
    (r'(?i)\b(remember|old\s*school|golden\s*age|timeless)', '怀旧/经典'),

    # 存在主义/沉思
    (r'(?i)\b(existential|wistful|thought.?provoking|contemplative|philosophical'
     r'|surreal|dreamy|slow.?paced|mind.?bending|psychedelic|trippy)', '存在主义/沉思'),
    (r'(?i)\b(deep\b|meaningful|introspective|profound|spiritual)', '存在主义/沉思'),
]


def classify_emotion(text: str) -> str:
    """从用户发言文本中识别情感类别，返回情感类型。"""
    if not text or not isinstance(text, str):
        return '普通/综合'
    for pattern, category in _EMOTION_RULES:
        if re.search(pattern, text):
            return category
    return '普通/综合'


# ══════════════════════════════════════════════════════════════════
#  2. 电影类型分类（基于电影名称/描述关键词）
# ══════════════════════════════════════════════════════════════════

GENRE_CATEGORIES = [
    '动作', '喜剧', '剧情', '恐怖', '爱情',
    '科幻', '惊悚', '动画', '奇幻', '悬疑/犯罪',
]

# 关键词 → 影片类型映射
_GENRE_KEYWORD_MAP: list[tuple[re.Pattern, str]] = [
    # 动作
    (re.compile(r'(?i)\b(action|fight|war\b|combat|martial\s*arts|gun|battle|explosion'
                r'|superhero|spy|agent| assassin|hunter|soldier|commando)', re.I), '动作'),
    # 喜剧
    (re.compile(r'(?i)\b(comedy|funny|hilarious|comedian|slapstick|parody|satire'
                r'|spoof|sitcom|laugh)', re.I), '喜剧'),
    # 剧情
    (re.compile(r'(?i)\b(drama|biopic|historical|period|based\s*on\s*a\s*true'
                r'|inspired\s*by|epic\s*| Oscar|family|life\b)', re.I), '剧情'),
    # 恐怖
    (re.compile(r'(?i)\b(horror|scary|creepy|ghost|haunt|zombie|vampire|demon|witch'
                r'|possession|exorcist|cannibal|slasher|serial\s*killer|gore)', re.I), '恐怖'),
    # 爱情
    (re.compile(r'(?i)\b(romance|romantic|love\s*|rom.?com|date\s*|kiss|wedding'
                r'|marriage|boyfriend|girlfriend|rom.?comedy)', re.I), '爱情'),
    # 科幻
    (re.compile(r'(?i)\b(sci.?fi|science\s*fiction|space|alien|future|dystopia|cyberpunk'
                r'|time\s*travel|robot|ai\b|artificial|mars|galaxy|star\s*|clone|apocalypse)', re.I), '科幻'),
    # 惊悚
    (re.compile(r'(?i)\b(thriller|suspense|mystery|psychological|mind.?bending'
                r'|twist|noir|paranoia|survival|cat.?.?mouse|chase)', re.I), '惊悚'),
    # 动画
    (re.compile(r'(?i)\b(animation|animated|cartoon|pixar|disney|gibli|anime|claymation'
                r'|stop.?motion|cgi|dreamworks)', re.I), '动画'),
    # 奇幻
    (re.compile(r'(?i)\b(fantasy|magic|wizard|dragon|sword|mythical|fairy|supernatural'
                r'|medieval|quest|enchanted|sorcerer|elvish|mythology)', re.I), '奇幻'),
    # 悬疑/犯罪
    (re.compile(r'(?i)\b(murder|detective|criminal|crime|investigation|noir|heist|gangster'
                r'|mafia|courtroom|trial|lawyer|mystery|cop\b|forensic)', re.I), '悬疑/犯罪'),
]


def classify_genre(movie_text: str) -> list[str]:
    """
    从一段电影推荐文案中提取所有出现的电影类型。
    返回类型列表（可能多个，去重）。
    """
    if not movie_text or not isinstance(movie_text, str):
        return ['剧情']
    genres = set()
    for pattern, genre in _GENRE_KEYWORD_MAP:
        if pattern.search(movie_text):
            genres.add(genre)
    return list(genres) if genres else ['剧情']


# ══════════════════════════════════════════════════════════════════
#  3. 数据加载与解析
# ══════════════════════════════════════════════════════════════════

def parse_raw_field(value):
    """
    解析 CSV 中的 processed/raw 字段（字符串表示的 list）。
    返回 Python list 或 None。
    """
    if pd.isna(value) or not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def extract_user_text_from_context(context_raw) -> str | None:
    """
    从 SYSTEM 行的 context_raw 中提取第一条 USER 发言文本。
    context_raw 格式: "[['USER', 'text...'], ...]"
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
    从 SYSTEM 行的 raw 字段提取推荐内容文本（电影名列表）。
    raw 格式: "['SYSTEM', 'movie1 \\nmovie2 ...']"
    """
    parsed = parse_raw_field(raw_value)
    if not parsed or not isinstance(parsed, list) or len(parsed) < 2:
        return None
    if parsed[0] == 'SYSTEM':
        return parsed[1]
    return None


def extract_user_text_direct(processed_value) -> str | None:
    """
    从 USER 行 (is_seeker=True) 的 processed 字段提取用户发言文本。
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
    核心流水线：遍历 DataFrame，构造分析记录。

    每条记录 = {
        'age_segment': 年龄段,
        'holiday_name': 节假日名称（非节假日='非节假日'）,
        'emotion': 情感类型,
        'genre': 影片类型,
    }

    逻辑：
      - 只处理 SYSTEM 行 (is_seeker=False)
      - 从 context_raw 提取用户情感
      - 从 raw 提取电影推荐 → 分类类型
      - 从 utc_time 判断是否节假日
    """
    records = []

    # 预计算用户年龄段
    # user_seg: dict[user_id, age_segment]

    for idx, row in df.iterrows():
        # 只处理推荐行
        if row.get('is_seeker') != False:
            continue

        user_id = row.get('user_id')
        age_seg = user_seg.get(user_id, 'unknown')

        # ---- 情感分析 ----
        user_text = extract_user_text_from_context(row.get('context_raw'))
        if not user_text:
            user_text = extract_user_text_from_context(row.get('context_processed'))
        emotion = classify_emotion(user_text or '')

        # ---- 电影类型提取 ----
        sys_text = extract_movie_text_from_system(row.get('raw'))
        if not sys_text:
            continue  # 没有推荐内容的行跳过

        genres = classify_genre(sys_text)

        # ---- 节假日判断 ----
        utc_val = row.get('utc_time')
        holiday_name = '非节假日'
        try:
            dt = datetime.utcfromtimestamp(int(utc_val))
            if cal.is_holiday(dt):
                names = cal.get_holiday_names(dt)
                if names:
                    holiday_name = names[0]  # 取第一个节日名
        except (ValueError, TypeError, OverflowError):
            pass

        # ---- 产出记录（每条推荐可能对应多个类型） ----
        for genre in genres:
            records.append({
                'age_segment': age_seg,
                'holiday_name': holiday_name,
                'emotion': emotion,
                'genre': genre,
            })

    return records


def generate_heatmaps(records: list[dict]):
    """
    对每个年龄段，生成 3 张热图，保存为 PNG 文件。
    """
    df_rec = pd.DataFrame(records)
    print(f"  总分析记录数: {len(df_rec)}")
    print(f"  年龄段分布:\n{df_rec['age_segment'].value_counts().to_string()}\n")

    age_groups = df_rec['age_segment'].unique()
    # 确保顺序
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

        # ── 热图 1: 节假日 × 情感 ──
        _plot_heatmap(
            data=sub,
            index='holiday_name',
            columns='emotion',
            title=f'{age} — 假日场景下用户情感分析热图',
            filename=f'heatmap1_holiday_emotion_{safe_age}.png',
        )

        # ── 热图 2: 情感 × 影片类型 ──
        _plot_heatmap(
            data=sub,
            index='emotion',
            columns='genre',
            title=f'{age} — 不同情感对用户观影类型影响热图',
            filename=f'heatmap2_emotion_genre_{safe_age}.png',
        )

        # ── 热图 3: 节假日 × 影片类型 ──
        _plot_heatmap(
            data=sub,
            index='holiday_name',
            columns='genre',
            title=f'{age} — 不同假日对用户观影类型影响热图',
            filename=f'heatmap3_holiday_genre_{safe_age}.png',
        )


def _plot_heatmap(data: pd.DataFrame, index: str, columns: str,
                  title: str, filename: str):
    """生成一张热图并保存。"""
    # 透视表：计数
    pivot = data.pivot_table(
        index=index,
        columns=columns,
        aggfunc='size',
        fill_value=0,
    )

    # 按行总和降序排序（出现多的靠上）
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    if pivot.empty or pivot.shape[0] == 0 or pivot.shape[1] == 0:
        print(f"    ⚠ 跳过 {filename}: 透视表为空")
        return

    # 绘图
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
    print("  LLM-Movie 数据分析脚本")
    print("=" * 55)

    # ── 步骤 1: 年龄分段 ──
    print("\n[1/4] 用户年龄分段...")
    user_seg_df = age_segment.segment_users(CSV_PATH)
    user_seg = dict(zip(user_seg_df['user_id'], user_seg_df['age_segment']))
    known = user_seg_df[user_seg_df['age_segment'] != 'unknown']
    print(f"  总用户: {len(user_seg_df)}, 可分段: {len(known)} ({len(known)/max(len(user_seg_df),1)*100:.1f}%)")

    # ── 步骤 2: 加载节假日 ──
    print("\n[2/4] 加载节假日日历...")
    cal = HolidayCalendar(HOLIDAY_CSV)
    print(f"  共 {len(cal.all_holidays)} 个节假日")

    # ── 步骤 3: 构建分析数据 ──
    print("\n[3/4] 加载对话数据并构建分析记录...")
    df = pd.read_csv(CSV_PATH)
    records = build_analysis_data(df, cal, user_seg)
    print(f"  原始行数: {len(df)}, 分析记录数: {len(records)}")

    # ── 步骤 4: 生成热图 ──
    print("\n[4/4] 生成热图...")
    generate_heatmaps(records)

    print(f"\n{'='*55}")
    print(f"  [OK] 完成！所有图片保存至: {OUTPUT_DIR}")
    print(f"{'='*55}")

    # 输出分年龄段统计摘要
    print("\n\n== 摘要统计 ==")
    print("-" * 55)
    df_rec = pd.DataFrame(records)
    for age in sorted(df_rec['age_segment'].unique()):
        sub = df_rec[df_rec['age_segment'] == age]
        print(f"\n  [{age}] 共 {len(sub)} 条记录")
        print(f"    情感 TOP3: {sub['emotion'].value_counts().head(3).to_dict()}")
        print(f"    类型 TOP3: {sub['genre'].value_counts().head(3).to_dict()}")
        hcount = sub[sub['holiday_name'] != '非节假日']
        print(f"    节假日记录: {len(hcount)} 条")
        if len(hcount) > 0:
            print(f"    节假日 TOP3: {hcount['holiday_name'].value_counts().head(3).to_dict()}")


if __name__ == '__main__':
    main()
