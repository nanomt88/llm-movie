# -*- coding: utf-8 -*-
"""
LLM-Movie 情感分析脚本 v5.0 — BERT 情绪识别 + ONNX 加速版

功能：
  1. 加载 my-tran-holidy-data.csv + holiday.csv
  2. 按年龄段分组用户 (复用 age_segment.py)
  3. 使用 bhadresh-savani/bert-base-uncased-emotion 模型进行情绪识别
     识别 6 种情绪：快乐(joy)、悲伤(sadness)、喜爱(love)、愤怒(anger)、恐惧(fear)、惊讶(surprise)
  4. 电影类型从 movie_info.json 中提取（按 IMDB ID 查找）
  5. 对每组输出 3 张热图：
     热图 1 — 情绪 × 节假日（按 holiday.csv description 分组）
     热图 2 — 情绪 × 影片类型
     热图 3 — 节假日 × 影片类型

性能优化（相对 v4）：
  ① 设置 CPU 线程数 = 16（对齐 AMD 8C16T）
  ② ONNX Runtime 推理（optimum/onnxruntime），2~4× 加速
  ③ 批量推理 batch_size = 128
  ④ 未安装 optimum 时自动回退 PyTorch

使用方式:
  python analysis_heatmap_v5.py

依赖:
  pip install pandas numpy matplotlib seaborn transformers torch
  pip install optimum[onnxruntime]    # 可选，大幅加速
"""

import ast
import json
import os
import re
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
#  优化①：设置 CPU 线程数，对齐 AMD 8 核 16 线程
#  必须在任何 torch/tf 操作之前执行
# ══════════════════════════════════════════════════════════════════
import torch

torch.set_num_threads(16)
# 对 MKL/OpenMP 也生效
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'

# ── Hugging Face 镜像（国内加速下载） ─────────────────────────────
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Transformers + ONNX ─────────────────────────────────────────
from transformers import pipeline, AutoTokenizer

# 尝试加载 optimum.onnxruntime，若未安装则回退 PyTorch
try:
    from optimum.onnxruntime import ORTModelForSequenceClassification

    _HAS_OPTIMUM = True
except ImportError:
    _HAS_OPTIMUM = False

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
CSV_PATH = os.path.join(DATA_DIR, 'my-train-1percent-data.csv')
HOLIDAY_CSV = os.path.join(DATA_DIR, 'holiday.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  1. 情绪识别（基于 BERT + ONNX Runtime 优化）
# ══════════════════════════════════════════════════════════════════

# bhadresh-savani/bert-base-uncased-emotion 模型输出的 6 种情绪标签
# 与 Ekman (1992) 6 种基本情绪对比：
#   - 快乐(Happiness)  ←→ joy        ✓ 匹配
#   - 悲伤(Sadness)    ←→ sadness    ✓ 匹配
#   - 愤怒(Anger)      ←→ anger      ✓ 匹配
#   - 恐惧(Fear)       ←→ fear       ✓ 匹配
#   - 惊讶(Surprise)   ←→ surprise   ✓ 匹配（Ekman 中有）
#   - 厌恶(Disgust)    → 模型无此分类；模型额外包含 love(喜爱)
# 按照"若模型与 Ekman 不同，以模型为准"的原则，使用模型的 6 分类
EMOTION_CATEGORIES = ['悲伤', '快乐', '喜爱', '愤怒', '恐惧', '惊讶']

_MODEL_LABEL_MAP = {
    'sadness': '悲伤',
    'joy': '快乐',
    'love': '喜爱',
    'anger': '愤怒',
    'fear': '恐惧',
    'surprise': '惊讶',
}

# 延迟初始化
_emotion_pipeline_instance = None


def _get_emotion_pipeline():
    """
    获取情绪识别 pipeline（延迟加载）。

    优先使用 ONNX Runtime 加速（需要安装 optimum）：
      pip install optimum[onnxruntime]
    若未安装则自动回退标准 PyTorch pipeline。
    """
    global _emotion_pipeline_instance
    if _emotion_pipeline_instance is not None:
        return _emotion_pipeline_instance

    model_id = "bhadresh-savani/bert-base-uncased-emotion"

    if _HAS_OPTIMUM:
        print(f"  [加载模型] {model_id} (ONNX Runtime)...")
        try:
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                model_id,
                export=True,  # 首次自动导出 PyTorch → ONNX
                provider="CPUExecutionProvider",  # 明确指定 CPU
            )
            ort_tokenizer = AutoTokenizer.from_pretrained(model_id)
            _emotion_pipeline_instance = pipeline(
                "text-classification",
                model=ort_model,
                tokenizer=ort_tokenizer,
                top_k=None,
            )
            return _emotion_pipeline_instance
        except Exception as e:
            print(f"  [警告] ONNX 加载失败，回退 PyTorch: {e}")

    print(f"  [加载模型] {model_id} (PyTorch)...")
    if not _HAS_OPTIMUM:
        print("  [提示] 安装 optimum 可加速推理: pip install optimum[onnxruntime]")
    _emotion_pipeline_instance = pipeline(
        "text-classification",
        model=model_id,
        top_k=None,
    )
    return _emotion_pipeline_instance


def classify_emotion_batch(texts: list[str]) -> list[str]:
    """
    批量情绪识别（优化版）。

    优化策略：
      - ONNX Runtime 推理（较 PyTorch 快 2~4×）
      - 多线程并行（torch.set_num_threads=16）
      - 动态 batch_size 自适应 CPU 内存

    参数：
        texts: 原始文本列表
    返回：
        情绪标签列表（中文）
    """
    pipe = _get_emotion_pipeline()

    # 截断过长文本（BERT token 上限 512）
    cleaned = [t[:512] if t and isinstance(t, str) else '' for t in texts]

    if not any(cleaned):
        return ['快乐'] * len(texts)

    # ── 优化②：动态 batch_size ──
    # CPU 推荐 128（平衡吞吐与内存），GPU 可更高
    batch_size = 128

    # 若文本量较小则用小 batch 避免浪费
    if len(cleaned) < 64:
        batch_size = 32
    elif len(cleaned) < 256:
        batch_size = 64

    print(f"  [推理配置] {len(cleaned)} 条文本, 批次大小={batch_size}, "
          f"线程数={torch.get_num_threads()}, "
          f"后端={'ONNX' if _HAS_OPTIMUM else 'PyTorch'}")

    # ── 优化③：批量推理 ──
    '''
    # raw_results 是一个列表，长度等于输入文本数量  ； 'label': 情绪标签（英文）， 'score': 置信度分数（0~1 之间）
        raw_results = [
            # 第一条文本的所有情绪分类结果
            [
                {'label': 'sadness', 'score': 0.01},
                {'label': 'joy', 'score': 0.85},      # ← 最高分
                {'label': 'love', 'score': 0.05},
                {'label': 'anger', 'score': 0.02},
                {'label': 'fear', 'score': 0.03},
                {'label': 'surprise', 'score': 0.04},
            ],
            # 第二条文本的所有情绪分类结果
            [
                {'label': 'sadness', 'score': 0.70},   # ← 最高分
                {'label': 'joy', 'score': 0.10},
                {'label': 'love', 'score': 0.05},
                {'label': 'anger', 'score': 0.05},
                {'label': 'fear', 'score': 0.08},
                {'label': 'surprise', 'score': 0.02},
            ],
            # ... 更多文本的结果
        ]
    '''
    raw_results = pipe(cleaned, batch_size=batch_size, truncation=True)

    # emotions 是一个字符串列表，每个元素是中文情绪标签； 长度：与输入 texts 相同
    emotions = []
    for result in raw_results:
        # 找出 score 最高的那个情绪
        top = max(result, key=lambda x: x['score'])
        # 将英文标签映射为中文，添加到结果列表
        emotions.append(_MODEL_LABEL_MAP.get(top['label'], '快乐'))

    return emotions


# ══════════════════════════════════════════════════════════════════
#  2. 电影类型分类（基于 movie_info.json）
# ══════════════════════════════════════════════════════════════════

GENRE_CATEGORIES = [
    # 这个是默认的，下面是IMDB中提取的
    # '动作', '喜剧', '剧情', '恐怖', '爱情',
    # '科幻', '惊悚', '动画', '奇幻', '悬疑/犯罪',
    '冒险', '剧情', '动作', '动作冒险', '动画', '历史', '喜剧', '奇幻', '家庭', '恐怖', '悬疑', '惊悚', '战争', '爱情',
    '犯罪', '电视电影', '科幻', '纪录', '西部', '音乐'
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


# imdb 电影ID 正则表达式
_IMDB_ID_RE = re.compile(r'tt\d+')


def extract_imdb_ids(text: str) -> list[str]:
    """从文本中提取所有 IMDB ID (ttXXXXXXX) 并去重"""
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
    """从 movie_info 中按 IMDB ID 查找电影类型"""
    info = movie_info.get(imdb_id)
    if info is None:
        not_found_log.append(imdb_id)
        return None
    genres = info.get('genres', [])
    return genres


def classify_genre_from_ids(imdb_ids: list[str],
                            movie_info: dict,
                            not_found_log: list[str]) -> list[str]:
    """对一组 IMDB ID 查找电影类型，合并去重"""
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
    """将 CSV 中的字符串字段安全地转换为 Python 对象"""
    if pd.isna(value) or not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        print(f"警告: literal_eval 解析失败: {repr(value[:100])}, 错误: {e}")
        return None


def extract_user_text_from_context(context_raw) -> str | None:
    """从上下文字段中提取用户的实际说话内容"""
    ctx = parse_raw_field(context_raw)
    if not ctx or not isinstance(ctx, list):
        return None
    if len(ctx) >= 2 and ctx[0] == 'USER':
        return ctx[1]
    # user_messages = []
    # for entry in ctx:
    #     if isinstance(entry, list) and len(entry) >= 2 and entry[0] == 'USER':
    #         user_messages.append(entry[1])
    # if len(user_messages) >= 2:
    #     return  '\n'.join(user_messages)
    # elif len(user_messages) == 1:
    #     return user_messages[0]
    return None


# 这个地方可能有问题 TODO
def extract_movie_text_from_system(raw_value) -> str | None:
    """从系统推荐字段中提取电影相关的描述文本"""
    parsed = parse_raw_field(raw_value)
    if not parsed or not isinstance(parsed, list) or len(parsed) < 2:
        return None
    if parsed[0] == 'SYSTEM':
        return parsed[1]
    return None


def extract_user_text_direct(processed_value) -> str | None:
    """直接从处理后的字段中提取用户发言"""
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
    核心数据组装函数。

    分两阶段执行：
      阶段一 — 收集所有用户发言文本，批量进行 BERT 情绪识别
      阶段二 — 遍历原始数据，组装完整分析记录
    """
    movie_info = _load_movie_info()
    not_found_ids: list[str] = []

    # ── 阶段一：批量情绪识别 ──
    text_batch: list[str] = []
    #text_indices：列表，存储了需要分析的文本在原始 DataFrame 中的行索引，例如：[0, 5, 12, 28, ...]
    text_indices: list[int] = []

    for idx, row in df.iterrows():
        # 这个地方有问题，原本是 !=False ，改为 == False; TODO
        if row.get('is_seeker') == False:
            continue

        user_text = extract_user_text_from_context(row.get('processed'))
        if not user_text:
            user_text = extract_user_text_from_context(row.get('raw'))

        # sys_text = extract_movie_text_from_system(row.get('processed'))
        # if not sys_text:
        #     continue

        #system 回答中没有电影id，应该不用跳过  todo
        # imdb_ids = extract_imdb_ids(sys_text)
        # if not imdb_ids:
        #     print(f'  [警告] 找不到电影 ID: ' , sys_text , '  idx: ' , idx)
        #     continue

        text_batch.append(user_text or '')
        text_indices.append(idx)

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] [情绪识别] 共 {len(text_batch)} 条文本，正在推理 ...")
    emotion_labels = classify_emotion_batch(text_batch)
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] [情绪识别] 完成")

    #将两个数组中的值合并成为 字典类型； text_indices为索引，emotion_labels为情绪标签
    # 例如：{0: 'joy', 5: 'sadness', 12: 'anger', 28: 'neutral', ...}
    idx_to_emotion = dict(zip(text_indices, emotion_labels))

    # ── 阶段二：组装分析记录 ──
    records = []

    for idx, row in df.iterrows():
        if row.get('is_seeker') != True:
            continue
        if idx not in idx_to_emotion:
            continue

        # 这里这个user_id为 system_id ，有问题，需要修复   fixme
        user_id = row.get('user_id')
        age_seg = user_seg.get(user_id, 'unknown')
        emotion = idx_to_emotion[idx]

        # 获取当前行的位置
        current_pos = df.index.get_loc(idx)
        # 检查是否有下一行
        if current_pos >= len(df) - 1:
            print("  [警告] 最后一行，没有下一行")
            continue

        next_row = df.iloc[current_pos + 1]
        next_user_id = next_row.get('user_id')
        #print(f"当前: {row['user_id']}, 下一个: {next_user_id}")
        sys_text = extract_movie_text_from_system(next_row.get('processed'))
        imdb_ids = extract_imdb_ids(sys_text)
        genres = classify_genre_from_ids(imdb_ids, movie_info, not_found_ids)
        if not genres:
            continue

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
                'emotion': emotion,
                'genre': genre,
            })

    if not_found_ids:
        unique_not_found = sorted(set(not_found_ids))
        print(f"\n  [!] {len(unique_not_found)} 个 IMDB ID 在 movie_info.json 中未找到:")
        for tid in unique_not_found:
            print(f"      - {tid}", end=", ")

    return records


def generate_heatmaps(records: list[dict]):
    """根据分析记录生成多维度的热力图。"""
    df_rec = pd.DataFrame(records)
    print(f"  总分析记录数: {len(df_rec)}")
    print(f"  情绪分布:\n{df_rec['emotion'].value_counts().to_string()}\n")

    age_groups = df_rec['age_segment'].unique()
    age_order = [s for s in age_segment.AGE_SEGMENTS if s in age_groups]
    age_order += sorted(set(age_groups) - set(age_segment.AGE_SEGMENTS))

    for age in age_order:
        print(f"\n{'=' * 55}")
        print(f"  年龄段: {age}")
        print(f"{'=' * 55}")

        sub = df_rec[df_rec['age_segment'] == age]
        if len(sub) < 3:
            print(f"    数据不足 ({len(sub)} 条)，跳过")
            continue

        safe_age = _sanitize_filename(age)

        # ── 热图 1: 情绪 × 节假日 ──
        sub_holiday = sub[sub['holiday_name'] != '非节假日'].copy()
        if not sub_holiday.empty:
            pivot = sub_holiday.pivot_table(
                index='emotion',
                columns='holiday_name',
                aggfunc='size',
                fill_value=0,
            )
            row_order = [e for e in EMOTION_CATEGORIES if e in pivot.index]
            pivot = pivot.loc[row_order]
            col_order = pivot.sum(axis=0).sort_values(ascending=False).index
            pivot = pivot[col_order]

            if pivot.shape[0] > 0 and pivot.shape[1] > 0:
                fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.2),
                                                max(5, pivot.shape[0] * 0.6)))
                sns.heatmap(
                    pivot, annot=True, fmt='d', cmap='YlOrRd',
                    linewidths=0.5, ax=ax, cbar_kws={'label': '观影次数'},
                )
                ax.set_title(f'{age} — 不同节假日用户情绪热图（BERT 6 类）',
                             fontsize=14, pad=16)
                ax.set_xlabel('节假日', fontsize=11)
                ax.set_ylabel('情绪类型', fontsize=11)
                plt.xticks(rotation=30, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                out_path = os.path.join(OUTPUT_DIR,
                                        f'v5_heatmap1_emotion_holiday_{safe_age}.png')
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    [OK] v5_heatmap1_emotion_holiday_{safe_age}.png  "
                      f"({pivot.shape[0]}x{pivot.shape[1]})")
            else:
                print(f"    \u26a0 跳过热图 1: 透视表为空")
        else:
            print(f"    \u26a0 跳过热图 1: 无节假日数据")

        # ── 热图 2: 情绪 × 影片类型 ──
        _plot_heatmap(
            data=sub,
            index='emotion',
            columns='genre',
            title=f'{age} — 不同情绪对用户观影类型影响热图（BERT 6 类）',
            filename=f'v5_heatmap2_emotion_genre_{safe_age}.png',
        )

        # ── 热图 3: 节假日 × 影片类型 ──
        _plot_heatmap(
            data=sub,
            index='holiday_name',
            columns='genre',
            title=f'{age} — 不同假日对用户观影类型影响热图（BERT 6 类）',
            filename=f'v5_heatmap3_holiday_genre_{safe_age}.png',
        )


def _plot_heatmap(data: pd.DataFrame, index: str, columns: str,
                  title: str, filename: str):
    """通用的热力图绘制函数。"""
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
    print("  LLM-Movie 情绪分析脚本 v5.0（ONNX 加速）")
    print("  ─ 基于 bhadresh-savani/bert-base-uncased-emotion 模型")
    print(f"  ─ CPU 线程: {torch.get_num_threads()}"
          f"  |  推理后端: {'ONNX' if _HAS_OPTIMUM else 'PyTorch'}")
    print("=" * 55)

    print("\n[1/4] 用户年龄分段...")
    user_seg_df = age_segment.segment_users(CSV_PATH)
    user_seg = dict(zip(user_seg_df['user_id'], user_seg_df['age_segment']))
    known = user_seg_df[user_seg_df['age_segment'] != 'unknown']
    print(f"  总用户: {len(user_seg_df)}, 可分段: {len(known)} "
          f"({len(known) / max(len(user_seg_df), 1) * 100:.1f}%)")

    print("\n[2/4] 加载节假日日历...")
    cal = HolidayCalendar(HOLIDAY_CSV)
    print(f"  共 {len(cal.all_holidays)} 个节假日")

    print("\n[3/4] 加载 movie_info.json...")
    movie_info = _load_movie_info()
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] 共 {len(movie_info)} 部电影")

    print("\n[4/4] 加载对话数据并构建分析记录...")
    df = pd.read_csv(CSV_PATH)
    records = build_analysis_data(df, cal, user_seg)
    print(f"  [{datetime.now().strftime('%H:%M:%S')}]  原始行数: {len(df)}, "
          f"分析记录数: {len(records)}")

    print("\n[5/4] 生成热图...")
    generate_heatmaps(records)

    print(f"\n{'=' * 55}")
    print(f"  [OK] 完成！所有图片保存至: {OUTPUT_DIR}")
    print(f"{'=' * 55}")

    print("\n\n== 摘要统计 ==")
    print("-" * 55)
    df_rec = pd.DataFrame(records)
    for age in sorted(df_rec['age_segment'].unique()):
        sub = df_rec[df_rec['age_segment'] == age]
        print(f"\n  [{age}] 共 {len(sub)} 条记录")
        print(f"    情绪 TOP3: {sub['emotion'].value_counts().head(3).to_dict()}")
        print(f"    类型 TOP3: {sub['genre'].value_counts().head(3).to_dict()}")
        hcount = sub[sub['holiday_name'] != '非节假日']
        print(f"    节假日记录: {len(hcount)} 条")
        if len(hcount) > 0:
            print(f"    节假日 TOP3: {hcount['holiday_name'].value_counts().head(3).to_dict()}")


if __name__ == '__main__':
    main()
