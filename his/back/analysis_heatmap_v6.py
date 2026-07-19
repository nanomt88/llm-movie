# -*- coding: utf-8 -*-
"""
LLM-Movie 情感分析脚本 v6.0 — BERT + ONNX INT8 量化 + 全栈调优版

功能：
  1. 加载 my-tran-holidy-data.csv + holiday.csv
  2. 按年龄段分组用户 (复用 age_segment.py)
  3. 使用 bhadresh-savani/bert-base-uncased-emotion 模型进行情绪识别
     识别 6 种情绪：快乐(joy)、悲伤(sadness)、喜爱(love)、愤怒(anger)、恐惧(fear)、惊讶(surprise)
  4. 电影类型从 movie_info.json 中提取（按 IMDB ID 查找）
  5. 对每组输出 3 张热图

性能优化（相对 v5，预期 14min → 5~7min/12k条）：
  ① ONNX Session 参数调优：intra_op_num_threads=16, inter_op_num_threads=2,
      OR_PARALLEL 执行模式 + ORT_ENABLE_ALL 全图优化
  ② INT8 动态量化（AVX2 兼容），首次运行自动导出并缓存
  ③ 自适应 max_length：根据文本字分布动态截断，避免冗余 padding
  ④ 回退链：INT8 量化 → FP32 ONNX → PyTorch，逐级自动降级

使用方式:
  python analysis_heatmap_v6.py

依赖:
  pip install pandas numpy matplotlib seaborn transformers torch
  pip install optimum[onnxruntime]    # 必须，量化依赖
"""

import ast
import json
import os
import re
import shutil
import time
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
#  优化①：CPU 线程数设置（必须早于任何 torch/transformers 导入）
# ══════════════════════════════════════════════════════════════════
import torch
torch.set_num_threads(16)
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'

# ── Hugging Face 镜像（国内加速下载） ─────────────────────────────
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from transformers import pipeline, AutoTokenizer

# 尝试加载 optimum.onnxruntime
try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    _HAS_OPTIMUM = True
except ImportError:
    _HAS_OPTIMUM = False

# 尝试加载量化模块
try:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    _CAN_QUANTIZE = True
except ImportError:
    _CAN_QUANTIZE = False

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
from his.src import age_segment
from his.src.holiday_util import HolidayCalendar

# ══════════════════════════════════════════════════════════════════
#  0. 路径配置
# ══════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'my-tran-holidy-data.csv')
HOLIDAY_CSV = os.path.join(DATA_DIR, 'holiday.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
QUANTIZED_DIR = os.path.join(BASE_DIR, 'quantized_model')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  1. 情绪识别（BERT + ONNX INT8 量化 + Session 调优 + 自适应截断）
# ══════════════════════════════════════════════════════════════════

EMOTION_CATEGORIES = ['悲伤', '快乐', '喜爱', '愤怒', '恐惧', '惊讶']

_MODEL_LABEL_MAP = {
    'sadness': '悲伤',
    'joy': '快乐',
    'love': '喜爱',
    'anger': '愤怒',
    'fear': '恐惧',
    'surprise': '惊讶',
}

_emotion_pipeline_instance = None


def _get_ort_session_options():
    """
    优化②：ONNX Runtime Session 参数调优。
      - intra_op_num_threads = 16   算子内并行（对齐 CPU 核心数）
      - inter_op_num_threads = 2    算子间并行（不宜过多）
      - OR_PARALLEL                 并行执行模式
      - ORT_ENABLE_ALL              全量图优化（常量折叠、算子融合等）
    """
    import onnxruntime
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = 16
    opts.inter_op_num_threads = 2
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def _try_load_quantized_model(model_id: str, session_options):
    """尝试加载 INT8 量化模型（若已缓存），否则自动导出并缓存。"""
    if not _CAN_QUANTIZE:
        print("  [量化不可用] 需要安装 optimum: pip install optimum[onnxruntime]")
        return None

    # 已缓存 → 直接加载
    if os.path.exists(QUANTIZED_DIR):
        try:
            print(f"  [加载模型] INT8 量化模型 (缓存命中) ...")
            return ORTModelForSequenceClassification.from_pretrained(
                QUANTIZED_DIR,
                session_options=session_options,
            )
        except Exception as e:
            print(f"  [警告] 量化模型加载失败，回退: {e}")
            return None

    # 首次运行 → 导出 FP32 ONNX → 量化 INT8
    print(f"  [量化] 首次运行，开始导出 INT8 量化模型 ...")
    try:
        # Step 1: 导出并加载 FP32 ONNX
        print(f"  [量化] Step 1/3: 导出 FP32 ONNX ...")
        ort_fp32 = ORTModelForSequenceClassification.from_pretrained(
            model_id, export=True,
        )

        # Step 2: 从已加载的 ORT 模型创建量化器
        print(f"  [量化] Step 2/3: 创建量化器 ...")
        quantizer = ORTQuantizer.from_model(ort_fp32)

        # Step 3: 选择量化配置 — 优先 avx512（Zen4+），安全回退 avx2
        qconfig = None
        for qname in ['avx512', 'avx2']:
            try:
                qconfig = getattr(AutoQuantizationConfig, qname)(is_static=False)
                print(f"  [量化] Step 3/3: 量化配置 = {qname} ...")
                break
            except Exception:
                continue
        if qconfig is None:
            raise RuntimeError("无支持的量化配置")

        quantizer.quantize(save_dir=QUANTIZED_DIR, quantization_config=qconfig)
        del ort_fp32
        print(f"  [量化] 完成! 模型已保存到: {QUANTIZED_DIR}")

        return ORTModelForSequenceClassification.from_pretrained(
            QUANTIZED_DIR,
            session_options=session_options,
        )

    except Exception as e:
        print(f"  [量化] 失败 ({e})，清理临时文件并回退 FP32 ONNX")
        if os.path.exists(QUANTIZED_DIR):
            shutil.rmtree(QUANTIZED_DIR, ignore_errors=True)
        return None


def _get_emotion_pipeline():
    """
    获取情绪识别 pipeline（延迟加载）。

    优先链：INT8 量化 → FP32 ONNX（Session 调优） → PyTorch
    """
    global _emotion_pipeline_instance
    if _emotion_pipeline_instance is not None:
        return _emotion_pipeline_instance

    model_id = "bhadresh-savani/bert-base-uncased-emotion"

    if _HAS_OPTIMUM:
        session_options = _get_ort_session_options()

        # 尝试 INT8 量化模型
        ort_model = _try_load_quantized_model(model_id, session_options)

        # 回退 FP32 ONNX
        if ort_model is None:
            print(f"  [加载模型] {model_id} (ONNX Runtime FP32 + Session 调优)...")
            try:
                ort_model = ORTModelForSequenceClassification.from_pretrained(
                    model_id,
                    export=True,
                    session_options=session_options,
                    provider="CPUExecutionProvider",
                )
            except Exception as e:
                print(f"  [警告] ONNX 加载失败，回退 PyTorch: {e}")
                ort_model = None

        if ort_model is not None:
            ort_tokenizer = AutoTokenizer.from_pretrained(model_id)
            _emotion_pipeline_instance = pipeline(
                "text-classification",
                model=ort_model,
                tokenizer=ort_tokenizer,
                top_k=None,
            )
            return _emotion_pipeline_instance

    # 最终回退：纯 PyTorch
    print(f"  [加载模型] {model_id} (PyTorch)...")
    if not _HAS_OPTIMUM:
        print("  [提示] 安装 optimum 可大幅加速: pip install optimum[onnxruntime]")
    _emotion_pipeline_instance = pipeline(
        "text-classification",
        model=model_id,
        top_k=None,
    )
    return _emotion_pipeline_instance


def _estimate_adaptive_max_length(texts: list[str], sample_size: int = 100) -> int:
    """
    优化③：自适应 max_length。

    从文本样本中估算 95% 分位词数 × 1.3（BERT 平均每词子词数），
    在保证 95% 文本不被截断的前提下，尽可能缩短序列长度。
    BERT 注意力复杂度 O(n²)，序列越短加速越明显。
    """
    sample = [t for t in texts[:sample_size] if t and isinstance(t, str)]
    if not sample:
        return 512

    word_counts = np.array([len(t.split()) for t in sample])
    p95_words = int(np.percentile(word_counts, 95))
    max_len = int(p95_words * 1.3)   # word → BPE token 比例 ≈ 1.3

    # 区间安全钳
    clamped = max(64, min(512, max_len))
    return clamped


def classify_emotion_batch(texts: list[str]) -> list[str]:
    """
    批量情绪识别（全栈优化版）。

    优化策略：
      ① Session 参数调优（intra/inter thread, OR_PARALLEL, 全图优化）
      ② INT8 量化推理（首次运行自动导出并缓存）
      ③ 自适应 max_length（基于文本长度分布动态截断）
      ④ 动态 batch_size（随数据量自动调整）
    """
    pipe = _get_emotion_pipeline()

    cleaned = [t[:512] if t and isinstance(t, str) else '' for t in texts]
    if not any(cleaned):
        return ['快乐'] * len(texts)

    # 动态 batch_size
    batch_size = 128
    if len(cleaned) < 64:
        batch_size = 32
    elif len(cleaned) < 256:
        batch_size = 64

    # 自适应 max_length（优化③）
    # 估算 95% 分位 token 数，传给 tokenizer 做截断
    # BERT 注意力 O(n²)，从 512→128 可节省约 16 倍注意力计算
    max_length = _estimate_adaptive_max_length(cleaned)

    backend_str = 'INT8 量化' if (os.path.exists(QUANTIZED_DIR) and _CAN_QUANTIZE) else \
                  ('ONNX FP32' if _HAS_OPTIMUM else 'PyTorch')

    print(f"  [推理配置] {len(cleaned)} 条文本, "
          f"批次大小={batch_size}, "
          f"max_length={max_length}, "
          f"线程数={torch.get_num_threads()}, "
          f"后端={backend_str}")

    raw_results = pipe(cleaned,
                       batch_size=batch_size,
                       truncation=True,
                       max_length=max_length)

    emotions = []
    for result in raw_results:
        top = max(result, key=lambda x: x['score'])
        emotions.append(_MODEL_LABEL_MAP.get(top['label'], '快乐'))

    return emotions


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
    global _movie_info_cache
    if _movie_info_cache is None:
        with open(MOVIE_INFO_PATH, 'r', encoding='utf-8') as f:
            _movie_info_cache = json.load(f)
    return _movie_info_cache


_IMDB_ID_RE = re.compile(r'tt\d+')


def extract_imdb_ids(text: str) -> list[str]:
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
    info = movie_info.get(imdb_id)
    if info is None:
        not_found_log.append(imdb_id)
        return None
    return info.get('genres', [])


def classify_genre_from_ids(imdb_ids: list[str],
                             movie_info: dict,
                             not_found_log: list[str]) -> list[str]:
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
    if pd.isna(value) or not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def extract_user_text_from_context(context_raw) -> str | None:
    ctx = parse_raw_field(context_raw)
    if not ctx or not isinstance(ctx, list):
        return None
    for entry in ctx:
        if isinstance(entry, list) and len(entry) >= 2 and entry[0] == 'USER':
            return entry[1]
    return None


def extract_movie_text_from_system(raw_value) -> str | None:
    parsed = parse_raw_field(raw_value)
    if not parsed or not isinstance(parsed, list) or len(parsed) < 2:
        return None
    if parsed[0] == 'SYSTEM':
        return parsed[1]
    return None


def extract_user_text_direct(processed_value) -> str | None:
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
    movie_info = _load_movie_info()
    not_found_ids: list[str] = []

    # ── 阶段一：批量情绪识别 ──
    text_batch: list[str] = []
    text_indices: list[int] = []

    for idx, row in df.iterrows():
        if row.get('is_seeker') != False:
            continue

        user_text = extract_user_text_from_context(row.get('context_raw'))
        if not user_text:
            user_text = extract_user_text_from_context(row.get('context_processed'))

        sys_text = extract_movie_text_from_system(row.get('processed'))
        if not sys_text:
            continue
        imdb_ids = extract_imdb_ids(sys_text)
        if not imdb_ids:
            continue

        text_batch.append(user_text or '')
        text_indices.append(idx)

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] [情绪识别] 共 {len(text_batch)} 条文本，正在推理 ...")
    emotion_labels = classify_emotion_batch(text_batch)
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] [情绪识别] 完成")

    idx_to_emotion = dict(zip(text_indices, emotion_labels))

    # ── 阶段二：组装分析记录 ──
    records = []
    total_rows = len(df)
    last_print_time = time.time()
    processed_count = 0

    for idx, row in df.iterrows():
        processed_count += 1
        current_time = time.time()
        if current_time - last_print_time >= 10:
            print(f"  [进度] {processed_count}/{total_rows}")
            last_print_time = current_time

        if row.get('is_seeker') != False:
            continue
        if idx not in idx_to_emotion:
            continue

        user_id = row.get('user_id')
        age_seg = user_seg.get(user_id, 'unknown')
        emotion = idx_to_emotion[idx]

        sys_text = extract_movie_text_from_system(row.get('processed'))
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
            print(f"      - {tid}")

    return records


def generate_heatmaps(records: list[dict]):
    df_rec = pd.DataFrame(records)
    print(f"  总分析记录数: {len(df_rec)}")
    print(f"  情绪分布:\n{df_rec['emotion'].value_counts().to_string()}\n")

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
                                        f'v6_heatmap1_emotion_holiday_{safe_age}.png')
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    [OK] v6_heatmap1_emotion_holiday_{safe_age}.png  "
                      f"({pivot.shape[0]}x{pivot.shape[1]})")
            else:
                print(f"    ⚠ 跳过热图 1: 透视表为空")
        else:
            print(f"    ⚠ 跳过热图 1: 无节假日数据")

        # ── 热图 2: 情绪 × 影片类型 ──
        _plot_heatmap(
            data=sub,
            index='emotion',
            columns='genre',
            title=f'{age} — 不同情绪对用户观影类型影响热图（BERT 6 类）',
            filename=f'v6_heatmap2_emotion_genre_{safe_age}.png',
        )

        # ── 热图 3: 节假日 × 影片类型 ──
        _plot_heatmap(
            data=sub,
            index='holiday_name',
            columns='genre',
            title=f'{age} — 不同假日对用户观影类型影响热图（BERT 6 类）',
            filename=f'v6_heatmap3_holiday_genre_{safe_age}.png',
        )


def _plot_heatmap(data: pd.DataFrame, index: str, columns: str,
                  title: str, filename: str):
    pivot = data.pivot_table(
        index=index,
        columns=columns,
        aggfunc='size',
        fill_value=0,
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    if pivot.empty or pivot.shape[0] == 0 or pivot.shape[1] == 0:
        print(f"    ⚠ 跳过 {filename}: 透视表为空")
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
    print("  LLM-Movie 情绪分析脚本 v6.0（全栈优化版）")
    print("  ─ 基于 bhadresh-savani/bert-base-uncased-emotion 模型")
    print(f"  ─ CPU: AMD Ryzen 7 8845H (8C16T)"
          f"  |  推理后端: {'INT8量化' if os.path.exists(QUANTIZED_DIR) and _CAN_QUANTIZE else 'ONNX FP32' if _HAS_OPTIMUM else 'PyTorch'}")
    print("  ─ 优化: Session调优 + INT8量化 + 自适应minLength")
    print("=" * 55)

    print("\n[1/4] 用户年龄分段...")
    user_seg_df = age_segment.segment_users(CSV_PATH)
    user_seg = dict(zip(user_seg_df['user_id'], user_seg_df['age_segment']))
    known = user_seg_df[user_seg_df['age_segment'] != 'unknown']
    print(f"  总用户: {len(user_seg_df)}, 可分段: {len(known)} "
          f"({len(known)/max(len(user_seg_df),1)*100:.1f}%)")

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

    print(f"\n{'='*55}")
    print(f"  [OK] 完成！所有图片保存至: {OUTPUT_DIR}")
    print(f"{'='*55}")

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
