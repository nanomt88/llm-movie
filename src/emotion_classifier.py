# -*- coding: utf-8 -*-
"""
情绪识别模块 — 基于 bhadresh-savani/bert-base-uncased-emotion 模型。

支持 ONNX Runtime 加速（需要安装 optimum），自动回退 PyTorch。

文件输出：
  intermediate/idx_to_emotion.json — 行索引 → 情绪标签 映射
"""

import os
import sys
from datetime import datetime

import torch
from transformers import pipeline, AutoTokenizer

from src.config import (
    INTERMEDIATE_DIR, MODEL_LABEL_MAP,
    generate_run_id, log, log_error, log_warn,
)
from src.debug_dump import write_json, extract_run_id, load_latest_by_step

# 尝试加载 optimum.onnxruntime
try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    _HAS_OPTIMUM = True
except ImportError:
    _HAS_OPTIMUM = False

# 延迟初始化 pipeline
_emotion_pipeline_instance = None


def _get_emotion_pipeline():
    """
    获取情绪识别 pipeline（延迟加载）。

    优先使用 ONNX Runtime 加速，未安装 optimum 时自动回退 PyTorch。
    """
    global _emotion_pipeline_instance
    if _emotion_pipeline_instance is not None:
        return _emotion_pipeline_instance

    model_id = "bhadresh-savani/bert-base-uncased-emotion"

    if _HAS_OPTIMUM:
        log('Emotion', f'加载模型: {model_id} (ONNX Runtime)...')
        try:
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                model_id,
                export=True,
                provider="CPUExecutionProvider",
            )
            ort_tokenizer = AutoTokenizer.from_pretrained(model_id)
            _emotion_pipeline_instance = pipeline(
                "text-classification",
                model=ort_model,
                tokenizer=ort_tokenizer,
                top_k=None,
            )
            log('Emotion', f'  ONNX Runtime pipeline 创建成功')
            return _emotion_pipeline_instance
        except Exception as e:
            log_error('Emotion', f'ONNX 加载失败，回退 PyTorch: {e}')

    log('Emotion', f'加载模型: {model_id} (PyTorch)...')
    if not _HAS_OPTIMUM:
        log('Emotion', '  提示: 安装 optimum 可加速: pip install optimum[onnxruntime]')
    try:
        _emotion_pipeline_instance = pipeline(
            "text-classification",
            model=model_id,
            top_k=None,
        )
        return _emotion_pipeline_instance
    except Exception as e:
        log_error('Emotion', f'Pipeline 创建失败: {e}')
        raise


def classify_emotion_batch(texts: list[str]) -> list[str]:
    """
    批量情绪识别。

    优化策略：
      - ONNX Runtime 推理（2~4× 加速）
      - 多线程并行（torch.set_num_threads=16）
      - 动态 batch_size 自适应

    参数：
        texts: 原始文本列表
    返回：
        中文情绪标签列表
    """
    pipe = _get_emotion_pipeline()

    # 截断过长文本
    cleaned = [t[:512] if t and isinstance(t, str) else '' for t in texts]

    if not any(cleaned):
        log_warn('Emotion', '所有文本为空，返回默认情绪')
        return ['快乐'] * len(texts)

    # 动态 batch_size
    batch_size = 128
    if len(cleaned) < 64:
        batch_size = 32
    elif len(cleaned) < 256:
        batch_size = 64

    log('Emotion', f'推理配置: {len(cleaned)} 条, '
                   f'batch={batch_size}, '
                   f'线程={torch.get_num_threads()}, '
                   f'后端={"ONNX" if _HAS_OPTIMUM else "PyTorch"}')

    try:
        raw_results = pipe(cleaned, batch_size=batch_size, truncation=True)
    except Exception as e:
        log_error('Emotion', f'推理失败: {e}')
        raise

    emotions = []
    for i, result in enumerate(raw_results):
        try:
            top = max(result, key=lambda x: x['score'])
            emotions.append(MODEL_LABEL_MAP.get(top['label'], '快乐'))
        except Exception as e:
            log_error('Emotion', f'第 {i} 条结果解析失败: {e}')
            emotions.append('快乐')

    return emotions


def save_idx_to_emotion(text_indices: list[int],
                        emotion_labels: list[str],
                        run_id: str = None) -> str | None:
    """
    保存 idx→emotion 映射到 intermediate 文件。

    包含：
      - 总条数
      - 情绪分布统计
      - 前 200 条完整映射（后续模块恢复用）
      - 完整映射写入 full_emotion 文件

    参数：
        run_id : 流水号，为 None 时自动生成
    """
    idx_to_emotion = dict(zip(text_indices, emotion_labels))

    # 情绪分布统计
    distribution = {}
    for label in emotion_labels:
        distribution[label] = distribution.get(label, 0) + 1

    info = {
        'total': len(idx_to_emotion),
        'emotion_distribution': distribution,
        'samples': [{'idx': k, 'emotion': v}
                     for k, v in list(idx_to_emotion.items())[:200]],
    }
    path = write_json(info, 'idx_to_emotion', run_id=run_id, step='03')

    # 同时保存完整映射用于下一模块恢复
    full_path = write_json(
        {str(k): v for k, v in idx_to_emotion.items()},
        'idx_to_emotion_full', run_id=run_id, step='03',
    )

    return path


def run(text_batch: list[str],
        text_indices: list[int],
        save_files: bool = True,
        run_id: str = None) -> dict:
    """
    统一入口：批量情绪识别。

    参数：
        text_batch   : 用户文本列表
        text_indices : 对应 DataFrame 行索引列表
        save_files   : 是否保存 intermediate 文件
        run_id       : 流水号（同一轮流水线执行共享此 ID）

    返回：
        dict — {
            'run_id': str,
            'emotion_labels': list[str],
            'idx_to_emotion': dict,
            'files': {...}  (save_files=True 时)
        }
    """
    if run_id is None:
        run_id = generate_run_id()

    log('Emotion', '=' * 40)
    log('Emotion', f'开始情绪识别: {len(text_batch)} 条文本 (run_id={run_id})')

    if not text_batch:
        log_warn('Emotion', '文本批次为空，跳过')
        return {'run_id': run_id, 'emotion_labels': [], 'idx_to_emotion': {}}

    ts_start = datetime.now()
    log('Emotion', f'开始推理...')

    emotion_labels = classify_emotion_batch(text_batch)

    elapsed = (datetime.now() - ts_start).total_seconds()
    log('Emotion', f'推理完成，耗时 {elapsed:.1f} 秒')

    idx_to_emotion = dict(zip(text_indices, emotion_labels))
    result = {
        'run_id': run_id,
        'emotion_labels': emotion_labels,
        'idx_to_emotion': idx_to_emotion,
    }

    if save_files:
        log('Emotion', '保存 intermediate 文件...')
        path = save_idx_to_emotion(text_indices, emotion_labels, run_id=run_id)
        if path:
            result['files'] = {'idx_to_emotion': path}

    # 控制台输出情绪分布
    distribution = {}
    for label in emotion_labels:
        distribution[label] = distribution.get(label, 0) + 1
    log('Emotion', f'情绪分布: {distribution}')

    log('Emotion', '=' * 40)
    return result


def main():
    """
    独立运行入口。

    用法：python -m src.emotion_classifier
    说明：自动发现最新的 S02 text_batch_full 文件，加载并完成情绪分类，
          沿用上游的 run_id 保存 S03 文件。
    """
    data, filepath = load_latest_by_step('02', 'text_batch_full')
    if not data:
        log_error('Emotion', '未找到 text_batch_full 文件，无法独立运行')
        sys.exit(1)

    # 沿用上游的 run_id，保持文件分组
    upstream_run_id = extract_run_id(filepath)
    log('Emotion', f'从文件恢复 text_batch (run_id={upstream_run_id}): {filepath}')

    text_indices = [int(k) for k in data.keys()]
    text_batch = list(data.values())
    log('Emotion', f'恢复 {len(text_batch)} 条文本')

    result = run(text_batch, text_indices, save_files=True, run_id=upstream_run_id)
    print(f'\n分类完成: {len(result["emotion_labels"])} 条, run_id: {result["run_id"]}')


if __name__ == '__main__':
    main()
