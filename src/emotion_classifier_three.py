# -*- coding: utf-8 -*-
"""
情感三分类模块 — 基于词典（VADER + AFINN + 领域关键词规则）。

输出类别：正面 / 中性 / 负面

与 emotion_classifier.py（BERT 6 分类）使用相同的 run() 接口，
便于在 main.py 中通过参数切换。

文件输出：
  intermediate/idx_to_emotion.json — 行索引 → 情感标签 映射
"""

import os
import re
from datetime import datetime

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn

from src.config import (
    SENTIMENT_CATEGORIES,
    generate_run_id, log, log_error, log_warn,
)
from src.debug_dump import write_json, extract_run_id, load_latest_by_step

# ── 分析器初始化 ───────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()
_afinn = Afinn(language='en')

# ── 领域关键词规则 ─────────────────────────────────────────────
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

    # 加权融合
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


def classify_emotion_batch(texts: list[str]) -> list[str]:
    """
    批量情感三分类。

    参数：
        texts: 原始文本列表
    返回：
        中文情感标签列表（'正面'/'中性'/'负面'）
    """
    cleaned = [t if t and isinstance(t, str) else '' for t in texts]

    if not any(cleaned):
        log_warn('EmotionThree', '所有文本为空，返回默认情感')
        return ['中性'] * len(texts)

    labels = []
    for text in cleaned:
        label = classify_sentiment(text)
        labels.append(label)

    return labels


def save_idx_to_emotion(text_indices: list[int],
                        emotion_labels: list[str],
                        run_id: str = None) -> str | None:
    """
    保存 idx→emotion 映射到 intermediate 文件。

    包含：
      - 总条数
      - 情感分布统计
      - 前 200 条完整映射
      - 完整映射写入 full_emotion 文件

    参数：
        run_id : 流水号，为 None 时自动生成
    """
    idx_to_emotion = dict(zip(text_indices, emotion_labels))

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
    统一入口：批量情感三分类。

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

    log('EmotionThree', '=' * 40)
    log('EmotionThree', f'开始情感三分类: {len(text_batch)} 条文本 (run_id={run_id})')

    if not text_batch:
        log_warn('EmotionThree', '文本批次为空，跳过')
        return {'run_id': run_id, 'emotion_labels': [], 'idx_to_emotion': {}}

    ts_start = datetime.now()
    log('EmotionThree', f'开始推理（基于词典）...')

    emotion_labels = classify_emotion_batch(text_batch)

    elapsed = (datetime.now() - ts_start).total_seconds()
    log('EmotionThree', f'分类完成，耗时 {elapsed:.1f} 秒')

    idx_to_emotion = dict(zip(text_indices, emotion_labels))
    result = {
        'run_id': run_id,
        'emotion_labels': emotion_labels,
        'idx_to_emotion': idx_to_emotion,
    }

    if save_files:
        log('EmotionThree', '保存 intermediate 文件...')
        path = save_idx_to_emotion(text_indices, emotion_labels, run_id=run_id)
        if path:
            result['files'] = {'idx_to_emotion': path}

    # 控制台输出情感分布
    distribution = {}
    for label in emotion_labels:
        distribution[label] = distribution.get(label, 0) + 1
    log('EmotionThree', f'情感分布: {distribution}')

    log('EmotionThree', '=' * 40)
    return result


def main():
    """
    独立运行入口。

    用法：python -m src.emotion_classifier_three
    说明：自动发现最新的 S02 text_batch_full 文件，加载并完成情感三分类，
          沿用上游的 run_id 保存 S03 文件。
    """
    data, filepath = load_latest_by_step('02', 'text_batch_full')
    if not data:
        log_error('EmotionThree', '未找到 text_batch_full 文件，无法独立运行')
        import sys
        sys.exit(1)

    upstream_run_id = extract_run_id(filepath)
    log('EmotionThree', f'从文件恢复 text_batch (run_id={upstream_run_id}): {filepath}')

    text_indices = [int(k) for k in data.keys()]
    text_batch = list(data.values())
    log('EmotionThree', f'恢复 {len(text_batch)} 条文本')

    result = run(text_batch, text_indices, save_files=True, run_id=upstream_run_id)
    print(f'\n分类完成: {len(result["emotion_labels"])} 条, run_id: {result["run_id"]}')


if __name__ == '__main__':
    main()
