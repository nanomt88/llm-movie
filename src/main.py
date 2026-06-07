# -*- coding: utf-8 -*-
"""
主协调模块 — 依次调用各子模块的 run()，串联完整分析流水线。

数据流（纯内存传递，无文件 I/O）：
  data_loader → text_extractor → emotion_classifier
  → record_builder → heatmap_generator

文件保存由各模块内部自行处理（统一使用 run_id + 步骤编号命名）。

支持：
  - 完整流水线运行
  - 指定跳过某些步骤（--skip）
  - 指定仅运行某些步骤（--steps）
"""

import argparse
from datetime import datetime

from src.config import log, log_error, log_warn

# ── 模块入口 ──────────────────────────────────────────────
from src.data_loader import run as run_data_loader
from src.text_extractor import run as run_text_extractor
from src.emotion_classifier import run as run_emotion_classifier
from src.emotion_classifier_three import run as run_emotion_classifier_three
from src.record_builder import run as run_record_builder
from src.heatmap_generator import run as run_heatmap_generator


_STEP_NAMES = [
    'data_loader',
    'text_extractor',
    'emotion_classifier',
    'record_builder',
    'heatmap_generator',
]


def _emotion_distribution(labels: list[str]) -> dict:
    """统计情绪分布。"""
    dist = {}
    for label in labels:
        dist[label] = dist.get(label, 0) + 1
    return dist


def run_pipeline(skip_steps: list[str] | None = None,
                 emotion_mode: str = 'bert',
                 run_id: str = None) -> dict:
    """
    完整流水线执行（纯编排，无文件 I/O）。

    参数：
        skip_steps   : 要跳过的步骤名列表
        emotion_mode : 'bert'（BERT 6 分类）| 'lexicon'（词典三分类）
        run_id       : 流水号，为 None 时由 data_loader 生成

    返回：
        dict — 各步骤的结果摘要
    """
    if skip_steps is None:
        skip_steps = []

    summary = {}
    ts_total = datetime.now()

    # ── 第 0 步：生成 run_id（由 data_loader 负责）───────
    #     但先占位，data_loader 会返回它生成的 run_id

    # ── Step 1: 数据加载 ──────────────────────────────
    step_data = {}  # 保存各步骤返回的完整数据供下游使用

    if 'data_loader' not in skip_steps:
        log('Main', '\n' + '#' * 55)
        log('Main', '# 步骤 1/5: 数据加载')
        log('Main', '#' * 55)
        result = run_data_loader(save_files=True, run_id=run_id)
        step_data['data_loader'] = result
        run_id = result.get('run_id', run_id)
        summary['data_loader'] = result.get('files', {})
    else:
        log('Main', '步骤 1/5: 跳过数据加载')
        summary['data_loader'] = {}

    # ── Step 2: 文本提取 ──────────────────────────────
    if 'text_extractor' not in skip_steps:
        log('Main', '\n' + '#' * 55)
        log('Main', '# 步骤 2/5: 文本提取')
        log('Main', '#' * 55)

        df = step_data.get('data_loader', {}).get('df')
        if df is None:
            log_error('Main', '缺少 DataFrame，无法执行文本提取')
            summary['text_extractor'] = {}
        else:
            result = run_text_extractor(df=df, save_files=True, run_id=run_id)
            step_data['text_extractor'] = result
            summary['text_extractor'] = {
                'files': result.get('files', {}),
                'num_texts': len(result.get('text_batch', [])),
            }
    else:
        log('Main', '步骤 2/5: 跳过文本提取')
        summary['text_extractor'] = {}

    # ── Step 3: 情绪/情感识别 ──────────────────────────
    if 'emotion_classifier' not in skip_steps:
        mode_name = 'BERT 6 分类' if emotion_mode == 'bert' else '词典三分类'
        log('Main', '\n' + '#' * 55)
        log('Main', f'# 步骤 3/5: 情绪识别（{mode_name}）')
        log('Main', '#' * 55)

        te = step_data.get('text_extractor', {})
        text_batch = te.get('text_batch', [])
        text_indices = te.get('text_indices', [])

        if text_batch:
            classifier = (run_emotion_classifier if emotion_mode == 'bert'
                          else run_emotion_classifier_three)
            result = classifier(text_batch, text_indices, save_files=True,
                                run_id=run_id)
            step_data['emotion_classifier'] = result
            summary['emotion_classifier'] = {
                'files': result.get('files', {}),
                'num_labels': len(result.get('emotion_labels', [])),
                'distribution': _emotion_distribution(
                    result.get('emotion_labels', [])),
            }
        else:
            log_warn('Main', 'text_batch 为空，跳过情绪分类')
            summary['emotion_classifier'] = {}
    else:
        log('Main', '步骤 3/5: 跳过情绪识别')
        summary['emotion_classifier'] = {}

    # ── Step 4: 记录组装 ──────────────────────────────
    if 'record_builder' not in skip_steps:
        log('Main', '\n' + '#' * 55)
        log('Main', '# 步骤 4/5: 记录组装')
        log('Main', '#' * 55)

        dl = step_data.get('data_loader', {})
        df = dl.get('df')
        user_seg = dl.get('user_seg')
        cal = dl.get('cal')
        movie_info = dl.get('movie_info')
        idx_to_emotion = (step_data.get('emotion_classifier', {})
                          .get('idx_to_emotion'))

        if idx_to_emotion and df is not None:
            result = run_record_builder(
                df=df,
                user_seg=user_seg,
                idx_to_emotion=idx_to_emotion,
                cal=cal,
                movie_info=movie_info,
                save_files=True,
                run_id=run_id,
            )
            step_data['record_builder'] = result
            summary['record_builder'] = {
                'files': result.get('files', {}),
                'num_records': len(result.get('records', [])),
            }
        else:
            log_warn('Main', '缺少 idx_to_emotion 或 DataFrame，跳过记录组装')
            summary['record_builder'] = {}
    else:
        log('Main', '步骤 4/5: 跳过记录组装')
        summary['record_builder'] = {}

    # ── Step 5: 热图生成 ──────────────────────────────
    if 'heatmap_generator' not in skip_steps:
        log('Main', '\n' + '#' * 55)
        log('Main', '# 步骤 5/5: 热图生成')
        log('Main', '#' * 55)

        records = (step_data.get('record_builder', {}).get('records'))

        if records:
            result = run_heatmap_generator(records, run_id=run_id)
            summary['heatmap_generator'] = {
                'generated_count': result.get('generated_count', 0),
            }
        else:
            log_warn('Main', 'records 为空，跳过热图生成')
            summary['heatmap_generator'] = {}
    else:
        log('Main', '步骤 5/5: 跳过热图生成')
        summary['heatmap_generator'] = {}

    elapsed = (datetime.now() - ts_total).total_seconds()
    log('Main', f'\n流水线完成，总耗时: {elapsed:.1f} 秒 (run_id={run_id})')

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='LLM-Movie 分析流水线 (模块化 v6)',
    )
    parser.add_argument(
        '--skip', nargs='*', default=[],
        choices=_STEP_NAMES,
        help='跳过的步骤',
    )
    parser.add_argument(
        '--steps', nargs='*', default=[],
        choices=_STEP_NAMES,
        help='仅运行指定步骤',
    )
    parser.add_argument(
        '--emotion-mode', default='bert',
        choices=['bert', 'lexicon'],
        help='情绪识别模式: bert（BERT 6 分类，默认）| lexicon（词典三分类）',
    )
    parser.add_argument(
        '--run-id', default=None,
        help='指定流水号（默认自动生成）',
    )
    args = parser.parse_args()

    if args.steps:
        skip_steps = [s for s in _STEP_NAMES if s not in args.steps]
    else:
        skip_steps = args.skip

    summary = run_pipeline(skip_steps, emotion_mode=args.emotion_mode,
                           run_id=args.run_id)

    # 打印最终摘要
    log('Main', '\n' + '=' * 55)
    log('Main', '执行摘要')
    log('Main', '=' * 55)
    for step, info in summary.items():
        if isinstance(info, dict):
            log('Main', f'  {step}: {info}')
        else:
            log('Main', f'  {step}')


if __name__ == '__main__':
    main()
