# -*- coding: utf-8 -*-
"""
热图生成模块 — 根据分析记录生成三维分析热图。

输出：output/v6_heatmap*.png

文件输出：无 intermediate 文件，直接生成 PNG 图片。
"""

import os
import sys

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    OUTPUT_DIR, sanitize_filename, _setup_font,
    generate_run_id, log, log_error, log_warn,
)
from src.debug_dump import extract_run_id, load_latest_by_step, find_latest_by_step

# 设置中文字体
_setup_font()


def _plot_heatmap(data: pd.DataFrame, index: str, columns: str,
                  title: str, filename: str):
    """通用的热力图绘制函数。"""
    try:
        pivot = data.pivot_table(
            index=index,
            columns=columns,
            aggfunc='size',
            fill_value=0,
        )
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

        if pivot.empty or pivot.shape[0] == 0 or pivot.shape[1] == 0:
            log_warn('Heatmap', f'跳过 {filename}: 透视表为空')
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
        log('Heatmap', f'  [OK] {filename}  ({pivot.shape[0]}x{pivot.shape[1]})')

    except Exception as e:
        log_error('Heatmap', f'{filename} 生成失败: {e}')
        import traceback
        traceback.print_exc()


def generate_heatmaps(records: list[dict], run_id: str = None):
    """根据分析记录生成多维度的热力图。

    参数：
        run_id : 流水号，用于输出 PNG 命名
    """
    if run_id is None:
        run_id = generate_run_id()

    if not records:
        log_warn('Heatmap', '记录为空，跳过热图生成')
        return

    df_rec = pd.DataFrame(records)
    log('Heatmap', f'总分析记录数: {len(df_rec)}')
    log('Heatmap', f'情绪分布: {df_rec["emotion"].value_counts().to_dict()}')

    age_groups = df_rec['age_segment'].unique()

    # 按预定义年龄段排序
    try:
        from src import age_segment as age_mod
        age_order = [s for s in age_mod.AGE_SEGMENTS if s in age_groups]
        age_order += sorted(set(age_groups) - set(age_mod.AGE_SEGMENTS))
    except ImportError:
        age_order = sorted(age_groups)

    for age in age_order:
        try:
            log('Heatmap', f'\n{"=" * 55}')
            log('Heatmap', f'年龄段: {age}')
            log('Heatmap', f'{"=" * 55}')

            sub = df_rec[df_rec['age_segment'] == age]
            if len(sub) < 3:
                log('Heatmap', f'  数据不足 ({len(sub)} 条)，跳过')
                continue

            safe_age = sanitize_filename(age)

            # ── 热图 1: 情绪 × 节假日 ──
            sub_holiday = sub[sub['holiday_name'] != '非节假日'].copy()
            if not sub_holiday.empty:
                pivot = sub_holiday.pivot_table(
                    index='emotion',        # 纵轴
                    columns='holiday_name', # 横轴
                    aggfunc='size',
                    fill_value=0,
                )
                # row_order = [e for e in EMOTION_CATEGORIES if e in pivot.index]
                emotion_groups = df_rec['emotion'].unique();
                row_order = [e for e in emotion_groups if e in pivot.index]
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
                    fname = f'{run_id}_S05_heatmap1_emotion_holiday_{safe_age}.png'
                    out_path = os.path.join(OUTPUT_DIR, fname)
                    fig.savefig(out_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    log('Heatmap',
                        f'  [OK] {fname}  '
                        f'({pivot.shape[0]}x{pivot.shape[1]})')
                else:
                    log_warn('Heatmap', '跳过热图 1: 透视表为空')
            else:
                log('Heatmap', '跳过热图 1: 无节假日数据')

            # ── 热图 2: 情绪 × 影片类型 ──
            _plot_heatmap(
                data=sub,
                index='emotion',
                columns='genre',
                title=f'{age} — 不同情绪对用户观影类型影响热图（BERT 6 类）',
                filename=f'{run_id}_S05_heatmap2_emotion_genre_{safe_age}.png',
            )

            # ── 热图 3: 节假日 × 影片类型 ──
            _plot_heatmap(
                data=sub,
                index='genre',           # 纵轴
                columns='holiday_name',  # 横轴
                title=f'{age} — 不同假日对用户观影类型影响热图（BERT 6 类）',
                filename=f'{run_id}_S05_heatmap3_holiday_genre_{safe_age}.png',
            )

        except Exception as e:
            log_error('Heatmap', f'年龄段 {age} 处理失败: {e}')
            continue


def export_csv_data(records: list[dict], run_id: str = None) -> dict:
    """
    将热力图对应的透视表数据导出为 CSV 文件。

    每个年龄段生成 3 个 CSV，与 generate_heatmaps 的三个热力图对应：
      1. emotion × holiday_name（仅节假日数据）
      2. emotion × genre
      3. genre × holiday_name

    CSV 的行 = 热力图纵轴（index），列 = 热力图横轴（columns），值 = 观影计数。

    参数：
        records : 分析记录列表
        run_id  : 流水号，为 None 时自动生成

    返回：
        dict — {'exported_count': int}
    """
    if run_id is None:
        run_id = generate_run_id()

    if not records:
        log_warn('Heatmap', '记录为空，跳过 CSV 导出')
        return {'exported_count': 0}

    df_rec = pd.DataFrame(records)
    log('Heatmap', f'开始导出 CSV 数据, 总记录数: {len(df_rec)}')

    # 年龄段排序
    age_groups = df_rec['age_segment'].unique()
    try:
        from src import age_segment as age_mod
        age_order = [s for s in age_mod.AGE_SEGMENTS if s in age_groups]
        age_order += sorted(set(age_groups) - set(age_mod.AGE_SEGMENTS))
    except ImportError:
        age_order = sorted(age_groups)

    exported = 0

    for age in age_order:
        sub = df_rec[df_rec['age_segment'] == age]
        if len(sub) < 3:
            continue

        safe_age = sanitize_filename(age)

        # ── CSV 1: emotion × holiday_name（仅节假日） ──
        sub_holiday = sub[sub['holiday_name'] != '非节假日'].copy()
        if not sub_holiday.empty:
            pivot = sub_holiday.pivot_table(
                index='emotion',
                columns='holiday_name',
                aggfunc='size',
                fill_value=0,
            )
            emotion_groups = df_rec['emotion'].unique()
            row_order = [e for e in emotion_groups if e in pivot.index]
            pivot = pivot.loc[row_order]
            col_order = pivot.sum(axis=0).sort_values(ascending=False).index
            pivot = pivot[col_order]

            if pivot.shape[0] > 0 and pivot.shape[1] > 0:
                fname = f'{run_id}_S05_heatmap1_emotion_holiday_{safe_age}.csv'
                out_path = os.path.join(OUTPUT_DIR, fname)
                pivot.to_csv(out_path, encoding='utf-8-sig')
                log('Heatmap', f'  [CSV] {fname}  ({pivot.shape[0]}x{pivot.shape[1]})')
                exported += 1

        # ── CSV 2: emotion × genre ──
        pivot2 = sub.pivot_table(
            index='emotion',
            columns='genre',
            aggfunc='size',
            fill_value=0,
        )
        pivot2 = pivot2.loc[pivot2.sum(axis=1).sort_values(ascending=False).index]
        if pivot2.shape[0] > 0 and pivot2.shape[1] > 0:
            fname = f'{run_id}_S05_heatmap2_emotion_genre_{safe_age}.csv'
            out_path = os.path.join(OUTPUT_DIR, fname)
            pivot2.to_csv(out_path, encoding='utf-8-sig')
            log('Heatmap', f'  [CSV] {fname}  ({pivot2.shape[0]}x{pivot2.shape[1]})')
            exported += 1

        # ── CSV 3: genre × holiday_name ──
        pivot3 = sub.pivot_table(
            index='genre',
            columns='holiday_name',
            aggfunc='size',
            fill_value=0,
        )
        pivot3 = pivot3.loc[pivot3.sum(axis=1).sort_values(ascending=False).index]
        if pivot3.shape[0] > 0 and pivot3.shape[1] > 0:
            fname = f'{run_id}_S05_heatmap3_holiday_genre_{safe_age}.csv'
            out_path = os.path.join(OUTPUT_DIR, fname)
            pivot3.to_csv(out_path, encoding='utf-8-sig')
            log('Heatmap', f'  [CSV] {fname}  ({pivot3.shape[0]}x{pivot3.shape[1]})')
            exported += 1

    log('Heatmap', f'CSV 导出完成: {exported} 个文件')
    return {'exported_count': exported}


def run(records: list[dict], run_id: str = None) -> dict:
    """
    统一入口：生成热图 PNG + CSV 数据文件。

    参数：
        records : 分析记录列表
        run_id  : 流水号（同一轮流水线执行共享此 ID）

    返回：
        dict — {'run_id': str, 'generated_count': int, 'csv_count': int}
    """
    if run_id is None:
        run_id = generate_run_id()

    log('Heatmap', '=' * 40)
    log('Heatmap', f'开始生成热图和 CSV (run_id={run_id})')

    count_before = len(os.listdir(OUTPUT_DIR)) if os.path.isdir(OUTPUT_DIR) else 0

    generate_heatmaps(records, run_id=run_id)
    csv_result = export_csv_data(records, run_id=run_id)

    count_after = len(os.listdir(OUTPUT_DIR)) if os.path.isdir(OUTPUT_DIR) else 0
    new_files = count_after - count_before

    log('Heatmap', f'热图生成完成，新增 {new_files} 个文件（含 {csv_result["exported_count"]} 个 CSV）')
    log('Heatmap', '=' * 40)

    return {
        'run_id': run_id,
        'generated_count': new_files,
        'csv_count': csv_result['exported_count'],
    }


def main():
    """
    独立运行入口。

    用法：python -m src.heatmap_generator
    说明：自动发现最新的 S04 records_full.jsonl，加载完整记录，
          沿用上游的 run_id 生成 S05 热图 PNG。
    """
    # 先从完整 records_full 加载
    records = None

    filepath = find_latest_by_step('04', 'records_full', extension='.jsonl')
    if filepath is None:
        # 回退到 records_summary（仅样本）
        data, filepath = load_latest_by_step('04', 'records_summary')
        if data:
            records = data.get('samples', [])
            log('Heatmap', f'从摘要中恢复 {len(records)} 条样本记录')
    else:
        # 需要从 jsonl 文件完整读取（load_latest_by_step 返回的 data 是 None for jsonl）
        pattern = filepath  # filepath already points to the latest
        full_records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    full_records.append(json.loads(line))
        records = full_records
        log('Heatmap', f'从 {os.path.basename(filepath)} 恢复 {len(records)} 条记录')

    if not records:
        log_error('Heatmap', '记录为空，无法运行')
        sys.exit(1)

    upstream_run_id = extract_run_id(filepath)
    result = run(records, run_id=upstream_run_id)
    print(f'\n热图生成完成: {result["generated_count"]} 个文件'
          f'（含 {result["csv_count"]} 个 CSV）, run_id: {result["run_id"]}')


if __name__ == '__main__':
    main()