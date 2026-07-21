# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文等非 ASCII 字符
"""
Step 7.2: Valentine's Day vs Romance Genre — Statistical Verification
步骤 7.2：情人节 vs 爱情片 关联性统计验证

【背景】
  step5_genre.py 的 J1–J5 用"绝对计数 + 多类型累加"统计各节假日的类型提及次数。
  由于电影通常带 2–5 个类型、且剧情在全库占 49.6%，绝对计数天然被剧情主导，
  情人节的"爱情"日均值（69.25）反而低于非节假日基线（97.43），排名第 9。
  无法回答"情人节是否更倾向推荐爱情片"这一因果性问题。

【本步骤目标】
  验证"情人节当天系统是否更倾向于推荐爱情片"。
  采用份额（share）而非绝对计数，避免多类型累加偏置；
  对照窗口为同年情人节 ±14 天内的非节假日日期，
  排除总统日 / 超级碗周日 / 春节 / 元旦 等邻近节假日干扰。

【方法】
  - 实验组：2019–2022 共 4 个情人节
  - 对照组：每年 2 月 ±14 天内 r['period'] in {workday, weekend} 的日期
  - 每条 seeker 记录提取系统回复中的电影 ID（与 step5 完全一致口径），
    判定该记录是否至少推荐了一部爱情片（'爱情' in genres）
  - 主指标：romance_share = 记录中含爱情片的比例（records-with-romance / total-records）
  - 检验：2×2 列联表 Pearson χ² + Fisher 精确检验 + Cohen's h 效应量 + RR 相对风险

【输出】
  output/movie/step7/v_valentine_vs_control_romance.png  逐年柱状图（带 95% CI）
  output/movie/step7/v_valentine_vs_control_romance.csv  逐年统计 + 汇总检验结果
"""

import os            # 文件路径操作
import csv            # CSV 输出
from collections import defaultdict   # 带默认值的字典
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


# ── 实验组：情人节日期（2018 年的 holiday.csv 中无情人节，故从 2019 起）──
VALENTINE_DATES = {
    '2019-02-14', '2020-02-14', '2021-02-14', '2022-02-14',
}

# 对照窗口半径（天）：每年情人节前后 ±14 天
CONTROL_WINDOW_DAYS = 14


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 取一条 seeker 记录对应系统回复中所有电影的类型集合
# ═══════════════════════════════════════════════════════════════════════
def _genre_set_for_seeker(
    r: dict,
    conv_system: dict,
    movie_info: dict,
):
    """Return the set of genres for the movies in this seeker's system reply.
       返回该 seeker 对应系统回复中所有电影的类型集合。

    Returns None if the system reply has no movie IDs (record excluded).
    若系统回复中没有电影 ID（无法判定类型），返回 None（该记录排除）。
    """
    mids = get_system_movie_ids(r.get('conv_id', ''), conv_system)
    if not mids:                          # 系统回复无电影 ID
        return None
    genres = set()
    for mid in mids:
        info = movie_info.get(mid, {})
        if isinstance(info, dict):
            gl = info.get('genres', []) or []
            if gl:
                genres.update(g.strip() for g in gl if g.strip())
    return genres                         # 可能为空 set（电影在 movie_info 中缺失或无类型）


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 把 seekers 聚合为 (year, date, group, has_romance) 记录列表
# ═══════════════════════════════════════════════════════════════════════
def _aggregate_records(seekers, conv_system, movie_info):
    """Aggregate seekers into records with (year, date, group, has_romance).

    实验组：date ∈ VALENTINE_DATES
    对照组：同年 2 月内、距情人节 ±14 天、period ∈ {workday, weekend}
            （排除所有 period=='holiday' 的日期，含总统日、超级碗、春节等）

    排除规则：系统回复中无电影 ID 的记录（无法判定类型）。
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
        if m != 2:                        # 实验+对照窗口都在 2 月
            continue

        # 分组判定
        if date_str in VALENTINE_DATES:
            group = 'Valentine'
        elif r.get('period') in ('workday', 'weekend'):
            # 同年 2 月 ±14 天内才作为对照
            v_date = date(y, 2, 14)
            cur_date = date(y, m, d)
            delta = (cur_date - v_date).days
            if abs(delta) > CONTROL_WINDOW_DAYS or delta == 0:
                continue                  # 超出对照窗口或落在情人节当天
            group = 'Control'
        else:
            continue                      # period == 'holiday' 的其他节假日排除

        # 取该记录系统回复的电影类型集合
        genres = _genre_set_for_seeker(r, conv_system, movie_info)
        if genres is None:                # 无系统回复电影，无法判定
            continue

        records.append({
            'year': y,
            'date': date_str,
            'group': group,
            'has_romance': '爱情' in genres,    # 主指标：是否至少推荐一部爱情片
        })
    return records


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 逐年 × 分组 聚合统计
# ═══════════════════════════════════════════════════════════════════════
def _per_year_stats(records):
    """Per-year per-group: total records, romance records, romance share.
       逐年 × 分组聚合：总记录数、含爱情片记录数、爱情片份额。"""
    agg = defaultdict(lambda: defaultdict(int))   # (year, group) -> {total, romance}
    for r in records:
        key = (r['year'], r['group'])
        agg[key]['total'] += 1
        if r['has_romance']:
            agg[key]['romance'] += 1
    rows = []
    for (y, g), c in agg.items():
        rows.append({
            'year': y,
            'group': g,
            'n_records': c['total'],
            'n_romance': c['romance'],
            'romance_share': c['romance'] / c['total'] if c['total'] else 0.0,
        })
    df = pd.DataFrame(rows).sort_values(['year', 'group']).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Helper: 汇总统计 + 显著性检验
# ═══════════════════════════════════════════════════════════════════════
def _pooled_stats(records):
    """Pool all years and run 2x2 chi-square + Fisher + Cohen's h + RR.
       汇总全部年份做 2×2 列联表检验 + 效应量。"""
    val = [r for r in records if r['group'] == 'Valentine']
    ctrl = [r for r in records if r['group'] == 'Control']
    val_n = len(val)
    val_r = sum(1 for r in val if r['has_romance'])
    ctrl_n = len(ctrl)
    ctrl_r = sum(1 for r in ctrl if r['has_romance'])
    val_share = val_r / val_n if val_n else 0.0
    ctrl_share = ctrl_r / ctrl_n if ctrl_n else 0.0

    # 2×2 列联表
    #   [[val_romance,       val_non_romance],
    #    [ctrl_romance,      ctrl_non_romance]]
    table = np.array([
        [val_r,  val_n  - val_r],
        [ctrl_r, ctrl_n - ctrl_r],
    ])

    # Pearson 卡方检验（默认带 Yates 连续性修正，适合 2×2）
    chi2, p_chi2, dof, expected = stats.chi2_contingency(table)
    # Fisher 精确检验（小样本更可靠，双侧）
    odds_ratio, p_fisher = stats.fisher_exact(table, alternative='two-sided')

    # Cohen's h 效应量：h = 2·arcsin(√p1) − 2·arcsin(√p2)
    def _phi(p):
        return 2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0)))
    h = _phi(val_share) - _phi(ctrl_share)

    # 相对风险 RR = P(val) / P(ctrl)
    rr = val_share / ctrl_share if ctrl_share > 0 else float('inf')

    return {
        'valentine_n': val_n,
        'valentine_romance': val_r,
        'valentine_share': val_share,
        'control_n': ctrl_n,
        'control_romance': ctrl_r,
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
#  Helper: 逐年柱状图（带 95% 置信区间误差棒）
# ═══════════════════════════════════════════════════════════════════════
def _plot(per_year_df, pooled, path):
    """Bar chart: per-year Valentine vs Control romance share + 95% CI.
       逐年柱状图：情人节 vs 对照组爱情片份额，带 95% CI 误差棒。"""
    years = sorted(per_year_df['year'].unique())
    val_shares, ctrl_shares, val_errs, ctrl_errs = [], [], [], []
    for y in years:
        v = per_year_df[(per_year_df['year'] == y) & (per_year_df['group'] == 'Valentine')]
        c = per_year_df[(per_year_df['year'] == y) & (per_year_df['group'] == 'Control')]
        vs = v['romance_share'].iloc[0] if len(v) else 0.0
        vn = v['n_records'].iloc[0]    if len(v) else 0
        cs = c['romance_share'].iloc[0] if len(c) else 0.0
        cn = c['n_records'].iloc[0]    if len(c) else 0
        val_shares.append(vs)
        ctrl_shares.append(cs)
        # 95% CI 近似：1.96·√(p(1-p)/n)
        val_errs.append(1.96 * np.sqrt(vs * (1 - vs) / max(vn, 1)) if vn else 0.0)
        ctrl_errs.append(1.96 * np.sqrt(cs * (1 - cs) / max(cn, 1)) if cn else 0.0)

    x = np.arange(len(years))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, val_shares,  w, yerr=val_errs,  label='Valentine',
           color='#e74c3c', alpha=0.85, capsize=4)
    ax.bar(x + w / 2, ctrl_shares, w, yerr=ctrl_errs,
           label=f'Control (±{CONTROL_WINDOW_DAYS}d, non-holiday)',
           color='#3498db', alpha=0.85, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel('Romance Share (records with ≥1 romance movie)')
    ax.set_title(
        "Valentine vs Control — Romance Genre Share\n"
        f"pooled Δ={pooled['delta_share']:.4f}, RR={pooled['relative_risk']:.2f}, "
        f"p(χ²)={pooled['p_chi2']:.4f}, p(Fisher)={pooled['p_fisher']:.4f}, "
        f"Cohen's h={pooled['cohens_h']:.3f}",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ymax = max(val_shares + ctrl_shares) if (val_shares or ctrl_shares) else 0.0
    ax.set_ylim(0, ymax * 1.4 + 0.05)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Helper: CSV 输出（逐年 + 汇总检验）
# ═══════════════════════════════════════════════════════════════════════
def _save_csv(per_year_df, pooled, path):
    """Save per-year stats + pooled test results to CSV.
       保存逐年统计 + 汇总检验结果到 CSV。"""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['year', 'group', 'n_records', 'n_romance', 'romance_share'])
        for _, row in per_year_df.iterrows():
            w.writerow([row['year'], row['group'], row['n_records'],
                        row['n_romance'], f"{row['romance_share']:.4f}"])
        # 汇总
        w.writerow([])
        w.writerow(['# Pooled'])
        w.writerow(['group', 'n_records', 'n_romance', 'romance_share'])
        w.writerow(['Valentine', pooled['valentine_n'], pooled['valentine_romance'],
                    f"{pooled['valentine_share']:.4f}"])
        w.writerow(['Control', pooled['control_n'], pooled['control_romance'],
                    f"{pooled['control_share']:.4f}"])
        # 检验
        w.writerow([])
        w.writerow(['# Tests'])
        w.writerow(['delta_share',   f"{pooled['delta_share']:.4f}"])
        w.writerow(['relative_risk', f"{pooled['relative_risk']:.4f}"])
        w.writerow(['cohens_h',      f"{pooled['cohens_h']:.4f}"])
        w.writerow(['chi2',          f"{pooled['chi2']:.4f}"])
        w.writerow(['p_chi2',        f"{pooled['p_chi2']:.4f}"])
        w.writerow(['odds_ratio',    f"{pooled['odds_ratio']:.4f}"])
        w.writerow(['p_fisher',      f"{pooled['p_fisher']:.4f}"])
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main  主函数入口
# ═══════════════════════════════════════════════════════════════════════
def main(data: dict = None):
    log("=" * 60)
    log("Step 7.2: Valentine's Day vs Romance Genre — Verification")
    log("=" * 60)

    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']
    rows = data['rows']
    movie_info = data['movie_info']

    log(f"Building conv_system from {len(rows)} rows ...")
    conv_system = build_conv_system(rows)
    log(f"Processing {len(seekers)} seekers ...")

    records = _aggregate_records(seekers, conv_system, movie_info)
    n_val = sum(1 for r in records if r['group'] == 'Valentine')
    n_ctrl = sum(1 for r in records if r['group'] == 'Control')
    log(f"Valid records: Valentine={n_val}, Control={n_ctrl}")

    # 逐年统计
    per_year = _per_year_stats(records)
    log("")
    log("Per-year stats:")
    for _, row in per_year.iterrows():
        log(f"  {row['year']} {row['group']:>9}: n={row['n_records']:>5}, "
            f"romance={row['n_romance']:>4}, share={row['romance_share']:.4f}")

    # 汇总 + 检验
    pooled = _pooled_stats(records)
    log("")
    log("Pooled:")
    log(f"  Valentine: n={pooled['valentine_n']}, romance={pooled['valentine_romance']}, "
        f"share={pooled['valentine_share']:.4f}")
    log(f"  Control:   n={pooled['control_n']}, romance={pooled['control_romance']}, "
        f"share={pooled['control_share']:.4f}")
    log(f"  Δ={pooled['delta_share']:.4f}  RR={pooled['relative_risk']:.3f}  "
        f"Cohen's h={pooled['cohens_h']:.3f}")
    log(f"  χ²={pooled['chi2']:.3f}  p(χ²)={pooled['p_chi2']:.4f}  "
        f"OR={pooled['odds_ratio']:.3f}  p(Fisher)={pooled['p_fisher']:.4f}")

    # 判定结论
    if pooled['p_chi2'] < 0.05 and pooled['delta_share'] > 0:
        verdict = "Valentine's Day recommends SIGNIFICANTLY MORE romance movies (Δ>0, p<0.05)"
    elif pooled['p_chi2'] < 0.05 and pooled['delta_share'] < 0:
        verdict = "Valentine's Day recommends SIGNIFICANTLY FEWER romance movies (Δ<0, p<0.05)"
    else:
        verdict = "No significant association between Valentine's Day and romance recommendations"
    log("")
    log(f"  Verdict: {verdict}")

    # 输出
    _plot(per_year, pooled,
          os.path.join(STEP_OUT, 'v_valentine_vs_control_romance.png'))
    _save_csv(per_year, pooled,
              os.path.join(STEP_OUT, 'v_valentine_vs_control_romance.csv'))

    log("")
    log("=" * 60)
    log(f"Step 7.2 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
