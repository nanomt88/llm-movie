#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Master pipeline: runs Step 1 → Step 5 sequentially.
主流水线：按顺序依次运行步骤 1 到步骤 5。

Usage（使用方法）:
    python pipeline.py               # 运行所有步骤
    python pipeline.py --steps 1 3 5  # 只运行指定的步骤（1、3、5）
    python pipeline.py --skip 4      # 跳过步骤 4
"""

import sys          # 系统模块，用于程序退出等
import time         # 时间模块，用于计时和延迟
import argparse     # 命令行参数解析模块
import traceback    # 异常追踪模块，用于打印完整的错误堆栈

from movie.config import log, STEP_DIRS  # 导入日志函数和步骤输出目录配置


# 步骤编号 -> 步骤名称的映射字典
STEPS = {
    1: "Question Frequency Analysis",      # 步骤1：提问频率分析
    2: "Active Users Analysis",            # 步骤2：活跃用户分析
    3: "Conversation Analysis",            # 步骤3：会话分析
    4: "Age Distribution Analysis",        # 步骤4：年龄分布分析
    5: "Movie Genre Analysis",             # 步骤5：电影类型分析
    6: "Movie Daily Analysis",             # 步骤5：电影每日数据
}


def run_step(step_num: int) -> float:
    """Run a single step by number, return elapsed seconds.
       按编号运行单个步骤，返回耗时（秒）。
    Args:
        step_num: 步骤编号（1-5）
    Returns:
        该步骤执行所花费的秒数
    """
    name = STEPS.get(step_num, f"Step {step_num}")  # 获取步骤名称，未知步骤则显示 "Step N"
    log(f"{'='*60}")                                 # 打印分隔线
    log(f"Pipeline: Starting Step {step_num} — {name}")  # 打印开始信息
    log(f"{'='*60}")

    t0 = time.time()      # 记录开始时间
    try:
        # 根据步骤编号动态导入对应的模块并获取 main 函数引用
        if step_num == 1:
            from movie.step1_question_freq import main as m
        elif step_num == 2:
            from movie.step2_active_users import main as m
        elif step_num == 3:
            from movie.step3_conversation import main as m
        elif step_num == 4:
            from movie.step4_age import main as m
        elif step_num == 5:
            from movie.step5_genre import main as m
        elif step_num == 6:
            from movie.step6_yearly_monthly import main as m
        else:
            log(f"  Unknown step {step_num}")  # 未知步骤
            return 0.0

        m()  # 执行该步骤的 main 函数
    except Exception as e:
        log(f"  ERROR in Step {step_num}: {e}")   # 打印错误信息
        traceback.print_exc()                      # 打印完整异常堆栈
        return time.time() - t0                    # 返回已耗时（即使出错）

    elapsed = time.time() - t0   # 计算实际耗时
    log(f"Step {step_num} completed in {elapsed:.1f}s")  # 打印完成信息和耗时
    return elapsed


def main():
    """Parse CLI args and run selected steps in order.
       解析命令行参数并按顺序运行选中的步骤。"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="Movie Analysis Pipeline — Step 1 to Step 6")
    # --steps 参数：指定要运行的步骤编号列表
    parser.add_argument(
        '--steps', nargs='+', type=int,
        help='Steps to run (e.g. --steps 1 3 5). Default: all steps 1-6.')
    # --skip 参数：指定要跳过的步骤编号列表
    parser.add_argument(
        '--skip', nargs='+', type=int, default=[],
        help='Steps to skip (e.g. --skip 4)')
    # --timing 参数：是否报告每个步骤的计时（默认开启）
    parser.add_argument(
        '--timing', action='store_true', default=True,
        help='Report step timing (default: True)')

    args = parser.parse_args()  # 解析命令行参数

    # 确定要运行的步骤顺序
    if args.steps:
        # 如果指定了 --steps，取指定步骤和已知步骤的交集，排序后作为运行顺序
        step_order = sorted(set(args.steps) & set(STEPS.keys()))
    else:
        step_order = sorted(STEPS.keys())  # 默认运行所有步骤 1-5

    # 从运行列表中移除 --skip 指定的步骤
    step_order = [s for s in step_order if s not in set(args.skip)]

    if not step_order:
        log("No steps to run. Exiting.")  # 如果没有需要运行的步骤，直接退出
        return

    # 打印将要运行的步骤和输出目录
    log(f"Pipeline: steps to run = {step_order}")
    log(f"Output dirs: {', '.join(STEP_DIRS[s] for s in step_order)}")
    log("")

    timings = {}           # 步骤编号 -> 耗时的字典
    total_t0 = time.time() # 记录总开始时间

    # 依次运行每个选中的步骤
    for step in step_order:
        elapsed = run_step(step)   # 运行步骤并获取耗时
        timings[step] = elapsed    # 记录该步骤耗时
        log("")

    total_elapsed = time.time() - total_t0  # 计算总耗时

    # Summary（打印汇总信息）
    log("=" * 60)
    log("PIPELINE SUMMARY")      # 流水线执行摘要
    log("=" * 60)
    for step in step_order:
        name = STEPS.get(step, f"Step {step}")  # 步骤名称
        t = timings.get(step, 0.0)               # 步骤耗时
        log(f"  Step {step} ({name}): {t:.1f}s")
    log(f"  ─────────────────────────────")
    log(f"  Total: {total_elapsed:.1f}s")         # 总耗时
    log("")
    log(f"All outputs saved under:")              # 所有输出文件保存在
    for step in step_order:
        log(f"  {STEP_DIRS[step]}")               # 各步骤的输出目录


if __name__ == '__main__':
    main()  # 当脚本直接运行时执行 main 函数
