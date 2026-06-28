# -*- coding: utf-8 -*-
"""
Pipeline: chain all 4 analysis steps into a single run.
Imports and calls each step's main() in sequence with progress reporting.

Usage:
    python -m data_analyzer.pipeline
"""

import time
from datetime import timedelta

from data_analyzer.config import log, setup_matplotlib

setup_matplotlib()


def _run_step(step_num: int, total: int, name: str, module):
    """Run a single step with timing."""
    log("")
    log("=" * 65, "Pipeline")
    log(f"  Step {step_num}/{total}: {name}", "Pipeline")
    log(f"  Started at: {time.strftime('%H:%M:%S')}", "Pipeline")
    log("=" * 65, "Pipeline")

    t0 = time.time()
    try:
        module.main()
        elapsed = time.time() - t0
        log(f"  Step {step_num} completed in {timedelta(seconds=elapsed)}", "Pipeline")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        log(f"  Step {step_num} FAILED after {timedelta(seconds=elapsed)}: {e}", "Pipeline")
        import traceback
        traceback.print_exc()
        return False


def main():
    t_start = time.time()
    log("")
    log("=" * 65, "Pipeline")
    log("  Pipeline: Holiday Movie Data Analysis (2018-2022)", "Pipeline")
    log("  Starting all 4 analysis steps...", "Pipeline")
    log("=" * 65, "Pipeline")

    # ── Import all step modules ──────────────────────────────────────
    from data_analyzer import (
        step1_length_freq as s1,
        step2_time_turns as s2,
        step3_scenarios as s3,
        step4_holiday_compare as s4,
    )

    steps = [
        (1, "Question Length & Access Frequency (Dimensions 1+2)", s1),
        (2, "Time Distribution & Conversation Turns (Dimensions 3+4)", s2),
        (3, "Special Scenario Demand Analysis (Dimension 5)", s3),
        (4, "Holiday Comparison Analysis (Dimensions 6+7)", s4),
    ]

    total = len(steps)
    successes = 0
    failures = 0

    for step_num, step_name, step_module in steps:
        ok = _run_step(step_num, total, step_name, step_module)
        if ok:
            successes += 1
        else:
            failures += 1
            log(f"  ⚠ Continuing to next step despite failure...", "Pipeline")

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t_start
    log("")
    log("=" * 65, "Pipeline")
    log(f"  Pipeline Complete!", "Pipeline")
    log(f"  Steps: {successes}/{total} succeeded", "Pipeline")
    log(f"  Total time: {timedelta(seconds=total_time)}", "Pipeline")
    log(f"  Output directory: output/step{{1,2,3,4}}/", "Pipeline")
    log("=" * 65, "Pipeline")


if __name__ == '__main__':
    main()
