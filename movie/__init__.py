# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，支持中文等非 ASCII 字符

"""
Movie Analysis Pipeline Package.
电影分析流水线包。

Analysis dimensions for holiday vs non-holiday vs workday vs weekend
in Reddit movie conversation data (2018-2022).
分析维度：节假日 vs 非节假日 vs 工作日 vs 周末，基于 Reddit 电影讨论数据（2018-2022）。

Steps（分析步骤）:
  1. Question Frequency (weekly + hourly)    —— 提问频率分析（按周 + 按小时）
  2. Active Users (weekly + hourly)          —— 活跃用户分析（按周 + 按小时）
  3. Conversation Turns (turns, multi-turn time, single/cross-day)  —— 会话轮次分析
  4. Age Distribution (weekly + hourly)      —— 年龄分布分析（按周 + 按小时）
  5. Movie Genre Analysis (weekly + hourly)  —— 电影类型分析（按周 + 按小时）
  6. Daily Movie Analysis                    —— 电影每日数据分析
  7. High-Frequency Words & Word Cloud       —— 高频词分析与词云
  8. LDA Topic Model & Holiday Preference    —— LDA主题模型与节假日偏好
  9. Co-occurrence Network & Sentiment       —— 共现网络与情感分析
  10. Aspect-Based Sentiment Analysis        —— 基于方面的情感分析

Usage（使用方法）:
    python -m movie.pipeline
"""
