"""
Shared plotting utilities for the movie analysis pipeline.
图表绘制工具：颜色常量、热力图标注、通用折线图等。
"""

import numpy as np

# ── Chart color scheme（图表配色方案）────────────────────────────────────────
COLOR_HOLIDAY = '#ff6b6b'      # 节假日：红色
COLOR_NONHOLIDAY = '#74b9ff'   # 非节假日：蓝色
COLOR_WORKDAY = '#feca57'      # 工作日：黄色
COLOR_WEEKEND = '#48dbfb'      # 周末：青色
HOLIDAY_CMAP = 'Set2'          # 节假日组柱状图使用的色图


def annotate_heatmap(ax, data, fmt='.1f', fs=6):
    """在imshow热力图上标注数值"""
    arr = data.data if isinstance(data, np.ma.MaskedArray) else np.asarray(data)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v) and abs(v) > 1e-6:
                ax.text(j, i, format(float(v), fmt), ha='center', va='center',
                        fontsize=fs, color='black')
