# -*- coding: utf-8 -*-
"""
配置模块 — 路径、常量、CPU 线程、字体设置。

运行时机：模块加载时自动执行（无文件 I/O，纯配置）。
"""

import os
import sys
from datetime import datetime


def generate_run_id() -> str:
    """
    生成流水号（run_id），用于同一批次中间文件的命名分组。

    格式：YYYYMMDD_HHMMSS（例如 20260606_143022）

    所有中间文件名统一使用此 ID，方便识别哪些文件来自同一轮流水线执行。
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

# ══════════════════════════════════════════════════════════════════
#  CPU 线程数设置（必须在任何 torch/tf 操作之前）
# ══════════════════════════════════════════════════════════════════
import torch

torch.set_num_threads(16)
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'

# ── Hugging Face 镜像（国内加速下载） ─────────────────────────────
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ══════════════════════════════════════════════════════════════════
#  路径配置
# ══════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, 'intermediate')

CSV_PATH = os.path.join(DATA_DIR, 'my-tran-holidy-data.csv')
HOLIDAY_CSV = os.path.join(DATA_DIR, 'holiday.csv')
MOVIE_INFO_PATH = os.path.join(DATA_DIR, 'movie_info.json')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  情感标签常量
# ══════════════════════════════════════════════════════════════════

# bhadresh-savani/bert-base-uncased-emotion 模型输出的 6 种情绪标签
# 与 Ekman (1992) 6 种基本情绪对比：
#   - 快乐(Happiness)  ←→ joy        ✓ 匹配
#   - 悲伤(Sadness)    ←→ sadness    ✓ 匹配
#   - 愤怒(Anger)      ←→ anger      ✓ 匹配
#   - 恐惧(Fear)       ←→ fear       ✓ 匹配
#   - 惊讶(Surprise)   ←→ surprise   ✓ 匹配（Ekman 中有）
#   - 厌恶(Disgust)    → 模型无此分类；模型额外包含 love(喜爱)
# 按照"若模型与 Ekman 不同，以模型为准"的原则，使用模型的 6 分类
EMOTION_CATEGORIES = ['悲伤', '快乐', '喜爱', '愤怒', '恐惧', '惊讶']

MODEL_LABEL_MAP = {
    'sadness': '悲伤',
    'joy': '快乐',
    'love': '喜爱',
    'anger': '愤怒',
    'fear': '恐惧',
    'surprise': '惊讶',
}

# 基于词典的情感三分类（正面 / 中性 / 负面）
SENTIMENT_CATEGORIES = ['正面', '中性', '负面']

# ══════════════════════════════════════════════════════════════════
#  电影类型常量
# ══════════════════════════════════════════════════════════════════

GENRE_CATEGORIES = [
    '冒险', '剧情', '动作', '动作冒险', '动画', '历史', '喜剧', '奇幻',
    '家庭', '恐怖', '悬疑', '惊悚', '战争', '爱情', '犯罪', '电视电影',
    '科幻', '纪录', '西部', '音乐',
]

# ══════════════════════════════════════════════════════════════════
#  中文字体回退
# ══════════════════════════════════════════════════════════════════

_CN_FONTS = [
    'Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei',
    'Noto Sans CJK SC', 'Source Han Sans SC',
]

CN_FONT = None


def _setup_font():
    """尝试加载中文字体，设置 matplotlib 全局字体。"""
    import matplotlib
    for f in _CN_FONTS:
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            CN_FONT = f
            break
        except Exception:
            continue

    if CN_FONT:
        matplotlib.rcParams['font.sans-serif'] = [CN_FONT] + matplotlib.rcParams['font.sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False


# ══════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════

def sanitize_filename(name: str) -> str:
    """清理文件名中的特殊字符"""
    return name.replace('<', 'lt').replace('>', 'gt') \
               .replace('+', 'p').replace(' ', '_')


def log(module: str, message: str, level: str = "INFO"):
    """统一日志输出格式：带时间戳 + 模块名"""
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"  [{ts}] [{level}] [{module}] {message}")


def log_error(module: str, message: str):
    """错误日志"""
    log(module, message, "ERROR")


def log_warn(module: str, message: str):
    """警告日志"""
    log(module, message, "WARN")
