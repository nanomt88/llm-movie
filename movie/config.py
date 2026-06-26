# -*- coding: utf-8 -*-
# 设置文件编码为 UTF-8，确保中文字符正常显示

"""
Shared configuration for the movie analysis pipeline.
Path, constants, font setup, logging.
电影分析流水线共享配置：路径、常量、字体设置、日志。
"""

import os           # 操作系统接口，用于文件和路径操作
from datetime import datetime  # 日期时间模块，用于时间戳格式化

# ── Paths（路径配置）─────────────────────────────────────────────────────────────
# 项目根目录：当前文件所在目录的上一级（即 llm-movie/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据目录：项目根目录下的 data/ 文件夹
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
# 输出目录：项目根目录下的 output/movie/ 文件夹
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'movie')
# 自动创建输出目录（如果已存在则不报错）
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input files（输入文件路径）
HOLIDAY_CSV = os.path.join(DATA_DIR, 'holiday.csv')                     # 节假日定义表
HOLIDAY_WORKDAY_CSV = os.path.join(DATA_DIR, 'holiday-workday.csv')     # 节假日调休调整表
FULL_YEAR_CSV = os.path.join(DATA_DIR, 'conv', 'data_all.csv')          # 全年会话数据
HOLIDAY_CONV_CSV = os.path.join(DATA_DIR, 'conv', 'all_holiday_records_v3.csv')  # 节假日会话数据
MOVIE_INFO_PATH = os.path.join(DATA_DIR, 'movie_info.json')             # 电影信息（含类型）
ENTITY2ID_PATH = os.path.join(DATA_DIR, 'entity2id.json')               # 实体到ID映射
USER_PROFILES_PATH = os.path.join(DATA_DIR, 'user_profiles.json')       # 用户画像
USER_AGE_SEG_PATH = os.path.join(DATA_DIR, 'conv', 'totle_user_seg_v3.json')  # 用户年龄段划分

# ── Thresholds（阈值设定）────────────────────────────────────────────────────────
MIN_DATA_ROWS = 10    # 单个节假日最少数据行数，少于该值则跳过该节假日分析
MIN_DATA_DAYS = 3     # 单个节假日最少独立天数，少于该值则不进行该节日的逐日分析

# ── Age segments（年龄段划分，与现有项目保持一致）──────────────────────────────────
AGE_SEGMENTS = ['<18', '18-25', '26-35', '36-50', '50+', 'unknown']

# ── Output directories（输出子目录）──────────────────────────────────────────────
STEP_DIRS = {}                        # 步骤编号 -> 输出目录路径 的字典
for step in range(1, 7):              # 遍历步骤 1 到 6
    d = os.path.join(OUTPUT_DIR, f'step{step}')   # 每个步骤的输出子目录
    os.makedirs(d, exist_ok=True)     # 自动创建目录
    STEP_DIRS[step] = d               # 存入字典


# ── Matplotlib font setup（Matplotlib 字体配置）─────────────────────────────────
def setup_matplotlib():
    """Configure matplotlib: Agg backend + font.
       配置 matplotlib：使用 Agg 后端（无 GUI 渲染）+ 中文字体。"""
    import matplotlib                # 导入 matplotlib 库
    matplotlib.use('Agg')            # 使用 Agg 后端，适用于无图形界面的服务器环境
    import matplotlib.pyplot as plt  # 导入 pyplot 模块

    # Try Chinese fonts（尝试加载中文字体）
    cn_fonts = [                     # 中文字体候选列表，按优先级排序
        'Microsoft YaHei',           # 微软雅黑（Windows）
        'SimHei',                    # 黑体（Windows）
        'WenQuanYi Micro Hei',       # 文泉驿微米黑（Linux）
        'Noto Sans CJK SC',          # Google Noto 简体中文（跨平台）
        'Source Han Sans SC',        # 思源黑体简体中文（跨平台）
    ]
    for f in cn_fonts:               # 依次尝试每个字体
        try:
            # 查找字体文件，不自动回退到默认字体（找不到会抛出异常）
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            # 将找到的字体设为默认无衬线字体，放在列表最前面以优先使用
            matplotlib.rcParams['font.sans-serif'] = [f] + matplotlib.rcParams['font.sans-serif']
            # 设置坐标轴负号正常显示（不使用 Unicode 减号，避免字体问题）
            matplotlib.rcParams['axes.unicode_minus'] = False
            break                     # 找到可用字体后退出循环
        except Exception:
            continue                  # 当前字体不可用，尝试下一个

    # 设置全局 DPI（每英寸点数），提高保存图片的分辨率
    plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150})


# ── Logging（日志输出）───────────────────────────────────────────────────────────
def log(msg: str, module: str = "Movie"):
    """Print a timestamped log message.
       打印带时间戳的日志消息。
    Args:
        msg:    日志内容字符串
        module: 模块名称，默认为 "Movie"
    """
    ts = datetime.now().strftime('%H:%M:%S')   # 获取当前时间的 HH:MM:SS 格式
    print(f"[{ts}] [{module}] {msg}", flush=True)  # 输出日志并立即刷新
