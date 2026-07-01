# -*- coding: utf-8 -*-
"""
Step 7: High-Frequency Word Analysis & Word Cloud
步骤 7：高频词分析与词云

Analysis:
  - Overall word frequency from user seekers' text
  - Word cloud visualization (overall)
  - Holiday vs non-holiday word frequency comparison
  - Per-holiday word frequency vs baseline
  - Holiday vs workday vs weekend word frequency comparison

Output: output/movie/step7/*.png + CSV
"""

import os               # 文件路径操作
import csv              # CSV 读写
import re               # 正则表达式，用于分词
import string           # 字符串工具
from collections import Counter, defaultdict   # 计数器与默认字典

import numpy as np      # 数值计算
from wordcloud import WordCloud                # 词云生成

import matplotlib
matplotlib.use('Agg')                           # 非交互式后端（服务器环境）
import matplotlib.pyplot as plt

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log

# ── 初始化 ──────────────────────────────────────────────────────────
setup_matplotlib()
STEP_OUT = STEP_DIRS[7]                         # 输出目录：output/movie/step7/
os.makedirs(STEP_OUT, exist_ok=True)

# ── 停用词表 ────────────────────────────────────────────────────────
# 标准英文停用词 + 领域特定噪音词
STOPWORDS = set({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine',
    'yours', 'hers', 'its', 'ours', 'theirs', 'this', 'that', 'these',
    'those', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'for',
    'with', 'on', 'in', 'at', 'to', 'from', 'by', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'of', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very',
    'just', 'because', 'as', 'until', 'while', 'if', 'else', 'like',
    'also', 'any', 'many', 'much', 'one', 'two', 'three', 'who', 'what',
    'which', 'doesn', 'don', 'didn', 'won', 'can', 'couldn', 'wouldn',
    'shouldn', 'isn', 'aren', 'wasn', 'weren', 'hasn', 'haven', 'hadn',
    'im', 'ive', 'id', 'youre', 'youve', 'theyll', 'theyre', 'theyd',
    'its', 'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'couldnt',
    'shouldnt', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent',
    'hadnt', 'let', 'get', 'got', 'gotten', 'going', 'go', 'goes',
    'went', 'see', 'seen', 'saw', 'know', 'known', 'knew', 'make',
    'made', 'makes', 'want', 'wants', 'wanted', 'take', 'took', 'taken',
    'takes', 'need', 'needs', 'needed', 'say', 'says', 'said', 'find',
    'finds', 'found', 'give', 'gives', 'gave', 'given', 'think',
    'thinks', 'thought', 'tell', 'tells', 'told', 'look', 'looks',
    'looked', 'use', 'uses', 'used', 'try', 'tries', 'tried', 'thanks',
    'please', 'help', 'hello', 'hi', 'good', 'bad', 'really', 'actually',
    'well', 'even', 'still', 'though', 'although', 'thing', 'things',
    'something', 'anything', 'everything', 'nothing', 'someone', 'anyone',
    'everyone', 'some', 'any', 'every', 'much', 'lot', 'lots', 'little',
    'bit', 'way', 'ways', 'kind', 'kinds', 'type', 'types', 'part',
    'parts', 'place', 'places', 'time', 'times', 'day', 'days', 'year',
    'years', 'new', 'old', 'first', 'last', 'next', 'good', 'great',
    'best', 'better', 'worst', 'worse', 'many', 'much', 'always',
    'never', 'ever', 'often', 'sometimes', 'usually', 'already', 'yet',
    'also', 'back', 'around', 'away', 'here', 'there', 'everywhere',
    'please', 'thank', 'thanks', 'much', 'able', 'possible', 'need',
})

# 附加领域噪音词：与电影讨论常见但不携带偏好信号的词汇
DOMAIN_STOP = {'movie', 'movies', 'film', 'films', 'show', 'shows',
               'watch', 'watched', 'watching', 'watchlist', 'like',
               'liked', 'looking', 'look', 'recommend', 'recommended',
               'recommendation', 'recommendations', 'suggest', 'suggested',
               'suggestion', 'suggestions', 'anyone', 'somebody', 'know',
               'looking', 'searching', 'find', 'found', 'seen', 'seeing',
               'title', 'reddit', 'post', 'sub', 'amp', 'x200b', 'gt',
               'br', 've', 'll', 'don', 'doesn', 'didn', 'won', 'isn',
               'https', 'http', 'www', 'com', 'org', 'edit', 'update',
               'going', 'go', 'wanna', 'gonna', 'gotta', 'tryna', 'yall',
               'yeah', 'yea', 'yep', 'nope', 'nah', 'ok', 'okay', 'oh',
               'ohh', 'ah', 'hmm', 'haha', 'lol', 'lmao', 'lmfao',
               'pretty', 'quite', 'rather', 'supposed', 'suppose',
               'guess', 'wonder', 'wondering', 'curious', 'interested',
               'never', 'ever', 'even', 'lot', 'lots',
               'actually', 'basically', 'honestly', 'literally',
               'seriously', 'definitely', 'absolutely', 'probably',
               'maybe', 'perhaps', 'hopefully', 'unfortunately',
               'thats', 'theres', 'heres', 'heres', 'ive', 'im',
               'theyre', 'youre', 'were', 'dont', 'didnt', 'cant',
               'wont', 'youve', 'theyve', 'couldve', 'wouldve',
               'shouldve', 'mightve', 'mustve', 'couldnt', 'wouldnt',
               'shouldnt', 'mustnt', 'neednt', 'darent', 'mightnt',
               'shant', 'oughtnt', 'daren', 'needn', 'mightn',
               'sis', 'bro', 'dude', 'guys', 'guy', 'woman', 'man',
               'people', 'person', 'thing', 'stuff', 'something',
               'anything', 'everything', 'nothing', 'everyone',
               'anyone', 'someone', 'anybody', 'somebody', 'nobody',
               'everybody', 'also', 'else', 'though', 'although',
               'however', 'therefore', 'thus', 'hence', 'furthermore',
               'meanwhile', 'nevertheless', 'nonetheless',
               'moreover', 'besides', 'indeed', 'instead',
               'regarding', 'concerning', 'including', 'except',
               'without', 'within', 'upon', 'across', 'along', 'among',
               'amongst', 'throughout', 'outside', 'inside', 'around',
               'behind', 'beneath', 'beside', 'beyond', 'via',
                'versus', 'vs', 'per', 'via',
                # ── HTML/URL 分词残留物 ──
                'ntt', 'utm', 'nedit', 'nthanks', 'nthe', 'nany', 'nso',
                'nthank', 'nsome', 'nwhat', 'nalso', 'nif', 'nmovies',
                'nmy', 'nand', 'nfor', 'nit', 'nwe', 'ni', 'ctt', 'cxld',
                'thett', 'andtt', 'mitt', 'mett', 'nm',
                'njan', 'ndas', 'nhit', 'nsee', 'nnothing',
                'nband', 'npiece', 'npart', 'nage', 'nworld',
                # ── 带撇号的缩写（会漏过分词器）──
                "i'm", "i've", "it's", "don't", "can't", "won't",
                "didn't", "doesn't", "isn't", "aren't", "that's",
                "you're", "they're", "there's", "here's", "what's",
                "wasn't", "couldn't", "wouldn't", "shouldn't",
                "haven't", "hasn't", "hadn't",
                # ── 无语义偏好的通用词 ──
                'feel', 'etc', 'letterboxd',
                'web', 'context', 'source', 'medium',
                'recently', 'advance', 'main',
                'example', 'examples', 'comments', 'request',
                }

ALL_STOPWORDS = STOPWORDS | DOMAIN_STOP       # 合并停用词总表


# ── 词频计算 ────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Tokenize text: lowercase, split on non-alpha, remove short words.
       分词：小写化、按非字母字符分割、去除过短单词和停用词。"""
    if not text:
        return []
    text = text.lower()
    tokens = re.split(r'[^a-z\']+', text)     # 按非字母/撇号分割
    return [t.strip("'") for t in tokens
            if len(t.strip("'")) > 2          # 过滤过短词（≤2字符）
            and t.strip("'") not in ALL_STOPWORDS  # 排除停用词
            and not t.strip("'").isnumeric()]       # 排除纯数字


def compute_word_freq(seekers: list[dict], date_set: set = None) -> Counter:
    """Compute word frequency from seekers matching date_set.
       计算指定日期范围内用户提问的高频词。
    Args:
        seekers: 用户提问记录列表
        date_set: 可选，日期集合过滤器
    Returns:
        Counter of word frequencies.
    """
    counter: Counter = Counter()
    for r in seekers:
        if date_set is not None and r['date'] not in date_set:
            continue                            # 按日期过滤
        # 优先使用处理后的文本（proc_text），没有则回退到原始文本
        text = r.get('proc_text', '')
        if not text:
            text = r.get('raw_text', '')
        tokens = tokenize(text)
        counter.update(tokens)                  # 累加词频
    return counter


def compute_word_freq_by_period(
    seekers: list[dict], period: str
) -> Counter:
    """Compute word frequency for a specific period (holiday/workday/weekend).
       计算特定时段（节假日/工作日/周末）的高频词。"""
    # 收集该时段的所有日期
    dates = set(r['date'] for r in seekers if r['period'] == period)
    return compute_word_freq(seekers, dates)


def deduplicate_seekers(seekers: list[dict]) -> list[dict]:
    """Deduplicate seeker records by text content (proc_text then raw_text).
       去除文本内容重复的用户提问记录，避免同一提问被重复计数。

    Keeps the first occurrence of each unique text. Affects all downstream
    word frequency computations.
    只保留每条唯一文本的首条记录，影响所有下游词频统计结果。
    """
    seen = set()
    deduped = []
    for r in seekers:
        text = r.get('proc_text', '')
        if not text:
            text = r.get('raw_text', '')
        key = (text or '').strip().lower()      # 以标准化后的文本作为去重键
        if not key or key in seen:
            continue                            # 跳过重复
        seen.add(key)
        deduped.append(r)
    n_removed = len(seekers) - len(deduped)
    if n_removed > 0:
        log(f"  Deduplication: removed {n_removed} duplicate records "
            f"(left {len(deduped)})")
    return deduped


def compute_word_freq_by_holiday(
    seekers: list[dict], holiday_name: str
) -> Counter:
    """Compute word frequency for a specific holiday name.
       计算特定节假日的高频词。"""
    dates = set(
        r['date'] for r in seekers
        if r['is_holiday'] and r.get('holiday_name', '') == holiday_name
    )
    return compute_word_freq(seekers, dates)


# ── 可视化 ──────────────────────────────────────────────────────────

def plot_wordcloud(word_freq: dict, title: str, filename: str,
                   max_words: int = 200, dpi: int = 150):
    """Generate and save a word cloud image.
       生成并保存词云图片。
    Args:
        word_freq: 词频字典 {词: 频次}
        title: 图表标题
        filename: 输出文件名
        max_words: 最多显示的词数
        dpi: 图片分辨率"""
    if not word_freq:
        log(f"  No words to plot for {filename}")
        return
    wc = WordCloud(
        width=1600, height=900,
        background_color='white',
        max_words=max_words,
        colormap='viridis',
        random_state=42,
        collocations=False,                 # 不检测搭配词
        prefer_horizontal=0.7,              # 70% 水平排列
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=16)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log(f"Saved: {path}")


def plot_top_words_bar(
    freq_dicts: dict[str, dict],
    title: str, filename: str,
    top_n: int = 50,
):
    """Grouped bar chart comparing top word frequencies across groups.
       分组柱状图：比较不同分组（如节假日 vs 非节假日）的高频词频次。"""
    # 收集所有组的全部词汇
    all_words = set()
    for group, freq in freq_dicts.items():
        all_words.update(freq.keys())
    # 按跨组总频次排序取 top
    word_totals = {
        w: sum(freq_dicts[g].get(w, 0) for g in freq_dicts)
        for w in all_words
    }
    top_words = sorted(word_totals, key=word_totals.get, reverse=True)[:top_n]
    if not top_words:
        log(f"  No words to plot for {filename}")
        return

    groups = list(freq_dicts.keys())
    fig, ax = plt.subplots(figsize=(max(12, top_n * 0.5), 6))
    x = np.arange(len(top_words))
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)

    colors = ['#ff6b6b', '#74b9ff', '#feca57', '#48dbfb', '#a29bfe', '#fd79a8']

    for i, group in enumerate(groups):
        vals = [freq_dicts[group].get(w, 0) for w in top_words]
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=group,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(top_words, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Frequency')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def plot_holiday_elevated_words(
    h_avg: dict[str, float],
    nh_avg: dict[str, float],
    threshold: float = 1.5,
    top_n: int = 30,
    filename: str = 'w2_holiday_elevated_words.png',
):
    """Plot words where holiday avg > non-holiday avg * threshold.
       绘制节假日日均频次显著高于非节假日的单词。
    Args:
        h_avg: 节假日日均词频
        nh_avg: 非节假日日均词频
        threshold: 比值阈值（默认 1.5 倍）
        top_n: 展示前 N 个词"""
    candidates = []
    for w, h_val in h_avg.items():
        nh_val = nh_avg.get(w, 0)
        if h_val > nh_val * threshold:                 # 超过阈值才入选
            candidates.append((w, h_val, nh_val))
    # 按节假日词频降序排列
    candidates.sort(key=lambda x: x[1], reverse=True)

    top = candidates[:top_n]
    if not top:
        log(f"  No words pass threshold={threshold} for plot")
        return

    words = [t[0] for t in top]
    h_vals = [t[1] for t in top]
    nh_vals = [t[2] for t in top]

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.45), 6))
    x = np.arange(len(words))
    width = 0.35

    ax.bar(x - width / 2, h_vals, width, label='Holiday (avg daily)',
           color='#ff6b6b', alpha=0.85)
    ax.bar(x + width / 2, nh_vals, width, label='Non-holiday (avg daily)',
           color='#74b9ff', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Avg Daily Frequency')
    ax.set_title(f'Words Where Holiday Avg > Non-Holiday Avg × {threshold} '
                 f'(Top {top_n})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")

    # 在日志中打印 top 10 及其具体比值
    log(f"  Top holiday-elevated words (avg daily, threshold={threshold}):")
    for w, hv, nhv in top[:10]:
        ratio = hv / max(nhv, 0.001)
        log(f"    {w}: holiday={hv:.2f}, non-holiday={nhv:.2f}, ratio={ratio:.1f}x")


def _save_word_csv(
    filename: str,
    freq_dicts: dict[str, Counter],
    total_label: str = None,
):
    """Save word frequencies to CSV with one group per column.
       将词频保存到 CSV 文件，每个分组的词频为一列。"""
    all_words = set()
    for freq in freq_dicts.values():
        all_words.update(freq.keys())

    # 按跨组总频次降序排列
    sorted_words = sorted(
        all_words,
        key=lambda w: sum(freq_dicts[g].get(w, 0) for g in freq_dicts),
        reverse=True,
    )

    groups = sorted(freq_dicts.keys())
    path = os.path.join(STEP_OUT, filename)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        header = ['word'] + [f'{g}_freq' for g in groups]
        for word in sorted_words[:500]:          # 只保存前 500 词
            row = [word]
            for g in groups:
                row.append(freq_dicts[g].get(word, 0))
            w.writerow(row)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  分析维度
# ═══════════════════════════════════════════════════════════════════════

def dim_w1_overall_wordcloud(seekers: list[dict]):
    """Overall word frequency and word cloud.
       全局高频词统计和词云。"""
    log("=" * 50)
    log("W1: Overall Word Frequency & Word Cloud")

    freq = compute_word_freq(seekers)
    top = freq.most_common(30)
    log(f"  Top 10 words: {dict(top[:10])}")

    plot_wordcloud(
        dict(freq),
        'Overall Word Cloud — Movie Discussion (Reddit)',
        'w1_overall_wordcloud.png',
    )
    _save_word_csv('w1_overall_word_freq.csv',
                   {'overall': freq})


def dim_w2_holiday_vs_nonholiday_words(seekers: list[dict], ratio_threshold: float = 1.5):
    """Holiday vs non-holiday word frequency comparison.
       节假日 vs 非节假日词频对比。

    Args:
        seekers: 用户提问记录列表
        ratio_threshold: 节假日高出非节假日的倍数阈值（默认 1.5），用于筛选显著偏高词汇
    """
    log("=" * 50)
    log("W2: Holiday vs Non-Holiday Word Frequency")

    h_freq = compute_word_freq_by_period(seekers, 'holiday')
    # non_holiday = workday + weekend (period 只有 holiday/workday/weekend)
    nh_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_freq = compute_word_freq(seekers, nh_dates)

    # 按日期数归一化到日均词频
    h_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')

    h_avg = {w: c / max(len(h_dates), 1) for w, c in h_freq.items()}
    nh_avg = {w: c / max(len(nh_dates), 1) for w, c in nh_freq.items()}

    plot_top_words_bar(
        {'Holiday': h_avg, 'Non-holiday': nh_avg},
        'Top Words: Holiday vs Non-Holiday (Avg Daily)',
        'w2_holiday_vs_nonholiday_words.png',
    )

    # 绘制节假日显著偏高的词（可配置阈值）
    plot_holiday_elevated_words(h_avg, nh_avg, threshold=ratio_threshold,
                                filename='w2_holiday_elevated_words.png')

    # 输出节日特定高频词（ratio > 2x）
    log("  Top holiday-specific words (ratio > 2x baseline):")
    ratio_words = []
    for w in h_freq:
        ratio = h_avg.get(w, 0) / max(nh_avg.get(w, 0.001), 0.001)
        if ratio > 2.0 and h_freq[w] >= 5:         # 比值 > 2 且节假日频次 >= 5
            ratio_words.append((w, ratio, h_freq[w], nh_freq.get(w, 0)))
    ratio_words.sort(key=lambda x: x[1], reverse=True)
    for w, r, hc, nhc in ratio_words[:20]:
        log(f"    {w}: holiday={hc}, non-holiday={nhc}, ratio={r:.2f}")

    _save_word_csv('w2_holiday_vs_nonholiday_words.csv',
                   {'holiday': h_freq, 'non_holiday': nh_freq})


def dim_w3_holiday_workday_weekend_words(seekers: list[dict]):
    """Holiday vs workday vs weekend word frequency.
       节假日 vs 工作日 vs 周末词频对比。"""
    log("=" * 50)
    log("W3: Holiday vs Workday vs Weekend Word Frequency")

    freq_dict = {}
    for p in ['holiday', 'workday', 'weekend']:
        pf = compute_word_freq_by_period(seekers, p)
        p_dates = set(r['date'] for r in seekers if r['period'] == p)
        freq_dict[p.capitalize()] = {
            w: c / max(len(p_dates), 1) for w, c in pf.items()   # 归一化为日均
        }

    plot_top_words_bar(
        freq_dict,
        'Top Words: Holiday vs Workday vs Weekend (Avg Daily)',
        'w3_holiday_workday_weekend_words.png',
    )

    raw_dict = {}
    for p in ['holiday', 'workday', 'weekend']:
        raw_dict[p] = compute_word_freq_by_period(seekers, p)
    _save_word_csv('w3_holiday_workday_weekend_words.csv', raw_dict)


def dim_w4_per_holiday_words(seekers: list[dict], top_n: int = 30):
    """Per-holiday word frequency vs non-holiday baseline (one bar chart per holiday).
       各个节假日词频 vs 非节假日基线（每个节假日一个柱状图）。

    For each holiday, plots the top N words with highest composite score
    (holiday avg daily freq × min(fold_ratio, 20)), fully sorted by score.
    每个节假日展示综合得分最高的 N 个词，得分为日均频次 × 倍数（最高 20 倍）。
    """
    log("=" * 50)
    log("W4: Per-Holiday Word Frequency vs Non-Holiday")

    nh_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_freq = compute_word_freq(seekers, nh_dates)
    num_nh = max(len(nh_dates), 1)
    nh_avg = {w: c / num_nh for w, c in nh_freq.items()}

    # 按节假日名称分组
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['is_holiday']:
            name = r.get('holiday_name', '')[:8]     # 节假日名称截断到 8 字符
            holiday_groups[name].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items()
                      if len(v) >= MIN_DATA_ROWS}    # 过滤数据量不足的组

    if not holiday_groups:
        log("  No holiday groups with sufficient data")
        return

    holiday_names = sorted(holiday_groups.keys())
    n_holidays = len(holiday_names)

    # 计算各节假日的日均词频和 vs 基线的倍数
    holiday_avg = {}
    holiday_ratio = {}
    all_words = set()
    for hn in holiday_names:
        h_dates = set(r['date'] for r in holiday_groups[hn])
        hf = compute_word_freq(holiday_groups[hn], h_dates)
        h_d = max(len(h_dates), 1)
        ha = {w: c / h_d for w, c in hf.items()}
        holiday_avg[hn] = ha
        holiday_ratio[hn] = {w: ha[w] / max(nh_avg.get(w, 1e-6), 1e-6) for w in ha}
        all_words.update(hf.keys())

    # ── CSV：词 × 节假日矩阵（含倍数列）──
    csv_path = os.path.join(STEP_OUT, 'w4_per_holiday_words.csv')
    # 综合得分：跨节假日取 max(日均词频 × min(倍数, 20))
    word_score = {}
    for w in all_words:
        max_score = 0
        for hn in holiday_names:
            ha = holiday_avg[hn].get(w, 0)
            if ha > 0:
                ratio = min(holiday_ratio[hn].get(w, 1), 20)   # 倍数封顶 20
                max_score = max(max_score, ha * ratio)
        if max_score > 0:
            word_score[w] = max_score
    sorted_words = sorted(word_score, key=word_score.get, reverse=True)[:500]

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        cw = csv.writer(f)
        header = ['word', 'non_holiday_avg_daily']
        for hn in holiday_names:
            header.extend([f'{hn}_avg_daily', f'{hn}_ratio'])
        cw.writerow(header)
        for word in sorted_words:
            row = [word, f'{nh_avg.get(word, 0):.4f}']
            for hn in holiday_names:
                ha = holiday_avg[hn].get(word, 0)
                hr = holiday_ratio[hn].get(word, 0)
                row.extend([f'{ha:.4f}', f'{hr:.2f}'])
            cw.writerow(row)
    log(f"Saved: {csv_path}")

    # ── 各节假日柱状图（每个节假日一个子图）──
    n_cols = min(4, n_holidays)
    n_rows = (n_holidays + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 5))
    axes = axes.flatten() if n_holidays > 1 else [axes]

    for idx, hn in enumerate(holiday_names):
        ax = axes[idx]
        h_avg_dict = holiday_avg[hn]

        # 综合得分排序：日均频次 × min(倍数, 20)
        scored = []
        for w, ha in h_avg_dict.items():
            ratio = min(holiday_ratio[hn].get(w, 1), 20)
            scored.append((w, ha, ratio, ha * ratio))
        scored.sort(key=lambda x: x[3], reverse=True)
        top = scored[:top_n]

        if not top:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        words = [t[0] for t in top[::-1]]        # 反转用于水平柱状图（从大到小）
        h_vals = [t[1] for t in top[::-1]]
        ratios = [t[2] for t in top[::-1]]
        nh_vals = [nh_avg.get(w, 0) for w in words]

        y = np.arange(len(words))
        bar_height = 0.35

        # 节假日日均词频柱（红色）
        bars_h = ax.barh(y + bar_height / 2, h_vals, bar_height,
                         color='#ff6b6b', alpha=0.85, label='Holiday (avg/d)')
        # 非节假日基线柱（蓝色）
        ax.barh(y - bar_height / 2, nh_vals, bar_height,
                color='#74b9ff', alpha=0.85, label='Non-holiday (avg/d)')

        # 在节假日柱上标注倍数（仅 > 1.5 倍时显示）
        for i, (bar, ratio) in enumerate(zip(bars_h, ratios)):
            w = bar.get_width()
            label = f'{ratio:.1f}x' if ratio > 1.5 else ''
            if label:
                ax.text(w + max(h_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                        label, va='center', fontsize=7, color='#c0392b')

        ax.set_yticks(y)
        ax.set_yticklabels(words, fontsize=8)
        ax.set_xlabel('Avg Daily Frequency', fontsize=8)
        ax.set_title(f'{hn}', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', labelsize=7)
        ax.legend(fontsize=7, loc='lower right')

    # 隐藏多余的子图
    for idx in range(n_holidays, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Per-Holiday Top Words vs Non-Holiday Baseline (sorted by score)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    chart_path = os.path.join(STEP_OUT, 'w4_per_holiday_bar_charts.png')
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {chart_path}")

    # ── 日志输出：各节假日 top 词汇 ──
    log("  Per-holiday top elevated words:")
    for hn in holiday_names:
        scored = [(w, holiday_avg[hn][w], holiday_ratio[hn].get(w, 0))
                  for w in holiday_avg[hn]
                  if holiday_avg[hn][w] >= 1]
        scored.sort(key=lambda x: x[1] * min(x[2], 20), reverse=True)
        top = scored[:8]
        if top:
            log(f"    {hn}: {[(w, f'{h:.1f}/d', f'{r:.1f}x') for w, h, r in top]}")


def dim_w5_per_holiday_words_heatmap(seekers: list[dict]):
    """Per-holiday word frequency vs non-holiday baseline (log2 ratio heatmap).
       各节假日单词 log2 倍率热力图（颜色 = log2(节假日日均 / 非节假日日均)）。

    Values:
      0   = same as baseline            # 与基线相同
      +1  = 2x baseline                 # 基线的 2 倍
      +3  = 8x+ baseline                # 基线的 8 倍以上
      -1  = 0.5x baseline               # 基线的 0.5 倍
      -3  = 0.125x baseline             # 基线的 0.125 倍
    """
    log("=" * 50)
    log("W5: Per-Holiday Word Log2-Ratio Heatmap vs Non-Holiday")

    EPSILON = 1e-6  # 防止 log(0)，用于基线上为零的词

    # 计算非节假日日均词频（baseline）
    nh_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_freq = compute_word_freq(seekers, nh_dates)
    num_nh = max(len(nh_dates), 1)
    nh_avg = {w: c / num_nh for w, c in nh_freq.items()}

    # 按节假日名分组
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['is_holiday']:
            name = r.get('holiday_name', '')[:8]
            holiday_groups[name].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items()
                      if len(v) >= MIN_DATA_ROWS}

    if not holiday_groups:
        log("  No holiday groups with sufficient data")
        return

    holiday_names = sorted(holiday_groups.keys())
    n = len(holiday_names)

    # 计算各节假日的日均词频
    holiday_avg = {}
    all_words = set()
    for hn in holiday_names:
        h_dates = set(r['date'] for r in holiday_groups[hn])
        hf = compute_word_freq(holiday_groups[hn], h_dates)
        num_d = max(len(h_dates), 1)
        holiday_avg[hn] = {w: c / num_d for w, c in hf.items()}
        all_words.update(hf.keys())

    # 选择 top 60 词：按跨节假日最大综合得分排序
    word_score = {}
    for w in all_words:
        max_score = 0
        for hn in holiday_names:
            h_val = holiday_avg[hn].get(w, 0)
            log2r = np.log2((h_val + EPSILON) / (nh_avg.get(w, EPSILON) + EPSILON))
            log2r_capped = max(-3, min(3, log2r))   # 截断到 [-3, 3]
            if log2r_capped > 0.5 and h_val >= 1:   # 突出节日相关且频次不低的词
                score = h_val * log2r_capped
                max_score = max(max_score, score)
        if max_score > 0:
            word_score[w] = max_score

    top_words = sorted(word_score, key=word_score.get, reverse=True)[:60]

    if len(top_words) < 3:
        log("  Too few words with elevated holiday frequency")
        return

    # 构建 log2 倍率矩阵，截断到 [-3, 3]
    matrix = np.zeros((len(top_words), n))
    for j, hn in enumerate(holiday_names):
        for i, w in enumerate(top_words):
            h_val = holiday_avg[hn].get(w, EPSILON)
            nh_val = nh_avg.get(w, EPSILON)
            log2r = np.log2((h_val + EPSILON) / (nh_val + EPSILON))
            matrix[i, j] = max(-3, min(3, log2r))

    # 绘制热力图 — 固定对称色阶使跨运行结果可比
    fig_w = max(12, n * 1.2)
    fig_h = max(10, len(top_words) * 0.28 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)

    ax.set_xticks(range(n))
    ax.set_xticklabels(holiday_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(top_words)))
    ax.set_yticklabels(top_words, fontsize=7)
    ax.set_xlabel('Holiday')
    ax.set_ylabel('Word')
    ax.set_title('Per-Holiday Word Frequency — Log2 Ratio vs Non-Holiday Baseline\n'
                 '(0=baseline, +1=2x, +3=8x+, -1=0.5x, -3=0.125x)',
                 fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cbar.set_label('log2(Holiday / Non-Holiday)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'w5_per_holiday_words_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV：输出倍率数据
    csv_path = os.path.join(STEP_OUT, 'w5_per_holiday_words_heatmap.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        cw = csv.writer(f)
        header = ['word', 'non_holiday_avg_daily']
        for hn in holiday_names:
            header.extend([f'{hn}_avg_daily', f'{hn}_log2_ratio', f'{hn}_fold_ratio'])
        cw.writerow(header)
        for word in top_words:
            row = [word, f'{nh_avg.get(word, 0):.4f}']
            for hn in holiday_names:
                h_val = holiday_avg[hn].get(word, 0)
                nh_val = nh_avg.get(word, 0)
                l2r = np.log2((h_val + EPSILON) / (nh_val + EPSILON))
                fold = (h_val + EPSILON) / (nh_val + EPSILON)
                row.extend([f'{h_val:.4f}', f'{l2r:.2f}', f'{fold:.2f}'])
            cw.writerow(row)
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  W6 — 节假日观影画像（关键词分类）
# ═══════════════════════════════════════════════════════════════════════

# 以下字典定义了观影兴趣各维度的关键词映射表
# 每个分类包含一组相关词汇，用于识别用户讨论中涉及的方面

# 类型倾向词汇
_GENRE_WORDS = {
    'Horror':     {'horror', 'scary', 'creepy', 'scared', 'frightening',
                   'terrifying', 'spooky', 'ghost', 'haunted', 'haunting',
                   'paranormal', 'supernatural', 'slasher', 'gore', 'slashers',
                   'scariest', 'horrifying', 'demonic', 'possession', 'zombie',
                   'zombies', 'vampire', 'vampires', 'werewolf'},
    'Comedy':     {'comedy', 'comedies', 'funny', 'humor', 'hilarious',
                   'comic', 'laugh', 'comedic', 'lighthearted', 'sitcom'},
    'Thriller':   {'thriller', 'thrillers', 'suspense', 'suspenseful',
                   'twist', 'twists', 'mystery', 'mysteries', 'intense',
                   'tension', 'edge', 'thrilling'},
    'Action':     {'action', 'adventure', 'explosions', 'superhero',
                   'superheroes', 'battle', 'war'},
    'Sci-Fi':     {'sci', 'science', 'fiction', 'futuristic', 'alien',
                   'aliens', 'space', 'dystopian', 'dystopia', 'time',
                   'travel', 'technology', 'cyberpunk', 'sci-fi'},
    'Drama':      {'drama', 'dramas', 'emotional', 'tearjerker', 'moving',
                   'heartfelt', 'tragic', 'gritty'},
    'Romance':    {'romance', 'romantic', 'romcom', 'date', 'rom-com'},
    'Animation':  {'animated', 'animation', 'cartoon', 'pixar', 'anime'},
    'Fantasy':    {'fantasy', 'magical', 'magic', 'sorcery', 'epic',
                   'mythical', 'mythology'},
    'Crime':      {'crime', 'murder', 'detective', 'noir', 'gangster',
                   'mafia', 'heist', 'investigation'},
    'Documentary':{'documentary', 'documentaries', 'doc'},
    'Musical':    {'musical', 'musicals', 'soundtrack'},
}

# 观影情绪/氛围关键词
_MOOD_WORDS = {
    'Cozy/Family':  {'cozy', 'warm', 'comfort', 'comforting', 'heartwarming',
                     'wholesome', 'festive', 'cheerful', 'merry', 'joy',
                     'joyful', 'happy', 'feel-good'},
    'Dark':         {'dark', 'grim', 'bleak', 'disturbing', 'twisted',
                     'darkness', 'sinister'},
    'Uplifting':    {'uplifting', 'inspiring', 'inspirational', 'hopeful',
                     'optimistic', 'positive'},
    'Relaxing':     {'relaxing', 'calm', 'peaceful', 'chill', 'gentle',
                     'soothing', 'mindless'},
    'Exciting':     {'exciting', 'thrilling', 'intense', 'edge', 'action'},
    'Thoughtful':   {'thought-provoking', 'deep', 'philosophical', 'complex',
                     'profound', 'meaningful'},
    'Nostalgic':    {'nostalgia', 'nostalgic', 'childhood', 'retro',
                     'classic'},
    'Sad':          {'sad', 'depressing', 'depression', 'tragic', 'cry',
                     'sorrow', 'melancholy'},
}

# 观影场景关键词
_CONTEXT_WORDS = {
    'Family/Kids': {'family', 'kids', 'children', 'parents', 'family-friendly',
                    'kid-friendly', 'grandparents', 'parent'},
    'Friends/Social': {'friends', 'friend', 'group', 'party', 'together'},
    'Date Night':  {'date', 'partner', 'girlfriend', 'boyfriend', 'spouse',
                    'husband', 'wife', 'significant'},
    'Binge/Series':{'binge', 'series', 'show', 'shows', 'season', 'episode',
                    'marathon'},
    'Rewatch':     {'rewatch', 'rewatching', 'rewatched', 'revisit'},
    'Alone/Quiet': {'alone', 'solo', 'myself'},
}

# 视频平台关键词
_PLATFORM_WORDS = {
    'Netflix':  {'netflix'},
    'Prime':    {'prime', 'amazon'},
    'HBO':      {'hbo', 'max'},
    'Disney+':  {'disney'},
    'Hulu':     {'hulu'},
    'Apple TV': {'apple'},
    'Streaming':{'streaming', 'stream'},
}

# 影片品质/口碑关键词
_QUALITY_WORDS = {
    'Underrated Gems': {'underrated', 'hidden', 'gem', 'gems', 'underappreciated'},
    'Classic':         {'classic', 'classics', 'timeless', 'masterpiece', 'masterpieces'},
    'Cult/Indie':      {'cult', 'underground', 'obscure', 'indie'},
    'Mainstream':      {'popular', 'mainstream', 'blockbuster', 'hit'},
}

# 叙事/制作方面关键词
_NARRATIVE_WORDS = {
    'Plot/Story':     {'plot', 'story', 'storytelling', 'narrative', 'writing'},
    'Ending':         {'ending', 'ending', 'finale', 'climax', 'conclusion'},
    'Characters':     {'character', 'characters', 'characterization',
                       'protagonist', 'protagonists', 'cast'},
    'Cinematography': {'cinematography', 'visuals', 'visual', 'shot', 'shots',
                       'cinematographic', 'beautiful'},
    'Music/Audio':    {'soundtrack', 'score', 'music', 'sound'},
    'Acting':         {'acting', 'performance', 'performances', 'actor',
                       'actors', 'actress'},
    'Atmosphere':     {'atmosphere', 'vibe', 'vibes', 'mood', 'tone', 'ambiance'},
}


def _score_categories(
    word_freq: dict[str, float],
    cat_map: dict[str, set[str]],
) -> dict[str, float]:
    """Score each category by summing avg daily freq of its matched words.
       计算每个分类的得分：将分类中匹配词汇的日均频次累加求和。"""
    scores = {}
    for cat, keywords in cat_map.items():
        total = 0.0
        for w, f in word_freq.items():
            if w in keywords:
                total += f
        if total > 0:
            scores[cat] = total
    return scores


def dim_w6_holiday_viewing_profile(seekers: list[dict]):
    """Categorize elevated holiday words and summarize viewing differences.
       对每个节假日高频差异词汇进行分类和归纳，总结各个节假日的观影差异。

    Output:
      - w6_holiday_viewing_profile.csv — category scores per holiday
      - Console log with per-holiday viewing profiles
    """
    log("=" * 50)
    log("W6: Holiday Viewing Profile (Keyword Categories)")

    # ── 计算非节假日基线 ──
    nh_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_freq = compute_word_freq(seekers, nh_dates)
    num_nh = max(len(nh_dates), 1)
    nh_avg = {w: c / num_nh for w, c in nh_freq.items()}

    # 按节假日名分组
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['is_holiday']:
            name = r.get('holiday_name', '')[:8]
            holiday_groups[name].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items()
                      if len(v) >= MIN_DATA_ROWS}

    if not holiday_groups:
        log("  No holiday groups with sufficient data")
        return

    holiday_names = sorted(holiday_groups.keys())

    # 计算各节假日日均词频
    holiday_avg = {}
    for hn in holiday_names:
        h_dates = set(r['date'] for r in holiday_groups[hn])
        hf = compute_word_freq(holiday_groups[hn], h_dates)
        h_d = max(len(h_dates), 1)
        holiday_avg[hn] = {w: c / h_d for w, c in hf.items()}

    # ── 为每个节假日评定各分类得分 ──
    # 节假日的分类得分基于"升高词"（节假日日均 > 非节假日日均 × 1.5）
    cat_groups = {
        'Genre':      _GENRE_WORDS,
        'Mood':       _MOOD_WORDS,
        'Context':    _CONTEXT_WORDS,
        'Platform':   _PLATFORM_WORDS,
        'Quality':    _QUALITY_WORDS,
        'Narrative':  _NARRATIVE_WORDS,
    }

    csv_rows = []
    log_lines = []

    for hn in holiday_names:
        hn_short = hn[:8]
        ha = holiday_avg[hn]

        # 筛选升高的词（节假日日均 > 非节假日基线 × 1.5）
        elevated = {}
        for w, h_val in ha.items():
            nh_val = nh_avg.get(w, 0)
            if h_val > nh_val * 1.5 and h_val >= 0.5:
                elevated[w] = h_val
        if not elevated:
            continue

        # 对各分类组评分
        profile_parts = []
        profile_data = {'holiday': hn_short}

        for group_name, cat_map in cat_groups.items():
            scores = _score_categories(elevated, cat_map)
            if scores:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top3 = ranked[:3]
                for rank, (cat, score) in enumerate(top3, 1):
                    profile_data[f'{group_name}_top{rank}'] = cat
                    profile_data[f'{group_name}_top{rank}_score'] = round(score, 1)
                profile_parts.append(f"{group_name}: {', '.join(c for c, _ in top3)}")
            else:
                for rank in range(1, 4):
                    profile_data[f'{group_name}_top{rank}'] = ''
                    profile_data[f'{group_name}_top{rank}_score'] = 0.0

        # 收集匹配的关键词实例（用于提供上下文）
        matched_keywords = []
        for group_name, cat_map in cat_groups.items():
            for cat, keywords in cat_map.items():
                hits = [w for w in elevated if w in keywords]
                if hits:
                    matched_keywords.append(f"{cat}: {', '.join(sorted(hits)[:10])}")
        profile_data['matched_keywords'] = ' | '.join(matched_keywords)

        csv_rows.append(profile_data)

        # 构建可读的摘要输出
        log_lines.append(f"\n  ── {hn_short} ──")
        for pline in profile_parts:
            log_lines.append(f"    {pline}")

    # ── 控制台输出 ──
    for line in log_lines:
        log(line)

    # ── 保存 CSV ──
    csv_path = os.path.join(STEP_OUT, 'w6_holiday_viewing_profile.csv')
    fieldnames = ['holiday']
    for group_name in cat_groups:
        for rank in range(1, 4):
            fieldnames.append(f'{group_name}_top{rank}')
            fieldnames.append(f'{group_name}_top{rank}_score')
    fieldnames.append('matched_keywords')

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(csv_rows)
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None, ratio_threshold: float = 1.5):
    log("=" * 60)
    log("Step 7: High-Frequency Word Analysis & Word Cloud")
    log("=" * 60)

    # 加载数据
    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']
    log(f"Loaded {len(seekers)} seeker records")
    seekers = deduplicate_seekers(seekers)               # 先去重

    dim_w1_overall_wordcloud(seekers)                    # W1: 全局词云
    log("")
    dim_w2_holiday_vs_nonholiday_words(seekers, ratio_threshold=ratio_threshold)  # W2: 节假日对比
    log("")
    dim_w3_holiday_workday_weekend_words(seekers)        # W3: 三分段对比
    log("")
    dim_w4_per_holiday_words(seekers)                    # W4: 各节假日柱状图
    log("")
    dim_w5_per_holiday_words_heatmap(seekers)            # W5: 各节假日热力图
    log("")
    dim_w6_holiday_viewing_profile(seekers)              # W6: 观影画像

    log("")
    log("=" * 60)
    log(f"Step 7 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
