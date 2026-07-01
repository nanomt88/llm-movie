# -*- coding: utf-8 -*-
"""
Step 8: LDA Topic Model & Holiday Preference Mining
步骤 8：LDA 主题模型与节假日偏好挖掘

Analysis:
  - Build LDA topic model on user seekers' text
  - Overall topic distribution
  - Holiday vs non-holiday topic preference comparison
  - Holiday vs workday vs weekend topic comparison
  - Per-holiday topic distribution vs baseline
  - Topic intensity heatmap per holiday

Dependencies: gensim, nltk
Output: output/movie/step8/*.png + CSV
"""

import os           # 文件路径操作
import csv          # CSV 文件读写
import re           # 正则表达式，用于分词
import logging       # 日志控制
from collections import Counter, defaultdict   # 计数器与默认字典

import numpy as np  # 数值计算

# 抑制 gensim 的详细日志输出
logging.getLogger('gensim').setLevel(logging.WARNING)

from gensim import corpora                     # 词袋/字典构建
from gensim.models import LdaModel             # LDA 主题模型
from gensim.models.coherencemodel import CoherenceModel  # 主题一致性评估

import matplotlib
matplotlib.use('Agg')   # 使用非交互式后端（服务器环境）
import matplotlib.pyplot as plt
import seaborn as sns

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log

# ── 初始化 ──────────────────────────────────────────────────────────
setup_matplotlib()                      # 配置 matplotlib 中文字体等
STEP_OUT = STEP_DIRS[8]                 # 输出目录：output/movie/step8/
os.makedirs(STEP_OUT, exist_ok=True)    # 确保输出目录存在

# ── 模型超参数 ─────────────────────────────────────────────────────
NUM_TOPICS = 8          # LDA 主题数量
NUM_TOP_WORDS = 15      # 每个主题展示的前 N 个关键词
PASSES = 10             # LDA 训练迭代次数（越大越收敛）
MIN_WORD_COUNT = 3      # 词在文档中出现的最小次数（过滤低频词）
MIN_WORD_LEN = 3        # 词的最小字符长度
MAX_WORD_FRAC = 0.5     # 词出现在文档中的最大比例（过滤过于通用的词）

# ── 中文类型名 → 英文映射工具 ──────────────────────────────────────
from movie.utils.genre_map import to_en


def tokenize(text: str) -> list[str]:
    """Simple English tokenizer for LDA.
       面向 LDA 的简单英文分词。"""
    if not text:            # 空文本直接返回空列表
        return []
    text = text.lower()     # 统一转小写
    tokens = re.split(r"[^a-z']+", text)  # 按非字母/撇号字符分割
    # 过滤：去除首尾撇号、检查长度、只保留字母词、排除停用词
    return [t.strip("'") for t in tokens
            if len(t.strip("'")) >= MIN_WORD_LEN
            and t.strip("'").isalpha()
            and t.strip("'") not in _STOPWORDS]

# ── 英文停用词表（含电影领域常见高频词）───────────────────────────
_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'i', 'you', 'he', 'she',
    'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'its', 'our', 'their', 'mine', 'yours', 'this', 'that',
    'these', 'those', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
    'for', 'with', 'on', 'in', 'at', 'to', 'from', 'by', 'about', 'into',
    'through', 'during', 'before', 'after', 'of', 'up', 'down', 'out',
    'off', 'over', 'under', 'again', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
    'than', 'too', 'very', 'just', 'because', 'as', 'until', 'while',
    'if', 'else', 'like', 'also', 'any', 'many', 'much', 'who', 'what',
    'which', 'im', 'ive', 'id', 'youre', 'youve', 'dont', 'doesnt',
    'didnt', 'wont', 'wouldnt', 'couldnt', 'shouldnt', 'isnt', 'arent',
    'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'movie', 'movies',
    'film', 'films', 'show', 'shows', 'watch', 'watched', 'watching',
    'like', 'liked', 'looking', 'look', 'recommend', 'recommended',
    'suggest', 'suggested', 'anyone', 'know', 'seen', 'seeing', 'go',
    'going', 'want', 'wants', 'wanted', 'need', 'needs', 'needed',
    'get', 'got', 'gotten', 'make', 'made', 'say', 'says', 'said',
    'think', 'thinks', 'thought', 'tell', 'tells', 'told', 'find',
    'finds', 'found', 'give', 'gives', 'gave', 'given', 'try', 'tries',
    'tried', 'thanks', 'please', 'help', 'really', 'actually', 'well',
    'even', 'still', 'though', 'thing', 'things', 'something', 'lot',
    'lots', 'bit', 'way', 'ways', 'time', 'times', 'day', 'days',
    'year', 'years', 'new', 'old', 'first', 'last', 'next', 'great',
    'best', 'better', 'good', 'bad', 'ever', 'never', 'always', 'also',
    'much', 'many', 'back', 'around', 'away', 'maybe', 'perhaps',
    'probably', 'definitely', 'absolutely', 'basically', 'literally',
    'pretty', 'quite', 'rather', 'guess', 'wonder', 'suppose',
    'yeah', 'yea', 'ok', 'okay', 'oh', 'hmm', 'haha', 'lol',
    'actually', 'honestly', 'seriously', 'hopefully', 'unfortunately',
    'supposed', 'suppose', 'gonna', 'gotta', 'wanna', 'tryna', 'yall',
    'guys', 'guy', 'people', 'person', 'someone', 'anyone', 'everyone',
    'something', 'anything', 'everything', 'nothing',
    'let', 'took', 'take', 'takes', 'taken', 'using', 'use', 'used',
    'based', 'called', 'knows', 'looking', 'goes', 'went', 'seen',
    'watching', 'watches', 'making', 'taking', 'giving', 'saying',
    'telling', 'finding', 'trying', 'coming', 'going', 'getting',
    'putting', 'setting', 'running', 'living', 'playing', 'working',
    'feeling', 'starts', 'started', 'starting', 'ending', 'ended',
    'reddit', 'post', 'sub', 'title', 'amp', 'x200b', 'gt', 'br',
    'https', 'http', 'www', 'com', 'org', 'edit', 'updated',
    'nedit', 'nwhat', 'nthe', 'nso', 'nt', 've', 'll', 're', 'youtu',
    'incorporated', 'haven', 'ampx200b', 'request', 'one', 'two',
    'list', 'end', 'long', 'big', 'top', 'done', 'favorite',
    'netflix', 'prime', 'amazon', 'hulu', 'hbomax', 'disney',
    'series', 'tv', 'show', 'movie', 'movies', 'film', 'films',
}


# ── 模型构建 ────────────────────────────────────────────────────────

def _build_lda_model(documents: list[list[str]]) -> tuple:
    """Build LDA model from tokenized documents.
       从分词后的文档构建 LDA 模型。
    Returns:
        (lda_model, corpus, dictionary, bow_corpus)
    """
    # 创建词典：将分词后的文档映射为词 ID
    dictionary = corpora.Dictionary(documents)
    # 过滤极端词：低频(no_below)和太高频(no_above)的词
    dictionary.filter_extremes(no_below=MIN_WORD_COUNT, no_above=MAX_WORD_FRAC)
    dictionary.compactify()                     # 重新编号词 ID
    log(f"  Dictionary size: {len(dictionary)} tokens")

    # 将每个文档转为词袋向量 (doc2bow: 词ID→(ID, 频次))
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    # 移除空文档（分词后无有效词的文档）
    bow_corpus = [b for b in bow_corpus if b]
    log(f"  Non-empty documents: {len(bow_corpus)}")

    # 训练 LDA 模型
    lda_model = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,      # 主题数
        passes=PASSES,              # 训练轮数
        random_state=42,            # 固定随机种子，保证可复现
    )
    return lda_model, dictionary, bow_corpus


def _get_topic_term_matrix(lda_model, dictionary) -> np.ndarray:
    """Get topic-term matrix: (num_topics, vocab_size).
       获取主题-词项矩阵：(主题数, 词表大小) 每元素为主题 t 下词 w 的概率。"""
    num_terms = len(dictionary)
    matrix = np.zeros((lda_model.num_topics, num_terms))
    for topic_id in range(lda_model.num_topics):
        terms = lda_model.get_topic_terms(topic_id, topn=num_terms)
        for term_id, prob in terms:
            matrix[topic_id, term_id] = prob
    return matrix


def _get_doc_topic_dist(lda_model, bow_corpus) -> np.ndarray:
    """Get document-topic distribution: (num_docs, num_topics).
       获取文档-主题分布矩阵：(文档数, 主题数) 每行为文档在各主题上的概率分布。"""
    num_topics = lda_model.num_topics
    num_docs = len(bow_corpus)
    matrix = np.zeros((num_docs, num_topics))
    for i, bow in enumerate(bow_corpus):
        topics = lda_model.get_document_topics(bow, minimum_probability=0)
        for topic_id, prob in topics:
            matrix[i, topic_id] = prob
    return matrix


def _assign_topic_labels(
    lda_model, dictionary, num_words: int = NUM_TOP_WORDS
) -> dict[int, str]:
    """Assign interpretable labels to topics based on top words.
       基于主题的关键词，为每个主题分配可解释的标签。
    Returns:
        dict[topic_id] -> label string (e.g., "comedy-horror-action")
        标签由该主题的前 3 个关键词用连字符拼接而成
    """
    labels = {}
    for topic_id in range(lda_model.num_topics):
        words = lda_model.show_topic(topic_id, topn=num_words)
        top3 = [w for w, _ in words[:3]]       # 取前 3 个关键词
        labels[topic_id] = '-'.join(top3)
    return labels


# ── 可视化函数 ──────────────────────────────────────────────────────

def _plot_topic_term_heatmap(lda_model, dictionary, filename: str):
    """Heatmap of top words per topic.
       每个主题的关键词热力图：展示每个主题下 top 词的分布概率。"""
    num_words = 15
    topic_words = {}
    for topic_id in range(lda_model.num_topics):
        words = lda_model.show_topic(topic_id, topn=num_words)
        topic_words[topic_id] = [w for w, _ in words]

    # 收集所有关键词（去重），限制最多 30 个
    all_words = []
    for tid in range(lda_model.num_topics):
        for w in topic_words[tid]:
            if w not in all_words:
                all_words.append(w)

    unique_words = []
    for tid in range(lda_model.num_topics):
        for w in topic_words[tid]:
            if w not in unique_words:
                unique_words.append(w)
    unique_words = unique_words[:30]

    # 构建主题-词概率矩阵
    matrix = np.zeros((lda_model.num_topics, len(unique_words)))
    for tid in range(lda_model.num_topics):
        word_probs = dict(lda_model.show_topic(tid, topn=len(dictionary)))
        for j, w in enumerate(unique_words):
            matrix[tid, j] = word_probs.get(w, 0)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(max(14, len(unique_words) * 0.5),
                                     max(5, lda_model.num_topics * 0.6)))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=unique_words,
                yticklabels=[f'T{tid}' for tid in range(lda_model.num_topics)],
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Probability'})
    ax.set_title('Topic-Term Heatmap', fontsize=14)
    ax.set_xlabel('Top Terms')
    ax.set_ylabel('Topic')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_topic_distribution_bar(
    dist_dict: dict[str, np.ndarray],
    title: str, filename: str,
):
    """Bar chart comparing topic distributions across groups.
       分组柱状图：比较不同分组（如节假日 vs 非节假日）的主题分布。
    dist_dict: 分组名 → 文档-主题分布矩阵"""
    groups = list(dist_dict.keys())
    topics = range(dist_dict[groups[0]].shape[1])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(topics))
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)
    colors = ['#ff6b6b', '#74b9ff', '#feca57', '#48dbfb']

    for i, group in enumerate(groups):
        vals = dist_dict[group].mean(axis=0)   # 计算该组各主题的平均概率
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=group,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'T{i}' for i in topics], fontsize=9)
    ax.set_ylabel('Avg Topic Proportion')       # 平均主题占比
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_per_holiday_topic_heatmap(
    topic_by_holiday: dict[str, np.ndarray],
    global_doc_topic: np.ndarray,
    filename: str,
):
    """Heatmap of holiday topic intensity vs global baseline.
       主题强度热力图：各节假日 vs 全局基线（红=高于平均，蓝=低于平均）。"""
    names = sorted(topic_by_holiday.keys())
    if not names:
        return

    global_mean = global_doc_topic.mean(axis=0)           # 全局平均主题分布
    matrix = np.zeros((len(names), global_doc_topic.shape[1]))
    for i, name in enumerate(names):
        matrix[i] = topic_by_holiday[name].mean(axis=0) - global_mean  # 差值

    fig, ax = plt.subplots(figsize=(12, max(4, len(names) * 0.4 + 1)))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                xticklabels=[f'T{i}' for i in range(global_doc_topic.shape[1])],
                yticklabels=names, ax=ax,
                center=0, linewidths=0.5,
                cbar_kws={'label': 'Diff from Global Avg'})
    ax.set_title('Holiday Topic Intensity: Difference from Global Average\n(Red=stronger on holiday, Blue=weaker)',
                 fontsize=11)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Holiday')
    plt.xticks(rotation=0)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  分析维度
# ═══════════════════════════════════════════════════════════════════════

def dim_l1_build_model(seekers: list[dict], texts: list[str]):
    """Build LDA model and show topics.
       构建 LDA 模型并展示主题。"""
    log("=" * 50)
    log("L1: Build LDA Topic Model")

    # 对所有文本进行分词，过滤空文档
    documents = [tokenize(t) for t in texts]
    non_empty = [d for d in documents if d]
    log(f"  Tokenized {len(non_empty)}/{len(documents)} non-empty docs")

    global lda_model, dictionary, bow_corpus
    lda_model, dictionary, bow_corpus = _build_lda_model(non_empty)

    # 打印每个主题的关键词
    log(f"  Topics ({NUM_TOPICS}):")
    labels = _assign_topic_labels(lda_model, dictionary)
    for tid in range(lda_model.num_topics):
        words = lda_model.show_topic(tid, topn=NUM_TOP_WORDS)
        word_str = ' + '.join(f'{p:.2f}*{w}' for w, p in words[:8])
        log(f"    T{tid} ({labels[tid]}): {word_str}")

    # 保存主题词到 CSV
    csv_path = os.path.join(STEP_OUT, 'l1_topic_terms.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['topic_id', 'label'] + [f'top_word_{i+1}' for i in range(NUM_TOP_WORDS)])
        for tid in range(lda_model.num_topics):
            words = lda_model.show_topic(tid, topn=NUM_TOP_WORDS)
            w.writerow([tid, labels[tid]] + [w for w, _ in words])
    log(f"Saved: {csv_path}")

    # 获取文档-主题分布并绘制热力图
    doc_topic = _get_doc_topic_dist(lda_model, bow_corpus)
    _plot_topic_term_heatmap(lda_model, dictionary, 'l1_topic_term_heatmap.png')
    return lda_model, dictionary, bow_corpus, labels, doc_topic


def dim_l2_holiday_vs_nonholiday_topics(doc_topic: np.ndarray, seekers: list[dict]):
    """Holiday vs non-holiday topic distribution comparison.
       节假日 vs 非节假日主题分布对比。"""
    log("=" * 50)
    log("L2: Holiday vs Non-Holiday Topic Distribution")

    # 筛选出有文本内容的记录
    seekers_filtered = [r for r in seekers
                        if r.get('proc_text', '') or r.get('raw_text', '')]

    if len(seekers_filtered) != doc_topic.shape[0]:
        log(f"  Warning: doc_topic shape {doc_topic.shape[0]} != filtered seekers {len(seekers_filtered)}")
        return

    # 按 period 字段分组：holiday vs non-holiday
    h_mask = np.array([r['period'] == 'holiday' for r in seekers_filtered])
    nh_mask = ~h_mask

    if h_mask.sum() == 0 or nh_mask.sum() == 0:
        log("  Not enough data for comparison")
        return

    # 绘制对比柱状图
    _plot_topic_distribution_bar(
        {'Holiday': doc_topic[h_mask], 'Non-holiday': doc_topic[nh_mask]},
        'Topic Distribution: Holiday vs Non-Holiday',
        'l2_holiday_vs_nonholiday_topics.png',
    )

    # 保存统计到 CSV
    csv_path = os.path.join(STEP_OUT, 'l2_holiday_vs_nonholiday_topics.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['period', 'topic_id', 'avg_proportion', 'std_proportion'])
        for period, mask in [('holiday', h_mask), ('non_holiday', nh_mask)]:
            proportions = doc_topic[mask]
            for tid in range(proportions.shape[1]):
                avg = proportions[:, tid].mean()
                std = proportions[:, tid].std()
                w.writerow([period, tid, f'{avg:.4f}', f'{std:.4f}'])
    log(f"Saved: {csv_path}")


def dim_l3_holiday_workday_weekend_topics(doc_topic: np.ndarray, seekers: list[dict]):
    """Holiday vs workday vs weekend topic distribution.
       节假日 vs 工作日 vs 周末主题分布。"""
    log("=" * 50)
    log("L3: Holiday vs Workday vs Weekend Topic Distribution")

    # 筛选有文本内容的记录
    seekers_filtered = [r for r in seekers
                        if r.get('proc_text', '') or r.get('raw_text', '')]
    if len(seekers_filtered) != doc_topic.shape[0]:
        log("  Mismatch between doc_topic and seekers, skipping")
        return

    # 按三类时期分组
    topic_by_period = {}
    for p in ['holiday', 'workday', 'weekend']:
        mask = np.array([r['period'] == p for r in seekers_filtered])
        if mask.sum() > 0:
            topic_by_period[p.capitalize()] = doc_topic[mask]
            log(f"  {p}: {mask.sum()} docs")

    if len(topic_by_period) < 2:
        log("  Not enough period data")
        return

    _plot_topic_distribution_bar(
        topic_by_period,
        'Topic Distribution: Holiday vs Workday vs Weekend',
        'l3_holiday_workday_weekend_topics.png',
    )

    # 保存 CSV
    csv_path = os.path.join(STEP_OUT, 'l3_holiday_workday_weekend_topics.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['period', 'topic_id', 'avg_proportion', 'std_proportion'])
        for period, data in topic_by_period.items():
            for tid in range(data.shape[1]):
                w.writerow([period, tid, f'{data[:, tid].mean():.4f}',
                            f'{data[:, tid].std():.4f}'])
    log(f"Saved: {csv_path}")


def dim_l4_per_holiday_topics(doc_topic: np.ndarray, seekers: list[dict]):
    """Per-holiday topic distribution vs non-holiday baseline.
       各节假日主题分布 vs 非节假日基线热力图。"""
    log("=" * 50)
    log("L4: Per-Holiday Topic Intensity Heatmap")

    seekers_filtered = [r for r in seekers
                        if r.get('proc_text', '') or r.get('raw_text', '')]
    if len(seekers_filtered) != doc_topic.shape[0]:
        log("  Mismatch, skipping")
        return

    # 按节假日名称分组
    holiday_groups = defaultdict(list)
    holiday_indices = defaultdict(list)
    for i, r in enumerate(seekers_filtered):
        if r['is_holiday']:
            name = r.get('holiday_name', '')[:8]    # 截取前 8 字符
            holiday_groups[name].append(r)
            holiday_indices[name].append(i)

    # 过滤数据量不足的节假日组
    holiday_groups = {k: v for k, v in holiday_groups.items()
                      if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  No holiday groups with sufficient data")
        return

    # 收集各节假日的主题分布
    topic_by_holiday = {}
    for name, indices in holiday_indices.items():
        if name in holiday_groups:
            topic_by_holiday[name] = doc_topic[indices]

    _plot_per_holiday_topic_heatmap(
        topic_by_holiday, doc_topic,
        'l4_per_holiday_topic_heatmap.png',
    )

    # 保存 CSV（含全局基线行）
    csv_path = os.path.join(STEP_OUT, 'l4_per_holiday_topic_dist.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        n_topics = doc_topic.shape[1]
        w.writerow(['holiday_name', 'num_docs'] + [f'T{tid}_avg' for tid in range(n_topics)])
        for name in sorted(topic_by_holiday.keys()):
            data = topic_by_holiday[name]
            row = [name, data.shape[0]]
            for tid in range(n_topics):
                row.append(f'{data[:, tid].mean():.4f}')
            w.writerow(row)
        # 全局基线行
        row = ['global_baseline', doc_topic.shape[0]]
        for tid in range(n_topics):
            row.append(f'{doc_topic[:, tid].mean():.4f}')
        w.writerow(row)
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None):
    log("=" * 60)
    log("Step 8: LDA Topic Model & Holiday Preference Mining")
    log("=" * 60)

    # 加载数据
    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']
    log(f"Loaded {len(seekers)} seeker records")

    # 准备文本：优先使用处理后的文本，fallback 到原始文本
    texts = [r.get('proc_text', '') or r.get('raw_text', '') for r in seekers]

    # L1: 构建模型
    lda_model, dictionary, bow_corpus, labels, doc_topic = dim_l1_build_model(
        seekers, texts)
    log("")

    if doc_topic is None or doc_topic.shape[0] == 0:
        log("  ERROR: LDA model produced no valid topic distribution")
        return

    # L2: 节假日 vs 非节假日对比
    dim_l2_holiday_vs_nonholiday_topics(doc_topic, seekers)
    log("")

    # L3: 节假日 vs 工作日 vs 周末对比
    dim_l3_holiday_workday_weekend_topics(doc_topic, seekers)
    log("")

    # L4: 各节假日热力图
    dim_l4_per_holiday_topics(doc_topic, seekers)

    log("")
    log("=" * 60)
    log(f"Step 8 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
