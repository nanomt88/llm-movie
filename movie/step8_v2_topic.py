# -*- coding: utf-8 -*-
"""
Step 8 v2: LDA Topic Mining (Overview + User-Text Dual Track)
步骤 8 v2：LDA 主题挖掘（电影剧情简介 + 用户提问文本 双轨）

修正原 step8_lda.py 的核心问题：
  原 step8 直接对 proc_text 做 LDA，但 proc_text 中电影名已被替换为 tt ID
  （AGENTS.md 数据规则 4），导致主题词被 tt 残片和通用对话词主导，
  无法反映电影题材。实际输出 l1_topic_terms.csv 中 T1 标签为
  "ntt-trilogy-underrated"，第一个主题词 "ntt" 即 tt ID 切分残留。

本版采用两条互补的 LDA 轨道（AGENTS.md 规则 8）：

  Track A — overview（电影题材主题挖掘）
    conv_id → 同 conv_id 的系统回复 → 提取 tt ID → movie_info.json.overview
    → 按 session_id 聚合每个会话所有推荐电影的 overview → jieba 分词 → 中文 LDA
    主题词会是"战争/士兵/战场"、"太空/飞船/外星"、"黑帮/家族/权力"等
    有语义的电影题材簇。

  Track B — usertext（用户讨论话题挖掘）
    按 session_id 聚合用户 raw_text（保留原始电影名，无 tt ID）
    → 清洗 tt ID 残留（保险）→ 借用 step7 调好的 DOMAIN_STOP 做英文 LDA
    主题词会是"恐怖片求推荐"、"老电影"、"家庭观影"等用户需求簇。

Output: output/movie/step8_v2/*.png + CSV
Dependencies: gensim, jieba, nltk, matplotlib, seaborn
"""

import os           # 文件路径操作
import csv          # CSV 文件读写
import re           # 正则表达式（清洗 tt ID 残留）
import logging      # 日志控制
from collections import defaultdict  # 默认字典

import numpy as np  # 数值计算

# 抑制 gensim 的详细日志输出
logging.getLogger('gensim').setLevel(logging.WARNING)

from gensim import corpora                     # 词袋/字典构建
from gensim.models import LdaModel             # LDA 主题模型

import matplotlib
matplotlib.use('Agg')   # 使用非交互式后端（服务器环境）
import matplotlib.pyplot as plt
import seaborn as sns

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log
from movie.utils.text import (
    tokenize as _tokenize_en,            # 英文分词（带 nltk lemmatize）
    tokenize_cn as _tokenize_cn,          # 中文分词（jieba）
    build_conv_system,                    # 规则8：构建 conv_id → 系统回复 映射
    get_system_movie_ids,                  # 规则8：从系统回复提取 tt ID 集合
)

# ── 初始化 ──────────────────────────────────────────────────────────
setup_matplotlib()                      # 配置 matplotlib 中文字体等
# step8_v2 用独立输出目录，避免覆盖原 step8 的输出
# STEP_DIRS[8] = .../output/movie/step8，dirname = .../output/movie
STEP_OUT = os.path.join(os.path.dirname(STEP_DIRS[8]), 'step8_v2')
os.makedirs(STEP_OUT, exist_ok=True)

# ── 模型超参数 ─────────────────────────────────────────────────────
NUM_TOPICS_OVERVIEW = 8     # overview 中文 LDA 主题数
NUM_TOPICS_USERTEXT = 8     # 用户文本英文 LDA 主题数
NUM_TOP_WORDS = 15          # 每个主题展示的前 N 个关键词
PASSES = 10                 # LDA 训练迭代次数（越大越收敛）
MIN_WORD_LEN_EN = 3         # 英文词的最小字符长度
MIN_WORD_LEN_CN = 2         # 中文词的最小字符长度

# 英文 Track B 的 filter_extremes 参数
MIN_WORD_COUNT_EN = 3       # 英文：词在文档中出现的最小次数
MAX_WORD_FRAC_EN = 0.5      # 英文：词出现在文档中的最大比例

# 中文 Track A 的 filter_extremes 参数（更激进，避免词典膨胀导致概率稀释）
# no_above=0.1 过滤出现在 >10% 文档中的通用词（如"发现""开始"等无题材信号的词）
# no_below=10  过滤低频噪声词（56K 文档中 <10 次出现）
MIN_WORD_COUNT_CN = 10      # 中文：词在文档中出现的最小次数
MAX_WORD_FRAC_CN = 0.1      # 中文：词出现在文档中的最大比例

# 借用 step7_wordcloud 调好的英文领域停用词表
# （含论坛噪声/编码残留/无类型信号词/通用动词填充等）
try:
    from movie.step7_wordcloud import ALL_STOPWORDS as _EN_STOPWORDS
except ImportError:
    _EN_STOPWORDS = None    # tokenize 在 stopwords=None 时不做停用词过滤

# tt ID 残留清洗正则（raw_text 一般不应有 tt ID，但保险起见）
_TT_PATTERN = re.compile(r'tt\d{7,9}')


# ── 文档构建 ────────────────────────────────────────────────────────

def _build_session_overview_docs(seekers: list[dict], rows: list[dict],
                                 movie_info: dict):
    """Build per-session overview documents (Track A).
       按 session 聚合每个会话推荐电影的 overview 拼接文档（轨道 A）。

    流程（遵循 AGENTS.md 规则 8）：
      1. build_conv_system(rows) 构建 conv_id → 系统回复文本列表 映射
      2. 对每条 seeker，用其 conv_id 查系统回复
      3. 从系统回复提取 tt ID 集合
      4. 对每个 tt ID 查 movie_info.json 的 overview 字段
      5. 按 session_id 聚合：同一会话所有 seeker 的所有推荐电影 overview 拼接

    Returns:
        (documents, metas)
        documents: list[str] —— 每个会话的 overview 拼接文本
        metas:     list[dict] —— 每个文档对应的会话首条 seeker 元信息
                   （用于后续按时段/holiday 分组）
    """
    conv_system = build_conv_system(rows)
    log(f"  Built conv_system: {len(conv_system)} turn-level entries")

    session_overviews = defaultdict(list)   # session_id -> [overview strings]
    session_meta = {}                       # session_id -> 首条 seeker 元信息

    for r in seekers:
        sid = r['session_id']
        if sid not in session_meta:
            session_meta[sid] = r
        # 规则8：从同 conv_id 系统回复提取 tt ID
        tt_ids = get_system_movie_ids(r.get('conv_id', ''), conv_system)
        for tt in tt_ids:
            info = movie_info.get(tt, {})
            if isinstance(info, dict):
                ov = info.get('overview', '')
                if ov:
                    session_overviews[sid].append(ov)

    # 拼接每个 session 的 overview 为一个文档字符串
    documents = []
    metas = []
    for sid, ovs in session_overviews.items():
        if not ovs:
            continue
        text = '。'.join(ovs)   # 中文句号分隔，便于 jieba 断句
        documents.append(text)
        metas.append(session_meta[sid])

    log(f"  Built {len(documents)} session-level overview documents")
    # 统计覆盖的电影数
    unique_tts = set()
    for r in seekers:
        unique_tts.update(get_system_movie_ids(r.get('conv_id', ''), conv_system))
    log(f"  Covered {len(unique_tts)} unique recommended movies with overview")
    return documents, metas


def _build_session_usertext_docs(seekers: list[dict]):
    """Build per-session user-text documents (Track B).
       按 session 聚合用户 raw_text（保留原始电影名，无 tt ID）（轨道 B）。

    Returns:
        (documents, metas)
        documents: list[str] —— 每个会话所有 seeker raw_text 的拼接
        metas:     list[dict] —— 每个文档对应的会话首条 seeker 元信息
    """
    session_texts = defaultdict(list)
    session_meta = {}
    for r in seekers:
        sid = r['session_id']
        if sid not in session_meta:
            session_meta[sid] = r
        # 优先 raw_text（保留原始电影名如 "Inception"，而非 proc_text 中的 tt ID）
        text = r.get('raw_text', '') or r.get('proc_text', '')
        if text:
            session_texts[sid].append(text)

    documents = []
    metas = []
    for sid, texts in session_texts.items():
        joined = ' '.join(texts)
        # 清洗 tt ID 残留（raw_text 一般不应有，但保险起见）
        joined = _TT_PATTERN.sub(' ', joined)
        if joined.strip():
            documents.append(joined)
            metas.append(session_meta[sid])

    log(f"  Built {len(documents)} session-level user-text documents")
    return documents, metas


# ── LDA 通用工具 ───────────────────────────────────────────────────

def _build_lda_model(documents: list[list[str]], num_topics: int,
                     no_below: int = 3, no_above: float = 0.5):
    """Build LDA model from tokenized documents.
       从分词后的文档构建 LDA 模型。

    注意：本函数信任输入 documents 已过滤空文档（调用方负责），
    以保证返回的 bow_corpus 长度 == len(documents)，
    从而 doc_topic 行数与外部 metas 对齐（用于 L2~L4 分组对比）。
    filter_extremes 后可能仍有文档变空（所有词被 no_below/no_above 过滤），
    LDA 训练时空 BoW 文档不贡献信息但也不报错，doc_topic 该行全 0。

    Args:
        no_below: 词在至少 no_below 个文档中出现才保留（过滤低频词）
        no_above: 词出现在 > no_above 比例文档中则过滤（过滤通用词）
    Returns:
        (lda_model, dictionary, bow_corpus)
    """
    # 创建词典
    dictionary = corpora.Dictionary(documents)
    # 过滤极端词：低频(no_below)和太高频(no_above)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    dictionary.compactify()
    log(f"  Dictionary size: {len(dictionary)} tokens (no_below={no_below}, no_above={no_above})")

    # 词袋向量（保持长度 == len(documents)，与外部 metas 对齐）
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    non_empty_count = sum(1 for b in bow_corpus if b)
    log(f"  Documents: {len(bow_corpus)} total, {non_empty_count} non-empty")

    lda_model = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=PASSES,
        random_state=42,            # 固定随机种子，保证可复现
    )
    return lda_model, dictionary, bow_corpus


def _get_doc_topic_dist(lda_model, bow_corpus) -> np.ndarray:
    """Get document-topic distribution: (num_docs, num_topics).
       获取文档-主题分布矩阵。"""
    num_topics = lda_model.num_topics
    num_docs = len(bow_corpus)
    matrix = np.zeros((num_docs, num_topics))
    for i, bow in enumerate(bow_corpus):
        topics = lda_model.get_document_topics(bow, minimum_probability=0)
        for tid, prob in topics:
            matrix[i, tid] = prob
    return matrix


def _assign_topic_labels(lda_model, num_words: int = NUM_TOP_WORDS) -> dict:
    """Assign interpretable labels to topics based on top words.
       基于主题的关键词，为每个主题分配可解释的标签。"""
    labels = {}
    for tid in range(lda_model.num_topics):
        words = lda_model.show_topic(tid, topn=num_words)
        top3 = [w for w, _ in words[:3]]
        labels[tid] = '-'.join(top3)
    return labels


# ── 可视化（与 step8_lda 同构，独立维护避免耦合）────────────────────

def _plot_topic_term_heatmap(lda_model, dictionary, filename: str, title: str):
    """Heatmap of top words per topic.
       每个主题的关键词热力图。"""
    num_words = 15
    # 收集每个主题的 top 词
    topic_words = {}
    for tid in range(lda_model.num_topics):
        words = lda_model.show_topic(tid, topn=num_words)
        topic_words[tid] = [w for w, _ in words]

    # 收集所有 top 词（按出现顺序去重），限制最多 30 个
    all_words = []
    for tid in range(lda_model.num_topics):
        for w in topic_words[tid]:
            if w not in all_words:
                all_words.append(w)
    all_words = all_words[:30]

    # 构建主题-词概率矩阵
    matrix = np.zeros((lda_model.num_topics, len(all_words)))
    for tid in range(lda_model.num_topics):
        word_probs = dict(lda_model.show_topic(tid, topn=len(dictionary)))
        for j, w in enumerate(all_words):
            matrix[tid, j] = word_probs.get(w, 0)

    fig, ax = plt.subplots(figsize=(max(14, len(all_words) * 0.5),
                                     max(5, lda_model.num_topics * 0.6)))
    sns.heatmap(matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=all_words,
                yticklabels=[f'T{tid}' for tid in range(lda_model.num_topics)],
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Probability'})
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Top Terms')
    ax.set_ylabel('Topic')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_topic_distribution_bar(dist_dict: dict, title: str, filename: str):
    """Bar chart comparing topic distributions across groups.
       分组柱状图：比较不同分组的主题分布。"""
    groups = list(dist_dict.keys())
    if not groups:
        return
    topics = range(dist_dict[groups[0]].shape[1])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(topics))
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)
    colors = ['#ff6b6b', '#74b9ff', '#feca57', '#48dbfb']

    for i, group in enumerate(groups):
        vals = dist_dict[group].mean(axis=0)
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=group,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'T{i}' for i in topics], fontsize=9)
    ax.set_ylabel('Avg Topic Proportion')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_per_holiday_topic_heatmap(topic_by_holiday: dict,
                                    global_doc_topic: np.ndarray,
                                    filename: str, title: str):
    """Heatmap of holiday topic intensity vs global baseline.
       主题强度热力图：各节假日 vs 全局基线。"""
    names = sorted(topic_by_holiday.keys())
    if not names:
        return

    global_mean = global_doc_topic.mean(axis=0)
    matrix = np.zeros((len(names), global_doc_topic.shape[1]))
    for i, name in enumerate(names):
        matrix[i] = topic_by_holiday[name].mean(axis=0) - global_mean

    fig, ax = plt.subplots(figsize=(12, max(4, len(names) * 0.4 + 1)))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                xticklabels=[f'T{i}' for i in range(global_doc_topic.shape[1])],
                yticklabels=names, ax=ax,
                center=0, linewidths=0.5,
                cbar_kws={'label': 'Diff from Global Avg'})
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Holiday')
    plt.xticks(rotation=0)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  L1-A: 轨道 A — 电影 overview 中文 LDA（电影题材主题挖掘）
#  L1-A: Track A — Movie Overview Chinese LDA (movie subject-matter themes)
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】主题-词热力图 + 主题词列表
# 【统计口径】规则8：conv_id→系统回复→tt ID→movie_info.overview→jieba 分词→LDA
#   文档单位：按 session_id 聚合，一个会话所有推荐电影 overview 拼接为一个文档
# 【输出文件】PNG: l1a_overview_topic_term_heatmap.png, CSV: l1a_overview_topic_terms.csv
# 【特殊说明】overview 为中文字段，需用 tokenize_cn（jieba 分词）
# ═══════════════════════════════════════════════════════════════════════

def dim_l1a_build_overview_model(seekers: list[dict], rows: list[dict],
                                 movie_info: dict):
    """Build LDA on movie overview (Track A).
        构建 overview 中文 LDA 模型（轨道 A）。"""
    log("=" * 50)
    log("L1-A: Build LDA on Movie Overview (Chinese, jieba)")

    docs_text, metas = _build_session_overview_docs(seekers, rows, movie_info)
    if len(docs_text) < 10:
        log("  Not enough overview docs, skipping track A")
        return None, None, None, None, None, None

    # jieba 分词 + 同步过滤空文档与 metas（保证 doc_topic 行数与 metas 对齐）
    tokenized = [_tokenize_cn(t) for t in docs_text]
    non_empty_docs = []
    non_empty_metas = []
    for doc, meta in zip(tokenized, metas):
        if doc:
            non_empty_docs.append(doc)
            non_empty_metas.append(meta)
    log(f"  Tokenized {len(non_empty_docs)}/{len(tokenized)} non-empty docs")

    if len(non_empty_docs) < MIN_DATA_ROWS:
        log("  Too few non-empty docs for LDA, skipping track A")
        return None, None, None, None, None, None

    lda_model, dictionary, bow_corpus = _build_lda_model(
        non_empty_docs, NUM_TOPICS_OVERVIEW,
        no_below=MIN_WORD_COUNT_CN, no_above=MAX_WORD_FRAC_CN)

    # 打印每个主题的关键词（中文 LDA 概率值较小，用 4 位小数显示）
    log(f"  Topics ({NUM_TOPICS_OVERVIEW}):")
    labels = _assign_topic_labels(lda_model)
    for tid in range(lda_model.num_topics):
        words = lda_model.show_topic(tid, topn=NUM_TOP_WORDS)
        word_str = ' + '.join(f'{p:.4f}*{w}' for w, p in words[:8])
        log(f"    T{tid} ({labels[tid]}): {word_str}")

    # 保存主题词到 CSV
    csv_path = os.path.join(STEP_OUT, 'l1a_overview_topic_terms.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['topic_id', 'label'] +
                   [f'top_word_{i+1}' for i in range(NUM_TOP_WORDS)])
        for tid in range(lda_model.num_topics):
            words = lda_model.show_topic(tid, topn=NUM_TOP_WORDS)
            w.writerow([tid, labels[tid]] + [word for word, _ in words])
    log(f"Saved: {csv_path}")

    # 文档-主题分布 + 热力图
    doc_topic = _get_doc_topic_dist(lda_model, bow_corpus)
    _plot_topic_term_heatmap(
        lda_model, dictionary,
        'l1a_overview_topic_term_heatmap.png',
        'L1-A: Topic-Term Heatmap (Movie Overview, Chinese LDA)')

    return lda_model, dictionary, bow_corpus, labels, doc_topic, non_empty_metas


# ═══════════════════════════════════════════════════════════════════════
#  L1-B: 轨道 B — 用户 raw_text 英文 LDA（用户讨论话题挖掘）
#  L1-B: Track B — User raw_text English LDA (user discussion topics)
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】主题-词热力图 + 主题词列表
# 【统计口径】按 session_id 聚合 raw_text（保留原始电影名，无 tt ID）
#   清洗 tt ID 残留（保险）+ 借用 step7 DOMAIN_STOP 做英文停用词
# 【输出文件】PNG: l1b_usertext_topic_term_heatmap.png, CSV: l1b_usertext_topic_terms.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_l1b_build_usertext_model(seekers: list[dict]):
    """Build LDA on user raw_text (Track B).
        构建 raw_text 英文 LDA 模型（轨道 B）。"""
    log("=" * 50)
    log("L1-B: Build LDA on User Text (English, by session)")

    docs_text, metas = _build_session_usertext_docs(seekers)
    if len(docs_text) < 10:
        log("  Not enough usertext docs, skipping track B")
        return None, None, None, None, None, None

    # 英文分词 + step7 调好的领域停用词 + 同步过滤空文档与 metas
    tokenized = [_tokenize_en(t, min_len=MIN_WORD_LEN_EN,
                              stopwords=_EN_STOPWORDS) for t in docs_text]
    non_empty_docs = []
    non_empty_metas = []
    for doc, meta in zip(tokenized, metas):
        if doc:
            non_empty_docs.append(doc)
            non_empty_metas.append(meta)
    log(f"  Tokenized {len(non_empty_docs)}/{len(tokenized)} non-empty docs")

    if len(non_empty_docs) < MIN_DATA_ROWS:
        log("  Too few non-empty docs for LDA, skipping track B")
        return None, None, None, None, None, None

    lda_model, dictionary, bow_corpus = _build_lda_model(
        non_empty_docs, NUM_TOPICS_USERTEXT,
        no_below=MIN_WORD_COUNT_EN, no_above=MAX_WORD_FRAC_EN)

    log(f"  Topics ({NUM_TOPICS_USERTEXT}):")
    labels = _assign_topic_labels(lda_model)
    for tid in range(lda_model.num_topics):
        words = lda_model.show_topic(tid, topn=NUM_TOP_WORDS)
        word_str = ' + '.join(f'{p:.4f}*{w}' for w, p in words[:8])
        log(f"    T{tid} ({labels[tid]}): {word_str}")

    csv_path = os.path.join(STEP_OUT, 'l1b_usertext_topic_terms.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['topic_id', 'label'] +
                   [f'top_word_{i+1}' for i in range(NUM_TOP_WORDS)])
        for tid in range(lda_model.num_topics):
            words = lda_model.show_topic(tid, topn=NUM_TOP_WORDS)
            w.writerow([tid, labels[tid]] + [word for word, _ in words])
    log(f"Saved: {csv_path}")

    doc_topic = _get_doc_topic_dist(lda_model, bow_corpus)
    _plot_topic_term_heatmap(
        lda_model, dictionary,
        'l1b_usertext_topic_term_heatmap.png',
        'L1-B: Topic-Term Heatmap (User Text, English LDA)')

    return lda_model, dictionary, bow_corpus, labels, doc_topic, non_empty_metas


# ═══════════════════════════════════════════════════════════════════════
#  L2: 节假日 VS 非节假日 主题分布对比 (Bar)
#  L2: Holiday vs Non-Holiday Topic Distribution
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】分组柱状图：每个主题一个对比柱(holiday/non_holiday)
# 【统计口径】按 session 元信息的 period 字段分组，组内对各主题概率取均值
# 【输出文件】PNG: l2_<track>_holiday_vs_nonholiday.png, CSV: l2_<track>_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_l2_holiday_vs_nonholiday(doc_topic: np.ndarray, metas: list[dict],
                                 track_name: str):
    """Holiday vs non-holiday topic distribution.
        节假日 vs 非节假日主题分布对比。"""
    log("=" * 50)
    log(f"L2 ({track_name}): Holiday vs Non-Holiday")

    if doc_topic is None or len(metas) != doc_topic.shape[0]:
        log("  Mismatch or empty, skipping")
        return

    h_mask = np.array([r['period'] == 'holiday' for r in metas])
    nh_mask = ~h_mask
    log(f"  Holiday: {h_mask.sum()} docs, Non-holiday: {nh_mask.sum()} docs")

    if h_mask.sum() == 0 or nh_mask.sum() == 0:
        log("  Not enough data for comparison")
        return

    _plot_topic_distribution_bar(
        {'Holiday': doc_topic[h_mask], 'Non-holiday': doc_topic[nh_mask]},
        f'Topic Distribution: Holiday vs Non-Holiday ({track_name})',
        f'l2_{track_name}_holiday_vs_nonholiday.png')

    # 保存 CSV
    csv_path = os.path.join(STEP_OUT,
                            f'l2_{track_name}_holiday_vs_nonholiday.csv')
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


# ═══════════════════════════════════════════════════════════════════════
#  L3: 节假日 VS 工作日 VS 周末 主题分布
#  L3: Holiday vs Workday vs Weekend Topics
# ═══════════════════════════════════════════════════════════════════════

def dim_l3_holiday_workday_weekend(doc_topic: np.ndarray, metas: list[dict],
                                   track_name: str):
    """Holiday vs workday vs weekend topic distribution.
        节假日 vs 工作日 vs 周末主题分布。"""
    log("=" * 50)
    log(f"L3 ({track_name}): Holiday vs Workday vs Weekend")

    if doc_topic is None or len(metas) != doc_topic.shape[0]:
        log("  Mismatch, skipping")
        return

    topic_by_period = {}
    for p in ['holiday', 'workday', 'weekend']:
        mask = np.array([r['period'] == p for r in metas])
        if mask.sum() > 0:
            topic_by_period[p.capitalize()] = doc_topic[mask]
            log(f"  {p}: {mask.sum()} docs")

    if len(topic_by_period) < 2:
        log("  Not enough period data")
        return

    _plot_topic_distribution_bar(
        topic_by_period,
        f'Topic Distribution: Holiday vs Workday vs Weekend ({track_name})',
        f'l3_{track_name}_holiday_workday_weekend.png')

    csv_path = os.path.join(STEP_OUT,
                            f'l3_{track_name}_holiday_workday_weekend.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['period', 'topic_id', 'avg_proportion', 'std_proportion'])
        for period, data in topic_by_period.items():
            for tid in range(data.shape[1]):
                w.writerow([period, tid, f'{data[:, tid].mean():.4f}',
                            f'{data[:, tid].std():.4f}'])
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  L4: 各节假日主题分布热力图
#  L4: Per-Holiday Topic Intensity Heatmap
# ═══════════════════════════════════════════════════════════════════════

def dim_l4_per_holiday_topics(doc_topic: np.ndarray, metas: list[dict],
                              track_name: str):
    """Per-holiday topic distribution vs global baseline.
        各节假日主题分布 vs 全局基线热力图。"""
    log("=" * 50)
    log(f"L4 ({track_name}): Per-Holiday Topic Intensity Heatmap")

    if doc_topic is None or len(metas) != doc_topic.shape[0]:
        log("  Mismatch, skipping")
        return

    # 按节假日名称分组
    holiday_indices = defaultdict(list)
    for i, r in enumerate(metas):
        if r.get('is_holiday', False):
            name = r.get('holiday_name', '')[:8]    # 截取前 8 字符
            holiday_indices[name].append(i)

    # 过滤数据量不足的节假日组
    holiday_indices = {k: v for k, v in holiday_indices.items()
                       if len(v) >= MIN_DATA_ROWS}
    if not holiday_indices:
        log("  No holiday groups with sufficient data")
        return

    topic_by_holiday = {name: doc_topic[indices]
                        for name, indices in holiday_indices.items()}

    _plot_per_holiday_topic_heatmap(
        topic_by_holiday, doc_topic,
        f'l4_{track_name}_per_holiday_topic_heatmap.png',
        f'L4 ({track_name}): Holiday Topic Intensity vs Global Avg')

    # 保存 CSV（含全局基线行）
    csv_path = os.path.join(STEP_OUT,
                            f'l4_{track_name}_per_holiday_topic_dist.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        n_topics = doc_topic.shape[1]
        w.writerow(['holiday_name', 'num_docs'] +
                   [f'T{tid}_avg' for tid in range(n_topics)])
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
    """Step 8 v2 主入口：双轨 LDA 主题挖掘。

    独立运行方式：
        python -m movie.step8_v2_topic

    或在 pipeline 中通过 data dict 调用：
        from movie.step8_v2_topic import main as m
        m(data)   # data 由 movie.data_loader.load_all() 提供
    """
    log("=" * 60)
    log("Step 8 v2: LDA Topic Mining (Overview + User-Text Dual Track)")
    log("=" * 60)

    # 加载数据
    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']
    rows = data.get('rows', [])
    movie_info = data['movie_info']
    log(f"Loaded {len(seekers)} seekers, {len(rows)} rows, "
        f"{len(movie_info)} movies")

    # ── Track A: overview 中文 LDA（电影题材主题）─────────────────
    result_a = dim_l1a_build_overview_model(seekers, rows, movie_info)
    lda_a, dict_a, bow_a, labels_a, dt_a, metas_a = result_a
    log("")

    if dt_a is not None and dt_a.shape[0] > 0:
        dim_l2_holiday_vs_nonholiday(dt_a, metas_a, 'A_overview')
        log("")
        dim_l3_holiday_workday_weekend(dt_a, metas_a, 'A_overview')
        log("")
        dim_l4_per_holiday_topics(dt_a, metas_a, 'A_overview')
        log("")

    # ── Track B: usertext 英文 LDA（用户讨论话题）─────────────────
    result_b = dim_l1b_build_usertext_model(seekers)
    lda_b, dict_b, bow_b, labels_b, dt_b, metas_b = result_b
    log("")

    if dt_b is not None and dt_b.shape[0] > 0:
        dim_l2_holiday_vs_nonholiday(dt_b, metas_b, 'B_usertext')
        log("")
        dim_l3_holiday_workday_weekend(dt_b, metas_b, 'B_usertext')
        log("")
        dim_l4_per_holiday_topics(dt_b, metas_b, 'B_usertext')

    log("")
    log("=" * 60)
    log(f"Step 8 v2 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
