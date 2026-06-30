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

import os
import csv
import re
import logging
from collections import Counter, defaultdict

import numpy as np

# Suppress verbose gensim logs
logging.getLogger('gensim').setLevel(logging.WARNING)

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log

setup_matplotlib()
STEP_OUT = STEP_DIRS[8]
os.makedirs(STEP_OUT, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
NUM_TOPICS = 8          # Number of LDA topics
NUM_TOP_WORDS = 15      # Top words per topic
PASSES = 10             # LDA training passes
MIN_WORD_COUNT = 3      # Min word occurrences to include in dictionary
MIN_WORD_LEN = 3
MAX_WORD_FRAC = 0.5     # Max fraction of docs a word can appear in (filter too common)

# ── Chinese→English genre mapping ────────────────────────────────────
from movie.utils.genre_map import to_en


def tokenize(text: str) -> list[str]:
    """Simple English tokenizer for LDA.
       面向 LDA 的简单英文分词。"""
    if not text:
        return []
    text = text.lower()
    tokens = re.split(r"[^a-z']+", text)
    return [t.strip("'") for t in tokens
            if len(t.strip("'")) >= MIN_WORD_LEN
            and t.strip("'").isalpha()
            and t.strip("'") not in _STOPWORDS]

# Standard English stopwords for LDA
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


def _build_lda_model(documents: list[list[str]]) -> tuple:
    """Build LDA model from tokenized documents.
       从分词后的文档构建 LDA 模型。
    Returns:
        (lda_model, corpus, dictionary, bow_corpus)
    """
    dictionary = corpora.Dictionary(documents)
    # Filter extreme words
    dictionary.filter_extremes(no_below=MIN_WORD_COUNT, no_above=MAX_WORD_FRAC)
    dictionary.compactify()
    log(f"  Dictionary size: {len(dictionary)} tokens")

    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    # Remove empty documents
    bow_corpus = [b for b in bow_corpus if b]
    log(f"  Non-empty documents: {len(bow_corpus)}")

    lda_model = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=PASSES,
        random_state=42,
    )
    return lda_model, dictionary, bow_corpus


def _get_topic_term_matrix(lda_model, dictionary) -> np.ndarray:
    """Get topic-term matrix: (num_topics, vocab_size).
       获取主题-词项矩阵。"""
    num_terms = len(dictionary)
    matrix = np.zeros((lda_model.num_topics, num_terms))
    for topic_id in range(lda_model.num_topics):
        terms = lda_model.get_topic_terms(topic_id, topn=num_terms)
        for term_id, prob in terms:
            matrix[topic_id, term_id] = prob
    return matrix


def _get_doc_topic_dist(lda_model, bow_corpus) -> np.ndarray:
    """Get document-topic distribution: (num_docs, num_topics).
       获取文档-主题分布矩阵。"""
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
    """
    labels = {}
    for topic_id in range(lda_model.num_topics):
        words = lda_model.show_topic(topic_id, topn=num_words)
        top3 = [w for w, _ in words[:3]]
        labels[topic_id] = '-'.join(top3)
    return labels


def _plot_topic_term_heatmap(lda_model, dictionary, filename: str):
    """Heatmap of top words per topic.
       每个主题的关键词热力图。"""
    num_words = 15
    topic_words = {}
    for topic_id in range(lda_model.num_topics):
        words = lda_model.show_topic(topic_id, topn=num_words)
        topic_words[topic_id] = [w for w, _ in words]

    # Build matrix
    all_words = []
    for tid in range(lda_model.num_topics):
        for w in topic_words[tid]:
            if w not in all_words:
                all_words.append(w)
    # Keep only up to 10 unique words per topic
    unique_words = []
    for tid in range(lda_model.num_topics):
        for w in topic_words[tid]:
            if w not in unique_words:
                unique_words.append(w)
    unique_words = unique_words[:30]

    matrix = np.zeros((lda_model.num_topics, len(unique_words)))
    for tid in range(lda_model.num_topics):
        word_probs = dict(lda_model.show_topic(tid, topn=len(dictionary)))
        for j, w in enumerate(unique_words):
            matrix[tid, j] = word_probs.get(w, 0)

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
       分组柱状图：比较不同分组的主题分布。"""
    groups = list(dist_dict.keys())
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


def _plot_per_holiday_topic_heatmap(
    topic_by_holiday: dict[str, np.ndarray],
    global_doc_topic: np.ndarray,
    filename: str,
):
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
#  Analysis dimensions
# ═══════════════════════════════════════════════════════════════════════

def dim_l1_build_model(seekers: list[dict], texts: list[str]):
    """Build LDA model and show topics.
       构建 LDA 模型并展示主题。"""
    log("=" * 50)
    log("L1: Build LDA Topic Model")

    documents = [tokenize(t) for t in texts]
    non_empty = [d for d in documents if d]
    log(f"  Tokenized {len(non_empty)}/{len(documents)} non-empty docs")

    global lda_model, dictionary, bow_corpus
    lda_model, dictionary, bow_corpus = _build_lda_model(non_empty)

    # Print topics
    log(f"  Topics ({NUM_TOPICS}):")
    labels = _assign_topic_labels(lda_model, dictionary)
    for tid in range(lda_model.num_topics):
        words = lda_model.show_topic(tid, topn=NUM_TOP_WORDS)
        word_str = ' + '.join(f'{p:.2f}*{w}' for w, p in words[:8])
        log(f"    T{tid} ({labels[tid]}): {word_str}")

    # Save topic terms to CSV
    csv_path = os.path.join(STEP_OUT, 'l1_topic_terms.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['topic_id', 'label'] + [f'top_word_{i+1}' for i in range(NUM_TOP_WORDS)])
        for tid in range(lda_model.num_topics):
            words = lda_model.show_topic(tid, topn=NUM_TOP_WORDS)
            w.writerow([tid, labels[tid]] + [w for w, _ in words])
    log(f"Saved: {csv_path}")

    # Save document-topic distributions
    doc_topic = _get_doc_topic_dist(lda_model, bow_corpus)
    _plot_topic_term_heatmap(lda_model, dictionary, 'l1_topic_term_heatmap.png')
    return lda_model, dictionary, bow_corpus, labels, doc_topic


def dim_l2_holiday_vs_nonholiday_topics(doc_topic: np.ndarray, seekers: list[dict]):
    """Holiday vs non-holiday topic distribution comparison.
       节假日 vs 非节假日主题分布对比。"""
    log("=" * 50)
    log("L2: Holiday vs Non-Holiday Topic Distribution")

    holiday_mask = np.array([r['period'] == 'holiday' for r in seekers
                             if r.get('proc_text', '') or r.get('raw_text', '')])
    non_holiday_mask = ~holiday_mask

    # Align doc_topic with seekers
    # Only non-empty text docs survived in bow_corpus — we need seeker mask for those
    texts = [r.get('proc_text', '') or r.get('raw_text', '')
             for r in seekers]
    doc_texts = [t for t in texts if tokenize(t)]

    if len(doc_texts) != doc_topic.shape[0]:
        log(f"  Warning: doc_topic shape {doc_topic.shape[0]} != filtered seekers {len(doc_texts)}")
        return

    h_topics = np.array([dc for dc, r in zip(doc_topic, seekers)
                         if r.get('proc_text', '') or r.get('raw_text', '')])
    # Recompute masks on filtered
    seekers_filtered = [r for r in seekers
                        if r.get('proc_text', '') or r.get('raw_text', '')]
    if len(seekers_filtered) != len(h_topics):
        return

    h_mask = np.array([r['period'] == 'holiday' for r in seekers_filtered])
    nh_mask = ~h_mask

    if h_mask.sum() == 0 or nh_mask.sum() == 0:
        log("  Not enough data for comparison")
        return

    _plot_topic_distribution_bar(
        {'Holiday': h_topics[h_mask], 'Non-holiday': h_topics[nh_mask]},
        'Topic Distribution: Holiday vs Non-Holiday',
        'l2_holiday_vs_nonholiday_topics.png',
    )

    # Save CSV
    csv_path = os.path.join(STEP_OUT, 'l2_holiday_vs_nonholiday_topics.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['period', 'topic_id', 'avg_proportion', 'std_proportion'])
        for period, mask in [('holiday', h_mask), ('non_holiday', nh_mask)]:
            proportions = h_topics[mask]
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

    seekers_filtered = [r for r in seekers
                        if r.get('proc_text', '') or r.get('raw_text', '')]
    if len(seekers_filtered) != doc_topic.shape[0]:
        log("  Mismatch between doc_topic and seekers, skipping")
        return

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

    # Group by holiday name
    holiday_groups = defaultdict(list)
    holiday_indices = defaultdict(list)
    for i, r in enumerate(seekers_filtered):
        if r['is_holiday']:
            name = r.get('holiday_name', '')[:8]
            holiday_groups[name].append(r)
            holiday_indices[name].append(i)

    holiday_groups = {k: v for k, v in holiday_groups.items()
                      if len(v) >= MIN_DATA_ROWS}
    if not holiday_groups:
        log("  No holiday groups with sufficient data")
        return

    topic_by_holiday = {}
    for name, indices in holiday_indices.items():
        if name in holiday_groups:
            topic_by_holiday[name] = doc_topic[indices]

    _plot_per_holiday_topic_heatmap(
        topic_by_holiday, doc_topic,
        'l4_per_holiday_topic_heatmap.png',
    )

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
        # Global baseline row
        row = ['global_baseline', doc_topic.shape[0]]
        for tid in range(n_topics):
            row.append(f'{doc_topic[:, tid].mean():.4f}')
        w.writerow(row)
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None):
    log("=" * 60)
    log("Step 8: LDA Topic Model & Holiday Preference Mining")
    log("=" * 60)

    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']
    log(f"Loaded {len(seekers)} seeker records")

    # Prepare texts
    texts = [r.get('proc_text', '') or r.get('raw_text', '') for r in seekers]

    # L1: Build model
    lda_model, dictionary, bow_corpus, labels, doc_topic = dim_l1_build_model(
        seekers, texts)
    log("")

    if doc_topic is None or doc_topic.shape[0] == 0:
        log("  ERROR: LDA model produced no valid topic distribution")
        return

    # L2: Holiday vs non-holiday
    dim_l2_holiday_vs_nonholiday_topics(doc_topic, seekers)
    log("")

    # L3: Holiday vs workday vs weekend
    dim_l3_holiday_workday_weekend_topics(doc_topic, seekers)
    log("")

    # L4: Per-holiday heatmap
    dim_l4_per_holiday_topics(doc_topic, seekers)

    log("")
    log("=" * 60)
    log(f"Step 8 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
