# -*- coding: utf-8 -*-
"""
Step 10: Aspect-Based Sentiment Analysis & Holiday Differentiation
步骤 10：基于方面的情感分析与节假日差异化分析

Analysis:
  - Extract sentiment for movie aspects (genre, plot, cast, visual, audio, etc.)
  - ML-based aspect detection + sentiment classification via transformers
  - Holiday vs non-holiday aspect sentiment comparison
  - Per-holiday aspect sentiment profile
  - Aspect importance by holiday type

Dependencies: transformers, torch (falls back to VADER if unavailable)
Output: output/movie/step10/*.png + CSV
"""

import os
import csv
import re
import json
import warnings
from collections import defaultdict, Counter

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log
from movie.utils.genre_map import to_en, get_all_english_genres

setup_matplotlib()
STEP_OUT = STEP_DIRS[10]
os.makedirs(STEP_OUT, exist_ok=True)

# ── Aspect definitions ────────────────────────────────────────────────
# Each aspect has keywords/phrases that indicate the aspect is being discussed
# 每个方面定义了一组关键词/短语，用于识别文本中涉及的方面

ASPECTS = {
    'genre': {
        'keywords': [
            'genre', 'comedy', 'horror', 'action', 'drama', 'thriller',
            'sci-fi', 'fantasy', 'romance', 'musical', 'animation',
            'documentary', 'war', 'western', 'crime', 'mystery',
            'romantic', 'scary', 'funny', 'dark', 'gritty', 'lighthearted',
            'type of movie', 'kind of movie', 'genres',
        ],
        'label_cn': '类型',
        'label_en': 'Genre/Style',
    },
    'plot': {
        'keywords': [
            'plot', 'story', 'storyline', 'narrative', 'twist', 'ending',
            'predictable', 'unpredictable', 'engaging', 'boring',
            'confusing', 'complex', 'simple', 'well-written',
            'screenplay', 'script', 'writing', 'plotline',
            'story telling', 'storytelling',
        ],
        'label_cn': '剧情',
        'label_en': 'Plot/Story',
    },
    'cast': {
        'keywords': [
            'cast', 'actor', 'actress', 'performance', 'acting',
            'star', 'starring', 'lead', 'character', 'portrayal',
            'performances', 'voice acting', 'voice actor',
            'ensemble', 'supporting', 'role', 'played',
        ],
        'label_cn': '演员',
        'label_en': 'Cast/Acting',
    },
    'visual': {
        'keywords': [
            'visual', 'cinematography', 'cgi', 'effects', 'graphics',
            'animation', 'beautiful', 'stunning', 'visually',
            'special effects', 'vfx', 'shot', 'camera', 'scene',
            'color', 'lighting', 'art direction', 'production design',
        ],
        'label_cn': '视效',
        'label_en': 'Visual/Effects',
    },
    'audio': {
        'keywords': [
            'soundtrack', 'music', 'score', 'sound', 'audio',
            'song', 'songs', 'musical', 'theme', 'composer',
            'sound design', 'sound effects', 'ost', 'background music',
            'dubbing', 'dialogue', 'voice',
        ],
        'label_cn': '音效',
        'label_en': 'Audio/Music',
    },
    'direction': {
        'keywords': [
            'director', 'directing', 'directed', 'direction',
            'filmmaker', 'auteur', 'vision', 'style',
            'pacing', 'tone', 'atmosphere', 'mood',
            'creative', 'artistic', 'visionary',
        ],
        'label_cn': '导演',
        'label_en': 'Direction',
    },
    'emotion': {
        'keywords': [
            'funny', 'scary', 'sad', 'emotional', 'heartwarming',
            'exciting', 'thrilling', 'frightening', 'hilarious',
            'moving', 'touching', 'inspiring', 'depressing',
            'enjoyable', 'entertaining', 'boring', 'dull',
            'intense', 'suspenseful', 'chilling', 'creepy',
            'feel-good', 'uplifting', 'powerful',
        ],
        'label_cn': '情感',
        'label_en': 'Emotion/Tone',
    },
    'recommendation': {
        'keywords': [
            'recommend', 'suggest', 'must-watch', 'worth', 'classic',
            'masterpiece', 'underrated', 'overrated', 'favorite',
            'best', 'greatest', 'top', 'highly', 'should watch',
            'must see', 'underappreciated',
        ],
        'label_cn': '推荐',
        'label_en': 'Recommendation',
    },
    'comparison': {
        'keywords': [
            'similar', 'like', 'reminds', 'remind', 'compare',
            'comparison', 'alike', 'same vibe', 'similar to',
            'if you like', 'fans of', 'better than', 'worse than',
            'combination', 'mix of', 'blend of', 'cross between',
            'recommendation based',
        ],
        'label_cn': '比较',
        'label_en': 'Comparison',
    },
    'content': {
        'keywords': [
            'violent', 'gore', 'blood', 'mature', 'adult',
            'language', 'nudity', 'sexual', 'offensive',
            'disturbing', 'graphic', 'explicit', 'content',
            'age', 'appropriate', 'family-friendly', 'children',
        ],
        'label_cn': '内容',
        'label_en': 'Content/Warnings',
    },
}

ASPECT_NAMES = list(ASPECTS.keys())
ASPECT_EN_LABELS = [ASPECTS[a]['label_en'] for a in ASPECT_NAMES]

# ── Sentiment analyzer ────────────────────────────────────────────────

class SentimentAnalyzer:
    """Sentiment analysis using transformer model with fallback to VADER.
       使用 transformer 模型进行情感分析，不可用时回退到 VADER。"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.use_vader = False
        self._initialize()

    def _initialize(self):
        """Try to load transformer model, fall back to VADER.
           尝试加载 transformer 模型，回退到 VADER。"""
        # Try transformers (ML-based)
        try:
            from transformers import pipeline
            # Use a small, fast model for sentiment
            self.model = pipeline(
                'sentiment-analysis',
                model='distilbert-base-uncased-finetuned-sst-2-english',
                max_length=128,
                truncation=True,
            )
            log("  Sentiment: loaded distilbert transformer model", "ABSA")
            return
        except Exception as e:
            log(f"  Sentiment: transformers unavailable ({e}), trying VADER", "ABSA")

        # Fallback to VADER
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            self.use_vader = True
            log("  Sentiment: using VADER fallback", "ABSA")
            return
        except Exception:
            log("  Sentiment: no model available — using rule-based fallback", "ABSA")
            self.use_vader = False

    def predict(self, text: str) -> tuple[str, float]:
        """Predict sentiment for a text.
           预测文本的情感。
        Returns:
            (label, score): ('POSITIVE'|'NEGATIVE'|'NEUTRAL', confidence)
        """
        if not text or len(text.strip()) < 3:
            return ('NEUTRAL', 0.0)

        if self.model is not None:
            try:
                result = self.model(text[:512])[0]
                label = result['label']
                score = result['score']
                return (label, score)
            except Exception:
                return ('NEUTRAL', 0.0)

        if self.use_vader:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
            if compound >= 0.05:
                return ('POSITIVE', compound)
            elif compound <= -0.05:
                return ('NEGATIVE', abs(compound))
            else:
                return ('NEUTRAL', 0.0)

        # Simple rule-based fallback
        pos_words = {'good', 'great', 'amazing', 'excellent', 'wonderful',
                     'fantastic', 'love', 'best', 'beautiful', 'awesome',
                     'enjoy', 'enjoyed', 'fun', 'funny', 'interesting',
                     'recommend', 'perfect', 'brilliant', 'favorite', 'classic'}
        neg_words = {'bad', 'terrible', 'awful', 'horrible', 'worst',
                     'hate', 'boring', 'dull', 'poor', 'disappointing',
                     'waste', 'ugly', 'stupid', 'ridiculous', 'annoying'}
        words = set(text.lower().split())
        pos_count = len(words & pos_words)
        neg_count = len(words & neg_words)
        if pos_count > neg_count:
            return ('POSITIVE', min(pos_count / max(len(words), 1) * 5, 1.0))
        elif neg_count > pos_count:
            return ('NEGATIVE', min(neg_count / max(len(words), 1) * 5, 1.0))
        return ('NEUTRAL', 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Aspect detection & sentiment extraction
# ═══════════════════════════════════════════════════════════════════════

def detect_aspects(text: str) -> dict[str, str]:
    """Detect which aspects are mentioned in text and extract the relevant snippet.
       检测文本中涉及的方面，并提取相关片段。"""
    if not text:
        return {}
    text_lower = text.lower()
    detected = {}
    for aspect_name, aspect_config in ASPECTS.items():
        for kw in aspect_config['keywords']:
            if kw in text_lower:
                # Extract the sentence containing the keyword
                sentences = re.split(r'[.!?]+', text)
                for sent in sentences:
                    if kw in sent.lower():
                        detected[aspect_name] = sent.strip()
                        break
                else:
                    detected[aspect_name] = kw
                break
    return detected


def extract_aspect_sentiments(
    seekers: list[dict],
    analyzer: SentimentAnalyzer,
) -> list[dict]:
    """Extract aspect-specific sentiments from seeker records.
       从用户提问记录中提取方面级情感。
    Returns:
        List of dicts with: date, period, holiday_name, aspect, sentiment, score, snippet
    """
    results = []
    stats = Counter()
    for r in seekers:
        text = r.get('proc_text', '') or r.get('raw_text', '')
        if not text:
            continue
        aspects = detect_aspects(text)
        if not aspects:
            continue

        for aspect, snippet in aspects.items():
            sentiment, score = analyzer.predict(snippet)
            results.append({
                'date': r.get('date', ''),
                'period': r.get('period', ''),
                'is_holiday': r.get('is_holiday', False),
                'holiday_name': r.get('holiday_name', ''),
                'holiday_type': r.get('holiday_type', ''),
                'aspect': aspect,
                'aspect_label': ASPECTS[aspect]['label_en'],
                'sentiment': sentiment,
                'score': score,
                'snippet': snippet[:200],
            })
            stats[aspect] += 1

    log(f"  Detected aspect mentions: {dict(stats)}", "ABSA")
    return results


def _sentiment_to_numeric(sentiment: str) -> int:
    """Convert sentiment label to numeric value.
       将情感标签转换为数值。"""
    return {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}.get(sentiment, 0)


# ═══════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════

def _plot_aspect_sentiment_bars(
    aspect_data: dict[str, dict],
    title: str, filename: str,
):
    """Grouped bar chart of sentiment scores per aspect.
       各方面情感得分的分组柱状图。"""
    aspects = sorted(aspect_data.keys(), key=lambda a: sum(
        aspect_data[a].get(g, {}).get('mean', 0) for g in aspect_data[a]
    ), reverse=True)

    groups = sorted(set(
        g for a in aspects for g in aspect_data[a]
    ))

    if not aspects or not groups:
        return

    fig, ax = plt.subplots(figsize=(12, max(5, len(aspects) * 0.4 + 1)))
    x = np.arange(len(aspects))
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)
    colors = ['#ff6b6b', '#74b9ff', '#feca57', '#48dbfb']

    for i, group in enumerate(groups):
        means = []
        errors = []
        for a in aspects:
            stats = aspect_data[a].get(group, {})
            means.append(stats.get('mean', 0))
            errors.append(stats.get('std', 0))
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=errors, capsize=3,
               label=group, color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([ASPECTS[a]['label_en'] for a in aspects],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean Sentiment (-1 to +1)')
    ax.set_title(title, fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved: {path}")


def _plot_aspect_heatmap(
    aspect_matrix: dict[str, dict[str, float]],
    title: str, filename: str,
    cmap: str = 'RdBu_r',
):
    """Heatmap of aspect values (rows=holidays, cols=aspects).
       方面值热力图（行=节假日，列=方面）。"""
    names = sorted(aspect_matrix.keys())
    aspects = ASPECT_NAMES
    if not names:
        return

    matrix = np.zeros((len(names), len(aspects)))
    for i, name in enumerate(names):
        for j, a in enumerate(aspects):
            matrix[i, j] = aspect_matrix[name].get(a, {'mean': 0})['mean']

    fig, ax = plt.subplots(figsize=(max(10, len(aspects) * 0.8),
                                     max(4, len(names) * 0.4 + 1)))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap,
                xticklabels=[ASPECTS[a]['label_en'] for a in aspects],
                yticklabels=names, ax=ax,
                center=0, linewidths=0.5,
                cbar_kws={'label': 'Mean Sentiment'})
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Aspect')
    ax.set_ylabel('Holiday')
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    path = os.path.join(STEP_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


def _save_aspect_csv(
    aspect_records: list[dict],
    filename: str,
):
    """Save aspect sentiment records to CSV.
       将方面情感记录保存到 CSV。"""
    csv_path = os.path.join(STEP_OUT, filename)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'period', 'holiday_name', 'aspect', 'aspect_label',
                     'sentiment', 'score', 'snippet'])
        for rec in aspect_records:
            w.writerow([
                rec['date'], rec['period'], rec['holiday_name'],
                rec['aspect'], rec['aspect_label'],
                rec['sentiment'], f'{rec["score"]:.4f}',
                rec['snippet'],
            ])
    log(f"Saved: {csv_path}")


def _aggregate_aspect_sentiment(
    aspect_records: list[dict],
    filter_fn=None,
    group_by_period: bool = True,
) -> dict[str, dict[str, dict]]:
    """Aggregate aspect sentiment records into statistics.
       聚合方面情感记录为统计数据。
    Args:
        aspect_records: 方面情感记录列表
        filter_fn: 可选过滤函数
        group_by_period: 是否按 period 分组，否则合并为 'overall' 组
    Returns:
        dict[aspect] -> dict[group] -> {'mean', 'std', 'count', 'pos_ratio'}
    """
    filtered = aspect_records if filter_fn is None else [r for r in aspect_records if filter_fn(r)]
    if not filtered:
        return {}

    grouped = defaultdict(lambda: defaultdict(list))
    for rec in filtered:
        aspect = rec['aspect']
        if group_by_period:
            group_key = rec.get('period', 'unknown')
        else:
            group_key = 'overall'
        grouped[aspect][group_key].append(_sentiment_to_numeric(rec['sentiment']))

    result = {}
    for aspect, group_data in grouped.items():
        result[aspect] = {}
        for group, scores in group_data.items():
            arr = np.array(scores)
            result[aspect][group] = {
                'mean': float(arr.mean()),
                'std': float(arr.std()) if len(arr) > 1 else 0.0,
                'count': len(arr),
                'pos_ratio': float((arr > 0).sum() / max(len(arr), 1)),
            }
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Analysis dimensions
# ═══════════════════════════════════════════════════════════════════════

def dim_a1_aspect_distribution(aspect_records: list[dict]):
    """Overall aspect mention distribution.
       全局方面提及分布统计。"""
    log("=" * 50)
    log("A1: Aspect Mention Distribution")

    counter = Counter(r['aspect_label'] for r in aspect_records)
    total = sum(counter.values())
    log(f"  Total aspect mentions: {total}")
    for label, cnt in counter.most_common():
        pct = cnt / total * 100
        log(f"    {label}: {cnt} ({pct:.1f}%)")

    # Pie chart
    labels = [l for l, _ in counter.most_common()]
    values = [c for _, c in counter.most_common()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=None, autopct='%1.1f%%',
        colors=colors, startangle=90,
    )
    ax.legend(wedges, labels, title="Aspect", loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    ax.set_title('Aspect Distribution in User Requests', fontsize=13)
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'a1_aspect_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")


def dim_a2_overall_aspect_sentiment(aspect_records: list[dict]):
    """Overall sentiment per aspect.
       各方面总体情感。"""
    log("=" * 50)
    log("A2: Overall Aspect Sentiment")

    # Overall (no period grouping)
    agg = _aggregate_aspect_sentiment(aspect_records, group_by_period=False)
    if not agg:
        log("  No aspect data")
        return

    log("  Sentiment by aspect (mean ± std):")
    for aspect in sorted(agg.keys()):
        stats = agg[aspect].get('overall', {})
        log(f"    {ASPECTS[aspect]['label_en']}: {stats.get('mean', 0):.3f} ± {stats.get('std', 0):.3f} "
            f"(n={stats.get('count', 0)})")

    _plot_aspect_sentiment_bars(
        agg,
        'Overall Sentiment by Aspect',
        'a2_overall_aspect_sentiment.png',
    )


def dim_a3_holiday_vs_nonholiday_aspect(aspect_records: list[dict]):
    """Holiday vs non-holiday aspect sentiment comparison.
       节假日 vs 非节假日方面情感对比。"""
    log("=" * 50)
    log("A3: Holiday vs Non-Holiday Aspect Sentiment")

    h_agg = _aggregate_aspect_sentiment(
        aspect_records, lambda r: r['period'] == 'holiday'
    )
    nh_agg = _aggregate_aspect_sentiment(
        aspect_records, lambda r: r['period'] != 'holiday'
    )

    # Merge into single dict for plotting
    merged = {}
    for aspect in set(list(h_agg.keys()) + list(nh_agg.keys())):
        merged[aspect] = {}
        if aspect in h_agg and 'holiday' in h_agg[aspect]:
            merged[aspect]['Holiday'] = h_agg[aspect]['holiday']
        if aspect in nh_agg:
            for group, stats in nh_agg[aspect].items():
                merged[aspect]['Non-holiday'] = stats

    if merged:
        _plot_aspect_sentiment_bars(
            merged,
            'Aspect Sentiment: Holiday vs Non-Holiday',
            'a3_holiday_vs_nonholiday_aspect.png',
        )

    # CSV
    csv_path = os.path.join(STEP_OUT, 'a3_holiday_vs_nonholiday_aspect.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['aspect', 'aspect_label', 'period', 'mean_sentiment',
                     'std_sentiment', 'count', 'positive_ratio'])
        for aspect in sorted(merged.keys()):
            for group, stats in merged[aspect].items():
                w.writerow([
                    aspect, ASPECTS[aspect]['label_en'], group,
                    f'{stats["mean"]:.4f}', f'{stats["std"]:.4f}',
                    stats['count'], f'{stats["pos_ratio"]:.3f}',
                ])
    log(f"Saved: {csv_path}")


def dim_a4_holiday_workday_weekend_aspect(aspect_records: list[dict]):
    """Holiday vs workday vs weekend aspect sentiment.
       节假日 vs 工作日 vs 周末方面情感。"""
    log("=" * 50)
    log("A4: Holiday vs Workday vs Weekend Aspect Sentiment")

    period_map = {'holiday': 'Holiday', 'workday': 'Workday', 'weekend': 'Weekend'}
    merged = {}
    for p_code, p_label in period_map.items():
        p_agg = _aggregate_aspect_sentiment(
            aspect_records, lambda r, pc=p_code: r['period'] == pc
        )
        for aspect, group_data in p_agg.items():
            if aspect not in merged:
                merged[aspect] = {}
            for group, stats in group_data.items():
                merged[aspect][p_label] = stats

    if merged:
        _plot_aspect_sentiment_bars(
            merged,
            'Aspect Sentiment: Holiday vs Workday vs Weekend',
            'a4_holiday_workday_weekend_aspect.png',
        )

    csv_path = os.path.join(STEP_OUT, 'a4_holiday_workday_weekend_aspect.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['aspect', 'aspect_label', 'period', 'mean_sentiment',
                     'std_sentiment', 'count', 'positive_ratio'])
        for aspect in sorted(merged.keys()):
            for group, stats in merged[aspect].items():
                w.writerow([
                    aspect, ASPECTS[aspect]['label_en'], group,
                    f'{stats["mean"]:.4f}', f'{stats["std"]:.4f}',
                    stats['count'], f'{stats["pos_ratio"]:.3f}',
                ])
    log(f"Saved: {csv_path}")


def dim_a5_per_holiday_aspect_heatmap(aspect_records: list[dict]):
    """Per-holiday aspect sentiment heatmap.
       各节假日方面情感热力图。"""
    log("=" * 50)
    log("A5: Per-Holiday Aspect Sentiment Heatmap")

    # Group by holiday name
    holiday_aspects = defaultdict(lambda: defaultdict(list))
    for rec in aspect_records:
        if rec['is_holiday'] and rec['holiday_name']:
            name = rec['holiday_name'][:8]
            holiday_aspects[name][rec['aspect']].append(
                _sentiment_to_numeric(rec['sentiment'])
            )

    # Filter: need MIN_DATA_ROWS data points per holiday
    holiday_aspects = {
        k: v for k, v in holiday_aspects.items()
        if sum(len(scores) for scores in v.values()) >= MIN_DATA_ROWS
    }

    if not holiday_aspects:
        log("  No holiday groups with sufficient data")
        return

    # Build matrix
    holiday_names = sorted(holiday_aspects.keys())
    matrix = np.zeros((len(holiday_names), len(ASPECT_NAMES)))
    for i, name in enumerate(holiday_names):
        for j, aspect in enumerate(ASPECT_NAMES):
            scores = holiday_aspects[name].get(aspect, [])
            if scores:
                matrix[i, j] = np.mean(scores)
            else:
                matrix[i, j] = np.nan

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(ASPECT_NAMES) * 0.8),
                                     max(5, len(holiday_names) * 0.5 + 1)))
    mask = np.isnan(matrix)
    cmap = plt.cm.RdBu_r
    cmap.set_bad('lightgray')
    sns.heatmap(
        np.ma.masked_invalid(matrix),
        annot=True, fmt='.2f', cmap=cmap,
        xticklabels=[ASPECTS[a]['label_en'] for a in ASPECT_NAMES],
        yticklabels=holiday_names, ax=ax,
        center=0, linewidths=0.5, mask=mask,
        cbar_kws={'label': 'Mean Sentiment (-1 to +1)'},
    )
    ax.set_title('Per-Holiday Aspect Sentiment Heatmap', fontsize=13)
    ax.set_xlabel('Aspect')
    ax.set_ylabel('Holiday')
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    path = os.path.join(STEP_OUT, 'a5_per_holiday_aspect_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {path}")

    # CSV
    csv_path = os.path.join(STEP_OUT, 'a5_per_holiday_aspect_sentiment.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        header = ['holiday_name']
        for a in ASPECT_NAMES:
            header.append(f'{a}_mean')
            header.append(f'{a}_count')
        w.writerow(header)
        for name in holiday_names:
            row = [name]
            for a in ASPECT_NAMES:
                scores = holiday_aspects[name].get(a, [])
                if scores:
                    row.append(f'{np.mean(scores):.4f}')
                    row.append(len(scores))
                else:
                    row.append('')
                    row.append(0)
            w.writerow(row)
    log(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main(data: dict = None):
    log("=" * 60)
    log("Step 10: Aspect-Based Sentiment Analysis & Holiday Differentiation")
    log("=" * 60)

    if data is None:
        from movie.data_loader import load_all
        data = load_all()
    seekers = data['seekers']
    log(f"Loaded {len(seekers)} seeker records")

    # Initialize sentiment analyzer (ML-based)
    log("Initializing sentiment analyzer...")
    analyzer = SentimentAnalyzer()
    log("")

    # Extract aspect sentiments
    log("Extracting aspect sentiments from seekers...")
    aspect_records = extract_aspect_sentiments(seekers, analyzer)
    log(f"  Total aspect-sentiment records: {len(aspect_records)}")
    log("")

    if not aspect_records:
        log("  No aspect data found. Skipping all analysis.")
        return

    # Save raw data
    _save_aspect_csv(aspect_records, 'a0_aspect_sentiments_raw.csv')

    # A1: Aspect distribution
    dim_a1_aspect_distribution(aspect_records)
    log("")

    # A2: Overall aspect sentiment
    dim_a2_overall_aspect_sentiment(aspect_records)
    log("")

    # A3: Holiday vs non-holiday
    dim_a3_holiday_vs_nonholiday_aspect(aspect_records)
    log("")

    # A4: Holiday vs workday vs weekend
    dim_a4_holiday_workday_weekend_aspect(aspect_records)
    log("")

    # A5: Per-holiday heatmap
    dim_a5_per_holiday_aspect_heatmap(aspect_records)

    log("")
    log("=" * 60)
    log(f"Step 10 complete! Results saved to {STEP_OUT}")
    log("=" * 60)


if __name__ == '__main__':
    main()
