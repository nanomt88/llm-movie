# -*- coding: utf-8 -*-
"""
Module 2: Sentiment & Intensity Analyzer
Extends the existing emotion_classifier_three.py pattern.
Adds sentiment intensity classification on top of 3-class sentiment.

Classification outputs:
  - sentiment: 'positive' / 'neutral' / 'negative'
  - intensity: 'mild' / 'moderate' / 'strong'

Uses VADER + AFINN hybrid approach (same as existing code).
"""

import re
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn

from data_analyzer.config import INTENSITY_MILD, INTENSITY_MODERATE, log

# ── Singleton analyzers ───────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()
_afinn = Afinn(language='en')

# ── Domain keyword boosters ───────────────────────────────────────────
_POSITIVE_BOOST = re.compile(
    r'(?i)\b(great|amazing|awesome|fantastic|wonderful|excellent|love|perfect|'
    r'brilliant|incredible|must.?watch|highly.?recommend|best|masterpiece|'
    r'favorite|beautiful|outstanding|superb|terrific|magnificent|splendid)\b'
)
_NEGATIVE_BOOST = re.compile(
    r'(?i)\b(terrible|awful|horrible|boring|waste|disappointing|worst|'
    r'trash|garbage|dreadful|atrocious|pathetic|miserable|abysmal|'
    r'painful|annoying|frustrating|ridiculous|stupid)\b'
)


# ── Sentiment Classification ──────────────────────────────────────────
def classify_sentiment(text: str) -> str:
    """
    Hybrid VADER + AFINN + keyword rules.
    Returns 'positive' / 'neutral' / 'negative'.
    """
    if not text or not isinstance(text, str):
        return 'neutral'

    # Keyword pre-check (high confidence override)
    has_pos = bool(_POSITIVE_BOOST.search(text))
    has_neg = bool(_NEGATIVE_BOOST.search(text))
    if has_pos and not has_neg:
        return 'positive'
    if has_neg and not has_pos:
        return 'negative'

    # VADER
    vader_scores = _vader.polarity_scores(text)
    vader_comp = vader_scores['compound']

    # AFINN
    afinn_score = _afinn.score(text)

    # High confidence VADER wins
    if abs(vader_comp) > 0.5:
        if vader_comp >= 0.05:
            return 'positive'
        elif vader_comp <= -0.05:
            return 'negative'
        return 'neutral'

    # Weighted fusion
    afinn_norm = np.clip(afinn_score / 10.0, -1.0, 1.0)
    combined = 0.6 * vader_comp + 0.4 * afinn_norm

    if combined >= 0.1:
        return 'positive'
    elif combined <= -0.1:
        return 'negative'
    return 'neutral'


# ── Sentiment Score (numeric) ─────────────────────────────────────────
def get_sentiment_score(text: str) -> float:
    """
    Get a continuous sentiment score in [-1, 1].
    Uses VADER compound as the primary signal.
    """
    if not text or not isinstance(text, str):
        return 0.0
    scores = _vader.polarity_scores(text)
    return scores['compound']


# ── Intensity Classification ──────────────────────────────────────────
def classify_intensity(text: str) -> str:
    """
    Classify emotional intensity based on VADER compound absolute value.
    Returns 'mild' / 'moderate' / 'strong'.
    """
    score = abs(get_sentiment_score(text))
    if score < INTENSITY_MILD:
        return 'mild'
    elif score < INTENSITY_MODERATE:
        return 'moderate'
    else:
        return 'strong'


# ── Batch API ─────────────────────────────────────────────────────────
def analyze_batch(texts: list[str]) -> list[dict]:
    """
    Analyze a batch of texts, returning sentiment + intensity for each.

    Args:
        texts: List of raw text strings.

    Returns:
        List of dicts: [{'sentiment': ..., 'intensity': ..., 'score': ...}, ...]
    """
    results = []
    for text in texts:
        results.append({
            'sentiment': classify_sentiment(text),
            'intensity': classify_intensity(text),
            'score': get_sentiment_score(text),
        })
    return results


def print_summary(results: list[dict], label: str = ""):
    """Print a quick summary of sentiment distribution."""
    from collections import Counter
    sents = Counter(r['sentiment'] for r in results)
    ints = Counter(r['intensity'] for r in results)
    total = len(results)
    log(f"Sentiment distribution {label}: {dict(sents)}")
    log(f"Intensity distribution {label}: {dict(ints)}")
    if total:
        log(f"  Positive: {sents.get('positive', 0)/total*100:.1f}%")
        log(f"  Negative: {sents.get('negative', 0)/total*100:.1f}%")
        log(f"  Neutral:  {sents.get('neutral', 0)/total*100:.1f}%")


# ── Standalone test ───────────────────────────────────────────────────
def main():
    log("=== Sentiment Analyzer Test ===", "Sentiment")
    test_texts = [
        "I absolutely love this movie! It's the best thing I've seen all year!",
        "This is terrible, what a waste of time. I hated every minute of it.",
        "I'm looking for a movie to watch with my family this weekend.",
        "Can anyone recommend a good horror movie that actually gives you chills?",
        "It was okay, nothing special but not bad either.",
    ]
    results = analyze_batch(test_texts)
    for text, res in zip(test_texts, results):
        print(f"  [{res['sentiment']:>8} / {res['intensity']:>8}] {text[:60]}...")
    print_summary(results)
    log("Sentiment analyzer ready!")


if __name__ == '__main__':
    main()
