# -*- coding: utf-8 -*-
"""
ABSA NLI / contextual classifier (Phase 2, ③.2).
方面上下文判定器：判定候选方面句是否真为评价（精度过滤）。

实现：
  - 主路径（默认）：规则版 — 候选句须含评价标记（opinion 形容词/系动词/评价动词），
    过滤纯提及句（如 "What would the twist be?"）。
  - 可选升级：transformer 句向量语义相似度（需下载模型，默认关闭以加速对比）。
"""

import re
from typing import Optional

# 评价标记词（出现任一即判为评价句）
# opinion 形容词 + 系动词/评价动词 + 程度副词
_OPINION_ADJ = {
    'good', 'great', 'bad', 'terrible', 'awful', 'horrible', 'worst', 'best',
    'amazing', 'fantastic', 'wonderful', 'excellent', 'brilliant', 'superb',
    'outstanding', 'phenomenal', 'incredible', 'stunning', 'beautiful',
    'breathtaking', 'gorgeous', 'magnificent', 'marvelous', 'splendid',
    'mediocre', 'poor', 'disappointing', 'disappointed', 'boring', 'dull',
    'flat', 'weak', 'strong', 'powerful', 'moving', 'touching', 'hilarious',
    'funny', 'scary', 'creepy', 'disturbing', 'confusing', 'predictable',
    'unpredictable', 'engaging', 'gripping', 'thrilling', 'intense', 'suspenseful',
    'cheesy', 'campy', 'dated', 'timeless', 'classic', 'masterpiece',
    'underrated', 'overrated', 'masterful', 'visually', 'cinematic',
    'overlong', 'rushed', 'slow', 'fast', 'tight', 'loose', 'coherent',
    'incoherent', 'messy', 'polished', 'rough', 'raw', 'subtle', 'nuanced',
}

_COPULA = {
    'was', 'were', 'is', 'are', 'am', 'been', 'being', 'be',
    'felt', 'seemed', 'looked', 'sounded', 'appeared', 'proved',
    'became', 'grew', 'turned',
}

_EVALUATIVE_VERB = {
    'loved', 'liked', 'hated', 'disliked', 'enjoyed', 'adored', 'despised',
    'recommended', 'suggest', 'prefer', 'rated', 'praised', 'panned',
    'delivered', 'nailed', 'killed', 'crushed', 'knocked', 'blew',
    'shined', 'stood', 'aged', 'holds', 'holds up', 'worked', 'failed',
    'suffered', 'dragged', 'rushed', 'pacing',
}

_DEGREE = {'really', 'very', 'so', 'too', 'quite', 'pretty', 'incredibly',
           'extremely', 'absolutely', 'totally', 'utterly', 'genuinely'}

# 否定（出现否定+opinion 仍算评价，只是反向）
_NEGATION = {'not', "n't", 'never', 'no', 'hardly', 'barely'}

_EVAL_TOKENS = _OPINION_ADJ | _COPULA | _EVALUATIVE_VERB | _DEGREE

# 转换为词集合查询用
_TOKEN_RE = re.compile(r"[a-z']+")


def _tokens(text: str) -> set:
    return {m.group(0) for m in _TOKEN_RE.finditer(text.lower())}


def has_evaluative_context(sentence: str) -> bool:
    """Rule-based: True if sentence contains evaluative/opinion markers.
       规则版：句含评价标记词则 True（判为真评价）。
       过滤纯提及句（无评价词的方面提及，如 "What would the twist be?"）。"""
    toks = _tokens(sentence)
    if toks & _EVAL_TOKENS:
        return True
    # 否定+任意形容词也判为评价（"not bad"）
    if toks & _NEGATION and toks & _OPINION_ADJ:
        return True
    return False


class AspectClassifier:
    """Phase 2 ③.2: contextual aspect classifier.
       主路径规则版（has_evaluative_context）；可选 transformer 句向量相似度。"""

    def __init__(self, use_model: bool = False, threshold: float = 0.5):
        self.use_model = use_model
        self.threshold = threshold
        self.model = None
        self.tokenizer = None
        self._aspect_proto_emb: dict = {}
        if use_model:
            self._try_load_model()

    def _try_load_model(self):
        """尝试加载 transformer 句向量模型（可选，失败则回退规则版）。"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            name = 'sentence-transformers/all-MiniLM-L6-v2'
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModel.from_pretrained(name)
            self._torch = torch
        except Exception as e:
            import warnings
            warnings.warn(f"absa_nli: model load failed ({e}), fallback to rule-based")
            self.use_model = False
            self.model = None

    def _mean_pool(self, outputs, attention_mask):
        import torch
        t = outputs.last_hidden_state  # [B, L, H]
        mask = attention_mask.unsqueeze(-1).expand(t.size()).float()
        return torch.sum(t * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def _embed(self, texts: list[str]) -> 'torch.Tensor':
        import torch
        if not texts:
            return torch.zeros(0, 384)
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=128, return_tensors='pt')
        with torch.no_grad():
            out = self.model(**enc)
        emb = self._mean_pool(out, enc['attention_mask'])
        return torch.nn.functional.normalize(emb, p=2, dim=1)

    def _aspect_proto(self, aspect: str, prototype_text: str) -> 'torch.Tensor':
        if aspect not in self._aspect_proto_emb:
            self._aspect_proto_emb[aspect] = self._embed([prototype_text])[0]
        return self._aspect_proto_emb[aspect]

    def score(self, aspect: str, sentence: str, prototype_text: str = '') -> float:
        """Return similarity/confidence score [0,1].
           模型版：cosine 相似度；规则版：1.0 if has_evaluative_context else 0.0。"""
        if self.use_model and self.model is not None and prototype_text:
            import torch
            sent_emb = self._embed([sentence])
            proto = self._aspect_proto(aspect, prototype_text)
            sim = torch.nn.functional.cosine_similarity(
                sent_emb, proto.unsqueeze(0)).item()
            return max(0.0, sim)
        # 规则版
        return 1.0 if has_evaluative_context(sentence) else 0.0

    def keep(self, aspect: str, sentence: str, prototype_text: str = '') -> bool:
        """Decide whether to keep a candidate aspect mention."""
        return self.score(aspect, sentence, prototype_text) >= self.threshold

    def filter_candidates(
        self, candidates: list[dict], prototypes: dict[str, str],
    ) -> tuple[list[dict], int]:
        """Filter candidates by contextual score.
        Returns (kept, n_filtered)。"""
        kept = []
        n_filtered = 0
        for c in candidates:
            proto = prototypes.get(c['aspect'], '')
            if self.keep(c['aspect'], c['snippet'], proto):
                c['nli_score'] = self.score(c['aspect'], c['snippet'], proto)
                kept.append(c)
            else:
                n_filtered += 1
        return kept, n_filtered
