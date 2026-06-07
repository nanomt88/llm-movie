# -*- coding: utf-8 -*-
"""
用户年龄分段模块：正则提取 + BERT 文本分类 + 关键词启发式，将用户划分到年龄段。

年龄段定义:
  ['<18', '18-25', '26-35', '36-50', '50+']

流程:
   1. 正则精确提取显式年龄（高置信度）
   2. BERT 模型文本分类推断隐含年龄（中-高置信度）
   3. 关键词启发式推断隐含年龄（低-中置信度，BERT 后备）
   4. 按 user_id 合并，显式 > BERT > 启发式
"""

import re
import ast
import os
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
from src.debug_dump import write_json
from src.config import (
    DATA_DIR, INTERMEDIATE_DIR,
    CSV_PATH, HOLIDAY_CSV, MOVIE_INFO_PATH,
    generate_run_id, log, log_error, log_warn,
)


# ------------------------------------------------------------------
# 年龄段定义
# ------------------------------------------------------------------
AGE_SEGMENTS = ['<18', '18-25', '26-35', '36-50', '50+']

SEGMENT_RANGES = {
    '<18':    (0, 17),
    '18-25':  (18, 25),
    '26-35':  (26, 35),
    '36-50':  (36, 50),
    '50+':    (51, 120),
}

# ------------------------------------------------------------------
# BERT 年龄分类器（懒加载单例）
# ------------------------------------------------------------------

_BERT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_age_bert_model')

# 模型全局缓存，只加载一次
_AGE_BERT_CLASSIFIER = None


class AgeBertClassifier:
    """
    基于 fine-tune DistilBERT 的年龄分类器。
    使用 `saved_age_bert_model/` 目录下的微调模型。
    """

    def __init__(self, model_path: str = _BERT_MODEL_PATH):
        warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.torch = torch
        self.device = torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 加载标签映射
        import json
        mapping_path = os.path.join(model_path, 'label_mapping.json')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            self.id2seg = {int(k): v for k, v in mapping['id2seg'].items()}
        else:
            self.id2seg = {i: s for i, s in enumerate(AGE_SEGMENTS)}

        self.seg2id = {v: k for k, v in self.id2seg.items()}

    def predict(self, text: str) -> tuple:
        """
        对单条文本预测年龄段。

        返回:
            (年龄段, 置信度分数 0-10, 每个年龄段概率 dict)
        """
        with self.torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=64,
                return_tensors='pt',
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            probs = self.torch.softmax(outputs.logits, dim=-1).squeeze(0)
            max_prob, pred_idx = self.torch.max(probs, dim=-1)
            confidence = round(float(max_prob) * 10, 1)  # 0-10 分
            seg = self.id2seg[int(pred_idx)]

            prob_dict = {self.id2seg[i]: float(probs[i]) for i in range(len(self.id2seg))}
            return seg, confidence, prob_dict

    def predict_texts(self, texts: list[str]) -> tuple:
        """
        对用户的多条文本预测年龄段（拼接后预测）。

        返回:
            (年龄段, 置信度分数 0-10, 概率 dict)
        """
        all_text = ' '.join(texts)
        return self.predict(all_text)


def _get_bert_classifier():
    """获取 BERT 分类器单例（懒加载）"""
    global _AGE_BERT_CLASSIFIER
    if _AGE_BERT_CLASSIFIER is None:
        if os.path.isdir(_BERT_MODEL_PATH) and os.path.exists(os.path.join(_BERT_MODEL_PATH, 'model.safetensors')):
            _AGE_BERT_CLASSIFIER = AgeBertClassifier()
        else:
            _AGE_BERT_CLASSIFIER = False  # 标记为不可用
    return _AGE_BERT_CLASSIFIER if _AGE_BERT_CLASSIFIER is not False else None


def infer_with_bert(texts: list[str]) -> tuple:
    """
    使用 BERT 模型推断用户年龄段（新函数）。

    参数:
        texts: 用户的多条发言文本列表

    返回:
        (年龄段, 置信度分数 0-10, 概率 dict, 原文拼接) 或 (None, 0, None, None)
    """
    classifier = _get_bert_classifier()
    if classifier is None:
        return None, 0, None, None

    try:
        seg, confidence, prob_dict = classifier.predict_texts(texts)
        all_text = ' '.join(texts)
        # 置信度阈值：低于 4 分 (40%) 返回 None 让后备机制处理
        if confidence >= 4.0:
            return seg, confidence, prob_dict, all_text
        return None, confidence, prob_dict, all_text
    except Exception:
        return None, 0, None, None


# ------------------------------------------------------------------
# 方法 1：显式年龄正则提取
# ------------------------------------------------------------------

# 模式 1: "I'm 25", "I am 30", "im 22"
# 模式 2: "25 years old", "30-year-old", "25 yo", "25 y/o"
# 模式 3: "my 8-year-old son", "my 10 year old kid"
# 模式 4: "my wife (35)" 等括号年龄
_EXPLICIT_AGE_PATTERNS = [
    # "I am 25", "I'm 30", "im 22"
    re.compile(r"(?i)(?:i\s*(?:'m|am)\s*)(\d{1,2})"),
    # "I am 25 years old", "30-year-old", "25 yo", "25 y/o", "25 yrs old"
    re.compile(r"(?i)(?:^|[\s,(])(\d{1,2})\s*(?:-?\s*years?\s*old|yo|y\.?\s*o\.?|yrs?\s*old)\b"),
    # "my 8-year-old son/daughter/kid", "his/her 8 year old"
    re.compile(r"(?i)(?:my|his|her|their)\s+(\d{1,2})\s*(?:-?\s*years?\s*-?\s*old|yo|y\.?o\.?)\s+(?:son|daughter|kid|child|grandson|granddaughter)"),
    # "age: 25", "aged 30"
    re.compile(r"(?i)(?:age|aged)\s*[:\-]?\s*(\d{1,2})"),
    # "(25)" in context of age (e.g. "my sister (25)")
    re.compile(r"(?i)(?:\b\w+\s*)\((\d{1,2})\)\s*(?:years?\s*old|yo)?"),
]


def _age_to_segment(age: int) -> str:
    """将具体年龄映射到年龄段"""
    for seg, (lo, hi) in SEGMENT_RANGES.items():
        if lo <= age <= hi:
            return seg
    return '50+'  # 120+ 也归入 50+


def extract_explicit_age(text: str):
    """从文本中提取显式年龄，返回 (年龄段, 具体年龄, 原文) 或 (None, None)"""
    for pat in _EXPLICIT_AGE_PATTERNS:
        for m in pat.finditer(text):
            age = int(m.group(1))
            if 1 <= age <= 120:  # 合理年龄范围
                return _age_to_segment(age), age, text
    return None, None, None


# ------------------------------------------------------------------
# 方法 2：关键词启发式推断
# ------------------------------------------------------------------

# 每条规则: (关键词列表, [(年龄段, 权重), ...])
# 权重表示该关键词映射到某年龄段的可信度
_AGE_RULES = [
    # === <18 相关 ===
    (r'\b(?:kid\s+movie|children\s+movie|cartoon|animation|disney|pixar|dreamworks)',
     [('<18', 2), ('18-25', 1)]),
    (r'\b(?:i\s+(?:am|\'?m)\s+a\s+child|i\s+(?:am|\'?m)\s+a\s+kid|i\s+(?:am|\'?m)\s+a\s+teenager|i\s+(?:am|\'?m)\s+a\s+teen)',
     [('<18', 3), ('18-25', 1)]),

    # === 18-25 相关 ===
    (r'\b(?:college|university|dorm|roommate|freshman|sophomore|high\s*school|teen|teenager|classmate)',
     [('18-25', 3), ('<18', 2)]),
    (r'\b(?:girlfriend|boyfriend|date\s*night|prom|crush|hangover|party|spring\s*break)',
     [('18-25', 2), ('26-35', 1)]),
    (r'\b(?:first\s*date|breakup|ex\s*(?:boyfriend|girlfriend))',
     [('18-25', 2), ('26-35', 1)]),

    # === 26-35 相关 ===
    (r'\b(?:my\s*(?:wife|husband)|dating|married|fianc[eé]|engagement|wedding)',
     [('26-35', 2), ('36-50', 1)]),
    (r'\b(?:baby|newborn|toddler|pregnant|maternity|parenthood)',
     [('26-35', 2), ('36-50', 1)]),
    (r'\b(?:hangover|night\s*out|bachelor|bachelorette)',
     [('26-35', 2), ('18-25', 1)]),

    # === 36-50 相关 ===
    (r'\b(?:kid|kids|children?|son|daughter|child|parent|mom\b|dad\b|mother|father|family\s+movie)',
     [('36-50', 3), ('26-35', 2)]),
    (r'\b(?:parent-teacher|pta| soccer\s*(?:mom|dad)|carpool|minivan|suburb)',
     [('36-50', 3), ('26-35', 2)]),
    (r'\b(?:mortgage|career|midlife|divorce|marriage\s*counseling)',
     [('36-50', 3), ('26-35', 1)]),
    (r'\b(?:nostalgia|nostalgic|90s|80s\s*movie|classic|grew\s*up\s*watching)',
     [('36-50', 2), ('26-35', 1), ('50+', 1)]),  # 怀旧可以跨年龄段

    # === 50+ 相关 ===
    (r'\b(?:grandma|grandpa|grandparent|grandchild|grandkid|grandson|granddaughter)',
     [('50+', 3), ('36-50', 1)]),
    (r'\b(?:retire|retired|retirement|senior|elderly|aging|old\s*age)',
     [('50+', 3)]),
    (r'\b(?:classic\s*film|golden\s*age|old\s*movie\s*1950|1960s|1970s)',
     [('50+', 2), ('36-50', 1)]),
]

# pat : _AGE_RULES 中每条规则的第一个元素，即正则表达式字符串（pattern）
# segs :  _AGE_RULES 中每条规则的第二个元素，即年龄段和权重的列表
_AGE_RULES_COMPILED = [(re.compile(pat, re.I), segs) for pat, segs in _AGE_RULES]


def infer_with_bert_fallback(texts: list[str]) -> tuple:
    """
    BERT 推断 + 关键词规则后备，返回 (年龄段, 置信度分数, 匹配关键词, 原文)。

    优先级: BERT (confidence>=4.0) > 关键词规则
    """
    all_text = ' '.join(texts)

    # ---- 1. BERT 优先 ----
    seg, confidence, prob_dict, _ = infer_with_bert(texts)
    if seg is not None:
        return seg, max(int(confidence), 2), ['bert_inference'], all_text

    # ---- 2. 关键词规则后备 ----
    scores = {s: 0 for s in AGE_SEGMENTS}
    matched_keywords = []

    for pat, segs in _AGE_RULES_COMPILED:
        m = pat.search(all_text)
        if m:
            matched_keywords.append(m.group(0))
            for seg, weight in segs:
                scores[seg] += weight

    if max(scores.values()) == 0:
        return None, 0, [], None

    best_seg = max(scores, key=lambda s: scores[s])
    best_score = scores[best_seg]
    return best_seg, best_score, matched_keywords, all_text


def infer_implicit_segment(texts: list[str]):
    """
    从用户的多条发言文本中推断年龄段。

    内部使用 BERT 模型进行文本分类，BERT 置信度不足时回退到关键词规则。
    保持原函数签名：(年龄段, 置信度分数, 匹配关键词, 原文)

    返回:
        (年龄段, 置信度分数, 匹配关键词列表, 原文)
    """
    return infer_with_bert_fallback(texts)


# ------------------------------------------------------------------
# 主流程：按 user_id 聚合，先显式后启发式
# ------------------------------------------------------------------

def segment_users(data_path: str) -> pd.DataFrame:
    """
    读取 CSV，返回每个 user_id 的年龄段分析结果。

    返回 DataFrame 列:
      - user_id:    用户 ID
      - age_segment: 年龄段标记
      - confidence:  confidence 值 (explicit=3, implicit=2/1)
      - source:      'explicit' / 'implicit' / 'unknown'
      - detail:      具体年龄或匹配关键词
      - msg_count:   该用户的发言条数
    """
    df = pd.read_csv(data_path)

    # defaultdict 是 Python collections 模块中的一个特殊字典类型，list为指定value的类型
    # 按 user_id 收集所有 USER 角色的文本
    # 格式为： { user1 : ['Hello']}
    user_bucket = defaultdict(list)
    for _, row in df.iterrows():
        try:
            parts = ast.literal_eval(row['processed'])
            if parts[0] == 'USER':
                user_bucket[row['user_id']].append(parts[1])
        except (ValueError, SyntaxError, TypeError):
            pass

    results = []
    for uid, texts in user_bucket.items():
        # ---- 方法 1: 显式提取 ----
        best_seg = None
        best_source = None
        best_detail = None
        # 置信度， 3-最高（对话中提到年龄）， 2-较高（权重推断出来的）
        best_conf = 0
        best_raw_text = None

        for t in texts:
            seg, age, raw_text = extract_explicit_age(t)
            if seg is not None:
                best_seg = seg
                best_source = 'explicit'
                best_detail = str(age)
                best_conf = 3
                best_raw_text = raw_text
                break  # 显式命中即停止

        # ---- 方法 2: 启发式推断 ----
        if best_seg is None:
            seg, score, kws, raw_text = infer_implicit_segment(texts)
            if seg is not None:
                # 置信度映射: 分数>=4 为高, >=2 为中, 否则低
                conf = 2 if score >= 4 else 1
                best_seg = seg
                best_source = 'implicit'
                best_detail = ', '.join(set(kws))
                best_conf = conf
                best_raw_text = raw_text

        results.append({
            'user_id': uid,
            'age_segment': best_seg or 'unknown',
            'confidence': best_conf,
            'source': best_source or 'unknown',
            'detail': best_detail or '',
            'msg_count': len(texts),
            'best_text' : best_raw_text or 'unknown',
            'raw_raw_text' : ' '.join(texts),
        })

    return pd.DataFrame(results)


def add_age_to_csv(data_path: str, out_path: str = None):
    """
    给原始 CSV 增加 age_segment 列，每行标注该用户的年龄段。
    """
    df_data = pd.read_csv(data_path)
    user_seg = segment_users(data_path)

    # 合并
    df_result = df_data.merge(user_seg[['user_id', 'age_segment', 'confidence', 'source']],
                               on='user_id', how='left')
    df_result['age_segment'] = df_result['age_segment'].fillna('unknown')

    if out_path:
        df_result.to_csv(out_path, index=False)
    return df_result


# ------------------------------------------------------------------
# 快捷函数
# ------------------------------------------------------------------

def get_age_of(text: str) -> tuple:
    """对单条文本直接判断年龄段，返回 (年龄段, 置信度, 来源)"""
    seg, age, raw_text = extract_explicit_age(text)
    if seg:
        return seg, 3, 'explicit'
    seg, score, kws,_  = infer_implicit_segment([text])
    if seg:
        return seg, (2 if score >= 4 else 1), 'implicit'
    return 'unknown', 0, 'unknown'

def write_age_to_file(user_seg: dict, csv_path: str = None):
    # 1. user_seg
    user_seg_data = {
        'total_users': len(user_seg),
        'age_distribution': {},
        'age_all' : {},
        # 'all': dict(list(user_seg.items())),
    }
    # 统计年龄段分布（需要从 age_segment 重新统计）
    try:
        user_seg_df = segment_users(csv_path)
        user_seg_data['age_distribution'] = user_seg_df['age_segment'].value_counts().to_dict()
        user_seg_data['age_all'] = dict(zip(user_seg['user_id'], user_seg['age_segment']));
    except Exception as e:
        log_warn('DataLoader', f'年龄段分布统计失败: {e}')
    path = write_json(user_seg_data, 'user_seg', run_id='totle', step='01')

# ------------------------------------------------------------------
# 年龄段差异对比工具
# ------------------------------------------------------------------

def compare_age_segments(file1_path: str, file2_path: str, output_csv: str = None) -> pd.DataFrame:
    """
    比较两个 JSON 文件中 age_all 标签下的差异。
    找出在两个文件中都存在（key 相同）但 value 不同的用户。

    参数:
        file1_path: 第一个 JSON 文件路径（基准文件）
        file2_path: 第二个 JSON 文件路径（对比文件）
        output_csv: 输出 CSV 文件路径（可选），如果提供则保存结果

    返回:
        DataFrame，包含列: user_id, value1, value2
        - user_id: 用户 ID
        - value1: 第一个文件中的年龄段值
        - value2: 第二个文件中的年龄段值
    """
    import json

    # 读取两个 JSON 文件
    with open(file1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    with open(file2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    # 提取 age_all 数据
    age_all_1 = data1.get('age_all', {})
    age_all_2 = data2.get('age_all', {})

    # 找出两个文件中都存在的 user_id
    common_users = set(age_all_1.keys()) & set(age_all_2.keys())

    # 找出 value 不同的用户
    differences = []
    for user_id in sorted(common_users):
        value1 = age_all_1[user_id]
        value2 = age_all_2[user_id]
        if value1 != value2:
            differences.append({
                'user_id': user_id,
                'file1_age': value1,
                'file2_age': value2
            })

    # 创建 DataFrame
    df_diff = pd.DataFrame(differences, columns=['user_id', 'file1_age', 'file2_age'])

    # 打印统计信息
    print(f"文件1: {file1_path}")
    print(f"  总用户数: {len(age_all_1)}")
    print(f"\n文件2: {file2_path}")
    print(f"  总用户数: {len(age_all_2)}")
    print(f"\n共同用户数: {len(common_users)}")
    print(f"年龄段不同的用户数: {len(df_diff)}")
    print(f"差异率: {len(df_diff)/len(common_users)*100:.2f}%" if common_users else "差异率: N/A")

    if len(df_diff) > 0:
        print(f"\n差异明细 (前10条):")
        print(df_diff.head(10).to_string(index=False))

        # 统计差异分布
        print(f"\nvfile1_age 的分布:")
        print(df_diff['file1_age'].value_counts().to_string())
        print(f"\nfile2_age 的分布:")
        print(df_diff['file2_age'].value_counts().to_string())

    # 保存到 CSV
    if output_csv:
        df_diff.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n差异明细已保存到: {output_csv}")

    return df_diff

def main():
    import os

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    csv_path = os.path.join(data_dir, 'my-train-data.csv')

    # ---- Demo ----
    print("=" * 60)
    print("单条文本测试")
    print("=" * 60)
    tests = [
        "I'm 25 years old, looking for horror movies",
        "my 8-year-old son loves cartoons",
        "Looking for a date night movie with my girlfriend",
        "Movies to watch with my grandpa",
        "Best college party movies for my dorm",
        "Family movie for my wife and kids",
        "I am 16 and looking for something scary",
    ]
    for t in tests:
        seg, conf, src = get_age_of(t)
        print(f"  {seg:6s} (conf={conf}, src={src}) | {t[:60]}")

    # ---- 批量分析 ----
    print("\n" + "=" * 60)
    print(f"批量分析: {csv_path}")
    print("=" * 60)

    user_df = segment_users(csv_path)
    total = len(user_df)
    known = user_df[user_df['age_segment'] != 'unknown']

    write_age_to_file(user_df, csv_path)

    print(f"总用户数: {total}")
    print(f"可分段:   {len(known)} ({len(known) / total * 100:.1f}%)")
    print(f"未知:     {total - len(known)}")
    print()

    # 各年龄段分布
    print("年龄段分布 (仅 known):")
    dist = known['age_segment'].value_counts().reindex(AGE_SEGMENTS, fill_value=0)
    for seg, cnt in dist.items():
        print(f"  {seg:6s}: {cnt:4d} ({cnt / len(known) * 100:5.1f}%)")

    print("\n按来源:")
    print(f"  {known['source'].value_counts().to_string()}")

# ------------------------------------------------------------------
# 使用示例
# ------------------------------------------------------------------
if __name__ == '__main__':
    # main()
    compare_age_segments('../output/intermediate/totle_holiday_S01_user_seg.json', 'output/intermediate/totle_S01_user_seg.json',
                         'output/intermediate/totle_user_seg_diff.csv')

    # 输出示例
    # print("\n示例结果 (前 5):")
    # for _, r in known.head(5).iterrows():
    #     print(f"  {r['user_id']:15s} → {r['age_segment']:6s} "
    #           # f"(conf={r['confidence']}, src={r['source']}, detail={r['detail'][:30]})")
    #           f"(conf={r['confidence']}, src={r['source']}, detail={r['detail'][:30]}), raw_text={r['raw_text']}")

    # 输出 年龄为 unknown的前3条
    # unknown = user_df[user_df['age_segment'] == 'unknown']

    # unknown = user_df[user_df['age_segment'] == list(SEGMENT_RANGES.keys())[3]]
    # for idx, r in unknown.head(10).iterrows():
    #     if idx < 0:
    #         continue
    #     print(f"  {r['user_id']:15s} → {r['age_segment']:6s} "
    #          #  f"(conf={r['confidence']}, src={r['source']}, detail={r['detail'][:30]}), "
    #           f"raw_text={r['raw_raw_text']}")

