# -*- coding: utf-8 -*-
"""
训练 BERT 年龄分类模型
使用 distilbert-base-multilingual-cased 对用户文本进行年龄段分类

生成训练数据策略:
  1. 显式年龄提取 (confidence=3) — 最高优先级
  2. 高置信度启发式推断 (score >= 4) — 弱监督标签

输出: saved_age_bert_model/ 下的模型文件
"""

import os
import sys
import json
import ast
import warnings
from collections import defaultdict

# 使用 HuggingFace 国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings('ignore')

# 加入项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.age_segment import extract_explicit_age, AGE_SEGMENTS, _AGE_RULES_COMPILED

# ------------------------------------------------------------------
# 关键词规则推断（不走 BERT，避免自反馈偏斜）
# ------------------------------------------------------------------
def _keyword_only_infer(texts: list[str]) -> tuple:
    """
    仅使用关键词规则推断年龄段，不调用 BERT。
    返回 (年龄段, 置信度分数, 匹配关键词) 或 (None, 0, [])。
    """
    all_text = ' '.join(texts)
    scores = {s: 0 for s in AGE_SEGMENTS}
    matched_keywords = []

    for pat, segs in _AGE_RULES_COMPILED:
        m = pat.search(all_text)
        if m:
            matched_keywords.append(m.group(0))
            for seg, weight in segs:
                scores[seg] += weight

    if max(scores.values()) == 0:
        return None, 0, []

    best_seg = max(scores, key=lambda s: scores[s])
    best_score = scores[best_seg]
    return best_seg, best_score, matched_keywords


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------
MODEL_NAME = 'distilbert-base-multilingual-cased'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_age_bert_model')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 3e-5

# 年龄段到 ID 映射
SEG2ID = {s: i for i, s in enumerate(AGE_SEGMENTS)}
ID2SEG = {i: s for i, s in enumerate(AGE_SEGMENTS)}


# ------------------------------------------------------------------
# 1. 从所有数据文件中提取用户文本与标签
# ------------------------------------------------------------------
def load_all_user_texts(data_dir: str):
    """从 data/ 下所有 CSV 中提取用户文本"""
    all_csv = [f for f in os.listdir(data_dir)
               if f.endswith('.csv') and f != 'holiday.csv' and f != 'holiday-weekend.csv']
    
    user_texts = defaultdict(list)
    for csv_file in all_csv:
        fpath = os.path.join(data_dir, csv_file)
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        for _, row in df.iterrows():
            try:
                parts = ast.literal_eval(row['processed'])
                if parts[0] == 'USER':
                    user_texts[row['user_id']].append(parts[1])
            except (ValueError, SyntaxError, TypeError, KeyError):
                pass
    
    return user_texts


def generate_labels(user_texts: dict):
    """
    为每个用户生成标签。
    策略: 显式年龄 > 关键词规则（不用 BERT，避免自反馈偏斜）
    返回 [(text, seg_id), ...]
    """
    samples = []
    
    stats = {'explicit': 0, 'keyword_high': 0, 'keyword_low': 0, 'none': 0}
    
    for uid, texts in user_texts.items():
        combined = ' '.join(texts)
        if not combined.strip():
            continue
        
        # 策略1: 显式年龄（最高置信度）
        seg, age, _ = extract_explicit_age(combined)
        if seg and seg in SEG2ID:
            samples.append((combined, SEG2ID[seg]))
            stats['explicit'] += 1
            continue
        
        # 策略2: 关键词规则推断（不用 BERT）
        seg, score, kws = _keyword_only_infer(texts)
        if seg and seg in SEG2ID and score >= 5:
            samples.append((combined, SEG2ID[seg]))
            stats['keyword_high'] += 1
            continue
        
        if seg and seg in SEG2ID and score >= 3:
            samples.append((combined, SEG2ID[seg]))
            stats['keyword_low'] += 1
            continue
        
        stats['none'] += 1
    
    print(f'Label stats: {stats}')
    print(f'Total training samples: {len(samples)}')
    
    # 打印每个年龄段的样本数
    seg_counts = defaultdict(int)
    for _, sid in samples:
        seg_counts[ID2SEG[sid]] += 1
    for s in AGE_SEGMENTS:
        print(f'  {s}: {seg_counts[s]}')
    
    return samples


# ------------------------------------------------------------------
# 2. 数据集
# ------------------------------------------------------------------
class AgeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }


# ------------------------------------------------------------------
# 3. 训练
# ------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """支持类别加权损失的自定义 Trainer"""
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device) if self.class_weights is not None else None)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1_weighted': f1}


def train():
    print('=' * 60)
    print('BERT 年龄分类模型训练')
    print('=' * 60)
    
    # 加载 tokenizer 和模型
    print(f'\n[1/5] 加载模型: {MODEL_NAME}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(AGE_SEGMENTS),
        id2label=ID2SEG,
        label2id=SEG2ID,
        ignore_mismatched_sizes=True,
    )
    
    # 加载数据
    print('\n[2/5] 提取用户文本并生成标签')
    user_texts = load_all_user_texts(DATA_DIR)
    print(f'  共 {len(user_texts)} 个用户')
    
    samples = generate_labels(user_texts)
    
    if len(samples) < 10:
        print('ERROR: 训练样本太少，无法训练')
        return
    
    texts, labels = zip(*samples)

    # 计算类别权重（用于加权损失，缓解样本不均衡）
    from collections import Counter
    label_counts = Counter(labels)
    total = len(labels)
    class_weights = torch.tensor([
        total / (len(label_counts) * label_counts[i]) for i in range(len(AGE_SEGMENTS))
    ], dtype=torch.float32)
    print(f'\n  类别权重:')
    for s in AGE_SEGMENTS:
        sid = SEG2ID[s]
        print(f'    {s}: {label_counts[s]} 样本, weight={class_weights[sid]:.4f}')

    # 分割训练/验证（用原始数据，不加过采样）
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f'\n  训练集: {len(train_texts)}')
    print(f'  验证集: {len(val_texts)}')
    
    # 创建数据集
    print('\n[3/5] 创建 PyTorch 数据集')
    train_dataset = AgeDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = AgeDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    # 训练参数
    print('\n[4/5] 配置训练器')
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,  # 约10%步数预热
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        report_to='none',
        fp16=False,  # CPU 训练
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
    )
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # 训练
    print('\n[5/5] 开始训练...')
    trainer.train()
    
    # 最终评估
    eval_result = trainer.evaluate()
    print(f'\n验证集结果: {eval_result}')
    
    # 保存模型（先清旧文件避免锁冲突）
    if os.path.exists(OUTPUT_DIR):
        # 只删模型权重文件，保留 checkpoints
        safetensors_path = os.path.join(OUTPUT_DIR, 'model.safetensors')
        if os.path.exists(safetensors_path):
            os.remove(safetensors_path)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 保存标签映射
    with open(os.path.join(OUTPUT_DIR, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump({'id2seg': ID2SEG, 'seg2id': SEG2ID}, f, ensure_ascii=False)
    
    print(f'\n模型已保存到: {OUTPUT_DIR}')
    
    # 在验证集上打印详细报告
    print('\n--- 分类详细报告 ---')
    val_preds = trainer.predict(val_dataset)
    val_logits = val_preds.predictions
    val_pred_labels = np.argmax(val_logits, axis=-1)
    print(classification_report(
        val_labels, val_pred_labels,
        target_names=AGE_SEGMENTS,
        zero_division=0,
    ))


if __name__ == '__main__':
    train()
