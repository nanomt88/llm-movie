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
import math             # 数学函数（log2）
import string           # 字符串工具
from collections import Counter, defaultdict   # 计数器与默认字典

import numpy as np      # 数值计算
from wordcloud import WordCloud                # 词云生成

import matplotlib
matplotlib.use('Agg')                           # 非交互式后端（服务器环境）
import matplotlib.pyplot as plt

from movie.config import STEP_DIRS, MIN_DATA_ROWS, setup_matplotlib, log
from movie.utils.text import tokenize, deduplicate_seekers
from movie.utils.plotting import annotate_heatmap

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
    'please', 'thank', 'thanks', 'much', 'able', 'possible', 'need','yes',
})

# 附加领域噪音词：与电影讨论常见但不携带偏好信号的词汇
DOMAIN_STOP = {'movie', 'movies', 'film', 'films',
               'watch', 'watched', 'watching', 'watchlist', 'like',
               'liked',                'looking', 'look', 'recommend', 'recommended',
               'recommendation', 'recommendations', 'suggest', 'suggested',
               'suggestion', 'suggestions', 'anyone', 'somebody', 'know',
               'searching', 'find', 'found', 'seen', 'seeing',
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
                'nhttps', 'nbut', 'nthis', "ni'm", "ni've", 'ttt',
                'njane', 'ncomedy', 'nwhen', 'nall', 'nthese', 'nhappy',
                'nthese', 'njan', 'nfeb', 'nmar', 'napr', 'nmay', 'njun',
                'njul', 'naug', 'nsep', 'noct', 'nnov', 'ndec',
                'chntb', 'cneon', 'sxsrf', 'sclient', 'htt',
                'mozambique', 'nellie', 'giphy', 'wiz', 'doo',
                 'rainforest', 'smoked',
                 'aboriginal', 'hispanic', 'eden', 'flame', 'heaps',
                 'thailand', 'january', 'trips', 'cells',
                 'outdoors', 'disabled', 'bite', 'holds', 'stood',
                 'encounters', 'palette', 'healthy',
                 'spring',
                 'ups',
                 'define',
                 'scares', 'approach',
                 'passengers', 'concentration', 'pursuit', 'elizabeth',
                 'nightclub', 'adjacent', 'objective',
                 'testing',
                 'geniuses', 'duology', 'progressive', 'partly',
                 'sclient', 'watcht',
                # ── 带撇号的缩写（会漏过分词器）──
                "i'm", "i've", "it's", "don't", "can't", "won't","he's"
                "didn't", "doesn't", "isn't", "aren't", "that's",
                "you're", "they're", "there's", "here's", "what's",
                "wasn't", "couldn't", "wouldn't", "shouldn't",
                "haven't", "hasn't", "hadn't", "i'll", "i'd", "year's",
                # ── 无语义偏好的通用词 ──
                'feel', 'etc', 'letterboxd',
                'web', 'context', 'medium',
                'recently', 'advance', 'main',
                'example', 'examples',  'request',
                # ── 流媒体平台/网站名（非推荐信号）──
                 'youtube',
                 'imdb', 'wikipedia', 'wiki', 'hollywood',
                # ── 通用填充词/语气词 ──
                'similar', 'check', 'list', 'ones', 'sure', 'right',
                'kinda', 'wow', 'hey', 'sorry', 'welcome', 'whatever',
                'anyway', 'obviously', 'exactly', 'particularly',
                'necessarily', 'completely', 'specifically', 'especially',
                'personally', 'tho', 'damn', 'fucking', 'shit', 'hell',
                 'cool', 'nice', 'fine', 'course',
                'ill', 'non', 'bonus',
                # ── 通用动词（无偏好信号）──
                'put', 'come', 'getting', 'seem', 'gets', 'came', 'fit',
                'comes', 'making', 'doing', 'started', 'happens', 'happen',
                'become', 'follow', 'following', 'share', 'explain',
                'knows', 'happened', 'saying', 'mention', 'mentioned',
                'called', 'adding', 'added', 'ask', 'asking', 'leave',
                 'reading', 'talking', 'search', 'finding', 'remember',
                 'forgot', 'hear', 'heard', 'sounds', 'seems', 'feels',
                 'felt', 'thinking', 'understand', 'believe', 'consider',
                'pick', 'wait', 'hoping', 'wanting', 'works', 'stop',
                'start', 'agree', 'removed', 'keep', 'point', 'sort',
                'prefer', 'preferably', 'appreciate', 'appreciated',
                'open', 'add',
                # ── 通用名词（无类型信号）──
                'others', 'reason', 'matter', 'fact', 'case', 'idea',
                'ideas', 'sense', 'name', 'line', 'lines', 'side',
                'category', 'level', 'job', 'taste', 'quality',
                'opinion', 'perspective', 'situation', 'attention',
                'problem', 'question', 'chance', 'moment', 'moments',
                'week', 'country', 'city', 'town', 'world', 'home',
                 'house', 'room', 'school',
                 'men', 'women', 'boy', 'girl', 'relationship',
                # ── 通用形容词/副词（无类型信号）──
                'long', 'big', 'short', 'small', 'high', 'low', 'full',
                'entire', 'whole', 'certain', 'particular', 'specific',
                'general', 'personal', 'real', 'realistic', 'true',
                'actual', 'multiple', 'single', 'half', 'early', 'late',
                'lately', 'currently', 'past', 'future', 'recent',
                'modern', 'older', 'young', 'fast', 'slow', 'easy',
                 'hard', 'close', 'huge', 'crazy', 'wrong',
                 # ── 电影/平台元词 ──
                 'trailer', 'trilogy', 'reviews', 'rec', 'recs', 'ref',
                 'comment', 'listed', 'description', 'subtitles',
                # ── 时间/季节词（非类型信号）──
                'april', 'june', 'autumn', 'sept', 'today', 'tonight',
                # ── 国籍/语言（非类型信号）──
                'american', 'english', 'french', 'japanese', 'korean',
                # ── 极低频噪音（编码残留/极罕见人名）──
                'wordsextra', 'urxf', 'gcrd', 'seligman', 'noblewoman',
                 'nymphomaniac', 'champion', 'boot',
                # ── W4 CSV 中新增噪音词 ──
                # 编码残留（ratio > 1000 且 non_holiday_avg ≈ 0）
                 'domina', 'balrog', 'droplabs',
                'cornfield', 'otto', 'alteration', 'serp', 'gws', 'rlz',
                'enus', 'mtt', 'tik', 'jpg', 'rly', 'nhappy', 'njane',
                'crazed', 'mongols', 'wyatt', 'hobo', 'knockoffs',
                # 论坛缩写/俚语
                'reco', 'thx', 'umm', 'serie',
                # 月份/星期（非类型信号）
                'dec', 'oct', 'feb', 'august', 'sunday', 'saturday',
                'monday', 'friday', 'thursday', 'tuesday', 'wednesday',
                # 地名（非类型信号）
                'texas', 'rome', 'greece', 'detroit', 'york', 'pacific',
                # 人名（非通用推荐信号）
                # 'neil', 'leslie', 'charlotte', 'bruno', 'hepburn', 'lars',
                # 'cary', 'jordan', 'reynolds', 'ruffalo', 'campbell',
                # 'churchill', 'norris', 'wayne', 'jerry', 'connor',
                # 通用名词（无类型偏好）
                'extended', 'edition', 'accurate', 'deserved', 'definition',
                'intention', 'teaching', 'incident', 'officer', 'articles',
                'table', 'field', 'poll', 'articles', 'native', 'wedding',
                'holiday', 'holidays', 'values', 'lessons', 'morning',
                'eight', 'sharing', 'disagree', 'replaced', 'mentor',
                 'mansion', 'bike', 'cinephiles',
                'library', 'elevator', 'gateway', 'cartel',
                'chocolate', 'cat', 'trans', 'anti', 'file',
                'surveillance', 'chainsaw', 'wolfenstein', 'synecdoche',
                'somthing', 'memorial', 'station', 'grant',
                'eleven', 'professor', 'captivating', 'disagree', 'summer',
                 'sibling', 'blue', 'intelligent', 'angry',
                 'muscular', 'honey', 'struck', 'scheming', 'infidelity',
                 'celebrate', 'desire',
                 'negotiation', 'puzzles',
                # ── 节假日自指词（非推荐信号）──
                 'christmas', 'halloween', 'thanksgiving',
                'valentines', 'valentine', 'easter', 'hanukkah',
                'holiday', 'holidays',
                 # ── 人名（继续停用）──
                 'neil', 'leslie', 'charlotte', 'von', 'joe',
                 'norris', 'jerry', 'lars',
                # ── 特定电影名（非通用偏好信号）──
                 'spiderman', 'enola', 'gladiator', 'lego', 'atmos',
                 'ranked',
                # ── 活动词 ──
                'camping', 'cake',
                 # ── 历史/政治/社会词 ──
                 'nazi', 'genocide', 'racist', 'politics',
                 # ── 高频噪声词补充（W1 top 500 中无 W6 信号的填充词）──
                 # 时态/时间填充
                 'now', 'since', 'ago', 'far', 'either', 'almost', 'enough',
                 'least', 'mostly', 'less', 'second', 'soon', 'yesterday',
                 'months', 'weeks', 'month', 'decade', 'later', 'finally',
                 'starting', 'decided', 'beginning', 'ended', 'starts',
                 'realize', 'realized', 'forward', 'despite', 'whether',
                 # 量词/限定词填充
                 'several', 'bunch', 'couple', 'ton', 'rest', 'deal',
                 'amount', 'common', 'various',
                 # 强化副词填充
                 'super', 'totally', 'extremely', 'highly', 'somewhat',
                 'somehow', 'possibly', 'generally', 'mainly', 'truly',
                 'genuinely', 'easily',
                 # 论坛/元信息填充
                 'idk', 'imo', 'spoilers', 'review', 'google', 'youtu',
                 'png', 'deleted', 'topic', 'subject', 'format', 'media',
                 'content', 'background',
                 # 通用动词填充
                 'mean', 'trying', 'having', 'coming', 'happening', 'showing',
                 'checking', 'describe', 'meant', 'focused', 'anymore',
                 'giving', 'tend', 'stuck', 'working', 'plays', 'played',
                 'playing', 'running',
                 # 通用名词填充
                 'view', 'terms', 'words', 'element', 'elements', 'points',
                 'head', 'middle', 'figure', 'choice', 'order', 'cause',
                 # 通用形容词填充
                 'different', 'available', 'familiar', 'normal', 'obvious',
                 'simple', 'strange', 'famous',
                 # ── P1: 编码残留补充（W1 高频 n 前缀噪声）──
                 'nplease', 'nlooking', 'nnot', 'njust', 'nin', 'pas',
                 # ── P1: 人名补充（W1 高频人名，非通用推荐信号）──
                 'john', 'david', 'tarantino', 'tom', 'michael', 'anderson',
                 'jones', 'bruno', 'hepburn', 'cary', 'jordan', 'reynolds',
                 'ruffalo', 'campbell', 'churchill', 'wayne', 'connor',
                 # ── P1: 地名/国籍补充（非类型信号）──
                 'british', 'asian', 'indian', 'german', 'spanish', 'america',
                 # ── P2: 通用高频动词补充（W1 高频、无类型信号）──
                 # 注意：act/star/move/direct/write/effect/visually/end 等已用作
                 # 关键词表的 lemma 形式，不能加入停用词，否则会反向杀死信号
                 'interest', 'set', 'mind', 'work', 'live', 'top', 'base',
                 'talk', 'play', 'forget', 'read', 'call', 'remove', 'turn',
                 'game', 'fall', 'change', 'feature', 'lose',
                 'miss', 'must', 'worth', 'include', 'involve', 'care', 'free',
                 'run', 'video', 'hour', 'win', 'plan', 'event', 'expect',
                 'learn', 'remind', 'hide', 'build', 'decide', 'aspect', 'stick',
                 'begin', 'explore', 'meet', 'catch', 'bring', 'grow', 'society',
                 'reality', 'culture', 'detail', 'reference', 'stand',
                 'none', 'drive', 'nature', 'face', 'speak',
                 'stay', 'hold', 'secret', 'note', 'sit', 'cut', 'force', 'hand',
                 'exist', 'mix', 'create', 'issue', 'class',
                 'itself', 'overall', 'service', 'weekend', 'save', 'car', 'reply',
                 'answer', 'lesser', 'choose', 'color', 'male', 'avoid',
                 'spend', 'discover', 'project', 'limit', 'towards', 'burn',
                 'location', 'money', 'throw', 'train', 'capture', 'escape', 'depict',
                 'present', 'body', 'notice', 'smart', 'fill', 'count', 'pay', 'engage',
                 'touch', 'channel', 'rich', 'listen', 'link', 'glad', 'major',
                 'vein', 'century', 'hunt', 'message', 'source', 'study',
                 'greatly', 'newer', 'deserve', 'buy', 'cold', 'aware',
                 'compare', 'sleep', 'form', 'rule', 'picture',
                 'complete', 'center', 'trip', 'cover', 'straight', 'delete', 'draw',
                 'walk', 'himself', 'box', 'clear', 'brain', 'pop', 'absolute', 'regard',
                 'viewer', 'mess', 'perfectly', 'animal', 'memory', 'street', 'lack',
                 'pull', 'promise', 'simply', 'wild', 'drop', 'reveal', 'trouble',
                 'somewhere', 'third', 'contain', 'kick', 'difficult',
                  'hook', 'cannot', 'btw',
                 # ── P2: 通用高频形容词补充（无类型信号）──
                 'black', 'human', 'white', 'sick', 'insane',
                }

ALL_STOPWORDS = STOPWORDS | DOMAIN_STOP       # 合并停用词总表


# ── 词频计算 ────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Tokenize text using shared utility function + domain stopwords.
       分词：使用共享工具 + 领域停用词表。

    本函数覆盖（shadow）了模块顶部从 movie.utils.text 导入的同名 tokenize，
    目的是将本模块自定义的 ALL_STOPWORDS（标准停用词 + 领域噪音词）注入分词器，
    使后续所有 compute_word_freq 调用自动剔除噪音词，无需重复传参。
    """
    from movie.utils.text import tokenize as _tokenize
    return _tokenize(text, stopwords=ALL_STOPWORDS)


def compute_word_freq(seekers: list[dict], date_set: set = None) -> Counter:
    """Compute word frequency from seekers matching date_set.
       计算指定日期范围内用户提问的高频词。
    Args:
        seekers: 用户提问记录列表
        date_set: 可选，日期集合过滤器（None 表示不过滤，统计全部）
    Returns:
        Counter of word frequencies. 词频计数器，已过滤极低频词。
    """
    counter: Counter = Counter()
    for r in seekers:
        if date_set is not None and r['date'] not in date_set:
            continue                            # 按日期过滤：仅统计 date_set 内的提问
        # 优先使用处理后的文本（proc_text，已规范化分词/小写化），没有则回退到原始文本
        text = r.get('proc_text', '')
        if not text:
            text = r.get('raw_text', '')
        tokens = tokenize(text)
        counter.update(tokens)                  # 累加该条提问的所有 token
    # 最低词频过滤：移除总频次 < MIN_TF 的极低频词
    # 极低频词通常是偶然提及/分词噪音/罕见人名，不携带稳定的偏好信号
    MIN_TF = 3
    counter = Counter({w: c for w, c in counter.items() if c >= MIN_TF})
    return counter


def compute_word_freq_by_period(
    seekers: list[dict], period: str
) -> Counter:
    """Compute word frequency for a specific period (holiday/workday/weekend).
       计算特定时段（节假日/工作日/周末）的高频词。"""
    # 收集该时段的所有日期
    dates = set(r['date'] for r in seekers if r['period'] == period)
    return compute_word_freq(seekers, dates)


# deduplicate_seekers is imported from movie.utils.text


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
    # 地板值：非节假日最小非零日均值，避免零出现时倍数膨胀
    nh_nonzero = [v for v in nh_avg.values() if v > 0]
    FLOOR = min(nh_nonzero) if nh_nonzero else 0.01
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
        ratio = hv / max(nhv, FLOOR)
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
#  W1: 全局高频词统计和词云
#  W1: Overall Word Frequency & Word Cloud
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】词云图 + 高频词列表
# 【统计口径】
#   对所有用户的 proc_text 进行词频统计
#   使用 WordCloud 库生成词云图
#   同时输出 CSV 格式的词频表
# 【输出文件】PNG: w1_overall_wordcloud.png, CSV: w1_overall_word_freq.csv
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


# ═══════════════════════════════════════════════════════════════════════
#  W2: 节假日 VS 非节假日 词频对比 (Bar)
#  W2: Holiday vs Non-Holiday Word Frequency
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】柱状图/表格: 对比节假日与非节假日的 top 词及比例
# 【统计口径】
#   按 period 分组计算词频
#   计算 ratio = holiday_freq / non_holiday_freq
#   筛选 ratio ≥ ratio_threshold(1.5) 或 ≤ 1/ratio_threshold 的差异词
# 【输出文件】CSV: w2_holiday_vs_nonholiday_words.csv
# 【特殊说明】_find_keywords() 辅助函数查找差异关键词
# ═══════════════════════════════════════════════════════════════════════

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
    # 非节假日 = 工作日 + 周末（period 字段只有 holiday/workday/weekend 三种取值）
    nh_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_freq = compute_word_freq(seekers, nh_dates)

    # 按日期数归一化到日均词频：消除节假日天数 ≠ 非节假日天数带来的总量偏差
    # 例如非节假日有 1800+ 天，节假日只有几十天，直接比 raw 频次毫无意义
    h_dates = set(r['date'] for r in seekers if r['period'] == 'holiday')

    h_avg = {w: c / max(len(h_dates), 1) for w, c in h_freq.items()}
    nh_avg = {w: c / max(len(nh_dates), 1) for w, c in nh_freq.items()}
    # 地板值 FLOOR：取非节假日最小非零日均频次，作为分母下限
    # 作用：当某词在非节假日基线为 0 时，避免 ratio = holiday / 0 → ∞ 的膨胀
    # 这样得到的 ratio 仍有可比性（不会因基线为零就飙到几十万倍）
    nh_nonzero = [v for v in nh_avg.values() if v > 0]
    FLOOR = min(nh_nonzero) if nh_nonzero else 0.01

    plot_top_words_bar(
        {'Holiday': h_avg, 'Non-holiday': nh_avg},
        'Top Words: Holiday vs Non-Holiday (Avg Daily)',
        'w2_holiday_vs_nonholiday_words.png',
    )

    # 绘制节假日日均显著高于非节假日的词（默认阈值 1.5 倍，可由参数调整）
    plot_holiday_elevated_words(h_avg, nh_avg, threshold=ratio_threshold,
                                filename='w2_holiday_elevated_words.png')

    # 输出节日特定高频词（ratio > 2x baseline，且节假日绝对频次 >= 5 防止偶发）
    log("  Top holiday-specific words (ratio > 2x baseline):")
    ratio_words = []
    for w in h_freq:
        ratio = h_avg.get(w, 0) / max(nh_avg.get(w, 0), FLOOR)  # 分母取 FLOOR 下限
        if ratio > 2.0 and h_freq[w] >= 5:         # 比值 > 2 且节假日频次 >= 5
            ratio_words.append((w, ratio, h_freq[w], nh_freq.get(w, 0)))
    ratio_words.sort(key=lambda x: x[1], reverse=True)
    for w, r, hc, nhc in ratio_words[:20]:
        log(f"    {w}: holiday={hc}, non-holiday={nhc}, ratio={r:.2f}")

    _save_word_csv('w2_holiday_vs_nonholiday_words.csv',
                   {'holiday': h_freq, 'non_holiday': nh_freq})


# ═══════════════════════════════════════════════════════════════════════
#  W3: 节假日 VS 工作日 VS 周末 词频对比
#  W3: Holiday vs Workday vs Weekend Word Frequency
# ═══════════════════════════════════════════════════════════════════════
# 【统计口径】3组(holiday/workday/weekend) 词频对比
# 【输出文件】CSV: w3_holiday_workday_weekend_words.csv
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
#  W4: 各节假日词频 VS 非节假日基线 (Bar per Holiday)
#  W4: Per-Holiday Word Frequency vs Non-Holiday
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】每个节假日一个水平柱状图: 显示 top_n 差异词及倍数
# 【统计口径】_find_keywords() 计算各节假日 vs 非节假日的词频倍率
# 【输出文件】PNG: w4_* (每个节假日独立图片), CSV: w4_*.csv
# ═══════════════════════════════════════════════════════════════════════

def dim_w4_per_holiday_words(seekers: list[dict], top_n: int = 30):
    """Per-holiday word frequency vs non-holiday baseline (one bar chart per holiday).
        各个节假日词频 vs 非节假日基线（每个节假日一个柱状图）。


    For each holiday, plots the top N words with highest composite score
    (holiday avg daily freq × log2(1 + fold_ratio)), fully sorted by score.
    每个节假日展示综合得分最高的 N 个词，得分为日均频次 × log2(1 + 倍数)。
    """
    log("=" * 50)
    log("W4: Per-Holiday Word Frequency vs Non-Holiday")

    nh_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_freq = compute_word_freq(seekers, nh_dates)
    num_nh = max(len(nh_dates), 1)
    nh_avg = {w: c / num_nh for w, c in nh_freq.items()}     # 非节假日日均词频基线

    # 地板值 FLOOR：取非节假日最小非零日均频次作为分母下限
    # 作用：当某词在非节假日基线为 0（从未被提及）时，避免 ratio = holiday/0 → ∞ 的膨胀
    # 取最小非零值而非任意固定值，是为了让 ratio 仍反映该词相对其他词的稀缺程度
    nh_nonzero_vals = [v for v in nh_avg.values() if v > 0]
    FLOOR = min(nh_nonzero_vals) if nh_nonzero_vals else 0.01

    # 按节假日名称分组（同一节假日跨多个年份汇总，便于稳定统计）
    holiday_groups = defaultdict(list)
    for r in seekers:
        if r['is_holiday']:
            name = r.get('holiday_name', '')[:8]     # 节假日名称截断到前 8 字符（按规则 14）
            holiday_groups[name].append(r)
    holiday_groups = {k: v for k, v in holiday_groups.items()
                      if len(v) >= MIN_DATA_ROWS}    # 过滤数据量不足的组，避免小样本噪音

    if not holiday_groups:
        log("  No holiday groups with sufficient data")
        return

    holiday_names = sorted(holiday_groups.keys())
    n_holidays = len(holiday_names)

    # 计算各节假日的日均词频 + 相对非节假日基线的倍数 ratio
    holiday_avg = {}     # {holiday_name: {word: 节假日日均词频}}
    holiday_ratio = {}   # {holiday_name: {word: ratio（节假日日均 / max(非节假日日均, FLOOR)）}}
    all_words = set()
    for hn in holiday_names:
        h_dates = set(r['date'] for r in holiday_groups[hn])
        hf = compute_word_freq(holiday_groups[hn], h_dates)
        h_d = max(len(h_dates), 1)
        ha = {w: c / h_d for w, c in hf.items()}    # 日均词频（消除节假日天数差异）
        holiday_avg[hn] = ha
        # ratio：节假日日均 / max(非节假日日均, FLOOR)
        # 分母取 max(..., FLOOR) 是关键：基线为零时用 FLOOR 代替 0，避免 ratio 爆炸
        holiday_ratio[hn] = {w: ha[w] / max(nh_avg.get(w, 0), FLOOR) for w in ha}
        all_words.update(hf.keys())

    # 最低文档频率过滤（MIN_DF）：只保留在 >= 2 个节假日中出现过的词
    # 作用：避免仅因个别用户在某一个节日偶然提及就进入分析结果
    MIN_DF = 2
    word_holiday_df = {}     # 文档频率：该词在多少个节假日中出现（>0）
    for w in all_words:
        word_holiday_df[w] = sum(1 for hn in holiday_names if holiday_avg[hn].get(w, 0) > 0)
    all_words = {w for w in all_words if word_holiday_df[w] >= MIN_DF}

    # ── CSV：词 × 节假日矩阵（含 ratio 列）──
    csv_path = os.path.join(STEP_OUT, 'w4_per_holiday_words.csv')
    # 综合得分公式：score = 日均词频(ha) × log2(1 + 倍数 ratio)
    # 设计意图（类 TF-IDF 思路）：
    #   - ha 项：奖励在节假日中确实高频的词（避免 ratio 高但绝对频次太低）
    #   - log2(1 + ratio) 项：对倍数做对数压缩，防止极高 ratio（如 100x）压制其他词
    #     ratio=1 → log2(2)=1（无差异时退化为纯频次）
    #     ratio=2 → log2(3)≈1.58（2 倍基线）
    #     ratio=8 → log2(9)≈3.17（8 倍基线）
    # 跨节假日取 max，使该词以其表现最强的节假日参与排序
    word_score = {}
    for w in all_words:
        max_score = 0
        for hn in holiday_names:
            ha = holiday_avg[hn].get(w, 0)
            if ha > 0:
                ratio = holiday_ratio[hn].get(w, 1)
                max_score = max(max_score, ha * math.log2(1 + ratio))
        if max_score > 0:
            word_score[w] = max_score

    # CSV 入选策略：各节假日 top 100 + 非节假日 top 100 的并集，保证兼顾节日特异词与高频通用词
    csv_top_n = 100
    selected_words = set()

    # 各节假日 top 100（按该节假日的综合得分 ha × log2(1 + ratio) 排序）
    for hn in holiday_names:
        scored = []
        for w, ha in holiday_avg[hn].items():
            ratio = holiday_ratio[hn].get(w, 1)
            scored.append((w, ha * math.log2(1 + ratio)))
        scored.sort(key=lambda x: x[1], reverse=True)
        for w, _ in scored[:csv_top_n]:
            selected_words.add(w)

    # 非节假日 top 100（按日均词频排序，反映基线本身的高频词）
    nh_scored = sorted(nh_avg.items(), key=lambda x: x[1], reverse=True)
    for w, _ in nh_scored[:csv_top_n]:
        selected_words.add(w)

    # 并集按全局综合得分降序排列（节日特异词在前，基线高频词在后）
    sorted_words = sorted(selected_words, key=lambda w: word_score.get(w, 0), reverse=True)

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
    log(f"Saved: {csv_path} ({len(sorted_words)} words)")

    # ── 各节假日柱状图（每个节假日一个子图）──
    n_cols = min(4, n_holidays)
    n_rows = (n_holidays + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 5))
    axes = axes.flatten() if n_holidays > 1 else [axes]

    for idx, hn in enumerate(holiday_names):
        ax = axes[idx]
        h_avg_dict = holiday_avg[hn]

        # 综合得分排序：score = 日均频次 ha × log2(1 + 倍数 ratio)
        # 与上方 CSV 排序公式一致，确保图表与 CSV 一致
        scored = []
        for w, ha in h_avg_dict.items():
            ratio = holiday_ratio[hn].get(w, 1)
            scored.append((w, ha, ratio, ha * math.log2(1 + ratio)))
        scored.sort(key=lambda x: x[3], reverse=True)
        top = scored[:top_n]

        if not top:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        # 反转列表：水平柱状图自上而下从大到小展示（最大的在顶部）
        words = [t[0] for t in top[::-1]]
        h_vals = [t[1] for t in top[::-1]]
        ratios = [t[2] for t in top[::-1]]
        nh_vals = [nh_avg.get(w, 0) for w in words]

        y = np.arange(len(words))
        bar_height = 0.35

        # 节假日日均词频柱（红色，位于 y 上半部）
        bars_h = ax.barh(y + bar_height / 2, h_vals, bar_height,
                         color='#ff6b6b', alpha=0.85, label='Holiday (avg/d)')
        # 非节假日基线柱（蓝色，位于 y 下半部）—— 与节假日柱并列对比
        ax.barh(y - bar_height / 2, nh_vals, bar_height,
                color='#74b9ff', alpha=0.85, label='Non-holiday (avg/d)')

        # 在节假日柱右端标注倍数（仅 ratio > 1.5 倍时显示，避免低差异词挤满标签）
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

    # 隐藏多余的子图（当节假日数不能整除 n_cols 时）
    for idx in range(n_holidays, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Per-Holiday Top Words vs Non-Holiday Baseline (sorted by score)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    chart_path = os.path.join(STEP_OUT, 'w4_per_holiday_bar_charts.png')
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved: {chart_path}")

    # ── 日志输出：各节假日 top 词汇（与前文公式一致，便于人工核对）──
    log("  Per-holiday top elevated words:")
    for hn in holiday_names:
        scored = [(w, holiday_avg[hn][w], holiday_ratio[hn].get(w, 0))
                  for w in holiday_avg[hn]
                  if holiday_avg[hn][w] >= 1]
        # 备选排序（截断 ratio 到 20）：scored.sort(key=lambda x: x[1] * min(x[2], 20), reverse=True)
        # 当前采用 log2(1 + ratio) 平滑：避免极高 ratio 主导排序
        scored.sort(key=lambda x: x[1] * math.log2(1 + x[2]), reverse=True)
        top = scored[:8]
        if top:
            log(f"    {hn}: {[(w, f'{h:.1f}/d', f'{r:.1f}x') for w, h, r in top]}")


# ═══════════════════════════════════════════════════════════════════════
#  W5: 各节假日 VS 非节假日 词频 log2 倍率热力图 (Heatmap)
#  W5: Per-Holiday Word Frequency Log2 Ratio Heatmap
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】热力图: 行=高频词, 列=节假日, 值=log2(holiday/non_holiday)
#   0 = 与基线相同, >0 = 节假日更突出, <0 = 非节假日更突出
# 【输出文件】PNG: w5_per_holiday_words_heatmap.png, CSV: w5_*.csv
# 【特殊说明】clustermap 行列聚类; 同时输出独立日志显示 top 词
# ═══════════════════════════════════════════════════════════════════════

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

    # EPSILON 地板值：取非节假日最小非零日均频次作为分子/分母下限
    # 作用：当 h_val 或 nh_val 为 0 时，避免 log2(0/x) = -∞ 或 log2(x/0) = +∞ 导致矩阵爆炸
    # 与 W2/W4 的 FLOOR 同思路，但在对数空间下用作加性平滑（additive smoothing）
    nh_dates = set(r['date'] for r in seekers if r['period'] != 'holiday')
    nh_freq = compute_word_freq(seekers, nh_dates)
    num_nh = max(len(nh_dates), 1)
    nh_avg = {w: c / num_nh for w, c in nh_freq.items()}
    nh_nonzero = [v for v in nh_avg.values() if v > 0]
    EPSILON = min(nh_nonzero) if nh_nonzero else 0.01

    # 按节假日名称分组（同 W4 逻辑）
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
    # 综合得分 = h_val × log2_ratio_capped，h_val 为节假日日均频次，log2_ratio_capped 为截断后的对数倍率
    word_score = {}
    for w in all_words:
        max_score = 0
        for hn in holiday_names:
            h_val = holiday_avg[hn].get(w, 0)
            # log2 倍率：log2((h + ε) / (nh + ε))，分子分母各加 ε 防止 log(0)
            log2r = np.log2((h_val + EPSILON) / (nh_avg.get(w, EPSILON) + EPSILON))
            # 截断到 [-3, 3]：对应 ratio ∈ [0.125x, 8x]，避免极少数极端值主导色阶
            log2r_capped = max(-3, min(3, log2r))
            # 双重过滤：log2r > 0.5（ratio > ~1.41x，确实升高）且 h_val >= 1（绝对频次不算太低）
            if log2r_capped > 0.5 and h_val >= 1:
                score = h_val * log2r_capped
                max_score = max(max_score, score)
        if max_score > 0:
            word_score[w] = max_score

    top_words = sorted(word_score, key=word_score.get, reverse=True)[:60]

    if len(top_words) < 3:
        log("  Too few words with elevated holiday frequency")
        return

    # 构建 log2 倍率矩阵：行=词，列=节假日，值=log2((h+ε)/(nh+ε))，截断到 [-3, 3]
    matrix = np.zeros((len(top_words), n))
    for j, hn in enumerate(holiday_names):
        for i, w in enumerate(top_words):
            h_val = holiday_avg[hn].get(w, EPSILON)
            nh_val = nh_avg.get(w, EPSILON)
            log2r = np.log2((h_val + EPSILON) / (nh_val + EPSILON))
            matrix[i, j] = max(-3, min(3, log2r))    # 截断保证色阶对称可比

    # 绘制热力图 — RdBu_r 配色 + 固定对称色阶 vmin=-3, vmax=3
    # 红色 = 节假日高于基线，蓝色 = 节假日低于基线，白色 = 与基线持平
    # 固定色阶使不同运行/不同节假日子集之间结果可比
    fig_w = max(12, n * 1.2)
    fig_h = max(10, len(top_words) * 0.28 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    annotate_heatmap(ax, matrix, fmt='.1f', fs=6)   # 在每格上叠加数值标注

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

# ══════════════════════════════════════════════════════════════════
# 以下 6 个字典定义了观影兴趣各维度的关键词映射表（W6 使用）。
# 每个分类包含一组相关英文词汇，用于识别用户讨论中涉及的方面。
# 词汇均为小写，以匹配分词器输出（tokenize 后已统一小写）。
# 评分逻辑：dim_w6 中将"节假日升高词"与各分类词典匹配，
# 累加该分类下所有匹配词的日均频次，得到该分类得分。
# ══════════════════════════════════════════════════════════════════

# ── 类型倾向词汇（电影题材/类型偏好）──
# 用于识别用户对某类电影题材的偏好
_GENRE_WORDS = {
    # 恐怖片：鬼魂、超自然、血腥、丧尸、吸血鬼、怪物等恐怖元素
    'Horror':     {'horror', 'creepy', 'spooky', 'ghost', 'haunt', 'haunted', 'haunting',
                   'paranormal', 'slasher', 'gore', 'slashers',
                   'scariest', 'horrify', 'horrifying', 'demonic', 'possession', 'zombie',
                   'zombies', 'vampire', 'vampires', 'werewolf',
                   'scary', 'scare', 'gory', 'monster', 'monsters',
                   'horrors', 'fear', 'supernatural', 'jumpscare', 'goosebumps',
                   'camps', 'surgeries'},
    # 喜剧片：幽默、搞笑、轻松、情景喜剧等喜剧元素
    'Comedy':     {'comedy', 'comedies', 'funny', 'humor', 'hilarious',
                   'comic', 'laugh', 'comedic', 'lighthearted', 'sitcom'},
    # 惊悚片：悬疑、反转、神秘、紧张感等惊悚元素
    'Thriller':   {'thriller', 'thrillers', 'suspense', 'suspenseful',
                   'twist', 'twists', 'mystery', 'mysteries', 'intense',
                   'tension', 'edge', 'thrill', 'thrilling', 'mysterious', 'paranoia',
                   'danger', 'pressure'},
    # 动作片：冒险、爆炸、超级英雄、战斗、战争、暴力、复仇等动作元素
    'Action':     {'action', 'adventure', 'explosions', 'superhero',
                   'superheroes', 'battle', 'war', 'fight', 'fighting', 'violent', 'revenge',
                   'race'},
    # 科幻片：科学、虚构、外星、太空、反乌托邦、时间旅行、赛博朋克等科幻元素
    # 注：'time' 在标准 STOPWORDS 中（太泛），不放入此表；'travel' 已覆盖时间旅行信号
    'Sci-Fi':     {'sci', 'science', 'fiction', 'futuristic', 'alien',
                   'aliens', 'space', 'dystopian', 'dystopia',
                   'travel', 'technology', 'cyberpunk', 'sci-fi',
                   'apocalyptic', 'universe', 'scifi', 'outbreak'},
    # 剧情片：情感、催泪、动人、悲剧、写实等戏剧元素
    'Drama':      {'drama', 'dramas', 'emotional', 'tearjerker', 'move', 'moving',
                   'heartfelt', 'tragic', 'gritty', 'dramatic', 'despair',
                   'betrayal', 'state', 'trend', 'faith', 'orphan', 'neighbor',
                   'nurse', 'theater', 'birthday'},
    # 爱情片：浪漫、爱情喜剧、约会等爱情元素
    'Romance':    {'romance', 'romantic', 'romcom', 'date', 'rom-com', 'love'},
    # 动画片：动画、卡通、皮克斯、日本动画等动画元素
    'Animation':  {'animate', 'animated', 'animation', 'cartoon', 'pixar', 'anime'},
    # 奇幻片：魔法、巫术、史诗、神话等奇幻元素
    'Fantasy':    {'fantasy', 'magical', 'magic', 'sorcery', 'epic',
                   'mythical', 'mythology', 'giant', 'wicked'},
    # 犯罪片：犯罪、谋杀、侦探、黑色电影、黑帮、抢劫、警察等犯罪元素
    'Crime':      {'crime', 'murder', 'detective', 'noir', 'gangster',
                   'mafia', 'heist', 'investigation', 'killer', 'kill',
                   'police', 'survival', 'hostage'},
    # 纪录片：纪录片、文献片
    'Documentary':{'documentary', 'documentaries', 'doc'},
    # 歌舞片：音乐剧、原声带
    'Musical':    {'musical', 'musicals', 'soundtrack', 'tracks'},
    # 西部片：西部题材
    'Western':    {'western', 'westerns'},
    # 历史片：历史题材、年代剧
    'Historical': {'historical', 'period'},
    # 青少年题材：青少年相关
    'Teen':       {'teen'},
}

# ── 观影情绪/氛围关键词（用户期望的情感基调）──
# 用于识别用户对影片情感氛围的偏好
_MOOD_WORDS = {
    # 温馨/家庭：舒适、温暖、治愈、家庭友好、欢乐、甜美等正向情绪
    'Cozy/Family':  {'cozy', 'warm', 'comfort', 'comforting', 'heartwarming',
                     'wholesome', 'festive', 'cheerful', 'merry', 'joy',
                     'joyful', 'happy', 'feel-good', 'sweet'},
    # 阴暗/沉重：黑暗、阴郁、压抑、扭曲、邪恶、超现实等负面情绪
    'Dark':         {'dark', 'grim', 'bleak', 'disturb', 'disturbing',
                     'darkness', 'sinister', 'evil', 'surreal'},
    # 振奋人心：鼓舞、励志、充满希望、积极、有力、美好等正向激励
    'Uplifting':    {'uplift', 'uplifting', 'inspire', 'inspiring', 'inspirational', 'hopeful',
                     'optimistic', 'positive', 'powerful', 'wonderful', 'dream'},
    # 轻松休闲：放松、平静、悠闲、温和、舒缓、不需思考等放松情绪
    'Relaxing':     {'relax', 'relaxing', 'calm', 'peaceful', 'chill', 'gentle',
                     'soothing', 'mindless'},
    # 紧张刺激：兴奋、惊险、紧张、意外、动作等高强度情绪
    'Exciting':     {'excite', 'exciting', 'intense', 'edge', 'action',
                     'surprise', 'surprised'},
    # 发人深省：引人深思、深度、哲学、复杂、深刻、有意义等思辨情绪
    'Thoughtful':   {'provoke', 'thought-provoking', 'deep', 'philosophical', 'complex',
                     'profound', 'meaningful'},
    # 怀旧情怀：怀旧、童年、复古、经典等回忆性情绪
    'Nostalgic':    {'nostalgia', 'nostalgic', 'childhood', 'retro',
                     'classic'},
    # 悲伤/抑郁：悲伤、压抑、悲剧、哭泣、哀愁、无聊、厌倦等负面情绪
    'Sad':          {'sad', 'depress', 'depressing', 'depression', 'tragic', 'cry',
                     'sorrow', 'melancholy', 'bore', 'boring', 'hate'},
    # 娱乐消遣：有趣、享受、娱乐性、消遣性等纯娱乐情绪
    'Fun/Enjoyable':{'fun', 'enjoy', 'enjoyed', 'enjoyable', 'entertain', 'entertaining'},
}

# ── 观影场景关键词（用户与谁一起看、什么情境下看）──
# 用于识别用户的观影场景与陪伴对象
_CONTEXT_WORDS = {
    # 家庭/儿童：与家人、父母、子女、儿童共同观看的家庭场景
    'Family/Kids': {'family', 'kids', 'children', 'parents', 'family-friendly',
                    'kid-friendly', 'grandparents', 'parent',
                    'kid', 'child', 'dad', 'father', 'mom', 'mother',
                    'brother', 'brothers', 'son', 'daughter'},
    # 朋友聚会：与朋友、群体、派对等社交场合共同观看
    'Friends/Social': {'friends', 'friend', 'group', 'party', 'together', 'mate'},
    # 约会场景：与伴侣、女友/男友、配偶等约会情境下观看
    'Date Night':  {'date', 'partner', 'girlfriend', 'boyfriend', 'spouse',
                    'husband', 'wife', 'significant'},
    # 刷剧/连续剧： binge-watching、剧集、季、集、马拉松式观看
    'Binge/Series':{'binge', 'series', 'show', 'shows', 'season', 'episode',
                    'marathon', 'episodes', 'seasons'},
    # 重温/重看：重新观看、回顾、再次欣赏
    'Rewatch':     {'rewatch', 'rewatching', 'rewatched', 'revisit'},
    # 独自观影：独自、一个人、安静观看的场景
    'Alone/Quiet': {'alone', 'solo', 'myself'},
}

# ── 视频平台关键词（流媒体平台提及）──
# 用于识别用户讨论中提及的具体流媒体平台
_PLATFORM_WORDS = {
    'Netflix':  {'netflix'},                          # Netflix 平台
    'Prime':    {'prime', 'amazon'},                  # Amazon Prime Video 平台
    'HBO':      {'hbo', 'max'},                       # HBO Max 平台
    'Disney+':  {'disney'},                           # Disney+ 平台
    'Hulu':     {'hulu'},                             # Hulu 平台
    'Apple TV': {'apple'},                            # Apple TV+ 平台
    'Streaming':{'streaming', 'stream'},             # 通用流媒体提及
}

# ── 影片品质/口碑关键词（用户对影片质量的评价倾向）──
# 用于识别用户对影片品质、口碑、知名度的偏好
_QUALITY_WORDS = {
    # 被低估的佳作：被低估、隐藏、宝石、未受赏识等"冷门好片"信号
    'Underrated Gems': {'underrate', 'underrated', 'hidden', 'gem', 'gems',
                        'underappreciate', 'underappreciated'},
    # 经典作品：经典、永恒、杰作、原创等"经典之作"信号
    'Classic':         {'classic', 'classics', 'timeless', 'masterpiece', 'masterpieces',
                       'original'},
    # 邪典/独立电影：邪典、地下、冷门、独立等"小众文艺"信号
    'Cult/Indie':      {'cult', 'underground', 'obscure', 'indie'},
    # 主流商业片：流行、主流、大片、爆款、系列、续集、翻拍等"商业化"信号
    'Mainstream':      {'popular', 'mainstream', 'blockbuster', 'hit',
                       'franchise', 'sequel', 'remake'},
    # 口碑极佳：惊人、超棒、奇幻、完美、出色、独特等高度好评
    'Great/Excellent': {'amazing', 'awesome', 'fantastic', 'perfect', 'brilliant',
                       'incredible', 'excellent', 'solid', 'decent', 'unique'},
    # 口碑不佳：糟糕、俗气、愚蠢、烂片等差评信号
    'Poor/Bad':       {'terrible', 'cheesy', 'dumb', 'stupid'},
}

# ── 叙事/制作方面关键词（影片制作与艺术元素）──
# 用于识别用户对影片叙事结构、制作工艺、艺术手法等维度的关注点
_NARRATIVE_WORDS = {
    # 剧情/故事：情节、故事叙述、剧本、前提、概念、旅程等叙事框架
    'Plot/Story':     {'plot', 'story', 'storytelling', 'narrative', 'write', 'writing',
                       'stories', 'storyline', 'premise', 'plots', 'concept', 'journey'},
    # 结局：结局、终章、高潮、结论等结尾处理
    'Ending':         {'end', 'ending', 'finale', 'climax', 'conclusion', 'endings'},
    # 角色塑造：角色、人物刻画、主角、配角、反派、英雄等人物维度
    'Characters':     {'character', 'characters', 'characterization',
                       'protagonist', 'protagonists', 'cast', 'villain', 'hero'},
    # 摄影/视觉：摄影、视觉效果、镜头、画面、美学、特效、CGI、风格等画面维度
    'Cinematography': {'cinematography', 'visuals', 'visual', 'shot', 'shots',
                       'cinematographic', 'beautiful', 'camera', 'visually',
                       'aesthetic', 'screen', 'effect', 'effects', 'cgi', 'style'},
    # 配乐/音效：原声带、配乐、音乐、声音、歌曲等听觉维度
    'Music/Audio':    {'soundtrack', 'score', 'music', 'sound', 'song'},
    # 表演/演员：演技、表演、演员、导演、主演、角色等演职人员维度
    'Acting':         {'act', 'acting', 'performance', 'performances', 'actor',
                       'actors', 'actress', 'director', 'directors',
                       'direct', 'directed', 'lead', 'role', 'star', 'starring'},
    # 氛围/基调：氛围、格调、情绪、基调、设定、主题等整体调性
    'Atmosphere':     {'atmosphere', 'vibe', 'vibes', 'mood', 'tone', 'ambiance',
                       'setting', 'theme', 'themes'},
}


def _score_categories(
    word_freq: dict[str, float],
    cat_map: dict[str, set[str]],
) -> dict[str, float]:
    """Score each category by summing avg daily freq of its matched words.
       计算每个分类的得分：将分类中匹配词汇的日均频次累加求和。

    评分公式：category_score = Σ freq(w) for w in word_freq if w in keywords
    即：遍历 word_freq 中的每个词，若该词落入某分类的关键词集合，
    则把它的日均频次累加到该分类得分上。返回所有得分 > 0 的分类。
    """
    scores = {}
    for cat, keywords in cat_map.items():
        total = 0.0
        for w, f in word_freq.items():
            if w in keywords:
                total += f
        if total > 0:
            scores[cat] = total
    return scores


# ═══════════════════════════════════════════════════════════════════════
#  W6: 各节假日观影特征归纳 (Category Scores)
#  W6: Holiday Viewing Profile Categorization
# ═══════════════════════════════════════════════════════════════════════
# 【图表类型】特征得分表/雷达图
# 【统计口径】对 W2/W4 筛选出的差异词进行人工分类(类别词库匹配)
#   类别: family, comedy, action, romance, holiday_themed, animation, documentary 等
#   计算各类别的总得分 = 该类别下所有词的日均提及次数之和
# 【输出文件】CSV: w6_holiday_viewing_profile.csv
# 【特殊说明】使用预定义的 category_keywords 字典进行匹配
# ═══════════════════════════════════════════════════════════════════════

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
    # 节假日的分类得分基于"升高词"（elevated words）：
    #   定义：节假日日均 > 非节假日日均 × 1.5（即比基线高 50% 以上）
    #   且节假日日均绝对值 >= 0.5（避免极低频词偶然升高进入画像）
    # 然后将升高词与各分类词典匹配，累加匹配词的日均频次得到分类得分
    cat_groups = {
        'Genre':      _GENRE_WORDS,        # 类型倾向（恐怖/喜剧/动作等 15 类）
        'Mood':       _MOOD_WORDS,         # 观影情绪（温馨/阴暗/振奋等 9 类）
        'Context':    _CONTEXT_WORDS,      # 观影场景（家庭/朋友/约会等 6 类）
        'Platform':   _PLATFORM_WORDS,     # 流媒体平台（Netflix/HBO 等 7 类）
        'Quality':    _QUALITY_WORDS,      # 品质口碑（被低估佳作/经典/邪典等 6 类）
        'Narrative':  _NARRATIVE_WORDS,    # 叙事制作（剧情/结局/角色/摄影等 7 类）
    }

    csv_rows = []
    log_lines = []

    for hn in holiday_names:
        hn_short = hn[:8]
        ha = holiday_avg[hn]

        # 筛选"升高词"：节假日日均 > 非节假日基线 × 1.5 且绝对值 >= 0.5
        # 双重过滤：1.5x 倍率过滤掉与基线持平的词，0.5 绝对值过滤掉极低频噪音
        elevated = {}
        for w, h_val in ha.items():
            nh_val = nh_avg.get(w, 0)
            if h_val > nh_val * 1.5 and h_val >= 0.5:
                elevated[w] = h_val
        if not elevated:
            continue

        # 对各分类组评分：每个分类组下按得分排名取 top3
        profile_parts = []
        profile_data = {'holiday': hn_short}
        max_score_all = 0  # 记录该节假日所有分类中的最高得分（用于后续可靠性过滤）

        for group_name, cat_map in cat_groups.items():
            scores = _score_categories(elevated, cat_map)
            if scores:
                # 按得分降序排列，取 top3 分类
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top3 = ranked[:3]
                for rank, (cat, score) in enumerate(top3, 1):
                    profile_data[f'{group_name}_top{rank}'] = cat
                    profile_data[f'{group_name}_top{rank}_score'] = round(score, 1)
                    if score > max_score_all:
                        max_score_all = score
                profile_parts.append(f"{group_name}: {', '.join(c for c, _ in top3)}")
            else:
                # 该分类组无任何匹配词，填充空值保持 CSV 列对齐
                for rank in range(1, 4):
                    profile_data[f'{group_name}_top{rank}'] = ''
                    profile_data[f'{group_name}_top{rank}_score'] = 0.0

        # 最低得分阈值过滤：所有分类最高得分 < 3 则跳过该节假日
        # 原因：得分过低说明升高词太少或日均频次太低，画像不可靠、易误导
        if max_score_all < 3:
            log(f"  ── {hn_short} ── (skipped: max score {max_score_all:.1f} < 3, insufficient data)")
            continue

        # 收集匹配的关键词实例（用于人工核对画像是否合理）
        # 每个分类最多列 10 个匹配词，拼接成可读字符串
        matched_keywords = []
        for group_name, cat_map in cat_groups.items():
            for cat, keywords in cat_map.items():
                hits = [w for w in elevated if w in keywords]
                if hits:
                    matched_keywords.append(f"{cat}: {', '.join(sorted(hits)[:10])}")
        profile_data['matched_keywords'] = ' | '.join(matched_keywords)

        csv_rows.append(profile_data)

        # 构建可读的摘要输出（控制台日志）
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

    # ── 关键日志：各节假日去重后用户提问数统计 ──
    # 同一会话中提问内容一致的已通过 deduplicate_seekers 去重
    holiday_counts = defaultdict(int)
    nh_count = 0
    for r in seekers:
        if r['is_holiday']:
            name = r.get('holiday_name', '')[:8]
            holiday_counts[name] += 1
        else:
            nh_count += 1
    log("── 节假日去重后用户提问数统计 ──")
    log(f"  非节假日提问总数: {nh_count}")
    for hn in sorted(holiday_counts.keys()):
        log(f"  {hn}: {holiday_counts[hn]}")
    log(f"  节假日提问总计: {sum(holiday_counts.values())}")
    log("────────────────────────────")

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
