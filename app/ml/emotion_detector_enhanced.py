"""
增强版情感检测模块
基于预训练 BERT 模型 + 规则增强的多策略情感分析

主要改进：
1. 使用预训练的中文/多语言情感分类模型
2. 支持多种模型后端（Transformers, 在线API）
3. 否定/讽刺检测
4. 情感强度量化
5. 上下文感知的情感分析
6. 完整的评估指标支持
"""

import re
import json
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据类 ====================

@dataclass
class EmotionLabel:
    """情感标签"""
    name: str
    confidence: float
    intensity: float = 0.5  # 0-1 强度


@dataclass
class EmotionResult:
    """情感检测结果"""
    primary_emotion: str
    confidence: float
    intensity: float
    all_emotions: Dict[str, float]
    valence: float = 0.0  # -1 (负面) 到 1 (正面)
    arousal: float = 0.0  # 0 (平静) 到 1 (激动)
    needs_support: bool = False
    negation_detected: bool = False
    sarcasm_probability: float = 0.0
    raw_model_output: Optional[Dict] = None


@dataclass
class EmotionMetrics:
    """情感检测评估指标"""
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None


# ==================== 情感检测器抽象基类 ====================

class BaseEmotionDetector(ABC):
    """情感检测器抽象基类"""
    
    # 标准情感类别映射到 VAD 空间
    EMOTION_VAD = {
        "joy": {"valence": 0.9, "arousal": 0.7},
        "hopeful": {"valence": 0.7, "arousal": 0.5},
        "curious": {"valence": 0.5, "arousal": 0.6},
        "neutral": {"valence": 0.0, "arousal": 0.3},
        "confusion": {"valence": -0.2, "arousal": 0.4},
        "anxiety": {"valence": -0.6, "arousal": 0.8},
        "sadness": {"valence": -0.8, "arousal": 0.2},
        "anger": {"valence": -0.7, "arousal": 0.9},
        "fear": {"valence": -0.8, "arousal": 0.8},
    }
    
    @abstractmethod
    def detect(self, text: str) -> EmotionResult:
        """检测文本情感"""
        pass
    
    def detect_batch(self, texts: List[str]) -> List[EmotionResult]:
        """批量检测"""
        return [self.detect(text) for text in texts]
    
    def _get_vad(self, emotion: str) -> Tuple[float, float]:
        """获取情感的 VAD 值"""
        vad = self.EMOTION_VAD.get(emotion, {"valence": 0.0, "arousal": 0.3})
        return vad["valence"], vad["arousal"]


# ==================== BERT 情感检测器 ====================

class BERTEmotionDetector(BaseEmotionDetector):
    """
    基于 BERT 的情感检测器
    支持多种预训练中文情感分类模型
    """
    
    # 推荐的预训练模型
    PRETRAINED_MODELS = {
        # 中文情感分类 - 使用更稳定可用的模型
        "chinese_sentiment": "uer/roberta-base-finetuned-jd-binary-chinese",
        "chinese_emotion": "touch20032003/xuyuan-trial-sentiment-bert-chinese",  # 8类中文情感
        "chinese_emotion_alt": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        # 多语言情感 - 这个模型比较稳定
        "multilingual": "nlptown/bert-base-multilingual-uncased-sentiment",
        "multilingual_sentiment": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        # 英文情感
        "english_emotion": "j-hartmann/emotion-english-distilroberta-base",
    }
    
    # 模型输出标签到标准情感的映射
    LABEL_MAPPINGS = {
        "chinese_sentiment": {
            "positive": "joy",
            "negative": "sadness",
            "LABEL_0": "sadness",
            "LABEL_1": "joy",
        },
        "chinese_emotion": {
            # touch20032003/xuyuan-trial-sentiment-bert-chinese 实际输出的8类标签
            "happiness": "joy",
            "sadness": "sadness",
            "fear": "fear",
            "disgust": "anger",
            "anger": "anger",
            "like": "hopeful",
            "surprise": "curious",
            "none": "neutral",
        },
        "chinese_emotion_alt": {
            "positive": "joy",
            "negative": "sadness",
            "neutral": "neutral",
        },
        "multilingual": {
            "1 star": "sadness",
            "2 stars": "anxiety",
            "3 stars": "neutral",
            "4 stars": "hopeful",
            "5 stars": "joy",
        },
        "multilingual_sentiment": {
            "positive": "joy",
            "negative": "sadness",
            "neutral": "neutral",
            "LABEL_0": "sadness",
            "LABEL_1": "neutral",
            "LABEL_2": "joy",
        },
        "english_emotion": {
            "anger": "anger",
            "disgust": "anger",
            "fear": "fear",
            "joy": "joy",
            "neutral": "neutral",
            "sadness": "sadness",
            "surprise": "curious",
        }
    }
    
    def __init__(
        self,
        model_name: str = "chinese_emotion",
        device: str = "auto",
        use_rule_enhancement: bool = True,
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            model_name: 模型名称（预定义 key 或 HuggingFace 路径）
            device: 运行设备
            use_rule_enhancement: 是否使用规则增强
            confidence_threshold: 置信度阈值
        """
        self.model_path = self.PRETRAINED_MODELS.get(model_name, model_name)
        self.model_key = model_name if model_name in self.PRETRAINED_MODELS else "custom"
        self.device = device
        self.use_rule_enhancement = use_rule_enhancement
        self.confidence_threshold = confidence_threshold
        
        self.model = None
        self.tokenizer = None
        self._pipeline = None
        
        # 规则增强组件
        if use_rule_enhancement:
            self.rule_detector = RuleBasedEmotionDetector()
            self.negation_handler = NegationHandler()
            self.sarcasm_detector = SarcasmDetector()
    
    def _load_model(self):
        """延迟加载模型，支持自动回退到备选模型"""
        if self._pipeline is not None:
            return
        
        try:
            from transformers import pipeline
            import torch
            
            # 确定设备
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 定义模型加载顺序（主模型 + 备选模型）
            models_to_try = [(self.model_key, self.model_path)]
            
            # 添加备选模型
            fallback_models = [
                ("multilingual_sentiment", "cardiffnlp/twitter-xlm-roberta-base-sentiment"),
                ("multilingual", "nlptown/bert-base-multilingual-uncased-sentiment"),
                ("chinese_sentiment", "uer/roberta-base-finetuned-jd-binary-chinese"),
            ]
            
            for key, path in fallback_models:
                if path != self.model_path:
                    models_to_try.append((key, path))
            
            # 尝试加载模型
            last_error = None
            for model_key, model_path in models_to_try:
                try:
                    logger.info(f"Loading emotion model: {model_path}")
                    
                    # 使用 pipeline 简化推理
                    self._pipeline = pipeline(
                        "text-classification",
                        model=model_path,
                        tokenizer=model_path,
                        device=0 if self.device == "cuda" else -1,
                        top_k=None,  # 返回所有类别的概率
                    )
                    
                    # 成功加载，更新模型key以便正确映射标签
                    self.model_key = model_key
                    self.model_path = model_path
                    logger.info(f"Model loaded successfully on {self.device}")
                    return
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to load {model_path}: {e}")
                    continue
            
            # 所有模型都失败了
            raise RuntimeError(f"Failed to load any emotion model. Last error: {last_error}")
            
        except ImportError:
            raise ImportError(
                "transformers is required. "
                "Install with: pip install transformers torch"
            )
    
    def detect(self, text: str) -> EmotionResult:
        """检测文本情感"""
        self._load_model()
        
        # 1. BERT 模型预测
        model_output = self._pipeline(text[:512])  # 截断到最大长度
        
        # 解析模型输出
        emotion_scores = self._parse_model_output(model_output)
        
        # 2. 规则增强（如果启用）
        if self.use_rule_enhancement:
            # 检测否定
            negation_detected = self.negation_handler.has_negation(text)
            if negation_detected:
                emotion_scores = self._apply_negation(emotion_scores)
            
            # 检测讽刺
            sarcasm_prob = self.sarcasm_detector.detect(text)
            
            # 融合规则检测结果
            rule_result = self.rule_detector.detect(text)
            emotion_scores = self._fuse_with_rules(emotion_scores, rule_result)
            
            # 检测是否需要支持
            needs_support = self._check_distress(text)
        else:
            negation_detected = False
            sarcasm_prob = 0.0
            needs_support = False
        
        # 3. 确定主要情感
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[primary_emotion]
        
        # 4. 计算 VAD
        valence, arousal = self._calculate_vad(emotion_scores)
        
        # 5. 计算强度
        intensity = self._calculate_intensity(text, emotion_scores)
        
        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=confidence,
            intensity=intensity,
            all_emotions=emotion_scores,
            valence=valence,
            arousal=arousal,
            needs_support=needs_support,
            negation_detected=negation_detected,
            sarcasm_probability=sarcasm_prob,
            raw_model_output=model_output
        )
    
    def _parse_model_output(self, output: List[Dict]) -> Dict[str, float]:
        """解析模型输出为标准情感分数"""
        label_map = self.LABEL_MAPPINGS.get(self.model_key, {})
        
        scores = defaultdict(float)
        raw_scores = {}  # 保存原始标签分数
        
        for item in output:
            if isinstance(item, list):
                # 处理 top_k=None 返回的列表格式
                for sub_item in item:
                    label = sub_item.get("label", "")
                    score = sub_item.get("score", 0.0)
                    raw_scores[label] = score
                    
                    # 映射到标准情感
                    mapped_label = label_map.get(label, label.lower())
                    if mapped_label in self.EMOTION_VAD:
                        scores[mapped_label] = max(scores[mapped_label], score)
            else:
                label = item.get("label", "")
                score = item.get("score", 0.0)
                raw_scores[label] = score
                mapped_label = label_map.get(label, label.lower())
                if mapped_label in self.EMOTION_VAD:
                    scores[mapped_label] = max(scores[mapped_label], score)
        
        # 检查是否已经有有效的映射结果
        has_valid_scores = any(s > 0.01 for s in scores.values())
        
        # 只有当没有有效映射且是简单的2-3类模型时，才进行扩展处理
        if not has_valid_scores and len(raw_scores) <= 3:
            # 尝试多种标签格式
            pos_score = raw_scores.get("positive", 0.0)
            neg_score = raw_scores.get("negative", 0.0)
            neu_score = raw_scores.get("neutral", 0.0)
            
            # LABEL_X 格式
            if pos_score == 0 and neg_score == 0 and neu_score == 0:
                neg_score = raw_scores.get("LABEL_0", 0.0)
                neu_score = raw_scores.get("LABEL_1", 0.0)
                pos_score = raw_scores.get("LABEL_2", 0.0)
            
            # 1-5 stars 格式
            if pos_score == 0 and neg_score == 0 and neu_score == 0:
                neg_score = raw_scores.get("1 star", 0.0) + raw_scores.get("2 stars", 0.0)
                neu_score = raw_scores.get("3 stars", 0.0)
                pos_score = raw_scores.get("4 stars", 0.0) + raw_scores.get("5 stars", 0.0)
            
            # 扩展到更丰富的情感类别
            if pos_score > 0 or neg_score > 0 or neu_score > 0:
                scores["joy"] = pos_score * 0.6
                scores["hopeful"] = pos_score * 0.3
                scores["curious"] = pos_score * 0.1
                scores["sadness"] = neg_score * 0.4
                scores["anxiety"] = neg_score * 0.35
                scores["anger"] = neg_score * 0.15
                scores["fear"] = neg_score * 0.1
                scores["neutral"] = neu_score * 0.7
                scores["confusion"] = neu_score * 0.3
        
        # 确保所有标准情感都有分数
        for emotion in self.EMOTION_VAD:
            if emotion not in scores:
                scores[emotion] = 0.0
        
        # 归一化
        total = sum(scores.values()) + 1e-8
        return {k: v / total for k, v in scores.items()}
    
    def _apply_negation(self, scores: Dict[str, float]) -> Dict[str, float]:
        """应用否定转换"""
        # 否定会翻转正负情感
        negation_map = {
            "joy": "sadness",
            "sadness": "joy",
            "hopeful": "anxiety",
            "anxiety": "hopeful",
        }
        
        new_scores = scores.copy()
        for pos, neg in negation_map.items():
            if pos in scores and neg in scores:
                # 交换分数的一部分
                swap_factor = 0.5
                pos_score = scores[pos]
                neg_score = scores[neg]
                new_scores[pos] = pos_score * (1 - swap_factor) + neg_score * swap_factor
                new_scores[neg] = neg_score * (1 - swap_factor) + pos_score * swap_factor
        
        return new_scores
    
    def _fuse_with_rules(
        self,
        model_scores: Dict[str, float],
        rule_result: Dict[str, float],
        model_weight: float = 0.7
    ) -> Dict[str, float]:
        """融合模型和规则检测结果"""
        fused = {}
        all_emotions = set(model_scores.keys()) | set(rule_result.keys())
        
        for emotion in all_emotions:
            model_score = model_scores.get(emotion, 0.0)
            rule_score = rule_result.get(emotion, 0.0)
            fused[emotion] = model_score * model_weight + rule_score * (1 - model_weight)
        
        # 归一化
        total = sum(fused.values()) + 1e-8
        return {k: v / total for k, v in fused.items()}
    
    def _calculate_vad(self, scores: Dict[str, float]) -> Tuple[float, float]:
        """计算加权 VAD 值"""
        valence = 0.0
        arousal = 0.0
        
        for emotion, score in scores.items():
            v, a = self._get_vad(emotion)
            valence += v * score
            arousal += a * score
        
        return valence, arousal
    
    def _calculate_intensity(
        self,
        text: str,
        scores: Dict[str, float]
    ) -> float:
        """计算情感强度"""
        # 基础强度 = 最高情感分数
        base_intensity = max(scores.values())
        
        # 强度修饰词
        intensifiers = ["很", "非常", "特别", "极其", "超级", "太", "really", "very", "extremely"]
        diminishers = ["有点", "稍微", "略微", "一点", "somewhat", "slightly", "a bit"]
        
        text_lower = text.lower()
        
        # 检测强度修饰
        intensity_modifier = 1.0
        for word in intensifiers:
            if word in text_lower:
                intensity_modifier = 1.3
                break
        for word in diminishers:
            if word in text_lower:
                intensity_modifier = 0.7
                break
        
        # 感叹号增加强度
        exclamation_count = text.count("!") + text.count("！")
        if exclamation_count > 0:
            intensity_modifier *= (1 + 0.1 * min(exclamation_count, 3))
        
        return min(base_intensity * intensity_modifier, 1.0)
    
    def _check_distress(self, text: str) -> bool:
        """检测严重情绪困扰"""
        distress_patterns = [
            r"不想活", r"活着没意思", r"想放弃", r"撑不下去",
            r"太绝望", r"没有希望", r"结束一切", r"伤害自己",
            r"自杀", r"割腕", r"跳楼", r"想死",
            r"don't want to live", r"kill myself", r"end it all"
        ]
        
        for pattern in distress_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


# ==================== 规则增强组件 ====================

class RuleBasedEmotionDetector:
    """基于规则的情感检测器（作为 BERT 的补充）"""
    
    EMOTION_LEXICON = {
        "joy": {
            "keywords": ["开心", "高兴", "快乐", "幸福", "欣喜", "满足", "愉快",
                        "happy", "joyful", "delighted", "pleased", "glad"],
            "patterns": [r"太好了", r"真棒", r"太开心", r"好高兴"]
        },
        "anxiety": {
            "keywords": ["焦虑", "担心", "害怕", "紧张", "不安", "恐惧", "压力",
                        "anxious", "worried", "afraid", "nervous", "stressed"],
            "patterns": [r"怎么办", r"好担心", r"很紧张", r"压力大"]
        },
        "confusion": {
            "keywords": ["迷茫", "困惑", "不知道", "不明白", "纠结", "犹豫",
                        "confused", "lost", "uncertain", "unsure"],
            "patterns": [r"该怎么", r"不知道该", r"到底应该", r"想不明白"]
        },
        "sadness": {
            "keywords": ["难过", "伤心", "悲伤", "失落", "沮丧", "痛苦", "孤独",
                        "sad", "depressed", "lonely", "hurt", "grief"],
            "patterns": [r"好难过", r"很伤心", r"太痛苦", r"感觉失去"]
        },
        "anger": {
            "keywords": ["生气", "愤怒", "气愤", "恼火", "不满", "委屈",
                        "angry", "mad", "frustrated", "annoyed"],
            "patterns": [r"太过分", r"凭什么", r"受不了", r"很生气"]
        },
        "hopeful": {
            "keywords": ["希望", "期待", "相信", "乐观", "向往", "憧憬",
                        "hope", "optimistic", "believe", "looking forward"],
            "patterns": [r"希望能", r"期待着", r"相信会"]
        },
        "curious": {
            "keywords": ["好奇", "想知道", "想了解", "有意思", "探索",
                        "curious", "interested", "wonder"],
            "patterns": [r"想知道", r"很好奇", r"怎么回事"]
        },
        "fear": {
            "keywords": ["恐惧", "害怕", "惊恐", "畏惧", "胆怯",
                        "fear", "scared", "terrified", "frightened"],
            "patterns": [r"太可怕", r"吓死", r"恐怖"]
        },
    }
    
    def detect(self, text: str) -> Dict[str, float]:
        """检测情感分数"""
        text_lower = text.lower()
        scores = defaultdict(float)
        
        for emotion, data in self.EMOTION_LEXICON.items():
            score = 0.0
            
            # 关键词匹配
            for keyword in data["keywords"]:
                if keyword in text_lower:
                    score += 1.0
            
            # 模式匹配
            for pattern in data["patterns"]:
                if re.search(pattern, text):
                    score += 1.5
            
            scores[emotion] = score
        
        # 归一化
        total = sum(scores.values()) + 1e-8
        if total < 0.5:  # 如果没有明显情感，默认 neutral
            scores["neutral"] = 1.0
            total = sum(scores.values())
        
        return {k: v / total for k, v in scores.items()}


class NegationHandler:
    """否定处理器"""
    
    NEGATION_WORDS = [
        "不", "没", "没有", "无", "非", "别", "莫", "勿", "未",
        "不是", "并非", "并不", "从不", "绝不", "毫不",
        "not", "no", "never", "don't", "doesn't", "didn't",
        "won't", "wouldn't", "can't", "couldn't", "shouldn't"
    ]
    
    def has_negation(self, text: str) -> bool:
        """检测文本是否包含否定"""
        text_lower = text.lower()
        
        for word in self.NEGATION_WORDS:
            if word in text_lower:
                # 检查是否是双重否定
                pattern = rf"{word}.*{word}"
                if re.search(pattern, text_lower):
                    return False  # 双重否定 = 肯定
                return True
        
        return False
    
    def get_negation_scope(self, text: str) -> List[Tuple[int, int]]:
        """获取否定词的作用范围"""
        scopes = []
        text_lower = text.lower()
        
        for word in self.NEGATION_WORDS:
            for match in re.finditer(re.escape(word), text_lower):
                start = match.start()
                # 否定范围通常到下一个标点或句末
                end_match = re.search(r'[，。！？,.!?]', text[start:])
                if end_match:
                    end = start + end_match.start()
                else:
                    end = min(start + 20, len(text))
                scopes.append((start, end))
        
        return scopes


class SarcasmDetector:
    """讽刺检测器"""
    
    # 讽刺的常见模式
    SARCASM_PATTERNS = [
        # 正话反说
        (r"真.{0,3}(好|棒|厉害|了不起)", r"(太|好|真).{0,5}(糟|差|烂|垃圾)"),
        # 反问 + 正面词
        (r"(难道不是|还不是|不就是).*(好|对|棒)", None),
        # 引号标记 - 匹配中英文引号
        (r'["""\u201c\u201d].{1,10}(好|棒|厉害)["""\u201c\u201d]', None),
        # 语气词组合
        (r"哦.*真.*(好|棒|厉害)", None),
        (r"呵呵", None),
        (r"yeah.+right", None),
        (r"oh.+great", None),
    ]
    
    # 讽刺指示词
    SARCASM_INDICATORS = [
        "呵呵", "哈哈哈哈", "笑死", "可笑", "讽刺",
        "yeah right", "sure thing", "of course", "obviously"
    ]
    
    def detect(self, text: str) -> float:
        """
        检测讽刺概率
        
        Returns:
            0-1 之间的讽刺概率
        """
        text_lower = text.lower()
        sarcasm_score = 0.0
        
        # 检查讽刺指示词
        for indicator in self.SARCASM_INDICATORS:
            if indicator in text_lower:
                sarcasm_score += 0.3
        
        # 检查讽刺模式
        for pattern_tuple in self.SARCASM_PATTERNS:
            pattern = pattern_tuple[0]
            if re.search(pattern, text_lower):
                sarcasm_score += 0.4
        
        # 检测情感不一致（需要上下文）
        # TODO: 可以通过分析前后文的情感极性来增强检测
        
        return min(sarcasm_score, 1.0)


# ==================== 上下文感知情感分析 ====================

class ContextAwareEmotionAnalyzer:
    """
    上下文感知的情感分析器
    考虑对话历史和情感轨迹
    """
    
    def __init__(
        self,
        base_detector: BaseEmotionDetector,
        history_window: int = 5,
        emotion_momentum: float = 0.3
    ):
        """
        Args:
            base_detector: 基础情感检测器
            history_window: 考虑的历史消息数量
            emotion_momentum: 情感惯性权重
        """
        self.detector = base_detector
        self.history_window = history_window
        self.emotion_momentum = emotion_momentum
        self.history: List[EmotionResult] = []
    
    def analyze(self, text: str) -> EmotionResult:
        """分析当前文本，考虑历史上下文"""
        # 获取当前检测结果
        current_result = self.detector.detect(text)
        
        # 应用情感惯性
        if self.history:
            adjusted_scores = self._apply_momentum(current_result.all_emotions)
            
            # 重新计算主要情感
            primary_emotion = max(adjusted_scores, key=adjusted_scores.get)
            
            current_result = EmotionResult(
                primary_emotion=primary_emotion,
                confidence=adjusted_scores[primary_emotion],
                intensity=current_result.intensity,
                all_emotions=adjusted_scores,
                valence=current_result.valence,
                arousal=current_result.arousal,
                needs_support=current_result.needs_support,
                negation_detected=current_result.negation_detected,
                sarcasm_probability=current_result.sarcasm_probability,
                raw_model_output=current_result.raw_model_output
            )
        
        # 更新历史
        self.history.append(current_result)
        if len(self.history) > self.history_window:
            self.history.pop(0)
        
        return current_result
    
    def _apply_momentum(self, current_scores: Dict[str, float]) -> Dict[str, float]:
        """应用情感惯性"""
        if not self.history:
            return current_scores
        
        # 计算历史情感的加权平均
        weights = [0.5 ** (len(self.history) - i) for i in range(len(self.history))]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        history_scores = defaultdict(float)
        for i, result in enumerate(self.history):
            for emotion, score in result.all_emotions.items():
                history_scores[emotion] += score * weights[i]
        
        # 融合当前和历史
        adjusted = {}
        all_emotions = set(current_scores.keys()) | set(history_scores.keys())
        
        for emotion in all_emotions:
            current = current_scores.get(emotion, 0.0)
            historical = history_scores.get(emotion, 0.0)
            adjusted[emotion] = (
                current * (1 - self.emotion_momentum) +
                historical * self.emotion_momentum
            )
        
        # 归一化
        total = sum(adjusted.values()) + 1e-8
        return {k: v / total for k, v in adjusted.items()}
    
    def get_emotional_trajectory(self) -> List[Dict]:
        """获取情感轨迹"""
        return [
            {
                "emotion": r.primary_emotion,
                "confidence": r.confidence,
                "valence": r.valence,
                "arousal": r.arousal
            }
            for r in self.history
        ]
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """获取整体主导情感"""
        if not self.history:
            return "neutral", 0.5
        
        emotion_counts = Counter(r.primary_emotion for r in self.history)
        dominant = emotion_counts.most_common(1)[0]
        return dominant[0], dominant[1] / len(self.history)
    
    def reset_history(self):
        """重置历史"""
        self.history = []


# ==================== 意图分类器 ====================

class EnhancedIntentClassifier:
    """
    增强版意图分类器
    支持多标签分类和意图强度
    """
    
    INTENT_DEFINITIONS = {
        "seeking_guidance": {
            "keywords": ["怎么办", "该如何", "应该怎样", "怎么做", "如何是好",
                        "what should", "how can", "what do I do", "help me"],
            "patterns": [r"我该", r"我应该", r"请告诉我", r"给我建议"],
            "description": "寻求指导或建议"
        },
        "seeking_clarity": {
            "keywords": ["想知道", "不明白", "为什么", "是什么意思", "什么是",
                        "want to know", "don't understand", "what does", "why"],
            "patterns": [r"这意味着", r"代表什么", r"有什么含义"],
            "description": "寻求澄清或理解"
        },
        "emotional_processing": {
            "keywords": ["感觉", "觉得", "情绪", "心情", "内心",
                        "feeling", "emotion", "mood", "heart"],
            "patterns": [r"我感到", r"让我很", r"心里"],
            "description": "处理情绪"
        },
        "decision_making": {
            "keywords": ["选择", "决定", "抉择", "取舍", "要不要",
                        "choose", "decide", "choice", "option"],
            "patterns": [r"A还是B", r"是否应该", r"要不要"],
            "description": "做决定"
        },
        "relationship_exploration": {
            "keywords": ["关系", "他", "她", "我们", "朋友", "家人", "爱人",
                        "relationship", "partner", "friend", "family"],
            "patterns": [r"和.{1,5}的关系", r"我们之间"],
            "description": "探索人际关系"
        },
        "self_reflection": {
            "keywords": ["我是", "我的", "自己", "内心", "真正的我",
                        "I am", "myself", "who I", "true self"],
            "patterns": [r"我到底", r"真正的", r"内心深处"],
            "description": "自我反思"
        },
        "future_planning": {
            "keywords": ["未来", "以后", "计划", "打算", "目标",
                        "future", "plan", "goal", "going to"],
            "patterns": [r"将来", r"接下来", r"下一步"],
            "description": "规划未来"
        },
        "validation_seeking": {
            "keywords": ["对吗", "是吧", "你觉得", "我做的",
                        "right", "correct", "do you think", "am I"],
            "patterns": [r"是不是", r"对不对", r"好吗"],
            "description": "寻求认可"
        }
    }
    
    def classify(
        self,
        text: str,
        return_all: bool = True
    ) -> Union[Dict[str, float], Tuple[str, float]]:
        """
        分类用户意图
        
        Args:
            text: 输入文本
            return_all: 是否返回所有意图的分数
            
        Returns:
            如果 return_all=True，返回 {intent: score}
            否则返回 (primary_intent, score)
        """
        text_lower = text.lower()
        scores = {}
        
        for intent, definition in self.INTENT_DEFINITIONS.items():
            score = 0.0
            
            # 关键词匹配
            for keyword in definition["keywords"]:
                if keyword in text_lower:
                    score += 1.0
            
            # 模式匹配
            for pattern in definition["patterns"]:
                if re.search(pattern, text):
                    score += 1.5
            
            scores[intent] = score
        
        # 归一化
        total = sum(scores.values()) + 1e-8
        scores = {k: v / total for k, v in scores.items()}
        
        if return_all:
            return scores
        else:
            primary = max(scores, key=scores.get)
            return primary, scores[primary]
    
    def get_multi_label(
        self,
        text: str,
        threshold: float = 0.15
    ) -> List[Tuple[str, float]]:
        """获取多标签分类结果"""
        scores = self.classify(text, return_all=True)
        
        # 返回超过阈值的意图
        return [
            (intent, score)
            for intent, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if score >= threshold
        ]


# ==================== 对话策略生成器 ====================

class DialogStrategyGenerator:
    """
    基于情感和意图生成对话策略
    """
    
    STRATEGY_TEMPLATES = {
        "anxiety": {
            "tone": "calm_supportive",
            "pace": "slow",
            "focus": "grounding",
            "response_style": "温和、接纳、不急于解决",
            "key_phrases": ["我理解这种感觉", "让我们一起慢慢看", "没有对错之分"],
            "avoid": ["你应该", "不要担心", "没什么大不了"],
            "techniques": ["深呼吸引导", "此刻专注", "正常化感受"]
        },
        "confusion": {
            "tone": "clarifying",
            "pace": "moderate",
            "focus": "structure",
            "response_style": "清晰、有条理、逐步引导",
            "key_phrases": ["让我确认一下", "一步一步来", "你觉得最核心的是什么"],
            "avoid": ["很简单", "显然", "你应该知道"],
            "techniques": ["复述确认", "分解问题", "优先排序"]
        },
        "sadness": {
            "tone": "empathetic",
            "pace": "slow",
            "focus": "validation",
            "response_style": "温暖、陪伴、不急于修复",
            "key_phrases": ["这听起来很不容易", "允许自己感受", "我在这里"],
            "avoid": ["振作起来", "往好处想", "时间会治愈"],
            "techniques": ["情感反映", "沉默陪伴", "温和询问"]
        },
        "anger": {
            "tone": "accepting",
            "pace": "moderate",
            "focus": "acknowledgment",
            "response_style": "承认、不评判、探索背后",
            "key_phrases": ["你的愤怒是可以理解的", "背后可能是什么", "你需要什么"],
            "avoid": ["冷静下来", "不值得生气", "对方也是"],
            "techniques": ["情绪命名", "需求探索", "边界确认"]
        },
        "hopeful": {
            "tone": "encouraging",
            "pace": "dynamic",
            "focus": "possibility",
            "response_style": "支持、具体化、务实乐观",
            "key_phrases": ["这个方向很好", "让我们看看怎么实现", "你已经有了想法"],
            "avoid": ["别高兴太早", "现实一点", "可能不行"],
            "techniques": ["愿景探索", "行动规划", "资源盘点"]
        },
        "curious": {
            "tone": "exploratory",
            "pace": "dynamic",
            "focus": "discovery",
            "response_style": "开放、丰富、游戏心态",
            "key_phrases": ["很有趣的角度", "还有什么可能", "让我们探索一下"],
            "avoid": ["标准答案是", "正确理解是", "你应该"],
            "techniques": ["开放提问", "多元视角", "联想扩展"]
        },
        "neutral": {
            "tone": "warm_curious",
            "pace": "moderate",
            "focus": "engagement",
            "response_style": "友好、有温度、适度引导",
            "key_phrases": ["我很好奇", "能多说说吗", "你怎么看"],
            "avoid": ["就这样吧", "下一个", "快点"],
            "techniques": ["温和邀请", "兴趣表达", "开放倾听"]
        },
        "fear": {
            "tone": "protective",
            "pace": "slow",
            "focus": "safety",
            "response_style": "安全、稳定、渐进",
            "key_phrases": ["你是安全的", "一小步一小步", "我们可以随时暂停"],
            "avoid": ["没什么可怕的", "勇敢一点", "面对它"],
            "techniques": ["安全建立", "渐进暴露", "资源确认"]
        }
    }
    
    def generate_strategy(
        self,
        emotion_result: EmotionResult,
        intent_scores: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        生成对话策略
        
        Args:
            emotion_result: 情感检测结果
            intent_scores: 意图分类分数
            
        Returns:
            对话策略字典
        """
        primary_emotion = emotion_result.primary_emotion
        
        # 获取基础策略
        base_strategy = self.STRATEGY_TEMPLATES.get(
            primary_emotion,
            self.STRATEGY_TEMPLATES["neutral"]
        ).copy()
        
        # 根据情感强度调整
        if emotion_result.intensity > 0.8:
            base_strategy["pace"] = "slow"
            base_strategy["additional_note"] = "情感强度较高，需要更多耐心和空间"
        
        # 根据 valence 调整
        if emotion_result.valence < -0.5:
            base_strategy["priority"] = "emotional_support_first"
        
        # 根据意图调整
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            base_strategy["detected_intent"] = primary_intent
            
            # 特殊意图处理
            if primary_intent == "decision_making":
                base_strategy["focus"] = "structured_reflection"
            elif primary_intent == "validation_seeking":
                base_strategy["additional_note"] = "用户可能需要认可，但避免简单迎合"
        
        # 添加元信息
        base_strategy["emotion_confidence"] = emotion_result.confidence
        base_strategy["needs_support"] = emotion_result.needs_support
        
        return base_strategy


# ==================== 全局实例 ====================

_emotion_detector: Optional[BERTEmotionDetector] = None
_context_analyzer: Optional[ContextAwareEmotionAnalyzer] = None


def get_emotion_detector(
    use_bert: bool = True,
    **kwargs
) -> BaseEmotionDetector:
    """获取情感检测器"""
    global _emotion_detector
    
    if _emotion_detector is None:
        if use_bert:
            _emotion_detector = BERTEmotionDetector(**kwargs)
        else:
            _emotion_detector = RuleBasedEmotionDetector()
    
    return _emotion_detector


def get_context_analyzer(**kwargs) -> ContextAwareEmotionAnalyzer:
    """获取上下文感知分析器"""
    global _context_analyzer
    
    if _context_analyzer is None:
        detector = get_emotion_detector()
        _context_analyzer = ContextAwareEmotionAnalyzer(detector, **kwargs)
    
    return _context_analyzer