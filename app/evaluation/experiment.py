"""
评估数据集和实验运行器
提供标准化的评估流程和数据集管理
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging

from .metrics import (
    RetrievalEvaluator, EmotionEvaluator, StatisticalTests,
    BaselineComparator, RetrievalGroundTruth, EmotionGroundTruth,
    RetrievalEvalResult, EmotionEvalResult
)

logger = logging.getLogger(__name__)


# ==================== 评估数据集 ====================

@dataclass
class EvaluationDataset:
    """评估数据集基类"""
    name: str
    description: str
    samples: List[Dict]
    metadata: Dict
    
    def __len__(self):
        return len(self.samples)
    
    def split(self, train_ratio: float = 0.8, seed: int = 42) -> tuple:
        """划分训练/测试集"""
        random.seed(seed)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)
        
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_samples = [self.samples[i] for i in train_indices]
        test_samples = [self.samples[i] for i in test_indices]
        
        return train_samples, test_samples
    
    def to_json(self, path: str):
        """保存数据集"""
        data = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "samples": self.samples
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> 'EvaluationDataset':
        """加载数据集"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(
            name=data["name"],
            description=data["description"],
            samples=data["samples"],
            metadata=data.get("metadata", {})
        )


class TarotRetrievalDataset(EvaluationDataset):
    """塔罗检索评估数据集"""
    
    # 示例标注数据（实际使用时需要人工标注）
    SAMPLE_ANNOTATIONS = [
        {
            "query_id": "q001",
            "query": "我最近工作压力很大，不知道该不该换工作",
            "relevant_doc_ids": ["card_fool", "card_wheel", "knowledge_choice"],
            "relevance_scores": {"card_fool": 3, "card_wheel": 2, "knowledge_choice": 2}
        },
        {
            "query_id": "q002",
            "query": "和男朋友吵架了，不知道要不要分手",
            "relevant_doc_ids": ["card_lovers", "card_tower", "knowledge_relationship"],
            "relevance_scores": {"card_lovers": 3, "card_tower": 2, "knowledge_relationship": 3}
        },
        {
            "query_id": "q003",
            "query": "对未来很迷茫，不知道自己想要什么",
            "relevant_doc_ids": ["card_hermit", "card_star", "knowledge_self_reflection"],
            "relevance_scores": {"card_hermit": 3, "card_star": 2, "knowledge_self_reflection": 3}
        },
        {
            "query_id": "q004",
            "query": "愚者牌代表什么",
            "relevant_doc_ids": ["card_fool"],
            "relevance_scores": {"card_fool": 3}
        },
        {
            "query_id": "q005",
            "query": "如何解读逆位的塔牌",
            "relevant_doc_ids": ["card_tower", "knowledge_reversed_cards"],
            "relevance_scores": {"card_tower": 3, "knowledge_reversed_cards": 3}
        }
    ]
    
    def __init__(self, samples: Optional[List[Dict]] = None):
        super().__init__(
            name="TarotRetrievalBenchmark",
            description="塔罗知识检索评估数据集",
            samples=samples or self.SAMPLE_ANNOTATIONS,
            metadata={"version": "1.0", "language": "zh"}
        )
    
    def to_ground_truth(self) -> List[RetrievalGroundTruth]:
        """转换为标准格式"""
        return [
            RetrievalGroundTruth(
                query_id=s["query_id"],
                query=s["query"],
                relevant_doc_ids=s["relevant_doc_ids"],
                relevance_scores=s.get("relevance_scores")
            )
            for s in self.samples
        ]


class EmotionDataset(EvaluationDataset):
    """情感检测评估数据集"""
    
    # 示例标注数据
    SAMPLE_ANNOTATIONS = [
        {"text_id": "e001", "text": "我好焦虑，不知道该怎么办", "emotion": "anxiety", "intensity": 0.8},
        {"text_id": "e002", "text": "感觉很迷茫，不知道未来在哪", "emotion": "confusion", "intensity": 0.7},
        {"text_id": "e003", "text": "太开心了，终于拿到offer了", "emotion": "joy", "intensity": 0.9},
        {"text_id": "e004", "text": "和他分手了，好难过", "emotion": "sadness", "intensity": 0.85},
        {"text_id": "e005", "text": "凭什么是这样，太不公平了", "emotion": "anger", "intensity": 0.8},
        {"text_id": "e006", "text": "希望一切都会好起来", "emotion": "hopeful", "intensity": 0.6},
        {"text_id": "e007", "text": "塔罗牌挺有意思的，想了解更多", "emotion": "curious", "intensity": 0.5},
        {"text_id": "e008", "text": "今天天气不错", "emotion": "neutral", "intensity": 0.3},
        {"text_id": "e009", "text": "压力太大了，感觉快撑不下去了", "emotion": "anxiety", "intensity": 0.95},
        {"text_id": "e010", "text": "虽然有点担心，但相信会顺利的", "emotion": "hopeful", "intensity": 0.6},
    ]
    
    def __init__(self, samples: Optional[List[Dict]] = None):
        super().__init__(
            name="TarotEmotionBenchmark",
            description="塔罗对话情感检测评估数据集",
            samples=samples or self.SAMPLE_ANNOTATIONS,
            metadata={"version": "1.0", "language": "zh"}
        )
    
    def to_ground_truth(self) -> List[EmotionGroundTruth]:
        """转换为标准格式"""
        return [
            EmotionGroundTruth(
                text_id=s["text_id"],
                text=s["text"],
                true_emotion=s["emotion"],
                true_intensity=s.get("intensity")
            )
            for s in self.samples
        ]


# ==================== 实验运行器 ====================

class ExperimentRunner:
    """
    实验运行器
    管理实验配置、运行和结果记录
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "./experiments"
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, Any] = {}
        self.start_time = None
        self.config: Dict = {}
    
    def set_config(self, config: Dict):
        """设置实验配置"""
        self.config = config
    
    def run_retrieval_experiment(
        self,
        retriever,
        dataset: TarotRetrievalDataset,
        method_name: str = "proposed"
    ) -> RetrievalEvalResult:
        """运行检索实验"""
        logger.info(f"Running retrieval experiment: {method_name}")
        
        ground_truth = dataset.to_ground_truth()
        predictions = []
        
        for gt in ground_truth:
            # 调用检索器
            results = retriever.retrieve(gt.query, top_k=10)
            doc_ids = [r.id for r in results]
            predictions.append((gt.query_id, doc_ids))
        
        # 评估
        evaluator = RetrievalEvaluator()
        eval_result = evaluator.evaluate(predictions, ground_truth)
        
        self.results[f"retrieval_{method_name}"] = {
            "metrics": {
                "precision_at_k": eval_result.precision_at_k,
                "recall_at_k": eval_result.recall_at_k,
                "ndcg_at_k": eval_result.ndcg_at_k,
                "mrr": eval_result.mrr,
                "map": eval_result.map_score
            },
            "per_query": eval_result.per_query_metrics
        }
        
        logger.info(f"\n{eval_result.summary}")
        return eval_result
    
    def run_emotion_experiment(
        self,
        detector,
        dataset: EmotionDataset,
        method_name: str = "proposed"
    ) -> EmotionEvalResult:
        """运行情感检测实验"""
        logger.info(f"Running emotion detection experiment: {method_name}")
        
        ground_truth = dataset.to_ground_truth()
        predictions = []
        true_labels = []
        
        for gt in ground_truth:
            # 调用检测器
            result = detector.detect(gt.text)
            predictions.append(result.primary_emotion)
            true_labels.append(gt.true_emotion)
        
        # 评估
        evaluator = EmotionEvaluator()
        eval_result = evaluator.evaluate(predictions, true_labels)
        
        self.results[f"emotion_{method_name}"] = {
            "metrics": {
                "accuracy": eval_result.accuracy,
                "macro_f1": eval_result.macro_f1,
                "weighted_f1": eval_result.weighted_f1
            },
            "per_class": eval_result.per_class_metrics,
            "predictions": predictions,
            "ground_truth": true_labels
        }
        
        logger.info(f"\n{eval_result.summary}")
        return eval_result
    
    def compare_methods(
        self,
        metric_name: str,
        method_names: List[str]
    ) -> Dict:
        """对比不同方法"""
        comparator = BaselineComparator()
        
        for method in method_names:
            key = f"emotion_{method}" if "emotion" in metric_name else f"retrieval_{method}"
            if key in self.results:
                comparator.add_result(method, self.results[key]["metrics"])
        
        comparisons = comparator.compare_all(metric_name)
        comparison_table = comparator.generate_comparison_table(metric_name)
        
        logger.info(f"\n{comparison_table}")
        
        return {
            "comparisons": comparisons,
            "table": comparison_table
        }
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.output_dir / f"{self.experiment_name}_{timestamp}.json"
        
        output = {
            "experiment_name": self.experiment_name,
            "timestamp": timestamp,
            "config": self.config,
            "results": self._serialize_results(self.results)
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {result_path}")
        return result_path
    
    def _serialize_results(self, results: Dict) -> Dict:
        """序列化结果（处理 numpy 类型）"""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        return convert(results)


# ==================== 基线方法实现 ====================

class BaselineRetriever:
    """基线检索方法（用于对比）"""
    
    def __init__(self, method: str = "bm25"):
        self.method = method
        self.documents = []
        self.bm25 = None
    
    def add_documents(self, documents: List[Dict]):
        self.documents = documents
        
        if self.method == "bm25":
            from ..rag.retriever_enhanced import BM25Retriever
            self.bm25 = BM25Retriever()
            texts = [d["text"] for d in documents]
            self.bm25.fit(texts)
    
    def retrieve(self, query: str, top_k: int = 5):
        if self.method == "bm25":
            results = self.bm25.retrieve(query, top_k)
            return [
                type('Result', (), {
                    'id': self.documents[idx]["id"],
                    'text': self.documents[idx]["text"],
                    'score': score
                })()
                for idx, score in results
            ]
        elif self.method == "random":
            import random
            indices = random.sample(range(len(self.documents)), min(top_k, len(self.documents)))
            return [
                type('Result', (), {
                    'id': self.documents[idx]["id"],
                    'text': self.documents[idx]["text"],
                    'score': 1.0 / (i + 1)
                })()
                for i, idx in enumerate(indices)
            ]
        return []


class BaselineEmotionDetector:
    """基线情感检测方法（用于对比）"""
    
    def __init__(self, method: str = "lexicon"):
        self.method = method
    
    def detect(self, text: str):
        if self.method == "lexicon":
            # 简单词典方法
            from ..ml.emotion_detector_enhanced import RuleBasedEmotionDetector
            detector = RuleBasedEmotionDetector()
            scores = detector.detect(text)
            primary = max(scores, key=scores.get)
            
            return type('Result', (), {
                'primary_emotion': primary,
                'confidence': scores[primary],
                'all_emotions': scores
            })()
        
        elif self.method == "random":
            emotions = ["joy", "sadness", "anxiety", "anger", "neutral", "confusion"]
            import random
            emotion = random.choice(emotions)
            
            return type('Result', (), {
                'primary_emotion': emotion,
                'confidence': random.random(),
                'all_emotions': {e: random.random() for e in emotions}
            })()
        
        return type('Result', (), {
            'primary_emotion': 'neutral',
            'confidence': 0.5,
            'all_emotions': {'neutral': 1.0}
        })()
