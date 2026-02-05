"""
评估框架模块
提供 RAG 检索和情感检测的完整评估指标

包含：
1. RAG 检索评估（Precision, Recall, NDCG, MRR, MAP）
2. 情感检测评估（Accuracy, F1, 混淆矩阵）
3. 统计显著性检验（t-test, Wilcoxon, Bootstrap）
"""

import json
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据类 ====================

@dataclass
class RetrievalGroundTruth:
    """检索评估的标注数据"""
    query_id: str
    query: str
    relevant_doc_ids: List[str]
    relevance_scores: Optional[Dict[str, int]] = None


@dataclass
class EmotionGroundTruth:
    """情感检测的标注数据"""
    text_id: str
    text: str
    true_emotion: str
    true_intensity: Optional[float] = None


@dataclass
class RetrievalEvalResult:
    """检索评估结果"""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float
    map_score: float
    per_query_metrics: Dict[str, Dict[str, float]]
    summary: str = ""


@dataclass
class EmotionEvalResult:
    """情感检测评估结果"""
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    class_names: List[str]
    summary: str = ""


@dataclass
class SignificanceTestResult:
    """统计显著性检验结果"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


# ==================== RAG 检索评估 ====================

class RetrievalEvaluator:
    """RAG 检索评估器"""
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
    
    def evaluate(
        self,
        predictions: List[Tuple[str, List[str]]],
        ground_truth: List[RetrievalGroundTruth]
    ) -> RetrievalEvalResult:
        """评估检索结果"""
        gt_map = {gt.query_id: gt for gt in ground_truth}
        
        per_query = {}
        all_precisions = {k: [] for k in self.k_values}
        all_recalls = {k: [] for k in self.k_values}
        all_ndcgs = {k: [] for k in self.k_values}
        all_rrs = []
        all_aps = []
        
        for query_id, pred_docs in predictions:
            if query_id not in gt_map:
                continue
            
            gt = gt_map[query_id]
            relevant_set = set(gt.relevant_doc_ids)
            relevance_scores = gt.relevance_scores or {d: 1 for d in gt.relevant_doc_ids}
            
            query_metrics = {}
            
            for k in self.k_values:
                top_k = pred_docs[:k]
                relevant_in_top_k = sum(1 for d in top_k if d in relevant_set)
                
                precision = relevant_in_top_k / k if k > 0 else 0
                recall = relevant_in_top_k / len(relevant_set) if relevant_set else 0
                
                all_precisions[k].append(precision)
                all_recalls[k].append(recall)
                query_metrics[f"P@{k}"] = precision
                query_metrics[f"R@{k}"] = recall
            
            for k in self.k_values:
                ndcg = self._calculate_ndcg(pred_docs[:k], relevance_scores, k)
                all_ndcgs[k].append(ndcg)
                query_metrics[f"NDCG@{k}"] = ndcg
            
            rr = self._calculate_rr(pred_docs, relevant_set)
            all_rrs.append(rr)
            query_metrics["RR"] = rr
            
            ap = self._calculate_ap(pred_docs, relevant_set)
            all_aps.append(ap)
            query_metrics["AP"] = ap
            
            per_query[query_id] = query_metrics
        
        precision_at_k = {k: np.mean(all_precisions[k]) for k in self.k_values}
        recall_at_k = {k: np.mean(all_recalls[k]) for k in self.k_values}
        ndcg_at_k = {k: np.mean(all_ndcgs[k]) for k in self.k_values}
        mrr = np.mean(all_rrs) if all_rrs else 0.0
        map_score = np.mean(all_aps) if all_aps else 0.0
        
        summary = self._generate_summary(
            precision_at_k, recall_at_k, ndcg_at_k, mrr, map_score, len(predictions)
        )
        
        return RetrievalEvalResult(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr,
            map_score=map_score,
            per_query_metrics=per_query,
            summary=summary
        )
    
    def _calculate_ndcg(self, pred_docs: List[str], relevance_scores: Dict[str, int], k: int) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(pred_docs):
            rel = relevance_scores.get(doc_id, 0)
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_rr(self, pred_docs: List[str], relevant_set: set) -> float:
        for i, doc_id in enumerate(pred_docs):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ap(self, pred_docs: List[str], relevant_set: set) -> float:
        if not relevant_set:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for i, doc_id in enumerate(pred_docs):
            if doc_id in relevant_set:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        
        return sum(precisions) / len(relevant_set) if precisions else 0.0
    
    def _generate_summary(self, precision, recall, ndcg, mrr, map_score, n_queries) -> str:
        lines = [
            "=" * 50,
            "RAG Retrieval Evaluation Summary",
            "=" * 50,
            f"Number of queries: {n_queries}",
            "\nPrecision@K:"
        ]
        for k, v in precision.items():
            lines.append(f"  P@{k}: {v:.4f}")
        lines.append("\nRecall@K:")
        for k, v in recall.items():
            lines.append(f"  R@{k}: {v:.4f}")
        lines.append("\nNDCG@K:")
        for k, v in ndcg.items():
            lines.append(f"  NDCG@{k}: {v:.4f}")
        lines.extend([f"\nMRR: {mrr:.4f}", f"MAP: {map_score:.4f}", "=" * 50])
        return "\n".join(lines)


# ==================== 情感检测评估 ====================

class EmotionEvaluator:
    """情感检测评估器"""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or [
            "joy", "hopeful", "curious", "neutral",
            "confusion", "anxiety", "sadness", "anger", "fear"
        ]
    
    def evaluate(self, predictions: List[str], ground_truth: List[str]) -> EmotionEvalResult:
        """评估情感检测结果"""
        assert len(predictions) == len(ground_truth)
        
        n_samples = len(predictions)
        all_classes = sorted(set(predictions) | set(ground_truth))
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        n_classes = len(all_classes)
        
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        for pred, true in zip(predictions, ground_truth):
            if pred in class_to_idx and true in class_to_idx:
                confusion[class_to_idx[true], class_to_idx[pred]] += 1
        
        per_class = {}
        precisions, recalls, f1s, supports = [], [], [], []
        
        for i, class_name in enumerate(all_classes):
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp
            support = confusion[i, :].sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class[class_name] = {
                "precision": precision, "recall": recall, "f1": f1, "support": int(support)
            }
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)
        
        accuracy = np.trace(confusion) / np.sum(confusion) if np.sum(confusion) > 0 else 0
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1s)
        
        total_support = sum(supports)
        weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support if total_support > 0 else 0
        
        summary = self._generate_summary(
            accuracy, macro_precision, macro_recall, macro_f1, weighted_f1, per_class, n_samples
        )
        
        return EmotionEvalResult(
            accuracy=accuracy,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            per_class_metrics=per_class,
            confusion_matrix=confusion,
            class_names=all_classes,
            summary=summary
        )
    
    def _generate_summary(self, accuracy, macro_p, macro_r, macro_f1, weighted_f1, per_class, n_samples) -> str:
        lines = [
            "=" * 50,
            "Emotion Detection Evaluation Summary",
            "=" * 50,
            f"Number of samples: {n_samples}",
            f"\nOverall Accuracy: {accuracy:.4f}",
            f"Macro Precision:  {macro_p:.4f}",
            f"Macro Recall:     {macro_r:.4f}",
            f"Macro F1:         {macro_f1:.4f}",
            f"Weighted F1:      {weighted_f1:.4f}",
            "\nPer-class Metrics:",
            f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
        ]
        for class_name, metrics in per_class.items():
            lines.append(
                f"{class_name:<12} {metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['support']:>10}"
            )
        lines.append("=" * 50)
        return "\n".join(lines)


# ==================== 统计显著性检验 ====================

class StatisticalTests:
    """统计显著性检验工具"""
    
    @staticmethod
    def paired_t_test(scores_a: List[float], scores_b: List[float], alpha: float = 0.05) -> SignificanceTestResult:
        """配对 t 检验"""
        from scipy import stats
        
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        
        # Cohen's d effect size
        diff = np.array(scores_a) - np.array(scores_b)
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
        
        return SignificanceTestResult(
            test_name="Paired t-test",
            statistic=float(t_stat),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            effect_size=float(effect_size)
        )
    
    @staticmethod
    def wilcoxon_test(scores_a: List[float], scores_b: List[float], alpha: float = 0.05) -> SignificanceTestResult:
        """Wilcoxon 符号秩检验（非参数）"""
        from scipy import stats
        
        stat, p_value = stats.wilcoxon(scores_a, scores_b)
        
        # Effect size (r = Z / sqrt(N))
        n = len(scores_a)
        z = stats.norm.ppf(1 - p_value / 2)
        effect_size = z / np.sqrt(n)
        
        return SignificanceTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=float(stat),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            effect_size=float(effect_size)
        )
    
    @staticmethod
    def bootstrap_ci(
        scores_a: List[float],
        scores_b: List[float],
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> SignificanceTestResult:
        """Bootstrap 置信区间"""
        np.random.seed(42)
        
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        n = len(scores_a)
        
        diffs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            diff = np.mean(scores_a[idx]) - np.mean(scores_b[idx])
            diffs.append(diff)
        
        diffs = np.array(diffs)
        alpha = 1 - confidence
        ci_lower = np.percentile(diffs, alpha / 2 * 100)
        ci_upper = np.percentile(diffs, (1 - alpha / 2) * 100)
        
        # 如果 CI 不包含 0，则显著
        is_significant = ci_lower > 0 or ci_upper < 0
        
        return SignificanceTestResult(
            test_name=f"Bootstrap {int(confidence*100)}% CI",
            statistic=float(np.mean(diffs)),
            p_value=float(np.mean(diffs <= 0) * 2),  # 近似 p-value
            is_significant=is_significant,
            confidence_interval=(float(ci_lower), float(ci_upper))
        )


# ==================== 基线对比 ====================

class BaselineComparator:
    """基线方法对比器"""
    
    def __init__(self):
        self.results: Dict[str, Dict] = {}
    
    def add_result(self, method_name: str, metrics: Dict[str, float]):
        """添加方法结果"""
        self.results[method_name] = metrics
    
    def compare_all(self, metric_name: str, alpha: float = 0.05) -> Dict[str, Dict]:
        """对比所有方法"""
        methods = list(self.results.keys())
        comparisons = {}
        
        for i, method_a in enumerate(methods):
            for method_b in methods[i+1:]:
                key = f"{method_a}_vs_{method_b}"
                
                scores_a = self.results[method_a].get(f"{metric_name}_per_sample", [])
                scores_b = self.results[method_b].get(f"{metric_name}_per_sample", [])
                
                if scores_a and scores_b and len(scores_a) == len(scores_b):
                    t_test = StatisticalTests.paired_t_test(scores_a, scores_b, alpha)
                    wilcoxon = StatisticalTests.wilcoxon_test(scores_a, scores_b, alpha)
                    bootstrap = StatisticalTests.bootstrap_ci(scores_a, scores_b)
                    
                    comparisons[key] = {
                        "t_test": t_test,
                        "wilcoxon": wilcoxon,
                        "bootstrap": bootstrap,
                        "mean_diff": np.mean(scores_a) - np.mean(scores_b)
                    }
        
        return comparisons
    
    def generate_comparison_table(self, metric_name: str) -> str:
        """生成对比表格"""
        methods = list(self.results.keys())
        
        lines = [
            "=" * 60,
            f"Method Comparison: {metric_name}",
            "=" * 60,
            f"{'Method':<20} {'Mean':>10} {'Std':>10} {'Best':>8}"
        ]
        
        best_score = -float('inf')
        best_method = ""
        
        for method, metrics in self.results.items():
            score = metrics.get(metric_name, 0)
            std = metrics.get(f"{metric_name}_std", 0)
            
            if score > best_score:
                best_score = score
                best_method = method
        
        for method, metrics in self.results.items():
            score = metrics.get(metric_name, 0)
            std = metrics.get(f"{metric_name}_std", 0)
            is_best = "✓" if method == best_method else ""
            lines.append(f"{method:<20} {score:>10.4f} {std:>10.4f} {is_best:>8}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
