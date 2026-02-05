"""
增强版 RAG (Retrieval-Augmented Generation) 模块
使用真实的 Sentence-BERT 进行语义嵌入，FAISS 进行高效相似度检索

主要改进：
1. 使用预训练的多语言 Sentence-BERT 模型 (paraphrase-multilingual-MiniLM-L12-v2)
2. 支持 FAISS 向量索引，提供高效的 ANN 检索
3. 支持混合检索策略（稠密检索 + BM25 稀疏检索）
4. 支持 OpenAI/Anthropic Embeddings API 作为备选
5. 完整的评估指标支持
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import re

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据类 ====================

@dataclass
class Document:
    """文档对象"""
    id: str
    text: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievedChunk:
    """检索结果"""
    id: str
    text: str
    score: float
    metadata: Dict
    retrieval_method: str = "dense"  # dense, sparse, hybrid


@dataclass
class RetrievalMetrics:
    """检索评估指标"""
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0


# ==================== Embedding 抽象基类 ====================

class BaseEmbedder(ABC):
    """嵌入模型抽象基类"""
    
    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本为向量"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度"""
        pass


# ==================== Sentence-BERT Embedder ====================

class SentenceBERTEmbedder(BaseEmbedder):
    """
    基于 Sentence-BERT 的嵌入模型
    支持多语言文本（中文、英文等）
    """
    
    # 推荐的多语言模型
    MULTILINGUAL_MODELS = {
        "default": "paraphrase-multilingual-MiniLM-L12-v2",  # 384维，支持50+语言
        "large": "paraphrase-multilingual-mpnet-base-v2",     # 768维，更高质量
        "chinese": "shibing624/text2vec-base-chinese",       # 768维，中文优化
    }
    
    def __init__(
        self, 
        model_name: str = "default",
        device: str = "auto",
        cache_dir: Optional[str] = None
    ):
        """
        初始化 Sentence-BERT 嵌入器
        
        Args:
            model_name: 模型名称，可以是预定义的 key 或 HuggingFace 模型路径
            device: 设备 ("cpu", "cuda", "auto")
            cache_dir: 模型缓存目录
        """
        self.model_name = self.MULTILINGUAL_MODELS.get(model_name, model_name)
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self._dimension = None
        
    def _load_model(self):
        """延迟加载模型"""
        if self.model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading Sentence-BERT model: {self.model_name}")
            
            # 确定设备
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            
            # 获取维度
            self._dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self._dimension}, Device: {self.device}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        编码文本列表为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度
            normalize: 是否归一化向量
            
        Returns:
            形状为 (n_texts, dimension) 的嵌入矩阵
        """
        self._load_model()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embeddings.astype(np.float32)
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        self._load_model()
        return self._dimension


# ==================== OpenAI Embedder ====================

class OpenAIEmbedder(BaseEmbedder):
    """
    基于 OpenAI API 的嵌入模型
    适用于需要高质量嵌入但不想本地部署模型的场景
    """
    
    MODELS = {
        "small": "text-embedding-3-small",   # 1536维
        "large": "text-embedding-3-large",   # 3072维
        "ada": "text-embedding-ada-002",     # 1536维 (legacy)
    }
    
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "small",
        dimensions: Optional[int] = None  # 仅 v3 模型支持自定义维度
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = self.MODELS.get(model, model)
        self.dimensions = dimensions
        self._client = None
        
    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai is required. Install with: pip install openai")
        return self._client
    
    def encode(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """编码文本（支持批处理）"""
        client = self._get_client()
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            kwargs = {"model": self.model_name, "input": batch}
            if self.dimensions and "3" in self.model_name:
                kwargs["dimensions"] = self.dimensions
                
            response = client.embeddings.create(**kwargs)
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def get_dimension(self) -> int:
        if self.dimensions:
            return self.dimensions
        return self.DIMENSIONS.get(self.model_name, 1536)


# ==================== BM25 稀疏检索 ====================

class BM25Retriever:
    """
    BM25 稀疏检索器
    用于混合检索策略中的词汇匹配部分
    """
    
    def __init__(
        self, 
        k1: float = 1.5, 
        b: float = 0.75,
        tokenizer: Optional[Callable] = None
    ):
        """
        Args:
            k1: 词频饱和参数
            b: 文档长度归一化参数
            tokenizer: 自定义分词器
        """
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or self._default_tokenizer
        
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.corpus_size: int = 0
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        
    def _default_tokenizer(self, text: str) -> List[str]:
        """默认分词器：支持中英文"""
        # 简单的中英文分词
        text = text.lower()
        # 英文按空格分词
        tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        # 中文按字符分词
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        tokens.extend(chinese_chars)
        return tokens
    
    def fit(self, documents: List[str]):
        """构建 BM25 索引"""
        self.corpus_size = len(documents)
        self.doc_term_freqs = []
        self.doc_lens = []
        self.doc_freqs = defaultdict(int)
        
        for doc in documents:
            tokens = self.tokenizer(doc)
            self.doc_lens.append(len(tokens))
            
            # 计算词频
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            self.doc_term_freqs.append(dict(term_freq))
            
            # 更新文档频率
            for token in set(tokens):
                self.doc_freqs[token] += 1
        
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
        
        # 计算 IDF
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        """计算查询与文档的 BM25 分数"""
        query_tokens = self.tokenizer(query)
        doc_term_freq = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        
        score = 0.0
        for token in query_tokens:
            if token not in doc_term_freq:
                continue
                
            tf = doc_term_freq[token]
            idf = self.idf.get(token, 0)
            
            # BM25 公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * numerator / denominator
            
        return score
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """检索最相关的文档"""
        scores = [(i, self.score(query, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ==================== FAISS 向量索引 ====================

class FAISSIndex:
    """
    FAISS 向量索引封装
    支持多种索引类型和 GPU 加速
    """
    
    INDEX_TYPES = {
        "flat": "IndexFlatIP",      # 精确搜索，适合小规模数据
        "ivf": "IndexIVFFlat",      # 倒排索引，适合中等规模
        "hnsw": "IndexHNSWFlat",    # HNSW 图索引，高召回率
        "pq": "IndexIVFPQ",         # 乘积量化，适合大规模数据
    }
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        use_gpu: bool = False,
        nlist: int = 100,  # IVF 的聚类数
        m: int = 8,        # HNSW 的连接数
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.nlist = nlist
        self.m = m
        self.index = None
        self._faiss = None
        
    def _import_faiss(self):
        if self._faiss is None:
            try:
                import faiss
                self._faiss = faiss
            except ImportError:
                raise ImportError(
                    "faiss is required. Install with: "
                    "pip install faiss-cpu (CPU) or pip install faiss-gpu (GPU)"
                )
        return self._faiss
    
    def build(self, embeddings: np.ndarray):
        """构建索引"""
        faiss = self._import_faiss()
        
        n_vectors = embeddings.shape[0]
        embeddings = embeddings.astype(np.float32)
        
        # 归一化向量（用于内积相似度）
        faiss.normalize_L2(embeddings)
        
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
            
        elif self.index_type == "ivf":
            # 调整 nlist 以适应数据量
            nlist = min(self.nlist, n_vectors // 10)
            nlist = max(nlist, 1)
            
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings)
            
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, self.m)
            
        elif self.index_type == "pq":
            nlist = min(self.nlist, n_vectors // 10)
            nlist = max(nlist, 1)
            
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, 8, 8)
            self.index.train(embeddings)
        
        # 添加向量
        self.index.add(embeddings)
        
        # GPU 加速
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                logger.warning(f"GPU not available, using CPU: {e}")
        
        logger.info(f"Built FAISS index: {self.index_type}, {n_vectors} vectors")
    
    def search(
        self, 
        query_embeddings: np.ndarray, 
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最近邻
        
        Returns:
            (distances, indices): 形状均为 (n_queries, top_k)
        """
        faiss = self._import_faiss()
        
        query_embeddings = query_embeddings.astype(np.float32)
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        faiss.normalize_L2(query_embeddings)
        
        # 设置搜索参数
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(10, self.nlist)
        
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices
    
    def save(self, path: str):
        """保存索引"""
        faiss = self._import_faiss()
        
        # 如果是 GPU 索引，先转回 CPU
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        else:
            index_to_save = self.index
            
        faiss.write_index(index_to_save, path)
    
    def load(self, path: str):
        """加载索引"""
        faiss = self._import_faiss()
        self.index = faiss.read_index(path)
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)


# ==================== 增强版塔罗知识检索器 ====================

class EnhancedTarotRetriever:
    """
    增强版塔罗知识检索器
    
    特性：
    1. 支持多种嵌入模型（Sentence-BERT, OpenAI）
    2. 支持混合检索（稠密 + 稀疏）
    3. 支持 FAISS 高效索引
    4. 完整的评估指标
    """
    
    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        index_type: str = "flat",
        use_gpu: bool = False,
        enable_hybrid: bool = True,
        hybrid_alpha: float = 0.7,  # 稠密检索权重
    ):
        """
        Args:
            embedder: 嵌入模型，默认使用 Sentence-BERT
            index_type: FAISS 索引类型
            use_gpu: 是否使用 GPU
            enable_hybrid: 是否启用混合检索
            hybrid_alpha: 混合检索中稠密检索的权重
        """
        self.embedder = embedder
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.enable_hybrid = enable_hybrid
        self.hybrid_alpha = hybrid_alpha
        
        self.documents: List[Document] = []
        self.faiss_index: Optional[FAISSIndex] = None
        self.bm25: Optional[BM25Retriever] = None
        self.is_initialized = False
        
    def _init_embedder(self):
        """初始化嵌入模型"""
        if self.embedder is None:
            self.embedder = SentenceBERTEmbedder(model_name="default")
    
    def add_documents(self, documents: List[Dict]):
        """
        添加文档到索引
        
        Args:
            documents: 文档列表，每个文档包含 id, text, metadata
        """
        for doc in documents:
            self.documents.append(Document(
                id=doc["id"],
                text=doc["text"],
                metadata=doc.get("metadata", {})
            ))
    
    def build_index(self, show_progress: bool = True):
        """构建索引"""
        self._init_embedder()
        
        if not self.documents:
            logger.warning("No documents to index")
            return
        
        logger.info(f"Building index for {len(self.documents)} documents...")
        
        # 1. 生成嵌入
        texts = [doc.text for doc in self.documents]
        embeddings = self.embedder.encode(texts, show_progress=show_progress)
        
        # 存储嵌入
        for i, doc in enumerate(self.documents):
            doc.embedding = embeddings[i]
        
        # 2. 构建 FAISS 索引
        dimension = self.embedder.get_dimension()
        self.faiss_index = FAISSIndex(
            dimension=dimension,
            index_type=self.index_type,
            use_gpu=self.use_gpu
        )
        self.faiss_index.build(embeddings)
        
        # 3. 构建 BM25 索引（如果启用混合检索）
        if self.enable_hybrid:
            self.bm25 = BM25Retriever()
            self.bm25.fit(texts)
        
        self.is_initialized = True
        logger.info("Index built successfully")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_fn: Optional[Callable[[Document], bool]] = None,
        method: str = "auto"  # auto, dense, sparse, hybrid
    ) -> List[RetrievedChunk]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            filter_fn: 过滤函数
            method: 检索方法
            
        Returns:
            检索结果列表
        """
        if not self.is_initialized:
            self.build_index()
        
        # 决定检索方法
        if method == "auto":
            method = "hybrid" if self.enable_hybrid else "dense"
        
        # 稠密检索
        dense_results = []
        if method in ["dense", "hybrid"]:
            dense_results = self._dense_retrieve(query, top_k * 2)
        
        # 稀疏检索
        sparse_results = []
        if method in ["sparse", "hybrid"] and self.bm25:
            sparse_results = self._sparse_retrieve(query, top_k * 2)
        
        # 合并结果
        if method == "hybrid":
            results = self._merge_results(dense_results, sparse_results, top_k)
        elif method == "dense":
            results = dense_results[:top_k]
        else:
            results = sparse_results[:top_k]
        
        # 应用过滤
        if filter_fn:
            results = [r for r in results if filter_fn(self._get_document(r.id))]
        
        return results[:top_k]
    
    def _dense_retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """稠密检索"""
        query_embedding = self.embedder.encode([query])
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # FAISS 返回 -1 表示无效
                continue
            doc = self.documents[idx]
            results.append(RetrievedChunk(
                id=doc.id,
                text=doc.text,
                score=float(distances[0][i]),
                metadata=doc.metadata,
                retrieval_method="dense"
            ))
        
        return results
    
    def _sparse_retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """稀疏检索"""
        bm25_results = self.bm25.retrieve(query, top_k)
        
        results = []
        for idx, score in bm25_results:
            doc = self.documents[idx]
            results.append(RetrievedChunk(
                id=doc.id,
                text=doc.text,
                score=score,
                metadata=doc.metadata,
                retrieval_method="sparse"
            ))
        
        return results
    
    def _merge_results(
        self,
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk],
        top_k: int
    ) -> List[RetrievedChunk]:
        """
        合并稠密和稀疏检索结果（RRF: Reciprocal Rank Fusion）
        """
        k = 60  # RRF 参数
        
        # 计算 RRF 分数
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, RetrievedChunk] = {}
        
        for rank, result in enumerate(dense_results):
            doc_scores[result.id] += self.hybrid_alpha / (k + rank + 1)
            doc_map[result.id] = result
        
        for rank, result in enumerate(sparse_results):
            doc_scores[result.id] += (1 - self.hybrid_alpha) / (k + rank + 1)
            if result.id not in doc_map:
                doc_map[result.id] = result
        
        # 排序并返回
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            chunk = doc_map[doc_id]
            results.append(RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                score=score,
                metadata=chunk.metadata,
                retrieval_method="hybrid"
            ))
        
        return results
    
    def _get_document(self, doc_id: str) -> Optional[Document]:
        """根据 ID 获取文档"""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def retrieve_for_card(
        self,
        card_name: str,
        context: str = "",
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """检索特定塔罗牌的相关知识"""
        query = f"{card_name} {context}".strip()
        
        # 首先精确匹配牌名
        exact_matches = []
        for doc in self.documents:
            if doc.metadata.get("type") == "card":
                if (card_name.lower() in doc.metadata.get("name", "").lower() or
                    card_name in doc.metadata.get("name_cn", "")):
                    exact_matches.append(RetrievedChunk(
                        id=doc.id,
                        text=doc.text,
                        score=1.0,
                        metadata=doc.metadata,
                        retrieval_method="exact"
                    ))
        
        # 然后语义检索
        semantic_results = self.retrieve(query, top_k=top_k)
        
        # 合并，去重
        seen_ids = {r.id for r in exact_matches}
        combined = exact_matches[:]
        for r in semantic_results:
            if r.id not in seen_ids:
                combined.append(r)
                seen_ids.add(r.id)
        
        return combined[:top_k]
    
    def save(self, path: str):
        """保存检索器状态"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存文档
        docs_data = [{
            "id": doc.id,
            "text": doc.text,
            "metadata": doc.metadata
        } for doc in self.documents]
        
        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        # 保存嵌入
        embeddings = np.array([doc.embedding for doc in self.documents if doc.embedding is not None])
        np.save(path / "embeddings.npy", embeddings)
        
        # 保存 FAISS 索引
        if self.faiss_index:
            self.faiss_index.save(str(path / "faiss.index"))
        
        # 保存 BM25
        if self.bm25:
            with open(path / "bm25.pkl", "wb") as f:
                pickle.dump(self.bm25, f)
        
        # 保存配置
        config = {
            "index_type": self.index_type,
            "enable_hybrid": self.enable_hybrid,
            "hybrid_alpha": self.hybrid_alpha
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)
        
        logger.info(f"Retriever saved to {path}")
    
    def load(self, path: str):
        """加载检索器状态"""
        path = Path(path)
        
        # 加载配置
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        self.index_type = config["index_type"]
        self.enable_hybrid = config["enable_hybrid"]
        self.hybrid_alpha = config["hybrid_alpha"]
        
        # 加载文档
        with open(path / "documents.json", "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        
        # 加载嵌入
        embeddings = np.load(path / "embeddings.npy")
        
        self.documents = []
        for i, doc_data in enumerate(docs_data):
            self.documents.append(Document(
                id=doc_data["id"],
                text=doc_data["text"],
                metadata=doc_data["metadata"],
                embedding=embeddings[i] if i < len(embeddings) else None
            ))
        
        # 初始化嵌入器
        self._init_embedder()
        
        # 加载 FAISS 索引
        self.faiss_index = FAISSIndex(
            dimension=self.embedder.get_dimension(),
            index_type=self.index_type,
            use_gpu=self.use_gpu
        )
        self.faiss_index.load(str(path / "faiss.index"))
        
        # 加载 BM25
        if self.enable_hybrid and (path / "bm25.pkl").exists():
            with open(path / "bm25.pkl", "rb") as f:
                self.bm25 = pickle.load(f)
        
        self.is_initialized = True
        logger.info(f"Retriever loaded from {path}")


# ==================== 全局实例 ====================

_retriever_instance: Optional[EnhancedTarotRetriever] = None


def get_enhanced_retriever(
    rebuild: bool = False,
    **kwargs
) -> EnhancedTarotRetriever:
    """获取增强版检索器单例"""
    global _retriever_instance
    
    if _retriever_instance is None or rebuild:
        _retriever_instance = EnhancedTarotRetriever(**kwargs)
    
    return _retriever_instance
