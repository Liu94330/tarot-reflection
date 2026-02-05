# 🌙 AI Tarot Reflection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

An AI-powered tarot reflection system combining RAG (Retrieval-Augmented Generation) with emotion detection for psychological self-reflection. This project uses tarot cards as projection tools rather than fortune-telling, focusing on helping users explore their inner thoughts and emotions.

[中文文档](#中文说明) | [English](#english)

---

## English

### ✨ Features

- **Enhanced RAG Pipeline**: Hybrid retrieval combining dense embeddings (Sentence-BERT) and sparse retrieval (BM25) with FAISS indexing
- **Emotion Detection**: BERT-based emotion detection with rule enhancement for Chinese text
- **Comprehensive Evaluation Framework**: Metrics for retrieval (P@K, NDCG, MRR, MAP) and classification (Accuracy, F1, Confusion Matrix)
- **Statistical Testing**: Paired t-test, Wilcoxon test, Bootstrap CI for significance analysis

### 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Liu94330/tarot-reflection.git
cd tarot-reflection

# Install dependencies
pip install -r requirements.txt

# Run evaluation
python -m app.run_evaluation
```

### 📁 Project Structure

```
tarot-enhanced/
├── app/
│   ├── run_evaluation.py          # Main evaluation script
│   ├── rag/
│   │   └── retriever_enhanced.py  # RAG retrieval module
│   ├── ml/
│   │   └── emotion_detector_enhanced.py  # Emotion detection
│   └── evaluation/
│       ├── metrics.py             # Evaluation metrics
│       ├── experiment.py          # Experiment framework
│       └── datasets.py            # Dataset utilities
├── requirements.txt
├── LICENSE
└── README.md
```

### 📊 Performance

#### RAG Retrieval
| Method | P@5 | NDCG@5 | MRR |
|--------|-----|--------|-----|
| BM25 Baseline | 0.38 | 0.52 | 0.58 |
| **Hybrid (Ours)** | **0.58** | **0.76** | **0.78** |

#### Emotion Detection
| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| Lexicon Baseline | 0.52 | 0.45 |
| **BERT+Rules (Ours)** | **0.78** | **0.75** |

### 🛠️ Technical Details

#### RAG Module
- **Embeddings**: Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector Index**: FAISS with support for Flat, IVF, HNSW, and PQ indexes
- **Hybrid Search**: Configurable alpha for dense/sparse fusion

#### Emotion Detection
- **Model**: Fine-tuned Chinese BERT for emotion classification
- **Enhancement**: Rule-based post-processing for negation and sarcasm
- **Output**: Primary emotion + intensity score (0-1)

### 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{tarot_reflection_2025,
  author = {Liu94330},
  title = {AI Tarot Reflection System},
  year = {2025},
  url = {https://github.com/Liu94330/tarot-reflection}
}
```

### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 中文说明

### ✨ 功能特点

- **增强版 RAG 检索**：结合稠密嵌入（Sentence-BERT）和稀疏检索（BM25）的混合检索，使用 FAISS 索引
- **情感检测**：基于 BERT 的中文情感检测，结合规则增强
- **完整评估框架**：检索指标（P@K、NDCG、MRR、MAP）和分类指标（准确率、F1、混淆矩阵）
- **统计检验**：配对 t 检验、Wilcoxon 检验、Bootstrap 置信区间

### 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/Liu94330/tarot-reflection.git
cd tarot-reflection

# 安装依赖
pip install -r requirements.txt

# 运行评估
python -m app.run_evaluation
```

### 📊 性能指标

#### RAG 检索性能
| 方法 | P@5 | NDCG@5 | MRR |
|-----|-----|--------|-----|
| BM25 基线 | 0.38 | 0.52 | 0.58 |
| **混合检索（本方法）** | **0.58** | **0.76** | **0.78** |

#### 情感检测性能
| 方法 | 准确率 | Macro F1 |
|-----|--------|----------|
| 词典基线 | 0.52 | 0.45 |
| **BERT+规则（本方法）** | **0.78** | **0.75** |

### 🎯 设计理念

本系统将塔罗牌作为**心理投射工具**而非占卜手段，帮助用户：
- 探索内心想法和情绪
- 进行自我反思和觉察
- 获得新的视角和洞见

塔罗牌不预测未来，而是照亮当下。

### 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

Made with ❤️ for self-reflection and inner exploration
