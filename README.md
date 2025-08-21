# PaperRag: Paper Keypoint Retrieval

This repository provides a FAISS-based retrieval system over **keypoints extracted from top-tier AI/ML conference papers**.  
Keypoints are short, information-rich statements (5–20 per paper) covering contributions, results, and insights.  

Embeddings are generated with **[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)** and indexed in FAISS for efficient similarity search.  
A reranker model **[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)** can be used to refine results.  

---
## Use Case Examples

### 1. Query and Review Generation

Researchers can issue a natural language query and automatically retrieve the most relevant keypoints across thousands of papers.  
This allows fast generation of concise literature summaries or draft reviews for specific topics.  

**Example:** Searching for `"diffusion models for video generation"` retrieves key contributions, performance metrics, and methodological insights from recent papers.  
These retrieved keypoints can then be synthesized into a structured review.

---

### 2. Field Analysis

The system can be used to analyze the distribution of research topics, methods, or datasets across conferences and years.  

<img width="1987" height="1490" alt="Field Analysis Output" src="https://github.com/user-attachments/assets/994b4a8f-eccd-49f4-9221-03c5d31de88b" />

---

### 3. Trend Analysis

Temporal analysis can reveal hot topics, rising techniques, or shifts in methodology over the years.  

<img width="1993" height="589" alt="Trend Analysis Output" src="https://github.com/user-attachments/assets/da940e48-6a5f-4829-9970-fa13e0897d64" />

---

### 4. Extended Applications

Beyond basic query and retrieval, the system supports advanced workflows like research direction mining, and paper polishing methods, etc. The application is limitless.

---

## 📥 Setup

### 1. Download Models from Hugging Face

```bash
# Clone embedding model
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-Embedding-0.6B Qwen3-Embedding-0.6B

# Clone reranker model
git clone https://huggingface.co/Qwen/Qwen3-Reranker-0.6B Qwen3-Reranker-0.6B
```
Or any other way you prefer.

---

### 2. Download FAISS Index + Metadata

Download the prebuilt FAISS index and metadata from Tsinghua Cloud:

[📂 Download link](https://cloud.tsinghua.edu.cn/d/635b2879fae7454ab9e1/)

Place files in a folder, e.g.:

```
faiss_index_qwen3_0.6B_Large/
    ├── index.faiss
    └── meta.json
```

---

## 📊 Dataset Overview

The FAISS index is built from **389,117 keypoints** (as of *Aug 21, 2025*), extracted from **21,825+ papers** across top conferences.

### Paper Counts by Year × Conference

| Year | ICLR | AAAI | ACL | CVPR | ICML | IJCAI | NeurIPS |
|------|------|------|-----|------|------|-------|---------|
| 2022 | 1093 | 1319 |  –  |  –   | 1233 | 862   | 2830    |
| 2023 | 1369 | 1405 | 911 | 2353 | 1828 | 851   | 3540    |
| 2024 | 1453 | 2331 | 865 | 2713 | 2610 | 1048  | 4492    |
| 2025 | 3699 | 3361 | 1601|  –   |  –   |  –    |   –     |

---

## 🔎 Keypoint Structure

Each paper contributes 5–20 keypoints. Example:

```json
{
  "file": "[NeurIPS 2022]Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation_f54e7434.json",
  "index": 5,
  "text": "DDB achieves state-of-the-art performance on GTA5→Cityscapes, GTA5+Synscapes→Cityscapes, and GTA5→Cityscapes+Mapillary benchmarks. It outperforms methods like ProDA, ADVENT, and CPSL by significant margins, particularly in handling class bias and multi-domain adaptation scenarios.",
  "title": "Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation",
  "authors": "Lin Chen, Zhixiang Wei, Xin Jin, Huaian Chen, Miao Zheng, Kai Chen, Yi Jin",
  "year": "2022",
  "venue": "NeurIPS",
  "field": ["Computer Vision", "Domain Adaptation", "Semantic Segmentation"],
  "first_author_affiliation": "University of Science and Technology of China"
}
```

---

## 🧠 Embedding

- Model: **Qwen3-Embedding-0.6B**
- Dimension: *1024*
- Storage: **FAISS index + JSON metadata**
- Current size: **389,117 keypoints** (expanding continuously)

---

## 🚀 Usage (see Jupyter Notebook Example)

```python
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import json

# Hardcoded paths
EMBEDDING_MODEL_PATH = "Qwen3-Embedding-0.6B"
RERANKER_MODEL_PATH = "Qwen3-Reranker-0.6B"
FAISS_INDEX_PATH = "faiss_index_qwen3_0.6B_Large/index.faiss"
FAISS_META_PATH = "faiss_index_qwen3_0.6B_Large/meta.json"

# --- Load FAISS index and metadata ---
index = faiss.read_index(FAISS_INDEX_PATH)
with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# --- Init embedding model (first LLM call in kernel must be embed) ---
embedder = LLM(model=EMBEDDING_MODEL_PATH, task="embed")

# --- Example query ---
QUERY = "LoRA fine-tune LLM with time-series data"

# --- Embed query ---
query_emb = embedder.embed([QUERY])[0].outputs.embedding
query_emb = np.array(query_emb, dtype=np.float32).reshape(1, -1)

# --- FAISS search ---
TOP_K = 5
_, idxs = index.search(query_emb, TOP_K)

# --- Retrieve metadata ---
results = [metadata[i] for i in idxs[0]]

# --- Print results ---
for rank, r in enumerate(results, 1):
    print(f"[{rank}] {r['title']} ({r['year']}): {r['text'][:200]}...")
```

---

## 📌 Notes
- The index is continuously updated as new conferences are published.
- Each keypoint is designed to be **retrieval-friendly** for downstream RAG and literature survey tasks.
- This system is optimized for **semantic retrieval**, not full-text search.

---
