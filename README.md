# Venture Description Analyzer

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![BERTopic](https://img.shields.io/badge/BERTopic-0.15%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

## About the Project

The Venture Description Analyzer is an NLP-based pipeline engineered to automatically cluster and categorize unstructured startup and business descriptions into meaningful thematic groups. Designed to facilitate scalable downstream quantitative analysis for venture capital evaluations, the engine leverages transformer embeddings and `BERTopic` to extract latent startup domains.

By transforming raw, unstructured text into enriched datasets with confidence scores and human-interpretable labels, this tool enables rapid market segmentation, startup landscape mapping, and venture trend analysis.

### Core Features
* **Semantic Representation:** Generates dense embeddings using `SentenceTransformers` (`all-MiniLM-L6-v2`) optimized for GPU acceleration.
* **Unsupervised Clustering:** Utilizes `BERTopic` configured for high interpretability (`min_topic_size=10`).
* **Noise Reduction:** Automatically filters out outlier clusters (Topic `-1`) to ensure high-confidence thematic grouping.
* **Data Enrichment:** Exports structured JSON/CSV datasets containing Topic IDs, probability metrics, and aggregate keyword labels.

## Prerequisites

* Python 3.8+
* CUDA-enabled GPU (Highly recommended for transformer inference)

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Harsh-7612/Venture-description-analysis.git](https://github.com/Harsh-7612/Venture-description-analysis.git)
   cd Venture-description-analysis
   ```
---

### 3. Code & Dependency Review

Based on the libraries and configurations implemented in your pipeline, several standard configuration files are missing.

**1. Create a `requirements.txt` file:**
Create this file at the root level to ensure reproducibility across different environments.
```text
pandas>=1.3.0
sentence-transformers>=2.2.2
bertopic>=0.15.0
torch>=1.9.0
scikit-learn>=1.0.2
```
