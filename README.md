# HQA-Cite: Hyperbolic Quantization Alignment for Citation Recommendation

This repository contains the official implementation of **HQA-Cite**, a three-stage pipeline for citation recommendation that leverages hyperbolic geometry, residual vector quantization, and multi-level self-attention.

## Overview

HQA-Cite operates in three sequential stages:

1. **Hyperbolic Embedding** (`step1_hyperbolic_embedding.py`): Learns paper representations on the Lorentz manifold via a two-phase approach.

2. **Residual Quantized VAE** (`step2_train_rqvae.py`): Compresses continuous Lorentz embeddings into compact discrete tokens using an 8-layer residual vector quantizer with contrastive learning.

3. **Token Self-Attention** (`step3_train_token_selfattn.py`): Recommends citations via two-level self-attention — token-level attention captures multi-grained semantics, neighbor-level attention builds context — with candidate-independent scoring for fast inference.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── run_pipeline.sh                    # End-to-end pipeline script
├── step1_hyperbolic_embedding.py      # Stage 1: Lorentz embedding
├── step2_train_rqvae.py               # Stage 2: RQ-VAE tokenization
├── step3_train_token_selfattn.py      # Stage 3: Self-attention recommendation
├── data_loader.py                     # Data loading and temporal split
├── evaluate.py                        # Evaluation metrics (Recall, NDCG, MAP, MRR)
└── data/
    ├── process_dblp.py                # Data preprocessing script
    └── processed/                     # Pre-processed datasets
        ├── machine_learning.json
        ├── computer_vision.json
        └── natural_language_processing.json
```

## Requirements

- Python 3.8+
- PyTorch >= 1.13.0
- transformers >= 4.20.0 (for SPECTER encoder in Stage 3)
- numpy, tqdm

```bash
pip install -r requirements.txt
```

## Datasets

We use three sub-datasets extracted from [DBLP Citation Network v12](https://www.aminer.cn/citation):

| Dataset | Papers | Citation Edges | Train (< 2016) | Test (>= 2016) |
|---------|--------|----------------|-----------------|-----------------|
| ML      | 75,952 | 763,167        | 65,312          | 9,616           |
| CV      | 88,374 | 1,062,298      | 81,409          | 6,778           |
| NLP     | 22,327 | 208,905        | 20,423          | 1,622           |

Download at https://drive.google.com/file/d/12Nma6m5u8TQSp7HmJ4iJdAVcB_bIBlYP/view?usp=sharing

## Quick Start

### Run the full pipeline on a single dataset

```bash
bash run_pipeline.sh --dataset NLP --device cuda:0
```

### Run step by step

```bash
# Step 1: Hyperbolic Embedding
python step1_hyperbolic_embedding.py \
    --dataset NLP --embedding_dim 768 --device cuda:0 \
    --save_path ./output/NLP_embeddings_v3_dim768.npy

# Step 2: RQ-VAE Tokenization
python step2_train_rqvae.py --dataset NLP --device cuda:0

# Step 3: Token Self-Attention Recommendation
python step3_train_token_selfattn.py \
    --dataset NLP --device cuda:0 \
    --K 20 --epochs 200 --lr 1e-3
```


## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 768 | Embedding dimensionality (Step 1) |
| `margin` | 0.5 | CML hinge loss margin (Step 1) |
| `num_neg` | 50 | Negative samples per positive (Step 1) |
| `num_emb` | 4096 x 8 | Codebook size per RQ level (Step 2) |
| `e_dim` | 512 | Quantization embedding dim (Step 2) |
| `K` | 20 | Number of SPECTER neighbors (Step 3) |
| `proj_dim` | 128 | Self-attention projection dim (Step 3) |
| `alpha` | 0.0-0.5 | Hybrid scoring weight (Step 3 eval) |


