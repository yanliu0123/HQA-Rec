#!/bin/bash
# ============================================================
# Three-Step Pipeline: Lorentz Embedding -> RQ-VAE -> Token Self-Attention
# ============================================================
# Usage:
#   bash run_pipeline.sh                          # NLP, all 3 steps
#   bash run_pipeline.sh --dataset ML             # ML, all 3 steps
#   bash run_pipeline.sh --step 2                 # NLP, from Step 2
#   bash run_pipeline.sh --dataset CV --step 3    # CV, only Step 3
#   bash run_pipeline.sh --all                    # All 3 datasets
# ============================================================

set -e

DEVICE="cuda:0"
DATASET="NLP"
START_STEP=1
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --step)    START_STEP="$2"; shift 2 ;;
        --device)  DEVICE="$2"; shift 2 ;;
        --all)     RUN_ALL=true; shift ;;
        *)         echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")"

run_dataset() {
    local DS=$1
    echo ""
    echo "######################################################"
    echo "# Dataset: $DS"
    echo "######################################################"

    # ========== Step 1: Hyperbolic Embedding ==========
    if [ "$START_STEP" -le 1 ]; then
        echo "========== [$DS] Step 1: Hyperbolic Embedding =========="
        PYTHONUNBUFFERED=1 python step1_hyperbolic_embedding.py \
            --dataset "$DS" \
            --embedding_dim 768 \
            --device "$DEVICE" \
            --save_path "./output/${DS}_embeddings_v3_dim768.npy" \
            --eval_freq 5 \
            --p1_epochs 50 \
            --p1_batch_size 2048 \
            --p1_patience 3 \
            --margin 0.5 \
            --num_neg 50 \
            --lambda_c 1.0 \
            --p2_epochs 500 \
            --p2_batch_size 10000 \
            --p2_lr 0.001 \
            --p2_momentum 0.95 \
            --p2_weight_decay 0.005 \
            --p2_patience 3
        echo "[$DS] Step 1 done."
    fi

    # ========== Step 2: RQ-VAE ==========
    if [ "$START_STEP" -le 2 ]; then
        echo "========== [$DS] Step 2: RQ-VAE =========="
        PYTHONUNBUFFERED=1 python step2_train_rqvae.py \
            --dataset "$DS" \
            --device "$DEVICE"
        echo "[$DS] Step 2 done."
    fi

    # ========== Step 3: Token Self-Attention ==========
    if [ "$START_STEP" -le 3 ]; then
        echo "========== [$DS] Step 3: Token Self-Attention =========="
        PYTHONUNBUFFERED=1 python step3_train_token_selfattn.py \
            --dataset "$DS" \
            --device "$DEVICE" \
            --K 20 \
            --epochs 200 \
            --lr 1e-3 \
            --n_neg 15 \
            --proj_dim 128 \
            --eval_every 20
        echo "[$DS] Step 3 done."
    fi
}

if [ "$RUN_ALL" = true ]; then
    for DS in NLP ML CV; do
        run_dataset "$DS"
    done
else
    run_dataset "$DATASET"
fi

echo ""
echo "========== Pipeline complete =========="
