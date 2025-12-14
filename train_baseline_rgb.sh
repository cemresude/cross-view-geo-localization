#!/bin/bash
# Baseline RGB-only Model Training Script

echo "======================================"
echo "Training RGB-only Baseline Model"
echo "======================================"

# Model comparison
echo ""
echo "📊 Comparing models before training..."
python compare_models.py

echo ""
echo "======================================"
echo "Starting RGB-only Training"
echo "======================================"

# Train RGB-only model
python train_cvusa.py \
    --name rgb_baseline_exp1 \
    --data_dir ./cvpr2017_cvusa/train \
    --gpu_ids 0 \
    --batchsize 8 \
    --h 384 \
    --w 384 \
    --views 2 \
    --pool avg \
    --lr 0.01 \
    --droprate 0.5 \
    --stride 2 \
    --erasing_p 0.5 \
    --color_jitter \
    --train_all

echo ""
echo "✅ RGB-only training completed!"
echo "Model saved in: model/rgb_baseline_exp1/"
