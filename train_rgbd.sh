#!/bin/bash
# RGBD Model Training Script

echo "======================================"
echo "Training RGBD Model"
echo "======================================"

# Check if depth maps exist
if [ ! -d "./cvpr2017_cvusa/train/satellite_depth" ]; then
    echo "⚠️  Warning: Depth maps not found!"
    echo "Please run MiDaS depth generation first."
    echo "See COLAB_SATELLITE_DRONE.ipynb Hücre 4.5"
    exit 1
fi

echo ""
echo "✅ Depth maps found, proceeding with RGBD training..."

# Train RGBD model
python train_cvusa.py \
    --name rgbd_exp1 \
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
echo "✅ RGBD training completed!"
echo "Model saved in: model/rgbd_exp1/"
