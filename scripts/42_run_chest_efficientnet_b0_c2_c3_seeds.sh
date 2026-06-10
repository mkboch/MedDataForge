#!/usr/bin/env bash
set -e

cd /home/manikm/meddataforge
source .venv/bin/activate

mkdir -p results/logs results/chest_baselines

for SEED in 20260602 20260603 20260604; do
  echo "===== EfficientNet-B0 chest C2 compiler-compatible seed ${SEED} ====="
  CUDA_VISIBLE_DEVICES=7 python scripts/27_train_skin_backbone_baseline.py \
    --split_csv data/training_splits/chest/C2_compiler_compatible_binary_pool.csv \
    --out_dir results/chest_baselines/C2_efficientnet_b0_compiler_compatible_binary_pool_seed${SEED} \
    --backbone efficientnet_b0 \
    --epochs 5 \
    --batch_size 192 \
    --num_workers 8 \
    --lr 3e-4 \
    --seed ${SEED} \
    --pretrained
done

for SEED in 20260602 20260603 20260604; do
  echo "===== EfficientNet-B0 chest C3 naive noisy pool seed ${SEED} ====="
  CUDA_VISIBLE_DEVICES=7 python scripts/27_train_skin_backbone_baseline.py \
    --split_csv data/training_splits/chest/C3_naive_binary_pool_all_abnormal_as_pneumonia.csv \
    --out_dir results/chest_baselines/C3_efficientnet_b0_naive_binary_pool_all_abnormal_seed${SEED} \
    --backbone efficientnet_b0 \
    --epochs 5 \
    --batch_size 192 \
    --num_workers 8 \
    --lr 3e-4 \
    --seed ${SEED} \
    --pretrained
done

echo "===== ALL CHEST EFFICIENTNET-B0 C2/C3 SEED RUNS COMPLETE ====="
