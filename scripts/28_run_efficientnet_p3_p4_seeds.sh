#!/usr/bin/env bash
set -e

cd /home/manikm/meddataforge
source .venv/bin/activate

mkdir -p results/logs results/skin_baselines

for SEED in 20260528 20260529 20260530; do
  echo "===== EfficientNet-B0 P3 seed ${SEED} ====="
  CUDA_VISIBLE_DEVICES=7 python scripts/27_train_skin_backbone_baseline.py \
    --split_csv data/training_splits/P3_compiler_filtered_pool_7class.csv \
    --out_dir results/skin_baselines/P3_efficientnet_b0_compiler_filtered_pool_7class_seed${SEED} \
    --backbone efficientnet_b0 \
    --epochs 5 \
    --batch_size 192 \
    --num_workers 8 \
    --lr 3e-4 \
    --seed ${SEED} \
    --pretrained
done

for SEED in 20260528 20260529 20260530; do
  echo "===== EfficientNet-B0 P4 seed ${SEED} ====="
  CUDA_VISIBLE_DEVICES=7 python scripts/27_train_skin_backbone_baseline.py \
    --split_csv data/training_splits/P4_naive_pool_including_flagged_8class.csv \
    --out_dir results/skin_baselines/P4_efficientnet_b0_naive_pool_flagged_8class_seed${SEED} \
    --backbone efficientnet_b0 \
    --epochs 5 \
    --batch_size 192 \
    --num_workers 8 \
    --lr 3e-4 \
    --seed ${SEED} \
    --pretrained
done

echo "===== ALL EFFICIENTNET-B0 P3/P4 SEED RUNS COMPLETE ====="
