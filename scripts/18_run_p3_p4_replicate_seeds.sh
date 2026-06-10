#!/usr/bin/env bash

cd /home/manikm/meddataforge || exit 1
source .venv/bin/activate

mkdir -p results/logs results/skin_baselines

echo "===== P3 seed 20260529 ====="
CUDA_VISIBLE_DEVICES=7 python scripts/15_train_skin_resnet_baseline.py \
  --split_csv data/training_splits/P3_compiler_filtered_pool_7class.csv \
  --out_dir results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class_seed20260529 \
  --epochs 5 \
  --batch_size 192 \
  --num_workers 8 \
  --lr 3e-4 \
  --seed 20260529 \
  --pretrained

echo "===== P3 seed 20260530 ====="
CUDA_VISIBLE_DEVICES=7 python scripts/15_train_skin_resnet_baseline.py \
  --split_csv data/training_splits/P3_compiler_filtered_pool_7class.csv \
  --out_dir results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class_seed20260530 \
  --epochs 5 \
  --batch_size 192 \
  --num_workers 8 \
  --lr 3e-4 \
  --seed 20260530 \
  --pretrained

echo "===== P4 seed 20260529 ====="
CUDA_VISIBLE_DEVICES=7 python scripts/15_train_skin_resnet_baseline.py \
  --split_csv data/training_splits/P4_naive_pool_including_flagged_8class.csv \
  --out_dir results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class_seed20260529 \
  --epochs 5 \
  --batch_size 192 \
  --num_workers 8 \
  --lr 3e-4 \
  --seed 20260529 \
  --pretrained

echo "===== P4 seed 20260530 ====="
CUDA_VISIBLE_DEVICES=7 python scripts/15_train_skin_resnet_baseline.py \
  --split_csv data/training_splits/P4_naive_pool_including_flagged_8class.csv \
  --out_dir results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class_seed20260530 \
  --epochs 5 \
  --batch_size 192 \
  --num_workers 8 \
  --lr 3e-4 \
  --seed 20260530 \
  --pretrained

echo "===== ALL P3/P4 REPLICATE SEED RUNS COMPLETE ====="
