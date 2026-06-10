#!/usr/bin/env bash
set -euo pipefail

cd /home/manikm/meddataforge
source .venv/bin/activate

mkdir -p results/logs results/chest_dedup_baselines results/tables results/figures

echo "===== START CHEST DEDUP OVERNIGHT RUN ====="
date

echo "===== SELECT GPU DYNAMICALLY ====="
GPU_ID=$(python - <<'PY'
import subprocess, re
out = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.used,memory.total",
    "--format=csv,noheader,nounits"
], text=True)
best = None
for line in out.strip().splitlines():
    idx, used, total = [x.strip() for x in line.split(",")]
    idx, used, total = int(idx), int(used), int(total)
    free = total - used
    if best is None or free > best[1]:
        best = (idx, free, used, total)
print(best[0])
PY
)
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Using physical GPU: ${GPU_ID}"
nvidia-smi

echo "===== STEP 1: MAKE DEDUP SPLITS ====="
python scripts/46_make_chest_dedup_splits.py

SEEDS=(20260609 20260610 20260611)
BACKBONES=("resnet18" "efficientnet_b0")

for BACKBONE in "${BACKBONES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "===== DEDUP C2 ${BACKBONE} seed ${SEED} ====="
    python scripts/27_train_skin_backbone_baseline.py \
      --split_csv data/training_splits/chest_dedup/C2_compiler_compatible_binary_pool_dedup.csv \
      --out_dir results/chest_dedup_baselines/C2_${BACKBONE}_dedup_seed${SEED} \
      --backbone "${BACKBONE}" \
      --epochs 5 \
      --batch_size 192 \
      --num_workers 8 \
      --lr 3e-4 \
      --seed "${SEED}" \
      --pretrained

    echo "===== DEDUP C3 ${BACKBONE} seed ${SEED} ====="
    python scripts/27_train_skin_backbone_baseline.py \
      --split_csv data/training_splits/chest_dedup/C3_naive_binary_pool_all_abnormal_dedup_same_c2test.csv \
      --out_dir results/chest_dedup_baselines/C3_${BACKBONE}_dedup_seed${SEED} \
      --backbone "${BACKBONE}" \
      --epochs 5 \
      --batch_size 192 \
      --num_workers 8 \
      --lr 3e-4 \
      --seed "${SEED}" \
      --pretrained
  done
done

echo "===== STEP 3: AGGREGATE DEDUP RESULTS ====="
python - <<'PY'
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

SEEDS = [20260609, 20260610, 20260611]
BACKBONES = ["resnet18", "efficientnet_b0"]

rows = []
for backbone in BACKBONES:
    for protocol in ["C2", "C3"]:
        for seed in SEEDS:
            d = Path(f"results/chest_dedup_baselines/{protocol}_{backbone}_dedup_seed{seed}")
            metric_path = d / "test_metrics.json"
            summary_path = d / "run_summary.json"
            if not metric_path.exists():
                rows.append({
                    "backbone": backbone,
                    "protocol": protocol,
                    "seed": seed,
                    "status": "missing",
                    "metric_file": str(metric_path),
                })
                continue
            m = json.loads(metric_path.read_text())
            s = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            rows.append({
                "backbone": "ResNet18" if backbone == "resnet18" else "EfficientNet-B0",
                "protocol": protocol,
                "seed": seed,
                "status": "ok",
                "setting": "Deduplicated compiler-compatible C2" if protocol == "C2" else "Deduplicated naive C3 on same C2 test",
                "metric_file": str(metric_path),
                "train_rows": s.get("train_rows", ""),
                "val_rows": s.get("val_rows", ""),
                "test_rows": m.get("n", s.get("test_rows", "")),
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
            })

df = pd.DataFrame(rows)
seed_csv = Path("results/tables/chest_dedup_seed_results.csv")
agg_csv = Path("results/tables/chest_dedup_aggregate_results.csv")
report_md = Path("results/tables/chest_dedup_result_report.md")
fig_path = Path("results/figures/fig_chest_dedup_macro_f1.png")
seed_csv.parent.mkdir(parents=True, exist_ok=True)
fig_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(seed_csv, index=False)

ok = df[df["status"].eq("ok")].copy()
agg = (
    ok.groupby(["backbone", "protocol", "setting"])
    .agg(
        n_runs=("macro_f1", "count"),
        test_rows_mean=("test_rows", "mean"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
    )
    .reset_index()
)
agg.to_csv(agg_csv, index=False)

diff_rows = []
for backbone, g in agg.groupby("backbone"):
    c2 = g[g["protocol"].eq("C2")].iloc[0]
    c3 = g[g["protocol"].eq("C3")].iloc[0]
    diff_rows.append({
        "backbone": backbone,
        "c2_macro_f1": c2["macro_f1_mean"],
        "c2_macro_f1_sd": c2["macro_f1_std"],
        "c3_macro_f1": c3["macro_f1_mean"],
        "c3_macro_f1_sd": c3["macro_f1_std"],
        "c2_minus_c3_macro_f1": c2["macro_f1_mean"] - c3["macro_f1_mean"],
    })
diff = pd.DataFrame(diff_rows)
diff.to_csv("results/tables/chest_dedup_c2_minus_c3_summary.csv", index=False)

plot = agg.copy()
plot["label"] = plot["backbone"] + " " + plot["protocol"]
plt.figure(figsize=(8,5))
plt.bar(plot["label"], plot["macro_f1_mean"], yerr=plot["macro_f1_std"], capsize=6)
plt.ylim(0, 1.05)
plt.ylabel("Macro F1, mean ± SD")
plt.title("Chest deduplicated hash-audited C2 vs C3")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(fig_path, dpi=250)
plt.close()

lines = []
lines.append("# Chest Deduplicated Hash-Audited Results\n")
lines.append("## Main differences\n")
lines.append(diff.to_markdown(index=False))
lines.append("\n## Aggregate results\n")
lines.append(agg.to_markdown(index=False))
lines.append("\n## Seed-level results\n")
show = ok.copy()
for c in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
    show[c] = show[c].map(lambda x: f"{x:.4f}")
lines.append(show.to_markdown(index=False))
lines.append(f"\n\nFigure: `{fig_path}`")
report_md.write_text("\n".join(lines), encoding="utf-8")

print("===== CHEST DEDUP AGGREGATION COMPLETE =====")
print("Seed CSV:", seed_csv)
print("Aggregate CSV:", agg_csv)
print("Diff CSV: results/tables/chest_dedup_c2_minus_c3_summary.csv")
print("Report:", report_md)
print("Figure:", fig_path)
print()
print("===== MAIN DIFFERENCES =====")
print(diff.to_string(index=False))
print()
print("===== AGGREGATE =====")
print(agg.to_string(index=False))
PY

echo "===== ALL CHEST DEDUP OVERNIGHT RUNS COMPLETE ====="
date
