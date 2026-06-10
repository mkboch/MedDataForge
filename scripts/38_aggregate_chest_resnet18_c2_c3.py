#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUNS = []
for seed in [20260602, 20260603, 20260604]:
    RUNS.append({
        "protocol": "C2",
        "seed": seed,
        "backbone": "ResNet18",
        "setting": "Compiler-compatible binary pool",
        "dir": Path(f"results/chest_baselines/C2_resnet18_compiler_compatible_binary_pool_seed{seed}"),
    })

for seed in [20260602, 20260603, 20260604]:
    RUNS.append({
        "protocol": "C3",
        "seed": seed,
        "backbone": "ResNet18",
        "setting": "Naive noisy binary pool, all abnormal as pneumonia",
        "dir": Path(f"results/chest_baselines/C3_resnet18_naive_binary_pool_all_abnormal_seed{seed}"),
    })

OUT_SEED = Path("results/tables/chest_resnet18_c2_c3_seed_results.csv")
OUT_AGG = Path("results/tables/chest_resnet18_c2_c3_aggregate_results.csv")
OUT_REPORT = Path("results/tables/chest_resnet18_c2_c3_report.md")
OUT_FIG = Path("results/figures/fig_chest_resnet18_c2_c3_macro_f1.png")

OUT_SEED.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main():
    rows = []

    for run in RUNS:
        metric_path = run["dir"] / "test_metrics.json"
        summary_path = run["dir"] / "run_summary.json"

        metrics = load_json(metric_path)
        summary = load_json(summary_path)

        rows.append({
            "protocol": run["protocol"],
            "seed": run["seed"],
            "backbone": run["backbone"],
            "setting": run["setting"],
            "metric_file": str(metric_path),
            "train_rows": summary.get("train_rows", ""),
            "val_rows": summary.get("val_rows", ""),
            "test_rows": metrics.get("n", summary.get("test_rows", "")),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_SEED, index=False)

    agg = (
        df.groupby(["backbone", "protocol", "setting"])
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
    agg.to_csv(OUT_AGG, index=False)

    c2 = agg[agg["protocol"].eq("C2")].iloc[0]
    c3 = agg[agg["protocol"].eq("C3")].iloc[0]
    diff = c2["macro_f1_mean"] - c3["macro_f1_mean"]

    fig_df = agg.sort_values("protocol")
    plt.figure(figsize=(8, 5))
    plt.bar(fig_df["protocol"], fig_df["macro_f1_mean"], yerr=fig_df["macro_f1_std"], capsize=8)
    plt.ylim(0, 1)
    plt.xlabel("Protocol")
    plt.ylabel("Test macro F1, mean ± SD")
    plt.title("Chest X-ray ResNet18: compatible vs naive binary pooling")
    for i, row in fig_df.reset_index(drop=True).iterrows():
        plt.text(i, row["macro_f1_mean"] + 0.035, f"{row['macro_f1_mean']:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=250)
    plt.close()

    lines = []
    lines.append("# Chest X-ray ResNet18 C2 vs C3 Result Report\n")
    lines.append("## Main comparison\n")
    lines.append(f"- **C2 compiler-compatible macro F1**: {c2['macro_f1_mean']:.4f} ± {c2['macro_f1_std']:.4f}")
    lines.append(f"- **C3 naive noisy-pool macro F1**: {c3['macro_f1_mean']:.4f} ± {c3['macro_f1_std']:.4f}")
    lines.append(f"- **Difference, C2 - C3**: {diff:.4f}")
    lines.append("\n## Interpretation\n")
    lines.append(
        "C2 uses only label-compatible normal and pneumonia/viral-pneumonia images. "
        "C3 intentionally collapses COVID-19 and lung opacity into pneumonia, representing unsafe naive pooling. "
        "If C2 exceeds C3, this supports the second-domain claim that compatibility-aware compilation can prevent harmful label-semantic collapse."
    )

    lines.append("\n## Aggregate table\n")
    show = agg.copy()
    for c in show.columns:
        if c.endswith("_mean") or c.endswith("_std"):
            show[c] = show[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    lines.append(show.to_markdown(index=False))

    lines.append("\n\n## Seed-level table\n")
    seed_show = df.copy()
    for c in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        seed_show[c] = seed_show[c].map(lambda x: f"{x:.4f}")
    lines.append(seed_show.to_markdown(index=False))
    lines.append(f"\n\nFigure: `{OUT_FIG}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("===== CHEST RESNET18 C2/C3 AGGREGATION COMPLETE =====")
    print(f"Seed CSV: {OUT_SEED}")
    print(f"Aggregate CSV: {OUT_AGG}")
    print(f"Report: {OUT_REPORT}")
    print(f"Figure: {OUT_FIG}")

    print("\n===== AGGREGATE RESULTS =====")
    print(agg.to_string(index=False))

    print("\n===== SEED-LEVEL RESULTS =====")
    print(df.to_string(index=False))

    print("\n===== MAIN DIFFERENCE =====")
    print(f"Chest ResNet18 C2 - C3 macro F1 difference: {diff:.4f}")


if __name__ == "__main__":
    main()
