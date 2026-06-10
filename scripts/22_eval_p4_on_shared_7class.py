#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

P4_RUNS = [
    {
        "seed": 20260528,
        "model_dir": "results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class",
    },
    {
        "seed": 20260529,
        "model_dir": "results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class_seed20260529",
    },
    {
        "seed": 20260530,
        "model_dir": "results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class_seed20260530",
    },
]

P3_METRICS = [
    {
        "seed": 20260528,
        "metric_file": "results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class/test_metrics.json",
    },
    {
        "seed": 20260529,
        "metric_file": "results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class_seed20260529/external_test_metrics.json",
    },
    {
        "seed": 20260530,
        "metric_file": "results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class_seed20260530/external_test_metrics.json",
    },
]

SPLIT_CSV = "data/training_splits/P4_naive_pool_including_flagged_8class.csv"
FILTERED_SPLIT_DIR = Path("data/training_splits/shared7_eval")
OUT_TABLE = Path("results/tables/p3_vs_p4_shared7_fair_comparison.csv")
OUT_AGG = Path("results/tables/p3_vs_p4_shared7_fair_comparison_aggregate.csv")
OUT_REPORT = Path("results/tables/p3_vs_p4_shared7_fair_comparison_report.md")

FILTERED_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    # Create a P4 evaluation file that excludes SCC from the test split.
    df = pd.read_csv(SPLIT_CSV, low_memory=False)
    test = df[df["split"].isin(["test", "external_test"])].copy()
    test_shared7 = test[test["canonical_label"] != "squamous_cell_carcinoma"].copy()

    # Keep train/val rows only to preserve file shape if needed, but script 16 uses only eval_split.
    eval_df = test_shared7.copy()
    eval_df["split"] = "shared7_test"

    filtered_csv = FILTERED_SPLIT_DIR / "P4_naive_pool_shared7_test_only.csv"
    eval_df.to_csv(filtered_csv, index=False)

    print("===== SHARED-7 TEST FILE CREATED =====")
    print(f"File: {filtered_csv}")
    print(f"Rows: {len(eval_df)}")
    print(eval_df["canonical_label"].value_counts().to_string())

    # Evaluate each P4 model on shared7_test.
    import subprocess
    p4_rows = []

    for run in P4_RUNS:
        out_json = Path(run["model_dir"]) / "shared7_test_metrics.json"

        cmd = [
            "python", "scripts/16_eval_saved_skin_model.py",
            "--split_csv", str(filtered_csv),
            "--model_dir", run["model_dir"],
            "--eval_split", "shared7_test",
            "--out_json", str(out_json),
            "--batch_size", "192",
            "--num_workers", "8",
        ]

        print("===== RUNNING =====")
        print(" ".join(cmd), flush=True)

        subprocess.run(cmd, check=True)

        metrics = load_json(out_json)
        p4_rows.append({
            "protocol": "P4_shared7_eval",
            "seed": run["seed"],
            "setting": "Naive pooled model evaluated on shared 7-class subset",
            "test_rows": metrics["n"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "metric_file": str(out_json),
        })

    p3_rows = []
    for run in P3_METRICS:
        metrics = load_json(run["metric_file"])
        p3_rows.append({
            "protocol": "P3",
            "seed": run["seed"],
            "setting": "Compiler-filtered pooled 7-class",
            "test_rows": metrics["n"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "metric_file": run["metric_file"],
        })

    result = pd.DataFrame(p3_rows + p4_rows)
    result.to_csv(OUT_TABLE, index=False)

    agg = (
        result.groupby(["protocol", "setting"])
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

    p3 = agg[agg["protocol"].eq("P3")].iloc[0]
    p4 = agg[agg["protocol"].eq("P4_shared7_eval")].iloc[0]
    diff = p3["macro_f1_mean"] - p4["macro_f1_mean"]

    lines = []
    lines.append("# P3 vs P4 Shared-7 Fairness Check\n")
    lines.append("## Main result\n")
    lines.append(
        "P4 was re-evaluated after excluding SCC rows from the test set, so the comparison uses the shared 7-class label space. "
        "This checks whether the compiler-filtered advantage is only due to P4 having an extra class."
    )
    lines.append(f"\n- **P3 macro F1**: {p3['macro_f1_mean']:.4f} ± {p3['macro_f1_std']:.4f}")
    lines.append(f"- **P4 shared-7 macro F1**: {p4['macro_f1_mean']:.4f} ± {p4['macro_f1_std']:.4f}")
    lines.append(f"- **Difference, P3 - P4 shared-7**: {diff:.4f}")

    lines.append("\n## Aggregate table\n")
    show = agg.copy()
    for c in show.columns:
        if c.endswith("_mean") or c.endswith("_std"):
            show[c] = show[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    lines.append(show.to_markdown(index=False))

    lines.append("\n\n## Seed-level table\n")
    seed_show = result.copy()
    for c in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        seed_show[c] = seed_show[c].map(lambda x: f"{x:.4f}")
    lines.append(seed_show.to_markdown(index=False))

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("===== SHARED-7 FAIR COMPARISON COMPLETE =====")
    print(f"Seed table: {OUT_TABLE}")
    print(f"Aggregate table: {OUT_AGG}")
    print(f"Report: {OUT_REPORT}")
    print("\n===== AGGREGATE =====")
    print(agg.to_string(index=False))
    print("\n===== MAIN DIFFERENCE =====")
    print(f"P3 - P4 shared-7 macro F1 difference: {diff:.4f}")


if __name__ == "__main__":
    main()
