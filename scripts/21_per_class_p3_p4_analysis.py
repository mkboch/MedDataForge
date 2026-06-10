#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUNS = [
    {
        "protocol": "P3",
        "seed": 20260528,
        "setting": "Compiler-filtered pooled 7-class",
        "metric_file": Path("results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class/test_metrics.json"),
    },
    {
        "protocol": "P3",
        "seed": 20260529,
        "setting": "Compiler-filtered pooled 7-class",
        "metric_file": Path("results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class_seed20260529/external_test_metrics.json"),
    },
    {
        "protocol": "P3",
        "seed": 20260530,
        "setting": "Compiler-filtered pooled 7-class",
        "metric_file": Path("results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class_seed20260530/external_test_metrics.json"),
    },
    {
        "protocol": "P4",
        "seed": 20260528,
        "setting": "Naive pooled with flagged SCC extension",
        "metric_file": Path("results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class/external_test_metrics.json"),
    },
    {
        "protocol": "P4",
        "seed": 20260529,
        "setting": "Naive pooled with flagged SCC extension",
        "metric_file": Path("results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class_seed20260529/external_test_metrics.json"),
    },
    {
        "protocol": "P4",
        "seed": 20260530,
        "setting": "Naive pooled with flagged SCC extension",
        "metric_file": Path("results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class_seed20260530/external_test_metrics.json"),
    },
]

OUT_DIR = Path("results/tables")
OUT_FIG = Path("results/figures/fig_p3_p4_per_class_f1.png")
OUT_SEED = OUT_DIR / "p3_p4_per_class_seed_level.csv"
OUT_AGG = OUT_DIR / "p3_p4_per_class_aggregate.csv"
OUT_DIFF = OUT_DIR / "p3_p4_per_class_differences.csv"
OUT_REPORT = OUT_DIR / "p3_p4_per_class_analysis_report.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main():
    rows = []

    for run in RUNS:
        metrics = load_json(run["metric_file"])
        report = metrics["classification_report"]

        for label, vals in report.items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            if not isinstance(vals, dict):
                continue

            rows.append({
                "protocol": run["protocol"],
                "seed": run["seed"],
                "setting": run["setting"],
                "class": label,
                "precision": vals.get("precision"),
                "recall": vals.get("recall"),
                "f1": vals.get("f1-score"),
                "support": vals.get("support"),
                "metric_file": str(run["metric_file"]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_SEED, index=False)

    agg = (
        df.groupby(["protocol", "setting", "class"])
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            support_mean=("support", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(OUT_AGG, index=False)

    p3 = agg[agg["protocol"].eq("P3")].copy()
    p4 = agg[agg["protocol"].eq("P4")].copy()

    diff = p3.merge(
        p4,
        on="class",
        suffixes=("_p3", "_p4"),
        how="inner",
    )
    diff["f1_diff_p3_minus_p4"] = diff["f1_mean_p3"] - diff["f1_mean_p4"]
    diff["recall_diff_p3_minus_p4"] = diff["recall_mean_p3"] - diff["recall_mean_p4"]
    diff["precision_diff_p3_minus_p4"] = diff["precision_mean_p3"] - diff["precision_mean_p4"]
    diff = diff.sort_values("f1_diff_p3_minus_p4", ascending=False)
    diff.to_csv(OUT_DIFF, index=False)

    # Plot per-class F1.
    plot = diff.sort_values("f1_diff_p3_minus_p4", ascending=True)
    y = range(len(plot))

    plt.figure(figsize=(10, max(5, 0.45 * len(plot))))
    plt.barh([i - 0.18 for i in y], plot["f1_mean_p3"], height=0.35, label="P3 compiler-filtered")
    plt.barh([i + 0.18 for i in y], plot["f1_mean_p4"], height=0.35, label="P4 naive flagged")
    plt.yticks(list(y), plot["class"])
    plt.xlabel("Per-class F1, mean across seeds")
    plt.title("Per-class F1: compiler-filtered pooling vs naive flagged pooling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=250)
    plt.close()

    lines = []
    lines.append("# P3 vs P4 Per-Class Analysis\n")
    lines.append("## Main interpretation\n")
    lines.append(
        "This analysis compares class-level performance across three seeds. "
        "Positive differences indicate that the compiler-filtered pooled protocol improves over naive pooling with the flagged SCC extension label."
    )

    lines.append("\n## Per-class F1 differences, P3 minus P4\n")
    show = diff[[
        "class",
        "f1_mean_p3",
        "f1_std_p3",
        "f1_mean_p4",
        "f1_std_p4",
        "f1_diff_p3_minus_p4",
        "recall_diff_p3_minus_p4",
        "precision_diff_p3_minus_p4",
    ]].copy()

    for c in show.columns:
        if c != "class":
            show[c] = show[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    lines.append(show.to_markdown(index=False))
    lines.append(f"\n\nFigure: `{OUT_FIG}`")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P3/P4 PER-CLASS ANALYSIS COMPLETE =====")
    print(f"Seed-level CSV: {OUT_SEED}")
    print(f"Aggregate CSV: {OUT_AGG}")
    print(f"Difference CSV: {OUT_DIFF}")
    print(f"Report: {OUT_REPORT}")
    print(f"Figure: {OUT_FIG}")

    print("\n===== PER-CLASS DIFFERENCES =====")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
