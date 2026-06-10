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
        "dir": Path("results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class"),
        "preferred_metric_files": ["test_metrics.json", "external_test_metrics.json"],
    },
    {
        "protocol": "P3",
        "seed": 20260529,
        "setting": "Compiler-filtered pooled 7-class",
        "dir": Path("results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class_seed20260529"),
        "preferred_metric_files": ["external_test_metrics.json", "test_metrics.json"],
    },
    {
        "protocol": "P3",
        "seed": 20260530,
        "setting": "Compiler-filtered pooled 7-class",
        "dir": Path("results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class_seed20260530"),
        "preferred_metric_files": ["external_test_metrics.json", "test_metrics.json"],
    },
    {
        "protocol": "P4",
        "seed": 20260528,
        "setting": "Naive pooled with flagged SCC extension",
        "dir": Path("results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class"),
        "preferred_metric_files": ["external_test_metrics.json", "test_metrics.json"],
    },
    {
        "protocol": "P4",
        "seed": 20260529,
        "setting": "Naive pooled with flagged SCC extension",
        "dir": Path("results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class_seed20260529"),
        "preferred_metric_files": ["external_test_metrics.json", "test_metrics.json"],
    },
    {
        "protocol": "P4",
        "seed": 20260530,
        "setting": "Naive pooled with flagged SCC extension",
        "dir": Path("results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class_seed20260530"),
        "preferred_metric_files": ["external_test_metrics.json", "test_metrics.json"],
    },
]

OUT_TABLE = Path("results/tables/p3_p4_seed_level_results.csv")
OUT_AGG = Path("results/tables/p3_p4_seed_aggregate_results.csv")
OUT_REPORT = Path("results/tables/p3_p4_seed_aggregate_report.md")
OUT_FIG = Path("results/figures/fig_p3_p4_macro_f1_seed_aggregate.png")

OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    return json.loads(path.read_text())


def find_metric_file(run):
    for name in run["preferred_metric_files"]:
        p = run["dir"] / name
        if p.exists():
            return p
    return None


def main():
    rows = []

    for run in RUNS:
        metric_path = find_metric_file(run)
        run_summary_path = run["dir"] / "run_summary.json"

        if metric_path is None:
            rows.append({
                "protocol": run["protocol"],
                "seed": run["seed"],
                "setting": run["setting"],
                "status": "missing_metrics",
                "metric_file": "",
            })
            continue

        metrics = load_json(metric_path)
        run_summary = load_json(run_summary_path) if run_summary_path.exists() else {}

        rows.append({
            "protocol": run["protocol"],
            "seed": run["seed"],
            "setting": run["setting"],
            "status": "ok",
            "metric_file": str(metric_path),
            "train_rows": run_summary.get("train_rows", ""),
            "val_rows": run_summary.get("val_rows", ""),
            "test_rows": metrics.get("n", run_summary.get("external_test_rows", "")),
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLE, index=False)

    ok = df[df["status"].eq("ok")].copy()

    agg = (
        ok.groupby(["protocol", "setting"])
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
    p4 = agg[agg["protocol"].eq("P4")].iloc[0]
    diff_mean = p3["macro_f1_mean"] - p4["macro_f1_mean"]

    # Figure with error bars.
    fig_df = agg.sort_values("protocol")
    plt.figure(figsize=(7, 5))
    plt.bar(fig_df["protocol"], fig_df["macro_f1_mean"], yerr=fig_df["macro_f1_std"], capsize=8)
    plt.ylim(0, 1)
    plt.xlabel("Protocol")
    plt.ylabel("Test macro F1, mean ± SD")
    plt.title("Compiler-filtered pooling vs naive flagged pooling across seeds")
    for i, row in fig_df.reset_index(drop=True).iterrows():
        plt.text(i, row["macro_f1_mean"] + 0.04, f"{row['macro_f1_mean']:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=250)
    plt.close()

    # Markdown report.
    lines = []
    lines.append("# P3 vs P4 Seed-Aggregated Result Report\n")
    lines.append("## Main comparison\n")
    lines.append(f"- **P3 mean macro F1**: {p3['macro_f1_mean']:.4f} ± {p3['macro_f1_std']:.4f}")
    lines.append(f"- **P4 mean macro F1**: {p4['macro_f1_mean']:.4f} ± {p4['macro_f1_std']:.4f}")
    lines.append(f"- **Mean macro F1 difference, P3 - P4**: {diff_mean:.4f}")
    lines.append("\n## Aggregate table\n")
    show_agg = agg.copy()
    for c in show_agg.columns:
        if c.endswith("_mean") or c.endswith("_std"):
            show_agg[c] = show_agg[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    lines.append(show_agg.to_markdown(index=False))

    lines.append("\n\n## Seed-level table\n")
    show = ok[[
        "protocol", "seed", "setting", "test_rows",
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
    ]].copy()
    for c in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        show[c] = show[c].map(lambda x: f"{x:.4f}")
    lines.append(show.to_markdown(index=False))

    lines.append(f"\n\nFigure: `{OUT_FIG}`")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("===== P3/P4 SEED AGGREGATION COMPLETE =====")
    print(f"Seed-level CSV: {OUT_TABLE}")
    print(f"Aggregate CSV: {OUT_AGG}")
    print(f"Report: {OUT_REPORT}")
    print(f"Figure: {OUT_FIG}")

    print("\n===== AGGREGATE RESULTS =====")
    print(agg.to_string(index=False))

    print("\n===== SEED-LEVEL RESULTS =====")
    print(ok.to_string(index=False))

    print("\n===== MAIN DIFFERENCE =====")
    print(f"P3 - P4 macro F1 mean difference: {diff_mean:.4f}")


if __name__ == "__main__":
    main()
