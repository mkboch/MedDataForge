#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report


P3_FILES = [
    ("P3", 20260528, "Compiler-filtered pooled 7-class", "results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class/test_metrics.json"),
    ("P3", 20260529, "Compiler-filtered pooled 7-class", "results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class_seed20260529/external_test_metrics.json"),
    ("P3", 20260530, "Compiler-filtered pooled 7-class", "results/skin_baselines/P3_resnet18_compiler_filtered_pool_7class_seed20260530/external_test_metrics.json"),
]

P4_SHARED7_FILES = [
    ("P4_shared7_corrected", 20260528, "Naive pooled model evaluated on shared 7-class subset", "results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class/shared7_test_metrics.json"),
    ("P4_shared7_corrected", 20260529, "Naive pooled model evaluated on shared 7-class subset", "results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class_seed20260529/shared7_test_metrics.json"),
    ("P4_shared7_corrected", 20260530, "Naive pooled model evaluated on shared 7-class subset", "results/skin_baselines/P4_resnet18_naive_pool_including_flagged_8class_seed20260530/shared7_test_metrics.json"),
]

SHARED7_LABELS = [
    "actinic_keratosis_or_intraepithelial_carcinoma",
    "basal_cell_carcinoma",
    "benign_keratosis_like_lesion",
    "dermatofibroma",
    "melanocytic_nevus",
    "melanoma",
    "vascular_lesion",
]

OUT_SEED = Path("results/tables/p3_vs_p4_shared7_corrected_seed_results.csv")
OUT_AGG = Path("results/tables/p3_vs_p4_shared7_corrected_aggregate.csv")
OUT_REPORT = Path("results/tables/p3_vs_p4_shared7_corrected_report.md")

OUT_SEED.parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    return json.loads(Path(path).read_text())


def extract_from_report(metrics, labels):
    report = metrics["classification_report"]

    # Recompute macro/weighted over only the 7 labels using existing per-class report values.
    rows = []
    total_support = 0
    for label in labels:
        vals = report[label]
        support = vals.get("support", 0)
        rows.append({
            "label": label,
            "precision": vals.get("precision", 0.0),
            "recall": vals.get("recall", 0.0),
            "f1": vals.get("f1-score", 0.0),
            "support": support,
        })
        total_support += support

    macro_precision = sum(r["precision"] for r in rows) / len(rows)
    macro_recall = sum(r["recall"] for r in rows) / len(rows)
    macro_f1 = sum(r["f1"] for r in rows) / len(rows)

    weighted_precision = sum(r["precision"] * r["support"] for r in rows) / max(1, total_support)
    weighted_recall = sum(r["recall"] * r["support"] for r in rows) / max(1, total_support)
    weighted_f1 = sum(r["f1"] * r["support"] for r in rows) / max(1, total_support)

    return {
        "n": int(total_support),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": macro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def main():
    rows = []

    for protocol, seed, setting, path in P3_FILES:
        metrics = load_json(path)
        corrected = extract_from_report(metrics, SHARED7_LABELS)
        rows.append({
            "protocol": protocol,
            "seed": seed,
            "setting": setting,
            "metric_file": path,
            **corrected,
        })

    for protocol, seed, setting, path in P4_SHARED7_FILES:
        metrics = load_json(path)
        corrected = extract_from_report(metrics, SHARED7_LABELS)
        rows.append({
            "protocol": protocol,
            "seed": seed,
            "setting": setting,
            "metric_file": path,
            **corrected,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_SEED, index=False)

    agg = (
        df.groupby(["protocol", "setting"])
        .agg(
            n_runs=("macro_f1", "count"),
            test_rows_mean=("n", "mean"),
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
    p4 = agg[agg["protocol"].eq("P4_shared7_corrected")].iloc[0]
    diff = p3["macro_f1_mean"] - p4["macro_f1_mean"]

    lines = []
    lines.append("# Corrected P3 vs P4 Shared-7 Comparison\n")
    lines.append("## Main result\n")
    lines.append(
        "This corrected comparison averages macro metrics over the seven shared evaluation classes only. "
        "P4 predictions into the removed SCC class still count as wrong for the true shared class, but SCC itself is not included as a zero-support class in the macro average."
    )
    lines.append(f"\n- **P3 macro F1**: {p3['macro_f1_mean']:.4f} ± {p3['macro_f1_std']:.4f}")
    lines.append(f"- **P4 corrected shared-7 macro F1**: {p4['macro_f1_mean']:.4f} ± {p4['macro_f1_std']:.4f}")
    lines.append(f"- **Difference, P3 - P4 corrected shared-7**: {diff:.4f}")

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

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("===== CORRECTED SHARED-7 COMPARISON COMPLETE =====")
    print(f"Seed table: {OUT_SEED}")
    print(f"Aggregate table: {OUT_AGG}")
    print(f"Report: {OUT_REPORT}")
    print("\n===== AGGREGATE =====")
    print(agg.to_string(index=False))
    print("\n===== MAIN DIFFERENCE =====")
    print(f"P3 - P4 corrected shared-7 macro F1 difference: {diff:.4f}")


if __name__ == "__main__":
    main()
