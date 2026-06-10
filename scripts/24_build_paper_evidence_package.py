#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

import pandas as pd


OUT = Path("results/paper_evidence_package")
OUT_TABLES = OUT / "tables"
OUT_FIGS = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)


FILES_TO_COPY = [
    ("results/tables/registry_audit_report.md", OUT / "registry_audit_report.md"),
    ("results/tables/table_registry_summary.csv", OUT_TABLES / "table_registry_summary.csv"),
    ("results/tables/table_locked_candidate_audit.csv", OUT_TABLES / "table_locked_candidate_audit.csv"),
    ("results/tables/skin_experiment_manifest_report.md", OUT / "skin_experiment_manifest_report.md"),
    ("results/tables/skin_image_linked_protocol_report.md", OUT / "skin_image_linked_protocol_report.md"),
    ("results/tables/skin_training_split_report.md", OUT / "skin_training_split_report.md"),
    ("results/tables/skin_baseline_results_report.md", OUT / "skin_baseline_results_report.md"),
    ("results/tables/p3_p4_seed_aggregate_report.md", OUT / "p3_p4_seed_aggregate_report.md"),
    ("results/tables/p3_p4_per_class_analysis_report.md", OUT / "p3_p4_per_class_analysis_report.md"),
    ("results/tables/p3_vs_p4_shared7_corrected_report.md", OUT / "p3_vs_p4_shared7_corrected_report.md"),
    ("results/tables/p3_p4_seed_aggregate_results.csv", OUT_TABLES / "p3_p4_seed_aggregate_results.csv"),
    ("results/tables/p3_vs_p4_shared7_corrected_aggregate.csv", OUT_TABLES / "p3_vs_p4_shared7_corrected_aggregate.csv"),
    ("results/tables/p3_p4_per_class_differences.csv", OUT_TABLES / "p3_p4_per_class_differences.csv"),
    ("results/tables/compiled_experiment_decisions.csv", OUT_TABLES / "compiled_experiment_decisions.csv"),
    ("results/tables/skin_multiclass_7_ham10000_isic2019_label_counts.csv", OUT_TABLES / "skin_multiclass_7_label_counts.csv"),
    ("results/figures/fig_skin_baseline_macro_f1.png", OUT_FIGS / "fig_skin_baseline_macro_f1.png"),
    ("results/figures/fig_p3_p4_macro_f1_seed_aggregate.png", OUT_FIGS / "fig_p3_p4_macro_f1_seed_aggregate.png"),
    ("results/figures/fig_p3_p4_per_class_f1.png", OUT_FIGS / "fig_p3_p4_per_class_f1.png"),
    ("results/figures/fig_locked_download_feasibility.png", OUT_FIGS / "fig_locked_download_feasibility.png"),
    ("results/figures/fig_locked_wave_priority.png", OUT_FIGS / "fig_locked_wave_priority.png"),
]


def copy_existing():
    copied = []
    missing = []
    for src, dst in FILES_TO_COPY:
        srcp = Path(src)
        if srcp.exists():
            shutil.copy2(srcp, dst)
            copied.append((str(srcp), str(dst)))
        else:
            missing.append(str(srcp))
    return copied, missing


def read_csv_safe(path):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def main():
    copied, missing = copy_existing()

    registry = read_csv_safe("results/tables/table_registry_summary.csv")
    p3p4 = read_csv_safe("results/tables/p3_p4_seed_aggregate_results.csv")
    shared7 = read_csv_safe("results/tables/p3_vs_p4_shared7_corrected_aggregate.csv")
    per_class = read_csv_safe("results/tables/p3_p4_per_class_differences.csv")

    summary = {
        "copied_files": len(copied),
        "missing_files": missing,
        "package_path": str(OUT),
    }

    if len(registry):
        r = registry.iloc[0].to_dict()
        summary["registry"] = {
            "total_extracted_datasets": int(r.get("total_extracted_datasets", 0)),
            "locked_primary_candidates": int(r.get("locked_primary_candidates", 0)),
            "locked_chest_xray_primary_candidates": int(r.get("locked_chest_xray_primary_candidates", 0)),
            "locked_skin_lesion_primary_candidates": int(r.get("locked_skin_lesion_primary_candidates", 0)),
            "wave1_all_candidates": int(r.get("wave1_all_candidates", 0)),
            "wave2_core_gated_or_manual_candidates": int(r.get("wave2_core_gated_or_manual_candidates", 0)),
        }

    if len(p3p4):
        p3 = p3p4[p3p4["protocol"].eq("P3")].iloc[0]
        p4 = p3p4[p3p4["protocol"].eq("P4")].iloc[0]
        summary["p3_vs_p4_all"] = {
            "p3_macro_f1_mean": float(p3["macro_f1_mean"]),
            "p3_macro_f1_std": float(p3["macro_f1_std"]),
            "p4_macro_f1_mean": float(p4["macro_f1_mean"]),
            "p4_macro_f1_std": float(p4["macro_f1_std"]),
            "difference": float(p3["macro_f1_mean"] - p4["macro_f1_mean"]),
        }

    if len(shared7):
        p3 = shared7[shared7["protocol"].eq("P3")].iloc[0]
        p4 = shared7[shared7["protocol"].eq("P4_shared7_corrected")].iloc[0]
        summary["p3_vs_p4_shared7_corrected"] = {
            "p3_macro_f1_mean": float(p3["macro_f1_mean"]),
            "p3_macro_f1_std": float(p3["macro_f1_std"]),
            "p4_macro_f1_mean": float(p4["macro_f1_mean"]),
            "p4_macro_f1_std": float(p4["macro_f1_std"]),
            "difference": float(p3["macro_f1_mean"] - p4["macro_f1_mean"]),
        }

    if len(per_class):
        top = per_class.sort_values("f1_diff_p3_minus_p4", ascending=False).head(5)
        summary["top_per_class_improvements"] = [
            {
                "class": row["class"],
                "f1_diff_p3_minus_p4": float(row["f1_diff_p3_minus_p4"]),
            }
            for _, row in top.iterrows()
        ]

    (OUT / "evidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# MedDataForge Paper Evidence Package\n")
    lines.append("## What this package contains\n")
    lines.append("This package freezes the current registry audit, compiler decisions, image-linked protocols, baseline model results, seed-level robustness checks, and per-class analyses.\n")

    lines.append("## Core findings\n")
    if "registry" in summary:
        rg = summary["registry"]
        lines.append(f"- The registry parser extracted **{rg['total_extracted_datasets']}** public medical imaging dataset records.")
        lines.append(f"- The locked chest X-ray and skin-lesion domains contained **{rg['locked_primary_candidates']}** primary candidate datasets.")
        lines.append(f"- Only **{rg['wave1_all_candidates']}** candidates were immediately first-wave accessible/easy, while **{rg['wave2_core_gated_or_manual_candidates']}** were gated or required manual access.")

    if "p3_vs_p4_all" in summary:
        s = summary["p3_vs_p4_all"]
        lines.append(
            f"- Across three seeds, compiler-filtered pooling achieved macro F1 "
            f"**{s['p3_macro_f1_mean']:.4f} ± {s['p3_macro_f1_std']:.4f}**, "
            f"compared with **{s['p4_macro_f1_mean']:.4f} ± {s['p4_macro_f1_std']:.4f}** "
            f"for naive pooling with the flagged SCC extension label."
        )

    if "p3_vs_p4_shared7_corrected" in summary:
        s = summary["p3_vs_p4_shared7_corrected"]
        lines.append(
            f"- In the corrected shared-7 comparison, P3 still outperformed P4 by "
            f"**{s['difference']:.4f} macro F1**, showing that the advantage is not only caused by P4 having an extra class."
        )

    if "top_per_class_improvements" in summary:
        lines.append("- The largest P3 per-class F1 improvements were:")
        for item in summary["top_per_class_improvements"]:
            lines.append(f"  - {item['class']}: +{item['f1_diff_p3_minus_p4']:.4f}")

    lines.append("\n## Suggested paper structure now\n")
    lines.append("1. Introduction: fragmentation ceiling and incompatibility hypothesis.")
    lines.append("2. Method: MedDataForge registry parser, metadata graph, label compiler, audit manifest, and experiment compiler.")
    lines.append("3. Registry audit: dataset availability and access fragmentation.")
    lines.append("4. Skin-lesion case study: HAM10000, ISIC2019, ISIC2020, Fitzpatrick17k compiler decisions.")
    lines.append("5. Experiments: P1/P2 directional transfer, P3 compiler-filtered pooling, P4 naive pooling.")
    lines.append("6. Robustness: three-seed P3/P4 comparison, corrected shared-7 evaluation, and per-class analysis.")
    lines.append("7. Limitations: only first-wave skin experiments so far; chest X-ray requires Kaggle/PhysioNet or credentialed datasets.")

    lines.append("\n## Files copied\n")
    for src, dst in copied:
        lines.append(f"- `{dst}`")

    if missing:
        lines.append("\n## Missing files\n")
        for m in missing:
            lines.append(f"- `{m}`")

    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print("===== PAPER EVIDENCE PACKAGE COMPLETE =====")
    print(f"Package: {OUT}")
    print(f"README: {OUT / 'README.md'}")
    print(f"Summary JSON: {OUT / 'evidence_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
