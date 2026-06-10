#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RAW = Path("data/registry/datasets_raw_v3.csv")
AUDITED = Path("data/registry/locked_domain_candidates_audited.csv")
MANIFEST_SUMMARY = Path("data/manifests/environment_and_manifest_summary.json")

OUT_TABLES = Path("results/tables")
OUT_FIGS = Path("results/figures")
OUT_REPORT = Path("results/tables/registry_audit_report.md")

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)


def save_bar(series, title, xlabel, ylabel, out_png, top_n=20):
    s = series.fillna("").replace("", "missing/unknown").value_counts().head(top_n)
    plt.figure(figsize=(10, max(4, 0.35 * len(s))))
    s.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    raw = pd.read_csv(RAW)
    audited = pd.read_csv(AUDITED)

    with open(MANIFEST_SUMMARY, "r") as f:
        manifest = json.load(f)

    registry_summary = {
        "total_extracted_datasets": len(raw),
        "datasets_with_primary_url": int(raw["primary_url"].fillna("").ne("").sum()),
        "datasets_with_paper_link": int(raw["has_paper_link"].sum()),
        "datasets_with_leaderboard_or_challenge": int(raw["has_leaderboard_or_challenge"].sum()),
        "datasets_with_license_link": int(raw["has_license_link"].sum()),
        "locked_primary_candidates": len(audited),
        "locked_chest_xray_primary_candidates": int((audited["locked_domain"] == "chest_xray").sum()),
        "locked_skin_lesion_primary_candidates": int((audited["locked_domain"] == "skin_lesion").sum()),
        "wave1_all_candidates": manifest["manifest_counts"]["wave1_all"],
        "wave1_no_kaggle_no_physionet_scriptable": manifest["manifest_counts"]["wave1a_no_kaggle_scriptable"],
        "wave1_kaggle_candidates": manifest["manifest_counts"]["wave1b_kaggle"],
        "wave2_core_gated_or_manual_candidates": manifest["manifest_counts"]["wave2_core_gated_or_manual"],
        "kaggle_json_exists": manifest["kaggle_json_exists"],
        "physionet_netrc_exists": manifest["netrc_exists_for_physionet_possible"],
    }

    pd.DataFrame([registry_summary]).to_csv(OUT_TABLES / "table_registry_summary.csv", index=False)

    show_cols = [
        "dataset_name", "locked_domain", "url_status", "http_status",
        "source_type_inferred", "download_feasibility",
        "first_wave_priority", "primary_url"
    ]

    audited[show_cols].to_csv(OUT_TABLES / "table_locked_candidate_audit.csv", index=False)

    audited[audited["first_wave_priority"].eq("wave1_scriptable_or_easy")][show_cols].to_csv(
        OUT_TABLES / "table_wave1_candidates.csv", index=False
    )

    audited[audited["first_wave_priority"].str.contains("wave2", na=False)][show_cols].to_csv(
        OUT_TABLES / "table_wave2_candidates.csv", index=False
    )

    save_bar(
        raw["section_h2"],
        "Extracted public medical imaging datasets by body-region section",
        "Number of datasets",
        "Section",
        OUT_FIGS / "fig_registry_by_section.png",
    )
    save_bar(
        raw["modality_inferred"],
        "Extracted datasets by inferred modality",
        "Number of datasets",
        "Inferred modality",
        OUT_FIGS / "fig_registry_by_modality.png",
    )
    save_bar(
        raw["task_inferred"],
        "Extracted datasets by inferred task type",
        "Number of datasets",
        "Inferred task",
        OUT_FIGS / "fig_registry_by_task.png",
    )
    save_bar(
        audited["download_feasibility"],
        "Download feasibility for locked primary candidates",
        "Number of datasets",
        "Feasibility category",
        OUT_FIGS / "fig_locked_download_feasibility.png",
    )
    save_bar(
        audited["first_wave_priority"],
        "Experiment wave priority for locked primary candidates",
        "Number of datasets",
        "Priority category",
        OUT_FIGS / "fig_locked_wave_priority.png",
    )

    lines = []
    lines.append("# MedDataForge Registry Audit Report\n")
    lines.append("## Key counts\n")
    for k, v in registry_summary.items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## Locked primary candidates by domain\n")
    lines.append(audited["locked_domain"].value_counts().to_markdown())

    lines.append("\n\n## Download feasibility\n")
    lines.append(audited["download_feasibility"].value_counts().to_markdown())

    lines.append("\n\n## Wave priority\n")
    lines.append(audited["first_wave_priority"].value_counts().to_markdown())

    lines.append("\n\n## Wave 1 candidates\n")
    lines.append(
        audited[audited["first_wave_priority"].eq("wave1_scriptable_or_easy")][show_cols].to_markdown(index=False)
    )

    lines.append("\n\n## Wave 2 candidates\n")
    lines.append(
        audited[audited["first_wave_priority"].str.contains("wave2", na=False)][show_cols].to_markdown(index=False)
    )

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("===== REGISTRY AUDIT REPORT COMPLETE =====")
    print(f"Summary table: {OUT_TABLES / 'table_registry_summary.csv'}")
    print(f"Locked audit table: {OUT_TABLES / 'table_locked_candidate_audit.csv'}")
    print(f"Wave 1 table: {OUT_TABLES / 'table_wave1_candidates.csv'}")
    print(f"Wave 2 table: {OUT_TABLES / 'table_wave2_candidates.csv'}")
    print(f"Markdown report: {OUT_REPORT}")
    print("Figures:")
    for p in sorted(OUT_FIGS.glob("fig_*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
