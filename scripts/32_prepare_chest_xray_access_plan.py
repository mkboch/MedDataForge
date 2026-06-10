#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


AUDIT = Path("data/registry/locked_domain_candidates_audited.csv")
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "chest_xray_access_and_experiment_plan.csv"
OUT_MD = OUT_DIR / "chest_xray_access_and_experiment_plan.md"
OUT_JSON = OUT_DIR / "chest_xray_access_environment_check.json"


CORE_CHEST = [
    "ChestX-ray8",
    "Chest X-Ray Images (Pneumonia)",
    "COVID-19 Radiography Database",
    "Indiana U. Chest X-rays",
    "CheXpert",
    "MIMIC-CXR-JPG",
    "PadChest",
    "VinDr-PCXR",
]


def classify_experiment_role(name):
    if name == "ChestX-ray8":
        return "core_multilabel_anchor"
    if name in {"CheXpert", "MIMIC-CXR-JPG", "PadChest", "VinDr-PCXR"}:
        return "core_multilabel_external_validation_if_accessible"
    if name in {"Chest X-Ray Images (Pneumonia)", "COVID-19 Radiography Database"}:
        return "binary_or_narrow_pneumonia_stress_test"
    if name == "Indiana U. Chest X-rays":
        return "report_caption_dataset_or_manual_label_review"
    return "manual_review"


def recommended_next_action(row):
    name = row["dataset_name"]
    source = str(row.get("source_type_inferred", ""))
    feasibility = str(row.get("download_feasibility", ""))

    if name == "ChestX-ray8":
        return "already likely scriptable through HuggingFace; use as first chest anchor"
    if source == "kaggle":
        return "requires Kaggle API token; can download after ~/.kaggle/kaggle.json is configured"
    if source == "physionet":
        return "requires PhysioNet credentials and approved access/DUA"
    if name == "CheXpert":
        return "requires Stanford terms/registration; manual download likely"
    if name == "PadChest":
        return "manual/registration review required"
    return feasibility or "manual review required"


def main():
    if not AUDIT.exists():
        raise SystemExit(f"Missing audit file: {AUDIT}")

    df = pd.read_csv(AUDIT)
    chest = df[(df["locked_domain"] == "chest_xray") & (df["dataset_name"].isin(CORE_CHEST))].copy()

    chest["chest_experiment_role"] = chest["dataset_name"].map(classify_experiment_role)
    chest["recommended_next_action"] = chest.apply(recommended_next_action, axis=1)

    keep = [
        "dataset_name",
        "source_type_inferred",
        "download_feasibility",
        "first_wave_priority",
        "url_status",
        "http_status",
        "primary_url",
        "chest_experiment_role",
        "recommended_next_action",
    ]
    keep = [c for c in keep if c in chest.columns]
    chest = chest[keep].sort_values(["first_wave_priority", "dataset_name"])

    chest.to_csv(OUT_CSV, index=False)

    env = {
        "kaggle_json_exists_home": Path.home().joinpath(".kaggle/kaggle.json").exists(),
        "kaggle_json_exists_project": Path("kaggle.json").exists(),
        "physionet_netrc_exists": Path.home().joinpath(".netrc").exists(),
        "chest_candidates_in_plan": int(len(chest)),
        "kaggle_candidates": chest[chest["source_type_inferred"].eq("kaggle")]["dataset_name"].tolist() if "source_type_inferred" in chest else [],
        "physionet_candidates": chest[chest["source_type_inferred"].eq("physionet")]["dataset_name"].tolist() if "source_type_inferred" in chest else [],
    }

    OUT_JSON.write_text(json.dumps(env, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Chest X-ray Access and Experiment Plan\n")
    lines.append("## Goal\n")
    lines.append(
        "The next limitation to fix is domain generality. The skin-lesion result is promising, but a Nature-level or Nature-adjacent methods paper needs a second domain. Chest X-ray is the best next domain because it has multiple public datasets with known label-space and access fragmentation."
    )

    lines.append("\n## Environment check\n")
    for k, v in env.items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## Candidate datasets\n")
    lines.append(chest.to_markdown(index=False))

    lines.append("\n## Recommended practical strategy\n")
    lines.append("1. Start with **ChestX-ray8** as the anchor because it is likely scriptable.")
    lines.append("2. Add Kaggle datasets if Kaggle API is configured:")
    lines.append("   - Chest X-Ray Images (Pneumonia)")
    lines.append("   - COVID-19 Radiography Database")
    lines.append("   - Indiana U. Chest X-rays")
    lines.append("3. Treat pneumonia/COVID datasets as narrow stress tests, not direct multilabel equivalents.")
    lines.append("4. Use CheXpert, MIMIC-CXR-JPG, PadChest, and VinDr-PCXR only if credentials/manual access are available.")
    lines.append("5. The first chest experiment should test whether the compiler rejects incompatible narrow pneumonia datasets for broad multilabel pooling.")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("===== CHEST X-RAY ACCESS PLAN COMPLETE =====")
    print(f"CSV: {OUT_CSV}")
    print(f"Markdown: {OUT_MD}")
    print(f"Environment JSON: {OUT_JSON}")
    print("\n===== ENVIRONMENT =====")
    print(json.dumps(env, indent=2))
    print("\n===== PLAN =====")
    print(chest.to_string(index=False))


if __name__ == "__main__":
    main()
