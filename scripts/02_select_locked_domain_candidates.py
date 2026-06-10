#!/usr/bin/env python3
import json
import re
from pathlib import Path

import pandas as pd


IN_CSV = Path("data/registry/datasets_raw_v3.csv")
OUT_DIR = Path("data/registry")


CHEST_INCLUDE = [
    "ChestX-ray14",
    "ChestX-ray8",
    "CheXpert",
    "CheXphoto",
    "MIMIC-CXR",
    "MIMIC-CXR-JPG",
    "PadChest",
    "VinDr-PCXR",
    "Chest X-Ray Images (Pneumonia)",
    "COVID-19 Radiography Database",
    "Indiana U. Chest X-rays",
    "Pulmonary Chest X-Ray Abnormalities",
    "RSNA Pneumonia Detection",
    "SIIM-ACR Pneumothorax Segmentation",
    "CANDID-PTX",
    "ChestX-Det",
]

SKIN_INCLUDE = [
    "ISIC",
    "HAM10000",
    "PAD-UFES-20",
    "PH 2",
    "Dermofit Image Library",
    "Dermoscopy and Dermatoscopy Atlas",
    "MED-NODE",
    "Melanoma Dataset",
    "Fitzpatrick 17k",
    "DDI",
    "SD-198",
]


CHEST_CLASSIFICATION_PRIORITY = [
    "ChestX-ray14",
    "ChestX-ray8",
    "CheXpert",
    "MIMIC-CXR-JPG",
    "PadChest",
    "VinDr-PCXR",
    "Chest X-Ray Images (Pneumonia)",
    "COVID-19 Radiography Database",
    "Indiana U. Chest X-rays",
]

SKIN_CLASSIFICATION_PRIORITY = [
    "ISIC",
    "HAM10000",
    "PAD-UFES-20",
    "PH 2",
    "Dermofit Image Library",
    "Dermoscopy and Dermatoscopy Atlas",
    "MED-NODE",
    "Melanoma Dataset",
    "Fitzpatrick 17k",
    "DDI",
]


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def match_any(name, candidates):
    n = norm(name)
    return any(norm(c) == n for c in candidates)


def infer_locked_domain(row):
    name = str(row.get("dataset_name", ""))
    desc = str(row.get("description", ""))
    text = f"{name} {desc}".lower()

    # Explicit exclusions caused by ambiguous names/section leakage.
    if name.strip().lower() in {"age", "atlas", "mela"}:
        return ""

    # Explicit exclusions caused by ambiguous names/section leakage.
    if name.strip().lower() in {"age", "atlas", "mela"}:
        return ""

    if match_any(name, CHEST_INCLUDE):
        return "chest_xray"
    if match_any(name, SKIN_INCLUDE):
        return "skin_lesion"

    chest_terms = [
        "chexpert", "mimic-cxr", "chestx-ray", "chest x-ray",
        "chest xray", "cxr", "padchest", "vindr-pcxr",
        "pneumonia detection", "pneumothorax"
    ]
    skin_terms = [
        "isic", "ham10000", "pad-ufes", "dermoscopy", "dermoscopic",
        "skin lesion", "melanoma", "dermatoscopy"
    ]

    if any(t in text for t in chest_terms):
        return "chest_xray"
    if any(t in text for t in skin_terms):
        return "skin_lesion"
    return ""


def infer_experiment_role(row):
    name = str(row.get("dataset_name", ""))
    if match_any(name, CHEST_CLASSIFICATION_PRIORITY):
        return "primary_classification_candidate"
    if match_any(name, SKIN_CLASSIFICATION_PRIORITY):
        return "primary_classification_candidate"

    task = str(row.get("task_inferred", "")).lower()
    if "classification" in task:
        return "secondary_classification_candidate"
    if "segmentation" in task or "detection" in task:
        return "not_primary_task_but_useful_for_graph"
    return "metadata_graph_only_or_manual_review"


def infer_access_bucket(row):
    access = str(row.get("access_inferred", ""))
    url = str(row.get("primary_url", "")).lower()
    name = str(row.get("dataset_name", "")).lower()

    if "physionet" in access or "physionet" in url:
        return "controlled_or_credential_required"
    if "kaggle" in access or "kaggle" in url:
        return "kaggle_api_required"
    if "stanford" in access or "stanford" in url or "aimi" in url:
        return "terms_or_registration_likely"
    if "isic" in access or "isic" in url:
        return "isic_archive_or_challenge_download"
    if "dataverse.harvard.edu" in url or "huggingface.co" in url or "github.com" in url or "zenodo.org" in url or "mendeley.com" in url:
        return "likely_scriptable_download_or_api"
    return "manual_review_needed"


def compatibility_notes(row):
    name = str(row.get("dataset_name", ""))
    domain = row.get("locked_domain", "")
    notes = []

    if domain == "chest_xray":
        if name in {"CheXpert", "MIMIC-CXR-JPG", "ChestX-ray14", "PadChest", "VinDr-PCXR"}:
            notes.append("multi-label chest finding label harmonization likely needed")
        if name in {"Chest X-Ray Images (Pneumonia)", "RSNA Pneumonia Detection"}:
            notes.append("narrow pneumonia-focused task may not align with broad multi-label datasets")
        if "Segmentation" in name or "Det" in name or "CANDID" in name or "SIIM" in name:
            notes.append("detection/segmentation dataset, not first classification task")
    elif domain == "skin_lesion":
        if name in {"ISIC", "HAM10000", "PAD-UFES-20", "PH 2"}:
            notes.append("diagnostic label mapping and year/source drift likely needed")
        if name in {"Fitzpatrick 17k", "DDI"}:
            notes.append("valuable for bias/fairness or source-shift analysis, label compatibility must be checked")
    return "; ".join(notes)


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Missing input: {IN_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)

    df["locked_domain"] = df.apply(infer_locked_domain, axis=1)
    candidates = df[df["locked_domain"].isin(["chest_xray", "skin_lesion"])].copy()

    candidates["experiment_role"] = candidates.apply(infer_experiment_role, axis=1)
    candidates["access_bucket"] = candidates.apply(infer_access_bucket, axis=1)
    candidates["compatibility_notes"] = candidates.apply(compatibility_notes, axis=1)

    sort_cols = ["locked_domain", "experiment_role", "dataset_name"]
    candidates = candidates.sort_values(sort_cols)

    chest = candidates[candidates["locked_domain"] == "chest_xray"].copy()
    skin = candidates[candidates["locked_domain"] == "skin_lesion"].copy()

    candidates.to_csv(OUT_DIR / "locked_domain_candidates.csv", index=False)
    chest.to_csv(OUT_DIR / "chest_xray_candidates.csv", index=False)
    skin.to_csv(OUT_DIR / "skin_lesion_candidates.csv", index=False)

    by_domain_and_role = {}
    for (domain, role), count in candidates.groupby(["locked_domain", "experiment_role"]).size().items():
        by_domain_and_role[f"{domain}__{role}"] = int(count)

    summary = {
        "total_locked_candidates": int(len(candidates)),
        "chest_xray_candidates": int(len(chest)),
        "skin_lesion_candidates": int(len(skin)),
        "primary_classification_candidates": int((candidates["experiment_role"] == "primary_classification_candidate").sum()),
        "by_domain_and_role": by_domain_and_role,
        "by_access_bucket": {str(k): int(v) for k, v in candidates["access_bucket"].value_counts().items()},
        "chest_primary_names": chest[chest["experiment_role"] == "primary_classification_candidate"]["dataset_name"].tolist(),
        "skin_primary_names": skin[skin["experiment_role"] == "primary_classification_candidate"]["dataset_name"].tolist(),
    }

    (OUT_DIR / "locked_domain_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== LOCKED DOMAIN SELECTION COMPLETE =====")
    print(json.dumps(summary, indent=2))

    show_cols = [
        "dataset_id", "dataset_name", "locked_domain", "experiment_role",
        "access_bucket", "modality_inferred", "task_inferred",
        "primary_url", "compatibility_notes"
    ]
    print("\n===== PRIMARY CANDIDATES =====")
    prim = candidates[candidates["experiment_role"] == "primary_classification_candidate"]
    print(prim[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
