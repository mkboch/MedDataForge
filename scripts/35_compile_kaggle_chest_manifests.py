#!/usr/bin/env python3
import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd


ROOTS = {
    "chest_xray_pneumonia": Path("data/images_raw/chest/kaggle/chest_xray_pneumonia"),
    "covid19_radiography": Path("data/images_raw/chest/kaggle/covid19_radiography"),
    "indiana_chest_xrays": Path("data/images_raw/chest/kaggle/indiana_chest_xrays"),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

OUT_ALL = Path("data/compiled_manifests/chest_kaggle_all_compiled_manifest.csv")
OUT_PNEUMONIA = Path("data/compiled_manifests/chest_binary_pneumonia_manifest.csv")
OUT_COVID = Path("data/compiled_manifests/chest_covid_radiography_multiclass_manifest.csv")
OUT_INDIANA = Path("data/compiled_manifests/chest_indiana_report_image_manifest.csv")
OUT_DECISIONS = Path("results/tables/chest_compiler_decisions.csv")
OUT_LABEL_COUNTS = Path("results/tables/chest_compiled_label_counts.csv")
OUT_REPORT = Path("results/tables/chest_compiler_manifest_report.md")
OUT_AUDIT = Path("results/compiler/compiler_audit_manifest_v0_5_chest_kaggle.json")

OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)


def canonicalize_text(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def is_real_image_file(p):
    s = str(p).replace("\\", "/")
    name = p.name
    return (
        p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and "__MACOSX" not in s
        and not name.startswith("._")
        and name != ".DS_Store"
    )


def list_images(root):
    return sorted([
        p for p in root.rglob("*")
        if is_real_image_file(p)
    ])


def list_covid_classification_images(root):
    """Return only radiograph images, not segmentation masks."""
    return sorted([
        p for p in root.rglob("*")
        if (
            is_real_image_file(p)
            and "/images/" in str(p).replace("\\", "/").lower()
            and "/masks/" not in str(p).replace("\\", "/").lower()
        )
    ])


def infer_pneumonia_row(path, root):
    rel = path.relative_to(root)
    parts = rel.parts

    lower_parts = [p.lower() for p in parts]

    split = "unknown"
    for s in ["train", "test", "val"]:
        if s in lower_parts:
            split = s
            break

    raw_label = "unknown"
    if "normal" in lower_parts:
        raw_label = "NORMAL"
    elif "pneumonia" in lower_parts:
        raw_label = "PNEUMONIA"

    canonical_label = {
        "NORMAL": "normal",
        "PNEUMONIA": "pneumonia",
    }.get(raw_label, "unknown")

    return {
        "dataset": "chest_xray_pneumonia",
        "source": "kaggle:paultimothymooney/chest-xray-pneumonia",
        "domain": "chest_xray",
        "task_family": "binary_pneumonia_classification",
        "compiler_role": "narrow_binary_pneumonia_dataset",
        "split_original": split,
        "raw_label": raw_label,
        "canonical_label": canonical_label,
        "image_path": str(path),
        "relative_path": str(rel),
        "compiler_status": "accepted_for_binary_pneumonia_only",
        "compatibility_warning": "narrow pneumonia task; not equivalent to broad multilabel chest abnormality datasets",
    }


def infer_covid_row(path, root):
    rel = path.relative_to(root)
    rel_str = str(rel)
    lower_path = rel_str.lower().replace("\\", "/")

    raw_label = "unknown"
    if "covid-19_radiography_dataset/covid/images/" in lower_path:
        raw_label = "COVID"
    elif "covid-19_radiography_dataset/viral pneumonia/images/" in lower_path:
        raw_label = "Viral Pneumonia"
    elif "covid-19_radiography_dataset/lung_opacity/images/" in lower_path:
        raw_label = "Lung_Opacity"
    elif "covid-19_radiography_dataset/normal/images/" in lower_path:
        raw_label = "Normal"

    canonical_label = {
        "COVID": "covid_19",
        "Viral Pneumonia": "viral_pneumonia",
        "Lung_Opacity": "lung_opacity",
        "Normal": "normal",
    }.get(raw_label, "unknown")

    return {
        "dataset": "covid19_radiography",
        "source": "kaggle:tawsifurrahman/covid19-radiography-database",
        "domain": "chest_xray",
        "task_family": "covid_pneumonia_opacity_normal_multiclass",
        "compiler_role": "narrow_covid_pneumonia_opacity_dataset",
        "split_original": "not_provided",
        "raw_label": raw_label,
        "canonical_label": canonical_label,
        "image_path": str(path),
        "relative_path": str(rel),
        "compiler_status": "accepted_for_covid_radiography_multiclass_only",
        "compatibility_warning": "COVID/opacity/pneumonia task; not equivalent to broad multilabel chest abnormality datasets",
    }


def load_indiana_metadata(root):
    projections = root / "indiana_projections.csv"
    reports = root / "indiana_reports.csv"

    proj = pd.read_csv(projections) if projections.exists() else pd.DataFrame()
    rep = pd.read_csv(reports) if reports.exists() else pd.DataFrame()

    return proj, rep


def compile_indiana(root):
    images = list_images(root)
    proj, rep = load_indiana_metadata(root)

    rows = []

    # Build projection lookup if columns are recognizable.
    proj_lookup = {}
    if len(proj):
        cols = {c.lower(): c for c in proj.columns}
        uid_col = cols.get("uid")
        filename_col = cols.get("filename")
        projection_col = cols.get("projection")
        if uid_col and filename_col:
            for _, r in proj.iterrows():
                fn = str(r[filename_col])
                proj_lookup[fn] = {
                    "uid": r.get(uid_col, ""),
                    "projection": r.get(projection_col, "") if projection_col else "",
                }

    report_lookup = {}
    if len(rep):
        cols = {c.lower(): c for c in rep.columns}
        uid_col = cols.get("uid")
        if uid_col:
            for _, r in rep.iterrows():
                uid = str(r[uid_col])
                report_lookup[uid] = r.to_dict()

    for path in images:
        rel = path.relative_to(root)
        fn = path.name
        meta = proj_lookup.get(fn, {})
        uid = str(meta.get("uid", ""))
        report = report_lookup.get(uid, {})

        rows.append({
            "dataset": "indiana_chest_xrays",
            "source": "kaggle:raddar/chest-xrays-indiana-university",
            "domain": "chest_xray",
            "task_family": "image_report_captioning_or_weak_label_extraction",
            "compiler_role": "report_image_dataset_not_direct_classification_equivalent",
            "split_original": "not_provided",
            "raw_label": "",
            "canonical_label": "",
            "image_path": str(path),
            "relative_path": str(rel),
            "uid": uid,
            "projection": meta.get("projection", ""),
            "has_report": bool(report),
            "compiler_status": "rejected_for_direct_label_pooling",
            "compatibility_warning": "image-report dataset; requires NLP label extraction before classification pooling",
        })

    return pd.DataFrame(rows), proj, rep


def main():
    print("===== CHEST MANIFEST COMPILATION START =====")

    rows = []

    # Pneumonia dataset
    pneumonia_root = ROOTS["chest_xray_pneumonia"]
    pneumonia_images = list_images(pneumonia_root)
    for p in pneumonia_images:
        rows.append(infer_pneumonia_row(p, pneumonia_root))
    pneumonia_df = pd.DataFrame([r for r in rows if r["dataset"] == "chest_xray_pneumonia"])

    # COVID dataset
    covid_root = ROOTS["covid19_radiography"]
    covid_images = list_covid_classification_images(covid_root)
    covid_rows = [infer_covid_row(p, covid_root) for p in covid_images]
    covid_df = pd.DataFrame(covid_rows)
    rows.extend(covid_rows)

    # Indiana dataset
    indiana_df, indiana_proj, indiana_reports = compile_indiana(ROOTS["indiana_chest_xrays"])
    rows.extend(indiana_df.to_dict("records"))

    all_df = pd.DataFrame(rows)
    all_df.to_csv(OUT_ALL, index=False)
    pneumonia_df.to_csv(OUT_PNEUMONIA, index=False)
    covid_df.to_csv(OUT_COVID, index=False)
    indiana_df.to_csv(OUT_INDIANA, index=False)

    label_counts = (
        all_df.groupby(["dataset", "task_family", "compiler_status", "canonical_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "count"], ascending=[True, False])
    )
    label_counts.to_csv(OUT_LABEL_COUNTS, index=False)

    decisions = pd.DataFrame([
        {
            "dataset": "chest_xray_pneumonia",
            "images": len(pneumonia_df),
            "task_family": "binary_pneumonia_classification",
            "compiler_decision": "accepted only for binary pneumonia stress-test protocol",
            "direct_pool_with_broad_multilabel_chest": "reject",
            "reason": "labels are normal vs pneumonia only, not a broad chest abnormality ontology",
        },
        {
            "dataset": "covid19_radiography",
            "images": len(covid_df),
            "task_family": "covid_pneumonia_opacity_normal_multiclass",
            "compiler_decision": "accepted only for COVID/pneumonia/opacity multiclass stress-test protocol",
            "direct_pool_with_broad_multilabel_chest": "reject",
            "reason": "labels are COVID, viral pneumonia, lung opacity, and normal, not equivalent to broad multilabel chest finding labels",
        },
        {
            "dataset": "indiana_chest_xrays",
            "images": len(indiana_df),
            "task_family": "image_report_captioning_or_weak_label_extraction",
            "compiler_decision": "reject for direct supervised classification pooling",
            "direct_pool_with_broad_multilabel_chest": "reject_without_label_extraction",
            "reason": "dataset contains image-report pairs and projections, not directly harmonized image-level classification labels",
        },
    ])
    decisions.to_csv(OUT_DECISIONS, index=False)

    audit = {
        "compiler_version": "0.5_chest_kaggle_manifest",
        "datasets": {
            "chest_xray_pneumonia": {
                "images": int(len(pneumonia_df)),
                "labels": pneumonia_df["canonical_label"].value_counts().to_dict(),
            },
            "covid19_radiography": {
                "images": int(len(covid_df)),
                "labels": covid_df["canonical_label"].value_counts().to_dict(),
            },
            "indiana_chest_xrays": {
                "images": int(len(indiana_df)),
                "metadata": {
                    "projection_rows": int(len(indiana_proj)),
                    "report_rows": int(len(indiana_reports)),
                    "images_with_report": int(indiana_df["has_report"].sum()) if "has_report" in indiana_df else 0,
                },
            },
        },
        "output_files": {
            "all_manifest": str(OUT_ALL),
            "pneumonia_manifest": str(OUT_PNEUMONIA),
            "covid_manifest": str(OUT_COVID),
            "indiana_manifest": str(OUT_INDIANA),
            "decisions": str(OUT_DECISIONS),
            "label_counts": str(OUT_LABEL_COUNTS),
        },
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Chest Kaggle Compiler Manifest Report\n")
    lines.append("## Summary\n")
    lines.append(f"- **Chest X-Ray Images (Pneumonia)**: {len(pneumonia_df)} images")
    lines.append(f"- **COVID-19 Radiography Database**: {len(covid_df)} images")
    lines.append(f"- **Indiana U. Chest X-rays**: {len(indiana_df)} images")
    lines.append("\n## Compiler decisions\n")
    lines.append(decisions.to_markdown(index=False))
    lines.append("\n## Label/task counts\n")
    lines.append(label_counts.to_markdown(index=False))
    lines.append("\n## Interpretation\n")
    lines.append(
        "These Kaggle chest datasets are all chest X-ray datasets, but the compiler does not treat them as directly equivalent. "
        "The pneumonia dataset supports a narrow binary pneumonia task, the COVID radiography dataset supports a COVID/pneumonia/opacity/normal task, "
        "and the Indiana dataset is an image-report dataset that requires label extraction before classification pooling. "
        "This provides second-domain evidence for MedDataForge's core claim: dataset compatibility must be compiled and audited before pooling."
    )

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("===== CHEST MANIFEST COMPILATION COMPLETE =====")
    print(json.dumps(audit, indent=2))
    print()
    print("===== COMPILER DECISIONS =====")
    print(decisions.to_string(index=False))
    print()
    print("===== LABEL COUNTS =====")
    print(label_counts.to_string(index=False))
    print()
    print(f"Report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
