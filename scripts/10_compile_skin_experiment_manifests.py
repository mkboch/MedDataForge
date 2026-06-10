#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


OUT_DIR = Path("data/compiled_manifests")
OUT_TABLES = Path("results/tables")
OUT_COMPILER = Path("results/compiler")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_COMPILER.mkdir(parents=True, exist_ok=True)


PATHS = {
    "ham10000": Path("data/metadata_raw_clean/HAM10000/HAM10000_metadata.tab"),
    "isic2019_gt": Path("data/metadata_raw_clean/ISIC/ISIC_2019_Training_GroundTruth.csv"),
    "isic2019_meta": Path("data/metadata_raw_clean/ISIC/ISIC_2019_Training_Metadata.csv"),
    "isic2020_gt": Path("data/metadata_raw_clean/ISIC/ISIC_2020_Training_GroundTruth_v2.csv"),
    "fitzpatrick17k": Path("data/metadata_raw_clean/Fitzpatrick_17k/fitzpatrick17k.csv"),
}


HAM_MAP = {
    "mel": "melanoma",
    "nv": "melanocytic_nevus",
    "bcc": "basal_cell_carcinoma",
    "akiec": "actinic_keratosis_or_intraepithelial_carcinoma",
    "bkl": "benign_keratosis_like_lesion",
    "df": "dermatofibroma",
    "vasc": "vascular_lesion",
}

ISIC2019_MAP = {
    "MEL": "melanoma",
    "NV": "melanocytic_nevus",
    "BCC": "basal_cell_carcinoma",
    "AK": "actinic_keratosis_or_intraepithelial_carcinoma",
    "BKL": "benign_keratosis_like_lesion",
    "DF": "dermatofibroma",
    "VASC": "vascular_lesion",
    "SCC": "squamous_cell_carcinoma",
    "UNK": "unknown",
}

FITZ_MAP = {
    "melanoma": "melanoma",
    "superficial spreading melanoma ssm": "melanoma",
    "lentigo maligna melanoma": "melanoma",
    "nodular melanoma": "melanoma",
    "basal cell carcinoma": "basal_cell_carcinoma",
    "squamous cell carcinoma": "squamous_cell_carcinoma",
    "actinic keratosis": "actinic_keratosis_or_intraepithelial_carcinoma",
    "seborrheic keratosis": "benign_keratosis_like_lesion",
    "dermatofibroma": "dermatofibroma",
    "vascular lesion": "vascular_lesion",
}


def read_csv(path, sep=","):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=sep, low_memory=False)


def load_ham10000():
    df = read_csv(PATHS["ham10000"], sep="\t")
    out = pd.DataFrame({
        "sample_id": df["image_id"].astype(str),
        "image_id": df["image_id"].astype(str),
        "dataset": "HAM10000",
        "source_file": str(PATHS["ham10000"]),
        "raw_label": df["dx"].astype(str),
        "canonical_label": df["dx"].astype(str).map(HAM_MAP),
        "task_family": "skin_lesion_classification",
        "image_available_status": "requires_ham10000_image_archives",
        "candidate_image_path": "",
        "patient_or_lesion_id": df.get("lesion_id", "").astype(str) if "lesion_id" in df.columns else "",
        "age": df.get("age", ""),
        "sex": df.get("sex", ""),
        "anatomical_site": df.get("localization", ""),
    })
    out["compiler_status"] = out["canonical_label"].apply(lambda x: "accepted_7class" if pd.notna(x) else "rejected_unmapped")
    return out


def load_isic2019():
    gt = read_csv(PATHS["isic2019_gt"])

    meta = None
    if PATHS["isic2019_meta"].exists():
        meta = read_csv(PATHS["isic2019_meta"])

    image_col = "image"
    if image_col not in gt.columns:
        image_col = gt.columns[0]

    rows = []
    for _, r in gt.iterrows():
        image_id = str(r[image_col])
        positive = []
        for c in ISIC2019_MAP:
            if c in gt.columns:
                try:
                    val = float(r[c])
                except Exception:
                    val = 0
                if val == 1:
                    positive.append(c)

        if len(positive) == 1:
            raw = positive[0]
            canonical = ISIC2019_MAP[raw]
            status = "accepted_7class" if canonical not in {"squamous_cell_carcinoma", "unknown"} else "flagged_extension_label"
        elif len(positive) == 0:
            raw = ""
            canonical = ""
            status = "rejected_no_positive_label"
        else:
            raw = "|".join(positive)
            canonical = ""
            status = "rejected_multi_positive_label"

        rows.append({
            "sample_id": image_id,
            "image_id": image_id,
            "dataset": "ISIC2019",
            "source_file": str(PATHS["isic2019_gt"]),
            "raw_label": raw,
            "canonical_label": canonical,
            "task_family": "skin_lesion_classification",
            "image_available_status": "requires_isic2019_training_images",
            "candidate_image_path": "",
            "patient_or_lesion_id": "",
            "age": "",
            "sex": "",
            "anatomical_site": "",
            "compiler_status": status,
        })

    out = pd.DataFrame(rows)

    if meta is not None and "image" in meta.columns:
        keep_cols = [c for c in ["image", "age_approx", "sex", "anatom_site_general"] if c in meta.columns]
        if keep_cols:
            merged = out.merge(meta[keep_cols], left_on="image_id", right_on="image", how="left")
            if "age_approx" in merged.columns:
                merged["age"] = merged["age_approx"]
            if "sex_y" in merged.columns:
                merged["sex"] = merged["sex_y"]
            elif "sex" in merged.columns:
                merged["sex"] = merged["sex"]
            if "anatom_site_general" in merged.columns:
                merged["anatomical_site"] = merged["anatom_site_general"]
            out = merged[[c for c in out.columns]]

    return out


def load_isic2020_binary():
    df = read_csv(PATHS["isic2020_gt"])

    image_col = "image_name" if "image_name" in df.columns else df.columns[0]
    target_col = "target" if "target" in df.columns else None

    if target_col is None:
        raise ValueError("ISIC2020 metadata has no target column.")

    out = pd.DataFrame({
        "sample_id": df[image_col].astype(str),
        "image_id": df[image_col].astype(str),
        "dataset": "ISIC2020",
        "source_file": str(PATHS["isic2020_gt"]),
        "raw_label": df[target_col].astype(str),
        "canonical_label": df[target_col].astype(str).map({"1": "melanoma", "0": "non_melanoma"}),
        "task_family": "skin_binary_melanoma_classification",
        "image_available_status": "requires_isic2020_training_jpeg",
        "candidate_image_path": "",
        "patient_or_lesion_id": df.get("patient_id", "").astype(str) if "patient_id" in df.columns else "",
        "age": df.get("age_approx", ""),
        "sex": df.get("sex", ""),
        "anatomical_site": df.get("anatom_site_general_challenge", df.get("anatom_site_general", "")),
    })
    out["compiler_status"] = "accepted_binary_melanoma"
    if "diagnosis" in df.columns:
        out["diagnosis_text"] = df["diagnosis"].astype(str)
    return out


def load_fitzpatrick_external():
    df = read_csv(PATHS["fitzpatrick17k"])
    raw = df["label"].astype(str)
    canonical = raw.str.lower().str.strip().map(FITZ_MAP)

    out = pd.DataFrame({
        "sample_id": df["md5hash"].astype(str) if "md5hash" in df.columns else df.index.astype(str),
        "image_id": df["md5hash"].astype(str) if "md5hash" in df.columns else df.index.astype(str),
        "dataset": "Fitzpatrick17k",
        "source_file": str(PATHS["fitzpatrick17k"]),
        "raw_label": raw,
        "canonical_label": canonical.fillna("outside_ham_isic_7_or_unmapped"),
        "task_family": "skin_broad_dermatology_external_stress_test",
        "image_available_status": "image_urls_in_metadata_may_need_download",
        "candidate_image_path": "",
        "patient_or_lesion_id": "",
        "age": "",
        "sex": "",
        "anatomical_site": "",
        "url": df["url"].astype(str) if "url" in df.columns else "",
        "fitzpatrick_scale": df["fitzpatrick_scale"] if "fitzpatrick_scale" in df.columns else "",
        "three_partition_label": df["three_partition_label"].astype(str) if "three_partition_label" in df.columns else "",
        "nine_partition_label": df["nine_partition_label"].astype(str) if "nine_partition_label" in df.columns else "",
    })

    out["compiler_status"] = out["canonical_label"].apply(
        lambda x: "mapped_overlap_label" if x != "outside_ham_isic_7_or_unmapped" else "reject_for_direct_pooling"
    )
    return out


def write_summary_table(df, name):
    label_counts = (
        df.groupby(["dataset", "canonical_label", "compiler_status"])
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "compiler_status", "canonical_label"])
    )
    label_counts.to_csv(OUT_TABLES / f"{name}_label_counts.csv", index=False)
    return label_counts


def main():
    ham = load_ham10000()
    isic2019 = load_isic2019()
    isic2020 = load_isic2020_binary()
    fitz = load_fitzpatrick_external()

    # Experiment A: compatible 7-class HAM10000 + ISIC2019.
    multiclass = pd.concat([ham, isic2019], ignore_index=True)
    multiclass["compiled_experiment_id"] = "skin_multiclass_7_ham10000_isic2019"
    multiclass["recommended_use"] = multiclass["compiler_status"].map({
        "accepted_7class": "use_for_7class_training_or_external_validation",
        "flagged_extension_label": "exclude_from_7class_or_use_extension_experiment",
        "rejected_no_positive_label": "exclude",
        "rejected_multi_positive_label": "exclude",
    }).fillna("exclude")

    # Experiment B: binary ISIC2020.
    binary2020 = isic2020.copy()
    binary2020["compiled_experiment_id"] = "skin_binary_melanoma_isic2020"
    binary2020["recommended_use"] = "use_for_binary_melanoma_task"

    # Experiment C: Fitzpatrick stress test only.
    fitz["compiled_experiment_id"] = "skin_fitzpatrick17k_external_stress"
    fitz["recommended_use"] = fitz["compiler_status"].map({
        "mapped_overlap_label": "use_only_as_external_stress_or_coarse_mapping",
        "reject_for_direct_pooling": "do_not_pool_with_dermoscopy_7class",
    })

    multiclass.to_csv(OUT_DIR / "skin_multiclass_7_ham10000_isic2019_manifest.csv", index=False)
    binary2020.to_csv(OUT_DIR / "skin_binary_melanoma_isic2020_manifest.csv", index=False)
    fitz.to_csv(OUT_DIR / "skin_fitzpatrick17k_external_stress_manifest.csv", index=False)

    all_manifest = pd.concat([multiclass, binary2020, fitz], ignore_index=True, sort=False)
    all_manifest.to_csv(OUT_DIR / "skin_all_compiled_metadata_manifest.csv", index=False)

    multiclass_counts = write_summary_table(multiclass, "skin_multiclass_7_ham10000_isic2019")
    binary_counts = write_summary_table(binary2020, "skin_binary_melanoma_isic2020")
    fitz_counts = write_summary_table(fitz, "skin_fitzpatrick17k_external_stress")

    audit = {
        "compiler_version": "0.2_metadata_manifest",
        "compiled_manifests": {
            "skin_multiclass_7_ham10000_isic2019": str(OUT_DIR / "skin_multiclass_7_ham10000_isic2019_manifest.csv"),
            "skin_binary_melanoma_isic2020": str(OUT_DIR / "skin_binary_melanoma_isic2020_manifest.csv"),
            "skin_fitzpatrick17k_external_stress": str(OUT_DIR / "skin_fitzpatrick17k_external_stress_manifest.csv"),
            "skin_all_compiled_metadata_manifest": str(OUT_DIR / "skin_all_compiled_metadata_manifest.csv"),
        },
        "decision_logic": {
            "HAM10000_ISIC2019": "accepted for 7-class mapping after excluding ISIC2019 SCC and UNK extension labels",
            "ISIC2020": "accepted as binary melanoma-only task, not directly pooled with 7-class datasets",
            "Fitzpatrick17k": "rejected for direct dermoscopy pooling; reserved for external stress/coarse mapping because it is broad dermatology",
        },
        "counts": {
            "multiclass_manifest_rows": int(len(multiclass)),
            "multiclass_accepted_7class_rows": int((multiclass["compiler_status"] == "accepted_7class").sum()),
            "multiclass_flagged_extension_rows": int((multiclass["compiler_status"] == "flagged_extension_label").sum()),
            "isic2020_binary_rows": int(len(binary2020)),
            "fitzpatrick_rows": int(len(fitz)),
            "fitzpatrick_mapped_overlap_rows": int((fitz["compiler_status"] == "mapped_overlap_label").sum()),
            "fitzpatrick_rejected_direct_pooling_rows": int((fitz["compiler_status"] == "reject_for_direct_pooling").sum()),
        },
    }

    (OUT_COMPILER / "compiler_audit_manifest_v0_2_skin_metadata.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    lines = []
    lines.append("# MedDataForge Skin-Lesion Experiment Manifest Report\n")
    lines.append("## Compiler decisions\n")
    lines.append("- **HAM10000 + ISIC2019**: accepted for 7-class diagnostic classification after excluding ISIC2019 extension labels SCC and UNK.")
    lines.append("- **ISIC2020**: accepted as binary melanoma-vs-non-melanoma, not directly pooled with the 7-class task.")
    lines.append("- **Fitzpatrick17k**: rejected for direct pooling with dermoscopy challenge datasets; retained as external stress/coarse-mapping dataset.\n")

    lines.append("## Counts\n")
    for k, v in audit["counts"].items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## 7-class manifest label counts\n")
    lines.append(multiclass_counts.to_markdown(index=False))

    lines.append("\n\n## ISIC2020 binary label counts\n")
    lines.append(binary_counts.to_markdown(index=False))

    lines.append("\n\n## Fitzpatrick17k direct-pooling decision counts\n")
    lines.append(fitz["compiler_status"].value_counts().to_markdown())

    report_path = OUT_TABLES / "skin_experiment_manifest_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("===== SKIN EXPERIMENT MANIFEST COMPILATION COMPLETE =====")
    print(json.dumps(audit, indent=2))
    print(f"Report: {report_path}")

    print("\n===== 7-CLASS LABEL COUNTS =====")
    print(multiclass_counts.to_string(index=False))

    print("\n===== BINARY ISIC2020 LABEL COUNTS =====")
    print(binary_counts.to_string(index=False))

    print("\n===== FITZPATRICK COMPILER STATUS =====")
    print(fitz["compiler_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
