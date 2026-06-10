#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


OUT_DIR = Path("results/compiler")
OUT_TABLES = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)


PATHS = {
    "chestxray8": Path("data/metadata_raw_clean/ChestX-ray8/Data_Entry_2017_v2020.csv"),
    "fitzpatrick17k": Path("data/metadata_raw_clean/Fitzpatrick_17k/fitzpatrick17k.csv"),
    "ham10000": Path("data/metadata_raw_clean/HAM10000/HAM10000_metadata.tab"),
    "isic2019_gt": Path("data/metadata_raw_clean/ISIC/ISIC_2019_Training_GroundTruth.csv"),
    "isic2019_meta": Path("data/metadata_raw_clean/ISIC/ISIC_2019_Training_Metadata.csv"),
    "isic2020_gt_v2": Path("data/metadata_raw_clean/ISIC/ISIC_2020_Training_GroundTruth_v2.csv"),
}


SKIN_CANONICAL_7 = {
    "mel": "melanoma",
    "nv": "melanocytic_nevus",
    "bcc": "basal_cell_carcinoma",
    "akiec": "actinic_keratosis_or_intraepithelial_carcinoma",
    "bkl": "benign_keratosis_like_lesion",
    "df": "dermatofibroma",
    "vasc": "vascular_lesion",
}

ISIC_ONEHOT_2019 = {
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

ISIC_ONEHOT_2018 = {
    "MEL": "melanoma",
    "NV": "melanocytic_nevus",
    "BCC": "basal_cell_carcinoma",
    "AKIEC": "actinic_keratosis_or_intraepithelial_carcinoma",
    "BKL": "benign_keratosis_like_lesion",
    "DF": "dermatofibroma",
    "VASC": "vascular_lesion",
}

FITZ_CANONICAL = {
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

CHEST_CANONICAL = {
    "No Finding": "no_finding",
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Effusion": "pleural_effusion",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Hernia": "hernia",
    "Infiltration": "infiltration",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pleural_Thickening": "pleural_thickening",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
}


def read_csv(path, sep=","):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=sep, low_memory=False)


def chestxray8_profile():
    df = read_csv(PATHS["chestxray8"])
    label_rows = []
    for labels in df["Finding Labels"].dropna().astype(str):
        for raw in labels.split("|"):
            raw = raw.strip()
            label_rows.append({
                "dataset": "ChestX-ray8",
                "raw_label": raw,
                "canonical_label": CHEST_CANONICAL.get(raw, "unmapped"),
            })
    counts = pd.DataFrame(label_rows).value_counts(["dataset", "raw_label", "canonical_label"]).reset_index(name="count")
    return df, counts


def ham10000_profile():
    df = read_csv(PATHS["ham10000"], sep="\t")
    rows = []
    for raw, count in df["dx"].astype(str).value_counts().items():
        rows.append({
            "dataset": "HAM10000",
            "raw_label": raw,
            "canonical_label": SKIN_CANONICAL_7.get(raw, "unmapped"),
            "count": int(count),
        })
    return df, pd.DataFrame(rows)


def isic2019_profile():
    df = read_csv(PATHS["isic2019_gt"])
    rows = []
    present = [c for c in ISIC_ONEHOT_2019 if c in df.columns]
    for c in present:
        count = int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
        rows.append({
            "dataset": "ISIC2019",
            "raw_label": c,
            "canonical_label": ISIC_ONEHOT_2019.get(c, "unmapped"),
            "count": count,
        })
    return df, pd.DataFrame(rows)


def isic2020_profile():
    df = read_csv(PATHS["isic2020_gt_v2"])
    rows = []

    if "diagnosis" in df.columns:
        vc = df["diagnosis"].astype(str).value_counts()
        for raw, count in vc.items():
            raw_low = raw.lower().strip()
            canonical = "melanoma" if raw_low == "melanoma" else "non_melanoma_or_other"
            rows.append({
                "dataset": "ISIC2020",
                "raw_label": raw,
                "canonical_label": canonical,
                "count": int(count),
            })

    if "target" in df.columns:
        vc = df["target"].astype(str).value_counts()
        for raw, count in vc.items():
            canonical = "melanoma" if raw == "1" else "non_melanoma"
            rows.append({
                "dataset": "ISIC2020_target",
                "raw_label": raw,
                "canonical_label": canonical,
                "count": int(count),
            })

    return df, pd.DataFrame(rows)


def fitzpatrick_profile():
    df = read_csv(PATHS["fitzpatrick17k"])
    rows = []

    if "label" in df.columns:
        for raw, count in df["label"].astype(str).value_counts().items():
            canonical = FITZ_CANONICAL.get(raw.lower().strip(), "outside_ham_isic_7_or_unmapped")
            rows.append({
                "dataset": "Fitzpatrick17k_label",
                "raw_label": raw,
                "canonical_label": canonical,
                "count": int(count),
            })

    if "three_partition_label" in df.columns:
        for raw, count in df["three_partition_label"].astype(str).value_counts().items():
            rows.append({
                "dataset": "Fitzpatrick17k_three_partition",
                "raw_label": raw,
                "canonical_label": raw.lower().replace(" ", "_"),
                "count": int(count),
            })

    if "nine_partition_label" in df.columns:
        for raw, count in df["nine_partition_label"].astype(str).value_counts().items():
            rows.append({
                "dataset": "Fitzpatrick17k_nine_partition",
                "raw_label": raw,
                "canonical_label": raw.lower().replace(" ", "_"),
                "count": int(count),
            })

    return df, pd.DataFrame(rows)


def make_pairwise_compat(label_df, domain):
    datasets = sorted(label_df["dataset"].unique().tolist())
    rows = []

    for a in datasets:
        la = set(label_df[label_df["dataset"] == a]["canonical_label"])
        la = {x for x in la if x and x != "unmapped"}
        for b in datasets:
            lb = set(label_df[label_df["dataset"] == b]["canonical_label"])
            lb = {x for x in lb if x and x != "unmapped"}
            inter = la & lb
            union = la | lb
            jacc = len(inter) / len(union) if union else 0.0
            rows.append({
                "domain": domain,
                "dataset_a": a,
                "dataset_b": b,
                "labels_a": len(la),
                "labels_b": len(lb),
                "intersection": len(inter),
                "union": len(union),
                "jaccard": round(jacc, 4),
                "shared_labels": ";".join(sorted(inter)),
                "a_only": ";".join(sorted(la - lb)),
                "b_only": ";".join(sorted(lb - la)),
            })

    return pd.DataFrame(rows)


def compile_skin_experiments(skin_labels):
    rows = []

    # Candidate experiment 1: HAM10000 + ISIC2019 multi-class mapping.
    multiclass7 = set(SKIN_CANONICAL_7.values())
    isic2019_labels = set(skin_labels[skin_labels["dataset"] == "ISIC2019"]["canonical_label"])
    ham_labels = set(skin_labels[skin_labels["dataset"] == "HAM10000"]["canonical_label"])

    rows.append({
        "experiment_id": "skin_multiclass_7_ham10000_isic2019",
        "domain": "skin_lesion",
        "candidate_datasets": "HAM10000;ISIC2019",
        "compiler_decision": "accept_with_label_mapping",
        "target_label_space": "7-class HAM10000/ISIC-style diagnostic labels",
        "shared_label_count": len(multiclass7 & isic2019_labels & ham_labels),
        "warning": "ISIC2019 includes SCC and UNK beyond HAM10000 7-class space; compiler must reject/drop or map separately.",
    })

    # Candidate experiment 2: ISIC2020 binary melanoma.
    rows.append({
        "experiment_id": "skin_binary_melanoma_isic2020",
        "domain": "skin_lesion",
        "candidate_datasets": "ISIC2020",
        "compiler_decision": "accept_binary_task_only",
        "target_label_space": "melanoma vs non-melanoma",
        "shared_label_count": 2,
        "warning": "Not directly compatible with 7-class multi-class datasets without collapsing labels.",
    })

    # Candidate experiment 3: Fitzpatrick broad dermatology vs ISIC/HAM.
    rows.append({
        "experiment_id": "skin_fitzpatrick17k_to_isic_ham",
        "domain": "skin_lesion",
        "candidate_datasets": "Fitzpatrick17k;HAM10000;ISIC2019",
        "compiler_decision": "reject_for_direct_pooling",
        "target_label_space": "diagnostic labels",
        "shared_label_count": "",
        "warning": "Fitzpatrick17k is broad clinical dermatology and not a dermoscopy-only 7-class lesion benchmark; use for external stress test or coarse malignant/benign/non-neoplastic mapping only.",
    })

    return pd.DataFrame(rows)


def compile_chest_experiments(chest_labels):
    rows = []
    rows.append({
        "experiment_id": "chest_multilabel_nih_only",
        "domain": "chest_xray",
        "candidate_datasets": "ChestX-ray8",
        "compiler_decision": "accept_single_dataset_baseline",
        "target_label_space": "15 NIH finding labels including No Finding",
        "shared_label_count": 15,
        "warning": "Cross-dataset experiment requires Kaggle or credentialed datasets such as CheXpert/MIMIC/PadChest/VinDr.",
    })
    rows.append({
        "experiment_id": "chest_cross_dataset_future_core",
        "domain": "chest_xray",
        "candidate_datasets": "ChestX-ray8;CheXpert;MIMIC-CXR-JPG;PadChest;VinDr-PCXR",
        "compiler_decision": "defer_until_access_available",
        "target_label_space": "shared thoracic finding labels",
        "shared_label_count": "",
        "warning": "Core compatible external datasets are currently gated/manual; this is a fragmentation finding and access-priority target.",
    })
    return pd.DataFrame(rows)


def main():
    chest_df, chest_labels = chestxray8_profile()
    ham_df, ham_labels = ham10000_profile()
    isic2019_df, isic2019_labels = isic2019_profile()
    isic2020_df, isic2020_labels = isic2020_profile()
    fitz_df, fitz_labels = fitzpatrick_profile()

    chest_labels.to_csv(OUT_TABLES / "label_map_chestxray8.csv", index=False)

    skin_labels = pd.concat(
        [ham_labels, isic2019_labels, isic2020_labels, fitz_labels],
        ignore_index=True,
    )
    skin_labels.to_csv(OUT_TABLES / "label_map_skin_wave1.csv", index=False)

    chest_pair = make_pairwise_compat(chest_labels, "chest_xray")
    skin_pair = make_pairwise_compat(skin_labels, "skin_lesion")
    chest_pair.to_csv(OUT_TABLES / "compatibility_pairwise_chest.csv", index=False)
    skin_pair.to_csv(OUT_TABLES / "compatibility_pairwise_skin.csv", index=False)

    exp_skin = compile_skin_experiments(skin_labels)
    exp_chest = compile_chest_experiments(chest_labels)
    experiments = pd.concat([exp_skin, exp_chest], ignore_index=True)
    experiments.to_csv(OUT_TABLES / "compiled_experiment_decisions.csv", index=False)

    audit = {
        "compiler_version": "0.1_metadata_only",
        "inputs": {k: str(v) for k, v in PATHS.items() if v.exists()},
        "datasets_profiled": {
            "ChestX-ray8": {
                "rows": int(len(chest_df)),
                "label_space": "multi-label thoracic findings",
                "labels": sorted(chest_labels["canonical_label"].unique().tolist()),
            },
            "HAM10000": {
                "rows": int(len(ham_df)),
                "label_space": "7-class skin lesion diagnosis",
                "labels": sorted(ham_labels["canonical_label"].unique().tolist()),
            },
            "ISIC2019": {
                "rows": int(len(isic2019_df)),
                "label_space": "one-hot skin lesion diagnosis with SCC/UNK extension",
                "labels": sorted(isic2019_labels["canonical_label"].unique().tolist()),
            },
            "ISIC2020": {
                "rows": int(len(isic2020_df)),
                "label_space": "binary melanoma target plus diagnosis metadata",
                "labels": sorted(isic2020_labels["canonical_label"].unique().tolist()),
            },
            "Fitzpatrick17k": {
                "rows": int(len(fitz_df)),
                "label_space": "broad dermatology labels plus coarse partitions",
                "labels_mapped_to_skin7_or_related": sorted(
                    x for x in fitz_labels["canonical_label"].unique().tolist()
                    if x not in {"outside_ham_isic_7_or_unmapped"}
                ),
            },
        },
        "compiled_experiment_decisions": experiments.to_dict(orient="records"),
    }

    (OUT_DIR / "compiler_audit_manifest_v0_1.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = []
    lines.append("# MedDataForge Label Compatibility Compiler Report\n")
    lines.append("## Main finding\n")
    lines.append(
        "The first metadata-only compiler pass shows that accessible Wave 1 datasets are not automatically poolable. "
        "HAM10000 and ISIC2019 share a mostly compatible multi-class lesion label space, ISIC2020 is primarily binary melanoma, "
        "and Fitzpatrick17k is a broader dermatology dataset that should not be naively pooled with dermoscopy challenge datasets."
    )

    lines.append("\n## Compiled experiment decisions\n")
    lines.append(experiments.to_markdown(index=False))

    lines.append("\n\n## Skin pairwise compatibility preview\n")
    preview = skin_pair[skin_pair["dataset_a"] != skin_pair["dataset_b"]].sort_values("jaccard", ascending=False).head(20)
    lines.append(preview.to_markdown(index=False))

    lines.append("\n\n## Chest experiment decision\n")
    lines.append(exp_chest.to_markdown(index=False))

    (OUT_TABLES / "label_compatibility_compiler_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("===== LABEL COMPATIBILITY COMPILER COMPLETE =====")
    print(f"Chest label map: {OUT_TABLES / 'label_map_chestxray8.csv'}")
    print(f"Skin label map: {OUT_TABLES / 'label_map_skin_wave1.csv'}")
    print(f"Skin pairwise compatibility: {OUT_TABLES / 'compatibility_pairwise_skin.csv'}")
    print(f"Compiled decisions: {OUT_TABLES / 'compiled_experiment_decisions.csv'}")
    print(f"Audit manifest: {OUT_DIR / 'compiler_audit_manifest_v0_1.json'}")
    print(f"Report: {OUT_TABLES / 'label_compatibility_compiler_report.md'}")

    print("\n===== COMPILED EXPERIMENT DECISIONS =====")
    print(experiments.to_string(index=False))

    print("\n===== TOP SKIN PAIRWISE COMPATIBILITY =====")
    print(
        skin_pair[skin_pair["dataset_a"] != skin_pair["dataset_b"]]
        .sort_values("jaccard", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
