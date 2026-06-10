#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd


IN_PROTOCOLS = Path("data/compiled_protocols/skin_multiclass_7_linked_manifest.csv")
OUT_DIR = Path("data/training_splits")
OUT_TABLES = Path("results/tables")
OUT_COMPILER = Path("results/compiler")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_COMPILER.mkdir(parents=True, exist_ok=True)

SEED = 20260528


def stratified_split(df, label_col="canonical_label", val_frac=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    parts = []

    for label, g in df.groupby(label_col):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)

        n = len(idx)
        n_val = max(1, int(round(n * val_frac))) if n >= 5 else max(0, int(round(n * val_frac)))

        val_idx = set(idx[:n_val])
        gg = g.copy()
        gg["split"] = ["val" if i in val_idx else "train" for i in gg.index]
        parts.append(gg)

    return pd.concat(parts, ignore_index=True)


def make_p1(df):
    accepted = df[df["compiler_status"].eq("accepted_7class")].copy()

    ham = accepted[accepted["dataset"].eq("HAM10000")].copy()
    isic = accepted[accepted["dataset"].eq("ISIC2019")].copy()

    ham_split = stratified_split(ham, val_frac=0.15, seed=SEED)
    ham_split["protocol_id"] = "P1_train_ham_validate_ham_test_isic"
    isic["protocol_id"] = "P1_train_ham_validate_ham_test_isic"
    isic["split"] = "external_test"

    return pd.concat([ham_split, isic], ignore_index=True)


def make_p2(df):
    accepted = df[df["compiler_status"].eq("accepted_7class")].copy()

    ham = accepted[accepted["dataset"].eq("HAM10000")].copy()
    isic = accepted[accepted["dataset"].eq("ISIC2019")].copy()

    isic_split = stratified_split(isic, val_frac=0.15, seed=SEED + 1)
    isic_split["protocol_id"] = "P2_train_isic_validate_isic_test_ham"
    ham["protocol_id"] = "P2_train_isic_validate_isic_test_ham"
    ham["split"] = "external_test"

    return pd.concat([isic_split, ham], ignore_index=True)


def make_p3_filtered_pool(df):
    accepted = df[df["compiler_status"].eq("accepted_7class")].copy()
    accepted["protocol_id"] = "P3_compiler_filtered_pool_7class"

    # Dataset-aware stratified split: each dataset contributes to train/val/test.
    parts = []
    rng = np.random.default_rng(SEED + 2)

    for (dataset, label), g in accepted.groupby(["dataset", "canonical_label"]):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * 0.15))) if n >= 5 else 0
        n_val = max(1, int(round(n * 0.15))) if n >= 5 else 0

        test_idx = set(idx[:n_test])
        val_idx = set(idx[n_test:n_test + n_val])

        gg = g.copy()
        splits = []
        for i in gg.index:
            if i in test_idx:
                splits.append("test")
            elif i in val_idx:
                splits.append("val")
            else:
                splits.append("train")
        gg["split"] = splits
        parts.append(gg)

    return pd.concat(parts, ignore_index=True)


def make_p4_naive_pool(df):
    # Includes flagged extension label SCC from the linked manifest.
    p = df.copy()
    p["protocol_id"] = "P4_naive_pool_including_flagged_8class"

    parts = []
    rng = np.random.default_rng(SEED + 3)

    for (dataset, label), g in p.groupby(["dataset", "canonical_label"]):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * 0.15))) if n >= 5 else 0
        n_val = max(1, int(round(n * 0.15))) if n >= 5 else 0

        test_idx = set(idx[:n_test])
        val_idx = set(idx[n_test:n_test + n_val])

        gg = g.copy()
        splits = []
        for i in gg.index:
            if i in test_idx:
                splits.append("test")
            elif i in val_idx:
                splits.append("val")
            else:
                splits.append("train")
        gg["split"] = splits
        parts.append(gg)

    return pd.concat(parts, ignore_index=True)


def validate_paths(df):
    exists = df["image_path"].map(lambda x: Path(str(x)).exists())
    return int(exists.sum()), int((~exists).sum())


def main():
    if not IN_PROTOCOLS.exists():
        raise SystemExit(f"Missing input protocols: {IN_PROTOCOLS}")

    df = pd.read_csv(IN_PROTOCOLS, low_memory=False)
    df = df[df["image_path"].notna()].copy()

    p1 = make_p1(df)
    p2 = make_p2(df)
    p3 = make_p3_filtered_pool(df)
    p4 = make_p4_naive_pool(df)

    outputs = {
        "P1_train_ham_validate_ham_test_isic": p1,
        "P2_train_isic_validate_isic_test_ham": p2,
        "P3_compiler_filtered_pool_7class": p3,
        "P4_naive_pool_including_flagged_8class": p4,
    }

    summary_rows = []
    audit = {
        "split_compiler_version": "0.4_training_splits",
        "seed": SEED,
        "outputs": {},
    }

    for name, out in outputs.items():
        out_path = OUT_DIR / f"{name}.csv"
        out.to_csv(out_path, index=False)

        found, missing = validate_paths(out)
        audit["outputs"][name] = {
            "path": str(out_path),
            "rows": int(len(out)),
            "image_paths_found": found,
            "image_paths_missing": missing,
            "num_classes": int(out["canonical_label"].nunique()),
            "classes": sorted(out["canonical_label"].dropna().unique().tolist()),
        }

        tmp = (
            out.groupby(["protocol_id", "split", "dataset", "canonical_label"])
            .size()
            .reset_index(name="count")
        )
        summary_rows.append(tmp)

    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(OUT_TABLES / "skin_training_split_summary.csv", index=False)

    (OUT_COMPILER / "compiler_audit_manifest_v0_4_training_splits.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    lines = []
    lines.append("# MedDataForge Skin Training Split Report\n")
    lines.append("## Split compiler summary\n")
    lines.append(f"- Seed: {SEED}")
    for name, info in audit["outputs"].items():
        lines.append(f"\n### {name}")
        for k, v in info.items():
            lines.append(f"- **{k}**: {v}")

    lines.append("\n## Split count table preview\n")
    lines.append(summary.head(120).to_markdown(index=False))

    report = OUT_TABLES / "skin_training_split_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print("===== SKIN TRAINING SPLIT COMPILATION COMPLETE =====")
    print(json.dumps(audit, indent=2))

    print("\n===== SPLIT SUMMARY PREVIEW =====")
    print(summary.head(160).to_string(index=False))

    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
