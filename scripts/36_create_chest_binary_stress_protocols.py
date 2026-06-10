#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd


PNEU = Path("data/compiled_manifests/chest_binary_pneumonia_manifest.csv")
COVID = Path("data/compiled_manifests/chest_covid_radiography_multiclass_manifest.csv")

OUT_DIR = Path("data/training_splits/chest")
OUT_TABLES = Path("results/tables")
OUT_COMPILER = Path("results/compiler")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_COMPILER.mkdir(parents=True, exist_ok=True)

SEED = 20260602


def stratified_split(df, label_col="canonical_label", val_frac=0.15, test_frac=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    parts = []

    for label, g in df.groupby(label_col):
        idx = np.array(g.index.tolist()).copy()
        rng.shuffle(idx)

        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))

        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test+n_val]
        train_idx = idx[n_test+n_val:]

        tmp = df.loc[train_idx].copy()
        tmp["split"] = "train"
        parts.append(tmp)

        tmp = df.loc[val_idx].copy()
        tmp["split"] = "val"
        parts.append(tmp)

        tmp = df.loc[test_idx].copy()
        tmp["split"] = "test"
        parts.append(tmp)

    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_pneumonia_train_val_external_covid(pneu, covid):
    # Use original train/val/test structure where available.
    p = pneu.copy()
    p = p[p["canonical_label"].isin(["normal", "pneumonia"])].copy()
    p["split"] = p["split_original"].replace({"test": "test_internal"}).fillna("unknown")

    # Use train + val for model development. Keep original test as internal_test, but main test is COVID external.
    p.loc[p["split"].eq("train"), "split"] = "train"
    p.loc[p["split"].eq("val"), "split"] = "val"
    p.loc[p["split"].eq("test_internal"), "split"] = "internal_test"

    # If val is tiny, keep it as val. If no val, stratify train.
    if (p["split"] == "val").sum() == 0:
        train_only = p[p["split"].eq("train")].copy()
        rest = p[~p["split"].eq("train")].copy()
        split_train = stratified_split(train_only, val_frac=0.15, test_frac=0.0, seed=SEED)
        split_train = split_train[split_train["split"].isin(["train", "val"])].copy()
        p = pd.concat([split_train, rest], ignore_index=True)

    c = covid[covid["canonical_label"].isin(["normal", "viral_pneumonia"])].copy()
    c["canonical_label"] = c["canonical_label"].replace({"viral_pneumonia": "pneumonia"})
    c["split"] = "external_test"
    c["compiler_status"] = "accepted_as_external_binary_pneumonia_compatible_subset"
    c["compatibility_warning"] = "restricted to normal vs viral pneumonia subset for binary pneumonia compatibility"

    out = pd.concat([p, c], ignore_index=True)
    out["protocol_id"] = "C1_train_pneumonia_test_covid_viral"
    return out


def make_compiler_compatible_pool(pneu, covid):
    p = pneu[pneu["canonical_label"].isin(["normal", "pneumonia"])].copy()
    c = covid[covid["canonical_label"].isin(["normal", "viral_pneumonia"])].copy()
    c["canonical_label"] = c["canonical_label"].replace({"viral_pneumonia": "pneumonia"})

    p["source_binary_mapping"] = "native_normal_pneumonia"
    c["source_binary_mapping"] = "covid_subset_normal_vs_viral_pneumonia_only"

    pool = pd.concat([p, c], ignore_index=True)
    pool["compiler_status"] = "accepted_for_compiler_compatible_binary_pool"
    pool["compatibility_warning"] = "COVID subset restricted to viral pneumonia only; COVID and lung opacity excluded"

    split = stratified_split(pool, val_frac=0.15, test_frac=0.15, seed=SEED)
    split["protocol_id"] = "C2_compiler_compatible_binary_pool"
    return split


def make_naive_binary_pool(pneu, covid):
    p = pneu[pneu["canonical_label"].isin(["normal", "pneumonia"])].copy()
    c = covid[covid["canonical_label"].isin(["normal", "viral_pneumonia", "covid_19", "lung_opacity"])].copy()

    # Naive mapping: all abnormal COVID labels collapsed into pneumonia.
    c["original_covid_label"] = c["canonical_label"]
    c["canonical_label"] = c["canonical_label"].replace({
        "viral_pneumonia": "pneumonia",
        "covid_19": "pneumonia",
        "lung_opacity": "pneumonia",
    })

    p["source_binary_mapping"] = "native_normal_pneumonia"
    c["source_binary_mapping"] = "naive_all_covid_abnormal_to_pneumonia"

    pool = pd.concat([p, c], ignore_index=True)
    pool["compiler_status"] = "naive_binary_pool_with_label_semantic_collapse"
    pool["compatibility_warning"] = "COVID and lung opacity collapsed into pneumonia; intentionally naive noisy pooling protocol"

    split = stratified_split(pool, val_frac=0.15, test_frac=0.15, seed=SEED)
    split["protocol_id"] = "C3_naive_binary_pool_all_abnormal_as_pneumonia"
    return split


def main():
    pneu = pd.read_csv(PNEU, low_memory=False)
    covid = pd.read_csv(COVID, low_memory=False)

    c1 = make_pneumonia_train_val_external_covid(pneu, covid)
    c2 = make_compiler_compatible_pool(pneu, covid)
    c3 = make_naive_binary_pool(pneu, covid)

    files = {
        "C1_train_pneumonia_test_covid_viral": OUT_DIR / "C1_train_pneumonia_test_covid_viral.csv",
        "C2_compiler_compatible_binary_pool": OUT_DIR / "C2_compiler_compatible_binary_pool.csv",
        "C3_naive_binary_pool_all_abnormal_as_pneumonia": OUT_DIR / "C3_naive_binary_pool_all_abnormal_as_pneumonia.csv",
    }

    c1.to_csv(files["C1_train_pneumonia_test_covid_viral"], index=False)
    c2.to_csv(files["C2_compiler_compatible_binary_pool"], index=False)
    c3.to_csv(files["C3_naive_binary_pool_all_abnormal_as_pneumonia"], index=False)

    all_df = pd.concat([c1, c2, c3], ignore_index=True)
    summary = (
        all_df.groupby(["protocol_id", "split", "dataset", "canonical_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["protocol_id", "split", "dataset", "canonical_label"])
    )
    summary.to_csv(OUT_TABLES / "chest_binary_stress_protocol_summary.csv", index=False)

    decisions = pd.DataFrame([
        {
            "protocol_id": "C1_train_pneumonia_test_covid_viral",
            "decision": "external compatibility stress test",
            "rationale": "train on narrow pneumonia dataset and externally test on COVID dataset restricted to normal vs viral pneumonia",
        },
        {
            "protocol_id": "C2_compiler_compatible_binary_pool",
            "decision": "compiler-compatible pool",
            "rationale": "pool only label-compatible normal and viral/pneumonia images",
        },
        {
            "protocol_id": "C3_naive_binary_pool_all_abnormal_as_pneumonia",
            "decision": "intentionally naive pool",
            "rationale": "collapse COVID-19 and lung opacity into pneumonia to simulate unsafe label pooling",
        },
    ])
    decisions.to_csv(OUT_TABLES / "chest_binary_stress_protocol_decisions.csv", index=False)

    audit = {
        "compiler_version": "0.6_chest_binary_stress_protocols",
        "seed": SEED,
        "files": {k: str(v) for k, v in files.items()},
        "summary_file": str(OUT_TABLES / "chest_binary_stress_protocol_summary.csv"),
        "decisions_file": str(OUT_TABLES / "chest_binary_stress_protocol_decisions.csv"),
        "row_counts": {
            "C1": int(len(c1)),
            "C2": int(len(c2)),
            "C3": int(len(c3)),
        },
    }
    (OUT_COMPILER / "compiler_audit_manifest_v0_6_chest_binary_protocols.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    lines = []
    lines.append("# Chest Binary Stress-Test Protocol Report\n")
    lines.append("## Purpose\n")
    lines.append(
        "These protocols test the MedDataForge compatibility hypothesis in a second domain. "
        "The compiler separates genuinely compatible binary pneumonia labels from naive pooling that collapses COVID-19 and lung opacity into pneumonia."
    )
    lines.append("\n## Protocol decisions\n")
    lines.append(decisions.to_markdown(index=False))
    lines.append("\n## Protocol counts\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("\n## Output files\n")
    for k, v in files.items():
        lines.append(f"- **{k}**: `{v}`")

    (OUT_TABLES / "chest_binary_stress_protocol_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("===== CHEST BINARY STRESS PROTOCOLS COMPLETE =====")
    print(json.dumps(audit, indent=2))
    print()
    print("===== DECISIONS =====")
    print(decisions.to_string(index=False))
    print()
    print("===== SUMMARY =====")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
