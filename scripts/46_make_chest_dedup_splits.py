#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np
from PIL import Image

SEED = 20260609

C2_IN = Path("data/training_splits/chest/C2_compiler_compatible_binary_pool.csv")
C3_IN = Path("data/training_splits/chest/C3_naive_binary_pool_all_abnormal_as_pneumonia.csv")

OUT_DIR = Path("data/training_splits/chest_dedup")
OUT_TABLES = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

C2_OUT = OUT_DIR / "C2_compiler_compatible_binary_pool_dedup.csv"
C3_OUT = OUT_DIR / "C3_naive_binary_pool_all_abnormal_dedup_same_c2test.csv"
REPORT_MD = OUT_TABLES / "chest_dedup_split_audit_report.md"
REPORT_JSON = OUT_TABLES / "chest_dedup_split_audit.json"


def file_md5(path, block_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def add_hashes(df, name):
    out = df.copy()
    hashes = []
    valid = []
    exists = []

    print(f"===== HASHING {name}: {len(out)} rows =====", flush=True)

    cache = {}
    for i, p in enumerate(out["image_path"].astype(str).tolist(), start=1):
        if i % 1000 == 0:
            print(f"{name}: hashed/checked {i}/{len(out)}", flush=True)

        path = Path(p)
        ex = path.exists()
        exists.append(ex)

        if not ex:
            valid.append(False)
            hashes.append("")
            continue

        if p in cache:
            h, ok = cache[p]
        else:
            ok = is_valid_image(path)
            h = file_md5(path) if ok else ""
            cache[p] = (h, ok)

        valid.append(ok)
        hashes.append(h)

    out["file_exists"] = exists
    out["valid_image"] = valid
    out["image_md5"] = hashes
    return out


def remove_conflicts_and_dedup(df, name):
    start = len(df)
    df = df[df["file_exists"] & df["valid_image"] & df["image_md5"].ne("")].copy()

    label_n = df.groupby("image_md5")["canonical_label"].nunique()
    conflict_hashes = set(label_n[label_n > 1].index)

    conflict_rows = df[df["image_md5"].isin(conflict_hashes)].copy()
    clean = df[~df["image_md5"].isin(conflict_hashes)].copy()

    # Keep one row per exact image content. Same hash + same label duplicates are removed.
    dedup = clean.sort_values(["canonical_label", "image_path"]).drop_duplicates("image_md5", keep="first").copy()

    summary = {
        "name": name,
        "input_rows": int(start),
        "valid_existing_rows": int(len(df)),
        "conflict_hashes": int(len(conflict_hashes)),
        "conflict_rows_removed": int(len(conflict_rows)),
        "dedup_rows": int(len(dedup)),
        "duplicate_rows_removed": int(len(clean) - len(dedup)),
    }
    return dedup, conflict_rows, summary


def stratified_three_way(df, seed=SEED, test_frac=0.15, val_frac=0.15):
    rng = np.random.default_rng(seed)
    parts = []

    for label, g in df.groupby("canonical_label"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)

        n = len(idx)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))

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

    return pd.concat(parts, ignore_index=True)


def stratified_train_val(df, seed=SEED, val_frac=0.15):
    rng = np.random.default_rng(seed)
    parts = []

    for label, g in df.groupby("canonical_label"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)

        n = len(idx)
        n_val = max(1, int(round(n * val_frac)))

        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        tmp = df.loc[train_idx].copy()
        tmp["split"] = "train"
        parts.append(tmp)

        tmp = df.loc[val_idx].copy()
        tmp["split"] = "val"
        parts.append(tmp)

    return pd.concat(parts, ignore_index=True)


def count_table(df):
    return (
        df.groupby(["split", "dataset", "canonical_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "dataset", "canonical_label"])
    )


def main():
    c2_raw = pd.read_csv(C2_IN, low_memory=False)
    c3_raw = pd.read_csv(C3_IN, low_memory=False)

    c2_hash = add_hashes(c2_raw, "C2")
    c3_hash = add_hashes(c3_raw, "C3")

    c2_dedup, c2_conflicts, c2_summary = remove_conflicts_and_dedup(c2_hash, "C2")
    c3_dedup, c3_conflicts, c3_summary = remove_conflicts_and_dedup(c3_hash, "C3")

    # C2 gets a clean train/val/test split.
    c2_split = stratified_three_way(c2_dedup, seed=SEED)

    # C3 uses the exact same C2 test hashes as its test set.
    c2_test = c2_split[c2_split["split"].eq("test")].copy()
    c2_test_hashes = set(c2_test["image_md5"])

    # Remove all C2-test hashes from C3 train/val pool to avoid hash leakage.
    c3_pool = c3_dedup[~c3_dedup["image_md5"].isin(c2_test_hashes)].copy()
    c3_train_val = stratified_train_val(c3_pool, seed=SEED, val_frac=0.15)

    c3_test = c2_test.copy()
    c3_test["split"] = "test"
    c3_test["protocol_note"] = "same_C2_dedup_test_for_fair_eval"

    c3_split = pd.concat([c3_train_val, c3_test], ignore_index=True)

    # Leakage checks.
    def leakage(split_df):
        rows = []
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            ha = set(split_df.loc[split_df["split"].eq(a), "image_md5"])
            hb = set(split_df.loc[split_df["split"].eq(b), "image_md5"])
            rows.append({"a": a, "b": b, "overlap_hashes": len(ha & hb)})
        return rows

    c2_leak = leakage(c2_split)
    c3_leak = leakage(c3_split)

    c2_split.to_csv(C2_OUT, index=False)
    c3_split.to_csv(C3_OUT, index=False)

    c2_counts = count_table(c2_split)
    c3_counts = count_table(c3_split)

    c2_counts.to_csv(OUT_TABLES / "chest_dedup_C2_split_counts.csv", index=False)
    c3_counts.to_csv(OUT_TABLES / "chest_dedup_C3_split_counts.csv", index=False)

    summary = {
        "seed": SEED,
        "c2_input": str(C2_IN),
        "c3_input": str(C3_IN),
        "c2_output": str(C2_OUT),
        "c3_output": str(C3_OUT),
        "c2_summary": c2_summary,
        "c3_summary": c3_summary,
        "c2_split_rows": int(len(c2_split)),
        "c3_split_rows": int(len(c3_split)),
        "c2_leakage": c2_leak,
        "c3_leakage": c3_leak,
        "c2_split_counts": c2_counts.to_dict(orient="records"),
        "c3_split_counts": c3_counts.to_dict(orient="records"),
    }

    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Chest Deduplicated Split Audit\n")
    lines.append("## Purpose\n")
    lines.append("This audit checks hash-level duplicate content and creates deduplicated C2/C3 chest X-ray splits. C3 is evaluated on the exact same deduplicated C2 test set.\n")
    lines.append("## Deduplication summary\n")
    lines.append(pd.DataFrame([c2_summary, c3_summary]).to_markdown(index=False))
    lines.append("\n## C2 hash leakage check\n")
    lines.append(pd.DataFrame(c2_leak).to_markdown(index=False))
    lines.append("\n## C3 hash leakage check\n")
    lines.append(pd.DataFrame(c3_leak).to_markdown(index=False))
    lines.append("\n## C2 split counts\n")
    lines.append(c2_counts.to_markdown(index=False))
    lines.append("\n## C3 split counts\n")
    lines.append(c3_counts.to_markdown(index=False))
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("===== CHEST DEDUP SPLIT AUDIT COMPLETE =====")
    print(json.dumps(summary, indent=2)[:8000])
    print(f"\nC2 split: {C2_OUT}")
    print(f"C3 split: {C3_OUT}")
    print(f"Report: {REPORT_MD}")


if __name__ == "__main__":
    main()
