#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


MANIFEST = Path("data/compiled_manifests/skin_multiclass_7_ham10000_isic2019_manifest.csv")
HAM_ROOT = Path("data/images_raw/skin/HAM10000")
ISIC_ROOT = Path("data/images_raw/skin/ISIC2019")

OUT_DIR = Path("data/compiled_protocols")
OUT_TABLES = Path("results/tables")
OUT_COMPILER = Path("results/compiler")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_COMPILER.mkdir(parents=True, exist_ok=True)


def build_image_index(root):
    exts = {".jpg", ".jpeg", ".png"}
    rows = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            stem = p.stem
            rows.append({
                "image_id": stem,
                "image_path": str(p),
                "filename": p.name,
            })
    return pd.DataFrame(rows)


def assign_protocols(df):
    accepted = df[df["compiler_status"].eq("accepted_7class")].copy()
    flagged = df[~df["compiler_status"].eq("accepted_7class")].copy()

    protocols = []

    # Protocol 1: train HAM, test ISIC.
    p1_train = accepted[accepted["dataset"].eq("HAM10000")].copy()
    p1_train["protocol_id"] = "P1_train_ham_test_isic"
    p1_train["split"] = "train"
    p1_test = accepted[accepted["dataset"].eq("ISIC2019")].copy()
    p1_test["protocol_id"] = "P1_train_ham_test_isic"
    p1_test["split"] = "external_test"
    protocols.append(pd.concat([p1_train, p1_test], ignore_index=True))

    # Protocol 2: train ISIC, test HAM.
    p2_train = accepted[accepted["dataset"].eq("ISIC2019")].copy()
    p2_train["protocol_id"] = "P2_train_isic_test_ham"
    p2_train["split"] = "train"
    p2_test = accepted[accepted["dataset"].eq("HAM10000")].copy()
    p2_test["protocol_id"] = "P2_train_isic_test_ham"
    p2_test["split"] = "external_test"
    protocols.append(pd.concat([p2_train, p2_test], ignore_index=True))

    # Protocol 3: compiler-filtered pooled data.
    p3 = accepted.copy()
    p3["protocol_id"] = "P3_compiler_filtered_pool"
    p3["split"] = "pool_train_eval_candidate"
    protocols.append(p3)

    # Protocol 4: naive pooled data, including flagged SCC extension if present.
    p4 = df.copy()
    p4["protocol_id"] = "P4_naive_pool_including_flagged"
    p4["split"] = "pool_train_eval_candidate"
    protocols.append(p4)

    all_protocols = pd.concat(protocols, ignore_index=True)

    if len(flagged):
        flagged = flagged.copy()
        flagged["protocol_id"] = "FLAGGED_EXTENSION_ROWS"
        flagged["split"] = "excluded_or_extension_experiment"
        all_protocols = pd.concat([all_protocols, flagged], ignore_index=True)

    return all_protocols


def main():
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")

    df = pd.read_csv(MANIFEST)

    print("Indexing HAM10000 images...", flush=True)
    ham_idx = build_image_index(HAM_ROOT)
    ham_idx["dataset"] = "HAM10000"

    print("Indexing ISIC2019 images...", flush=True)
    isic_idx = build_image_index(ISIC_ROOT)
    isic_idx["dataset"] = "ISIC2019"

    image_idx = pd.concat([ham_idx, isic_idx], ignore_index=True)
    image_idx.to_csv(OUT_DIR / "skin_image_index.csv", index=False)

    linked = df.merge(
        image_idx[["dataset", "image_id", "image_path"]],
        on=["dataset", "image_id"],
        how="left",
    )

    linked["image_link_status"] = linked["image_path"].apply(
        lambda x: "found" if isinstance(x, str) and len(x) > 0 else "missing"
    )

    linked.to_csv(OUT_DIR / "skin_multiclass_7_linked_manifest.csv", index=False)

    protocols = assign_protocols(linked)
    protocols.to_csv(OUT_DIR / "skin_multiclass_7_protocols.csv", index=False)

    summary_rows = []
    for group_cols in [
        ["dataset"],
        ["dataset", "compiler_status"],
        ["dataset", "canonical_label", "compiler_status"],
        ["protocol_id", "split"],
        ["protocol_id", "split", "dataset"],
        ["protocol_id", "split", "canonical_label"],
    ]:
        tmp = protocols.groupby(group_cols).size().reset_index(name="count")
        tmp.insert(0, "grouping", "+".join(group_cols))
        summary_rows.append(tmp)

    summary_table = pd.concat(summary_rows, ignore_index=True, sort=False)
    summary_table.to_csv(OUT_TABLES / "skin_protocol_summary_counts.csv", index=False)

    missing = linked[linked["image_link_status"].eq("missing")]
    missing.to_csv(OUT_TABLES / "skin_missing_image_links.csv", index=False)

    audit = {
        "compiler_version": "0.3_image_linked_protocols",
        "image_counts": {
            "HAM10000": int(len(ham_idx)),
            "ISIC2019": int(len(isic_idx)),
            "total_indexed": int(len(image_idx)),
        },
        "manifest_rows": int(len(df)),
        "linked_rows": int(len(linked)),
        "found_image_rows": int((linked["image_link_status"] == "found").sum()),
        "missing_image_rows": int((linked["image_link_status"] == "missing").sum()),
        "accepted_7class_rows": int((linked["compiler_status"] == "accepted_7class").sum()),
        "flagged_extension_rows": int((linked["compiler_status"] == "flagged_extension_label").sum()),
        "protocol_file": str(OUT_DIR / "skin_multiclass_7_protocols.csv"),
        "linked_manifest_file": str(OUT_DIR / "skin_multiclass_7_linked_manifest.csv"),
        "image_index_file": str(OUT_DIR / "skin_image_index.csv"),
    }

    (OUT_COMPILER / "compiler_audit_manifest_v0_3_image_protocols.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    lines = []
    lines.append("# MedDataForge Skin-Lesion Image-Linked Protocol Report\n")
    lines.append("## Image linking summary\n")
    for k, v in audit.items():
        if not isinstance(v, dict):
            lines.append(f"- **{k}**: {v}")
    lines.append("\n## Image counts\n")
    for k, v in audit["image_counts"].items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## Protocol summary\n")
    prot = protocols.groupby(["protocol_id", "split", "dataset"]).size().reset_index(name="count")
    lines.append(prot.to_markdown(index=False))

    lines.append("\n\n## Label counts by protocol\n")
    lab = protocols.groupby(["protocol_id", "split", "canonical_label"]).size().reset_index(name="count")
    lines.append(lab.head(100).to_markdown(index=False))

    report = OUT_TABLES / "skin_image_linked_protocol_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print("===== IMAGE LINKING + PROTOCOL COMPILATION COMPLETE =====")
    print(json.dumps(audit, indent=2))

    print("\n===== Protocol summary =====")
    print(protocols.groupby(["protocol_id", "split", "dataset"]).size().reset_index(name="count").to_string(index=False))

    print("\n===== Missing image links =====")
    print(f"missing rows: {len(missing)}")
    if len(missing):
        print(missing[["dataset", "image_id", "canonical_label", "compiler_status"]].head(30).to_string(index=False))

    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
