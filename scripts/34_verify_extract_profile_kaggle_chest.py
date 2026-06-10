#!/usr/bin/env python3
import json
import zipfile
from pathlib import Path
from collections import Counter, defaultdict

ARCHIVES = [
    {
        "dataset": "chest_xray_pneumonia",
        "archive": Path("data/images_raw_archives/chest/kaggle/chest_xray_pneumonia/chest-xray-pneumonia.zip"),
        "extract_to": Path("data/images_raw/chest/kaggle/chest_xray_pneumonia"),
    },
    {
        "dataset": "covid19_radiography",
        "archive": Path("data/images_raw_archives/chest/kaggle/covid19_radiography/covid19-radiography-database.zip"),
        "extract_to": Path("data/images_raw/chest/kaggle/covid19_radiography"),
    },
    {
        "dataset": "indiana_chest_xrays",
        "archive": Path("data/images_raw_archives/chest/kaggle/indiana_chest_xrays/chest-xrays-indiana-university.zip"),
        "extract_to": Path("data/images_raw/chest/kaggle/indiana_chest_xrays"),
    },
]

OUT_JSON = Path("results/tables/kaggle_chest_extract_profile.json")
OUT_MD = Path("results/tables/kaggle_chest_extract_profile.md")
OUT_TXT = Path("results/tables/kaggle_chest_file_tree_preview.txt")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TABLE_EXTS = {".csv", ".txt", ".json", ".xml", ".xlsx", ".tsv"}


def verify_zip(path):
    if not path.exists():
        return {"exists": False, "ok": False, "error": "missing archive", "members": 0, "bytes": 0}
    try:
        with zipfile.ZipFile(path, "r") as z:
            bad = z.testzip()
            return {
                "exists": True,
                "ok": bad is None,
                "error": "" if bad is None else f"bad member: {bad}",
                "members": len(z.namelist()),
                "bytes": path.stat().st_size,
            }
    except Exception as e:
        return {
            "exists": True,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "members": 0,
            "bytes": path.stat().st_size if path.exists() else 0,
        }


def extract_zip(path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(path, "r") as z:
        for member in z.namelist():
            target = out_dir / member
            if target.exists() and target.stat().st_size > 0:
                skipped += 1
                continue
            z.extract(member, out_dir)
            extracted += 1
    return {"newly_extracted": extracted, "already_existing": skipped}


def profile_tree(root):
    files = [p for p in root.rglob("*") if p.is_file()]
    ext_counts = Counter(p.suffix.lower() or "[no_ext]" for p in files)
    image_files = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    metadata_files = [p for p in files if p.suffix.lower() in TABLE_EXTS]

    top_dirs = Counter()
    class_like_dirs = Counter()

    for p in image_files:
        rel = p.relative_to(root)
        parts = rel.parts
        if len(parts) >= 2:
            top_dirs[parts[0]] += 1
            class_like_dirs["/".join(parts[:-1])] += 1
        elif len(parts) == 1:
            top_dirs["[root]"] += 1

    samples = [str(p) for p in image_files[:20]]
    metadata_samples = [str(p) for p in metadata_files[:30]]

    return {
        "root": str(root),
        "total_files": len(files),
        "total_images": len(image_files),
        "total_metadata_like_files": len(metadata_files),
        "extension_counts": dict(ext_counts.most_common(20)),
        "top_image_dirs": dict(top_dirs.most_common(30)),
        "class_like_image_dirs": dict(class_like_dirs.most_common(50)),
        "sample_images": samples,
        "metadata_like_files": metadata_samples,
    }


def main():
    results = []
    tree_lines = []

    print("===== VERIFYING ARCHIVES =====", flush=True)

    for item in ARCHIVES:
        verify = verify_zip(item["archive"])
        row = {
            "dataset": item["dataset"],
            "archive": str(item["archive"]),
            "extract_to": str(item["extract_to"]),
            "verify": verify,
        }
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)

    if not all(r["verify"]["ok"] for r in results):
        out = {"status": "verification_failed", "results": results}
        OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
        raise SystemExit("At least one ZIP failed verification. Stop before extraction.")

    print("===== EXTRACTING ARCHIVES =====", flush=True)

    for row, item in zip(results, ARCHIVES):
        info = extract_zip(item["archive"], item["extract_to"])
        row["extract"] = info
        print(json.dumps({"dataset": item["dataset"], "extract": info}, indent=2), flush=True)

    print("===== PROFILING EXTRACTED TREES =====", flush=True)

    for row, item in zip(results, ARCHIVES):
        profile = profile_tree(item["extract_to"])
        row["profile"] = profile
        print(json.dumps({
            "dataset": item["dataset"],
            "total_images": profile["total_images"],
            "total_files": profile["total_files"],
            "top_image_dirs": profile["top_image_dirs"],
            "metadata_like_files": profile["metadata_like_files"][:10],
        }, indent=2), flush=True)

        tree_lines.append(f"===== {item['dataset']} =====")
        for p in sorted([x for x in item["extract_to"].rglob("*") if x.is_file()])[:300]:
            tree_lines.append(str(p))
        tree_lines.append("")

    OUT_TXT.write_text("\n".join(tree_lines), encoding="utf-8")

    out = {
        "status": "complete",
        "datasets": results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Kaggle Chest X-ray Extract/Profile Report\n")
    lines.append("## Summary\n")
    for row in results:
        prof = row["profile"]
        lines.append(f"- **{row['dataset']}**: {prof['total_images']} images, {prof['total_files']} files")
    lines.append("\n## Dataset details\n")
    for row in results:
        prof = row["profile"]
        lines.append(f"\n### {row['dataset']}\n")
        lines.append(f"- Archive: `{row['archive']}`")
        lines.append(f"- Extracted to: `{row['extract_to']}`")
        lines.append(f"- Images: **{prof['total_images']}**")
        lines.append(f"- Files: **{prof['total_files']}**")
        lines.append("- Top image directories:")
        for k, v in list(prof["top_image_dirs"].items())[:20]:
            lines.append(f"  - `{k}`: {v}")
        lines.append("- Metadata-like files:")
        if prof["metadata_like_files"]:
            for m in prof["metadata_like_files"][:20]:
                lines.append(f"  - `{m}`")
        else:
            lines.append("  - none detected")
    lines.append(f"\nFile tree preview: `{OUT_TXT}`")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("===== KAGGLE CHEST VERIFY/EXTRACT/PROFILE COMPLETE =====")
    print(f"JSON: {OUT_JSON}")
    print(f"Markdown: {OUT_MD}")
    print(f"Tree preview: {OUT_TXT}")

    print("\n===== COPYABLE SUMMARY =====")
    for row in results:
        prof = row["profile"]
        print(f"{row['dataset']}: images={prof['total_images']} files={prof['total_files']} extract_to={row['extract_to']}")


if __name__ == "__main__":
    main()
