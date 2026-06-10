#!/usr/bin/env python3
import json
import shutil
import time
import zipfile
from pathlib import Path
from collections import Counter

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


def safe_extract_member(z, info, out_dir):
    target = out_dir / info.filename

    if info.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return "dir"

    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        current_size = target.stat().st_size
        expected_size = info.file_size
        if current_size == expected_size:
            return "skip_complete"
        target.unlink()

    with z.open(info, "r") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)

    return "extracted_or_repaired"


def count_images(root):
    return sum(
        1 for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


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

    return {
        "root": str(root),
        "total_files": len(files),
        "total_images": len(image_files),
        "total_metadata_like_files": len(metadata_files),
        "extension_counts": dict(ext_counts.most_common(30)),
        "top_image_dirs": dict(top_dirs.most_common(40)),
        "class_like_image_dirs": dict(class_like_dirs.most_common(80)),
        "sample_images": [str(p) for p in image_files[:20]],
        "metadata_like_files": [str(p) for p in metadata_files[:40]],
    }


def extract_archive(item):
    dataset = item["dataset"]
    archive = item["archive"]
    out_dir = item["extract_to"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")

    print(f"\n===== START DATASET: {dataset} =====", flush=True)
    print(f"Archive: {archive}", flush=True)
    print(f"Extract to: {out_dir}", flush=True)
    print(f"Archive size GB: {archive.stat().st_size / (1024**3):.2f}", flush=True)

    started = time.time()
    counts = Counter()

    with zipfile.ZipFile(archive, "r") as z:
        members = z.infolist()
        total = len(members)
        print(f"Total ZIP members: {total}", flush=True)

        for i, info in enumerate(members, start=1):
            status = safe_extract_member(z, info, out_dir)
            counts[status] += 1

            if i == 1 or i % 500 == 0 or i == total:
                elapsed = time.time() - started
                pct = 100.0 * i / max(1, total)
                img_count = count_images(out_dir)
                print(
                    f"[{dataset}] {i}/{total} ({pct:.1f}%) | "
                    f"extracted_or_repaired={counts['extracted_or_repaired']} | "
                    f"skipped_complete={counts['skip_complete']} | "
                    f"dirs={counts['dir']} | "
                    f"images_now={img_count} | "
                    f"elapsed_min={elapsed/60:.1f}",
                    flush=True,
                )

    profile = profile_tree(out_dir)

    print(f"===== FINISHED DATASET: {dataset} =====", flush=True)
    print(f"Images: {profile['total_images']}", flush=True)
    print(f"Files: {profile['total_files']}", flush=True)
    print(f"Top image dirs: {json.dumps(profile['top_image_dirs'], indent=2)}", flush=True)

    return {
        "dataset": dataset,
        "archive": str(archive),
        "extract_to": str(out_dir),
        "extract_counts": dict(counts),
        "profile": profile,
    }


def main():
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    print("===== RESUMABLE KAGGLE CHEST EXTRACTION START =====", flush=True)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    results = []
    for item in ARCHIVES:
        results.append(extract_archive(item))

    tree_lines = []
    for row in results:
        root = Path(row["extract_to"])
        tree_lines.append(f"===== {row['dataset']} =====")
        for p in sorted([x for x in root.rglob("*") if x.is_file()])[:400]:
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
        for k, v in list(prof["top_image_dirs"].items())[:30]:
            lines.append(f"  - `{k}`: {v}")
        lines.append("- Metadata-like files:")
        if prof["metadata_like_files"]:
            for m in prof["metadata_like_files"][:30]:
                lines.append(f"  - `{m}`")
        else:
            lines.append("  - none detected")
    lines.append(f"\nFile tree preview: `{OUT_TXT}`")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("\n===== KAGGLE CHEST VERIFY/EXTRACT/PROFILE COMPLETE =====", flush=True)
    print(f"JSON: {OUT_JSON}", flush=True)
    print(f"Markdown: {OUT_MD}", flush=True)
    print(f"Tree preview: {OUT_TXT}", flush=True)

    print("\n===== COPYABLE SUMMARY =====", flush=True)
    for row in results:
        prof = row["profile"]
        print(f"{row['dataset']}: images={prof['total_images']} files={prof['total_files']} extract_to={row['extract_to']}", flush=True)


if __name__ == "__main__":
    main()
