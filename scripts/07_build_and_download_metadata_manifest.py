#!/usr/bin/env python3
import json
import re
from pathlib import Path

import pandas as pd
import requests


IN_CSV = Path("data/acquisition/probe_wave1a_candidate_files.csv")
OUT_MANIFEST = Path("data/acquisition/metadata_download_manifest.csv")
OUT_DIR = Path("data/metadata_raw")
OUT_SUMMARY = Path("data/acquisition/metadata_download_summary.json")


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", x)
    return x.strip("_")


def select_metadata_files(df):
    rows = []

    for _, r in df.iterrows():
        dataset = str(r.get("dataset_name", ""))
        label = str(r.get("label", ""))
        url = str(r.get("url", ""))
        kind = str(r.get("kind", "")).lower()
        low = f"{dataset} {label} {url}".lower()

        keep = False
        reason = ""

        if dataset == "ChestX-ray8":
            if any(x in low for x in ["data_entry_2017_v2020.csv", "train_val_list.txt", "test_list.txt"]):
                keep = True
                reason = "nih_chestxray_metadata_or_split"

        elif dataset == "Fitzpatrick 17k":
            if "fitzpatrick17k.csv" in low:
                keep = True
                reason = "fitzpatrick17k_labels"

        elif dataset == "HAM10000":
            if "ham10000_metadata.tab" in low:
                keep = True
                reason = "ham10000_labels"
            elif "isic2018_task3_test_groundtruth.tab" in low:
                keep = True
                reason = "ham10000_related_isic2018_test_labels"

        elif dataset == "ISIC":
            # Keep classification metadata/ground-truth CSV/TAB files only.
            if kind in {"csv", "tab"} and any(x in low for x in [
                "isic2018_task3_training_groundtruth",
                "isic2018_task3_validation_groundtruth",
                "isic2018_task3_test_groundtruth",
                "isic_2019_training_groundtruth",
                "isic_2019_training_metadata",
                "isic_2019_test_groundtruth",
                "isic_2019_test_metadata",
                "isic_2020_training_groundtruth",
                "isic_2020_training_groundtruth_v2",
                "isic_2020_training_duplicates",
            ]):
                keep = True
                reason = "isic_classification_metadata"

        if keep:
            out = r.to_dict()
            out["selection_reason"] = reason
            rows.append(out)

    return pd.DataFrame(rows)


def download_file(url, out_path, timeout=120):
    headers = {"User-Agent": "MedDataForge-MetadataDownloader/0.1"}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as r:
            status = r.status_code
            if status >= 400:
                return {
                    "ok": False,
                    "status_code": status,
                    "bytes": 0,
                    "error": f"HTTP {status}",
                }
            n = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        n += len(chunk)
            return {
                "ok": True,
                "status_code": status,
                "bytes": n,
                "error": "",
            }
    except Exception as e:
        return {
            "ok": False,
            "status_code": "",
            "bytes": 0,
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Missing probe file: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    meta = select_metadata_files(df)

    if meta.empty:
        raise SystemExit("No metadata files selected. Probe output needs review.")

    # Add local output paths.
    local_paths = []
    for _, r in meta.iterrows():
        dataset_dir = OUT_DIR / safe_name(r["dataset_name"])
        label = safe_name(Path(str(r["label"])).name)
        if not label:
            label = safe_name(str(r["label"])) or "metadata_file"
        local_paths.append(str(dataset_dir / label))

    meta["local_path"] = local_paths
    meta.to_csv(OUT_MANIFEST, index=False)

    results = []
    for _, r in meta.iterrows():
        url = r["url"]
        out_path = Path(r["local_path"])
        print(f"Downloading metadata: {r['dataset_name']} | {r['label']} -> {out_path}", flush=True)

        if out_path.exists() and out_path.stat().st_size > 0:
            res = {
                "ok": True,
                "status_code": "cached",
                "bytes": out_path.stat().st_size,
                "error": "",
            }
        else:
            res = download_file(url, out_path)

        rr = r.to_dict()
        rr.update(res)
        results.append(rr)

    summary = {
        "metadata_files_selected": int(len(meta)),
        "metadata_files_downloaded_or_cached": int(sum(1 for x in results if x["ok"])),
        "metadata_files_failed": int(sum(1 for x in results if not x["ok"])),
        "by_dataset": pd.DataFrame(results).groupby("dataset_name").size().astype(int).to_dict(),
        "failed": [
            {
                "dataset_name": x["dataset_name"],
                "label": x["label"],
                "url": x["url"],
                "error": x["error"],
            }
            for x in results if not x["ok"]
        ],
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv("data/acquisition/metadata_download_results.csv", index=False)

    print("===== METADATA DOWNLOAD COMPLETE =====")
    print(json.dumps(summary, indent=2))

    print("\n===== Downloaded metadata files =====")
    for x in results:
        print(f"{x['dataset_name']} | {x['label']} | ok={x['ok']} | bytes={x['bytes']} | {x['local_path']}")


if __name__ == "__main__":
    main()
