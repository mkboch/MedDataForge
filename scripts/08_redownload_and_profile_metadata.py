#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


IN_MANIFEST = Path("data/acquisition/metadata_download_manifest.csv")
OUT_ROOT = Path("data/metadata_raw_clean")
OUT_PROFILE = Path("results/tables/metadata_file_profiles.csv")
OUT_LABEL_REPORT = Path("results/tables/label_space_report.md")
OUT_SUMMARY = Path("results/tables/metadata_profile_summary.json")


def safe_component(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", x)
    return x.strip("_") or "unknown"


def filename_from_url_or_label(url, label):
    path = urlparse(str(url)).path
    base = Path(path).name
    if base and "." in base:
        return safe_component(base)
    label_base = safe_component(label)
    return label_base


def download(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "cached", out_path.stat().st_size

    try:
        with requests.get(url, stream=True, timeout=180, allow_redirects=True, headers={"User-Agent": "MedDataForge/0.1"}) as r:
            if r.status_code >= 400:
                return False, f"HTTP {r.status_code}", 0
            n = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        n += len(chunk)
            return True, "", n
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", 0


def sniff_sep(path):
    suffix = path.suffix.lower()
    if suffix == ".tab" or suffix == ".tsv":
        return "\t"
    if suffix == ".csv":
        return ","
    # Try sniffing.
    try:
        sample = path.read_text(errors="ignore")[:4096]
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","


def read_table(path):
    sep = sniff_sep(path)
    try:
        return pd.read_csv(path, sep=sep, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception as e:
            raise e


def classify_dataset_file(dataset_name, filename):
    low = f"{dataset_name} {filename}".lower()

    if "chest" in dataset_name.lower() or "nih" in low:
        return "chest_xray"
    if any(x in dataset_name.lower() for x in ["fitzpatrick", "ham10000", "isic", "pad-ufes"]):
        return "skin_lesion"
    return "unknown"


def summarize_labels(dataset_name, df):
    cols = list(df.columns)
    lower_cols = {c.lower(): c for c in cols}
    label_info = {}

    if dataset_name == "ChestX-ray8":
        if "Finding Labels" in cols:
            all_labels = []
            for x in df["Finding Labels"].dropna().astype(str):
                all_labels.extend([y.strip() for y in x.split("|")])
            s = pd.Series(all_labels).value_counts()
            label_info["label_column"] = "Finding Labels"
            label_info["label_counts"] = s.to_dict()
            label_info["num_unique_labels"] = int(len(s))

    elif dataset_name == "Fitzpatrick 17k":
        # Known columns may include label, nine_partition_label, three_partition_label, fitzpatrick_scale.
        for c in ["label", "nine_partition_label", "three_partition_label", "fitzpatrick_scale"]:
            if c in cols:
                vc = df[c].dropna().astype(str).value_counts().head(50)
                label_info[f"{c}_counts"] = vc.to_dict()

    elif dataset_name == "HAM10000":
        if "dx" in cols:
            vc = df["dx"].dropna().astype(str).value_counts()
            label_info["label_column"] = "dx"
            label_info["label_counts"] = vc.to_dict()
            label_info["num_unique_labels"] = int(len(vc))

    elif dataset_name == "ISIC":
        # ISIC 2018/2019 one-hot diagnostic columns.
        possible = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC", "UNK"]
        present = [c for c in possible if c in cols]
        if present:
            counts = {}
            for c in present:
                try:
                    counts[c] = int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
                except Exception:
                    counts[c] = int((df[c].astype(str) == "1").sum())
            label_info["onehot_label_columns"] = present
            label_info["label_counts"] = counts
            label_info["num_unique_labels"] = int(len(present))

        # ISIC 2020 target column.
        if "target" in cols:
            vc = df["target"].dropna().astype(str).value_counts()
            label_info["target_counts"] = vc.to_dict()

        if "diagnosis" in cols:
            vc = df["diagnosis"].dropna().astype(str).value_counts().head(50)
            label_info["diagnosis_counts"] = vc.to_dict()

    return label_info


def main():
    if not IN_MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {IN_MANIFEST}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(IN_MANIFEST)
    download_rows = []

    for _, r in manifest.iterrows():
        dataset = str(r["dataset_name"])
        url = str(r["url"])
        label = str(r["label"])
        fname = filename_from_url_or_label(url, label)
        out_path = OUT_ROOT / safe_component(dataset) / fname

        ok, error, nbytes = download(url, out_path)
        row = r.to_dict()
        row.update({
            "clean_local_path": str(out_path),
            "clean_filename": fname,
            "download_ok": ok,
            "download_error": error,
            "download_bytes": nbytes,
        })
        download_rows.append(row)
        print(f"{dataset} | {fname} | ok={ok} | bytes={nbytes} | {error}", flush=True)

    download_df = pd.DataFrame(download_rows)
    download_df.to_csv("data/acquisition/metadata_download_results_clean.csv", index=False)

    profiles = []
    label_reports = []

    for _, r in download_df.iterrows():
        dataset = str(r["dataset_name"])
        path = Path(str(r["clean_local_path"]))
        if not r["download_ok"] or not path.exists():
            continue
        if path.suffix.lower() not in [".csv", ".tab", ".tsv", ".txt"]:
            continue

        try:
            df = read_table(path)
            label_info = summarize_labels(dataset, df)
            profiles.append({
                "dataset_name": dataset,
                "file": path.name,
                "path": str(path),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": "|".join(map(str, df.columns.tolist())),
                "domain": classify_dataset_file(dataset, path.name),
                "label_info_json": json.dumps(label_info, ensure_ascii=False),
            })

            label_reports.append({
                "dataset_name": dataset,
                "file": path.name,
                "rows": int(len(df)),
                "columns": df.columns.tolist(),
                "label_info": label_info,
            })
        except Exception as e:
            profiles.append({
                "dataset_name": dataset,
                "file": path.name,
                "path": str(path),
                "rows": "",
                "columns": "",
                "column_names": "",
                "domain": classify_dataset_file(dataset, path.name),
                "label_info_json": "{}",
                "read_error": f"{type(e).__name__}: {e}",
            })

    prof = pd.DataFrame(profiles)
    prof.to_csv(OUT_PROFILE, index=False)

    lines = []
    lines.append("# MedDataForge Metadata and Label-Space Profile\n")
    lines.append("## Metadata files profiled\n")
    if not prof.empty:
        lines.append(prof[["dataset_name", "file", "rows", "columns", "domain"]].to_markdown(index=False))
    else:
        lines.append("No metadata profiles were generated.")

    lines.append("\n## Label-space summaries\n")
    for item in label_reports:
        lines.append(f"\n### {item['dataset_name']} | {item['file']}\n")
        lines.append(f"- Rows: {item['rows']}")
        lines.append(f"- Columns: {', '.join(map(str, item['columns'][:40]))}")
        li = item["label_info"]
        if not li:
            lines.append("- Label summary: not detected")
        else:
            lines.append("```json")
            lines.append(json.dumps(li, indent=2, ensure_ascii=False)[:6000])
            lines.append("```")

    OUT_LABEL_REPORT.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "manifest_rows": int(len(manifest)),
        "downloaded_or_cached": int(download_df["download_ok"].sum()),
        "failed": int((~download_df["download_ok"]).sum()),
        "profiled_files": int(len(prof)),
        "datasets_profiled": sorted(prof["dataset_name"].dropna().unique().tolist()) if not prof.empty else [],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== METADATA PROFILE COMPLETE =====")
    print(json.dumps(summary, indent=2))
    print(f"Profile CSV: {OUT_PROFILE}")
    print(f"Label report: {OUT_LABEL_REPORT}")
    print("\n===== Profile preview =====")
    if not prof.empty:
        print(prof[["dataset_name", "file", "rows", "columns", "domain"]].to_string(index=False))


if __name__ == "__main__":
    main()
