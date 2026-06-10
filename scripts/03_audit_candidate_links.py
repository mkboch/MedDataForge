#!/usr/bin/env python3
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


IN_CSV = Path("data/registry/locked_domain_candidates.csv")
OUT_CSV = Path("data/registry/locked_domain_candidates_audited.csv")
OUT_SUMMARY = Path("data/registry/link_audit_summary.json")


def safe_request(url, timeout=12):
    result = {
        "url_status": "not_checked",
        "http_status": "",
        "final_url": "",
        "content_type": "",
        "error": "",
    }

    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        result["url_status"] = "invalid_or_missing"
        return result

    headers = {
        "User-Agent": "MedDataForge-LinkAudit/0.1 (+research metadata audit)"
    }

    try:
        # Some sites reject HEAD, so try GET with stream.
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
        result["http_status"] = str(r.status_code)
        result["final_url"] = str(r.url)
        result["content_type"] = r.headers.get("content-type", "")
        if 200 <= r.status_code < 400:
            result["url_status"] = "alive"
        elif r.status_code in {401, 403}:
            result["url_status"] = "alive_but_forbidden_or_login"
        elif r.status_code == 404:
            result["url_status"] = "not_found"
        else:
            result["url_status"] = f"http_{r.status_code}"
        r.close()
    except Exception as e:
        result["url_status"] = "request_error"
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def infer_source_type(url):
    u = str(url).lower()
    host = urlparse(u).netloc

    if "kaggle.com" in u:
        return "kaggle"
    if "physionet.org" in u:
        return "physionet"
    if "stanford" in u or "aimi" in u:
        return "stanford_aimi"
    if "challenge.isic-archive.com" in u or "isic-archive.com" in u:
        return "isic"
    if "dataverse.harvard.edu" in u:
        return "harvard_dataverse"
    if "mendeley.com" in u:
        return "mendeley_data"
    if "huggingface.co" in u:
        return "huggingface"
    if "github.com" in u:
        return "github"
    if "zenodo.org" in u:
        return "zenodo"
    if "grand-challenge.org" in u:
        return "grand_challenge"
    if "bimcv.cipf.es" in u:
        return "bimcv"
    return host or "unknown"


def infer_download_feasibility(row):
    name = str(row.get("dataset_name", ""))
    url = str(row.get("primary_url", ""))
    source_type = row.get("source_type_inferred", "")
    access_bucket = str(row.get("access_bucket", ""))

    if source_type == "kaggle":
        return "scriptable_if_kaggle_api_configured"
    if source_type == "physionet":
        return "scriptable_if_physionet_credentials_and_dua_available"
    if source_type == "stanford_aimi":
        return "manual_terms_or_registration_likely"
    if source_type == "isic":
        return "likely_scriptable_from_isic_challenge_archive"
    if source_type in {"harvard_dataverse", "mendeley_data", "huggingface", "github", "zenodo"}:
        return "likely_scriptable_or_api_download"
    if source_type in {"grand_challenge", "bimcv"}:
        return "manual_review_or_registration_possible"

    if "manual" in access_bucket:
        return "manual_review_needed"
    return "unknown_needs_manual_review"


def infer_first_wave(row):
    name = str(row.get("dataset_name", ""))
    domain = str(row.get("locked_domain", ""))

    # Prioritize datasets that are realistic to access and useful for first proof-of-concept.
    chest_first = {
        "ChestX-ray8",
        "ChestX-ray14",
        "Chest X-Ray Images (Pneumonia)",
        "COVID-19 Radiography Database",
        "Indiana U. Chest X-rays",
    }
    chest_gated_but_core = {
        "CheXpert",
        "MIMIC-CXR-JPG",
        "PadChest",
        "VinDr-PCXR",
    }
    skin_first = {
        "ISIC",
        "HAM10000",
        "PAD-UFES-20",
        "Fitzpatrick 17k",
        "DDI",
    }

    if domain == "chest_xray" and name in chest_first:
        return "wave1_scriptable_or_easy"
    if domain == "chest_xray" and name in chest_gated_but_core:
        return "wave2_core_but_access_gated_or_manual"
    if domain == "skin_lesion" and name in skin_first:
        return "wave1_scriptable_or_easy"
    if domain == "skin_lesion":
        return "wave2_manual_review"
    return "not_prioritized"


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Missing input CSV: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    primary = df[df["experiment_role"].eq("primary_classification_candidate")].copy()

    audit_rows = []
    for _, row in primary.iterrows():
        url = row.get("primary_url", "")
        print(f"Auditing: {row.get('dataset_name')} -> {url}", flush=True)
        req = safe_request(url)
        source_type = infer_source_type(url)

        out = row.to_dict()
        out.update(req)
        out["source_type_inferred"] = source_type
        out["download_feasibility"] = infer_download_feasibility(out)
        out["first_wave_priority"] = infer_first_wave(out)
        audit_rows.append(out)

        time.sleep(0.5)

    audited = pd.DataFrame(audit_rows)
    audited.to_csv(OUT_CSV, index=False)

    summary = {
        "audited_primary_candidates": int(len(audited)),
        "by_locked_domain": {str(k): int(v) for k, v in audited["locked_domain"].value_counts().items()},
        "by_url_status": {str(k): int(v) for k, v in audited["url_status"].value_counts().items()},
        "by_source_type": {str(k): int(v) for k, v in audited["source_type_inferred"].value_counts().items()},
        "by_download_feasibility": {str(k): int(v) for k, v in audited["download_feasibility"].value_counts().items()},
        "by_first_wave_priority": {str(k): int(v) for k, v in audited["first_wave_priority"].value_counts().items()},
        "wave1_names": audited[audited["first_wave_priority"].eq("wave1_scriptable_or_easy")]["dataset_name"].tolist(),
        "wave2_core_names": audited[audited["first_wave_priority"].eq("wave2_core_but_access_gated_or_manual")]["dataset_name"].tolist(),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== LINK AUDIT COMPLETE =====")
    print(json.dumps(summary, indent=2))

    show_cols = [
        "dataset_name", "locked_domain", "url_status", "http_status",
        "source_type_inferred", "download_feasibility",
        "first_wave_priority", "primary_url"
    ]
    print("\n===== AUDITED PRIMARY CANDIDATES =====")
    print(audited[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
