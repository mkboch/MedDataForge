#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests


DEFAULT_URL = "https://raw.githubusercontent.com/m-aryayi/Medical-Imaging-Datasets/main/README.md"


def clean_text(x: str) -> str:
    if x is None:
        return ""
    x = re.sub(r"<[^>]+>", " ", str(x))
    x = x.replace("&nbsp;", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def extract_markdown_links(text: str):
    links = []
    for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        label = clean_text(m.group(1))
        url = m.group(2).strip()
        if url and not url.startswith("#"):
            links.append({"label": label, "url": url})
    return links


def infer_url_type(url: str, label: str = "") -> str:
    u = (url or "").lower()
    l = (label or "").lower()
    if any(x in u or x in l for x in [
        "paper", "arxiv", "pubmed", "doi.org", "nature.com",
        "springer", "ieee", "sciencedirect", "pmc.ncbi"
    ]):
        return "paper"
    if any(x in u or x in l for x in [
        "leaderboard", "challenge", "grand-challenge",
        "kaggle.com/competitions"
    ]):
        return "leaderboard_or_challenge"
    if any(x in u or x in l for x in [
        "license", "licence", "creativecommons", "cc-by"
    ]):
        return "license"
    return "dataset_or_other"


def keyword_extract(text: str):
    m = re.search(r"(?:keyword|keywords|keyboard)\s*[:：]\s*(.*)", text, flags=re.I)
    if not m:
        return ""
    val = clean_text(m.group(1))
    val = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", val)
    return val.strip(" -*|")


def detect_modalities(text: str):
    t = text.lower()
    mapping = {
        "xray": ["x-ray", "xray", "radiograph", "cxr", "chest x ray", "chest x-ray"],
        "ct": ["ct", "computed tomography"],
        "mri": ["mri", "magnetic resonance", "t1", "t2", "flair"],
        "ultrasound": ["ultrasound", "sonography"],
        "dermoscopy": ["dermoscopy", "dermoscopic", "skin lesion"],
        "fundus": ["fundus", "retina", "retinal"],
        "pathology": ["histology", "histopathology", "pathology", "whole slide", "wsi", "microscopy"],
        "pet": ["pet", "positron emission"],
        "oct": ["oct", "optical coherence tomography"],
    }
    found = []
    for key, pats in mapping.items():
        if any(p in t for p in pats):
            found.append(key)
    return ";".join(sorted(set(found)))


def detect_tasks(text: str):
    t = text.lower()
    mapping = {
        "classification": ["classification", "classify", "diagnosis", "diagnostic"],
        "segmentation": ["segmentation", "segment", "mask"],
        "detection": ["detection", "detect", "localization", "localisation", "bounding box"],
        "registration": ["registration", "register"],
        "prediction": ["prediction", "prognosis", "survival", "outcome"],
        "reconstruction": ["reconstruction", "reconstruct"],
    }
    found = []
    for key, pats in mapping.items():
        if any(p in t for p in pats):
            found.append(key)
    return ";".join(sorted(set(found)))


def detect_domain(text: str):
    t = text.lower()
    mapping = {
        "chest_xray": [
            "chest x-ray", "chest xray", "cxr", "chest radiograph",
            "chexpert", "mimic-cxr", "nih chest", "padchest",
            "vindr", "vinbigdata", "chestx-ray14"
        ],
        "skin_lesion": [
            "skin lesion", "dermoscopy", "dermoscopic", "isic",
            "melanoma", "ham10000", "pad-ufes"
        ],
        "brain": ["brain", "brats", "glioma", "alzheimer", "stroke", "intracranial"],
        "retina": ["retina", "retinal", "fundus", "diabetic retinopathy"],
        "breast": ["breast", "mammography", "mammogram"],
        "lung": ["lung", "pulmonary"],
        "liver": ["liver", "hepatic"],
        "colon": ["colon", "colonoscopy"],
    }
    found = []
    for key, pats in mapping.items():
        if any(p in t for p in pats):
            found.append(key)
    return ";".join(sorted(set(found)))


def detect_access(text: str, urls):
    t = text.lower()
    all_urls = " ".join([u.get("url", "") for u in urls]).lower()
    if "kaggle.com" in all_urls:
        return "kaggle_required"
    if "physionet.org" in all_urls:
        return "physionet_possible_credential_required"
    if "stanford" in all_urls or "aimi" in all_urls:
        return "possible_terms_required"
    if any(x in t for x in [
        "request access", "registration", "license agreement",
        "data use agreement", "restricted"
    ]):
        return "restricted_or_registration"
    return "unknown_or_open"


def parse_readme(md: str):
    records = []
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    current_h4 = ""
    pending = None

    def flush_pending():
        nonlocal pending
        if pending:
            text_blob = " ".join([
                pending.get("dataset_name", ""),
                pending.get("description", ""),
                pending.get("keywords_raw", "")
            ])
            links = pending.get("links", [])
            pending["modality_inferred"] = detect_modalities(text_blob)
            pending["task_inferred"] = detect_tasks(text_blob)
            pending["domain_inferred"] = detect_domain(text_blob)
            pending["access_inferred"] = detect_access(text_blob, links)
            pending["num_links"] = len(links)
            pending["links_json"] = json.dumps(links, ensure_ascii=False)

            url_types = [infer_url_type(x.get("url", ""), x.get("label", "")) for x in links]
            pending["has_paper_link"] = int("paper" in url_types)
            pending["has_leaderboard_or_challenge"] = int("leaderboard_or_challenge" in url_types)
            pending["has_license_link"] = int("license" in url_types)

            primary = ""
            for x, typ in zip(links, url_types):
                if typ == "dataset_or_other":
                    primary = x.get("url", "")
                    break
            if not primary and links:
                primary = links[0].get("url", "")
            pending["primary_url"] = primary

            records.append(pending)
            pending = None

    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            continue

        h = re.match(r"^(#{1,6})\s+(.*)$", line)
        if h:
            level = len(h.group(1))
            title = clean_text(re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", h.group(2)))
            if level == 1:
                current_h1 = title
            elif level == 2:
                current_h2 = title
            elif level == 3:
                current_h3 = title
            elif level >= 4:
                current_h4 = title
            continue

        links = extract_markdown_links(line)

        starts_candidate = False
        if re.match(r"^[-*+]\s+", line) and links:
            starts_candidate = True
        elif links and len(clean_text(line)) > 20 and not re.search(r"badge|license|licence", line, flags=re.I):
            starts_candidate = True

        if starts_candidate:
            flush_pending()
            first = links[0]
            name = clean_text(first["label"])
            no_md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
            no_bullet = re.sub(r"^[-*+]\s+", "", no_md).strip()
            desc = clean_text(no_bullet)
            kw = keyword_extract(line)
            pending = {
                "dataset_name": name,
                "section_h1": current_h1,
                "section_h2": current_h2,
                "section_h3": current_h3,
                "section_h4": current_h4,
                "description": desc,
                "keywords_raw": kw,
                "source_line": line,
                "links": links,
            }
            continue

        if pending:
            line_links = extract_markdown_links(line)
            if line_links:
                pending["links"].extend(line_links)
            kw = keyword_extract(line)
            if kw:
                pending["keywords_raw"] = (pending.get("keywords_raw", "") + "; " + kw).strip("; ")
            if len(clean_text(line)) > 5:
                pending["description"] = clean_text(pending.get("description", "") + " " + line)

    flush_pending()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme_url", default=DEFAULT_URL)
    ap.add_argument("--out_dir", default="data/registry")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching README: {args.readme_url}")
    r = requests.get(args.readme_url, timeout=60)
    r.raise_for_status()
    md = r.text
    (out_dir / "source_README.md").write_text(md, encoding="utf-8")

    records = parse_readme(md)
    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit("No dataset records extracted. Parser needs adjustment.")

    df.insert(0, "dataset_id", [f"D{i:05d}" for i in range(1, len(df) + 1)])
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").map(lambda x: clean_text(str(x)))

    raw_csv = out_dir / "datasets_raw.csv"
    raw_json = out_dir / "datasets_raw.json"
    df.to_csv(raw_csv, index=False)
    df.to_json(raw_json, orient="records", indent=2, force_ascii=False)

    focus = df[
        df["domain_inferred"].str.contains("chest_xray|skin_lesion", na=False, regex=True)
        | df["dataset_name"].str.lower().str.contains("chex|mimic|nih|padchest|vin|isic|ham10000|pad-ufes|chestx", regex=True)
        | df["description"].str.lower().str.contains("chex|mimic-cxr|nih chest|padchest|vinbigdata|vindr|isic|ham10000|skin lesion|melanoma|chestx", regex=True)
    ].copy()

    focus_csv = out_dir / "datasets_focus_chest_skin.csv"
    focus.to_csv(focus_csv, index=False)

    summary = {
        "total_records_extracted": int(len(df)),
        "focus_chest_skin_records": int(len(focus)),
        "records_with_primary_url": int((df["primary_url"] != "").sum()),
        "records_with_paper_link": int(df["has_paper_link"].sum()),
        "records_with_leaderboard_or_challenge": int(df["has_leaderboard_or_challenge"].sum()),
        "records_with_license_link": int(df["has_license_link"].sum()),
        "top_domains": df["domain_inferred"].value_counts().head(20).to_dict(),
        "top_modalities": df["modality_inferred"].value_counts().head(20).to_dict(),
        "top_tasks": df["task_inferred"].value_counts().head(20).to_dict(),
    }

    summary_path = out_dir / "registry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== EXTRACTION COMPLETE =====")
    print(f"Raw CSV: {raw_csv}")
    print(f"Raw JSON: {raw_json}")
    print(f"Focus CSV: {focus_csv}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
