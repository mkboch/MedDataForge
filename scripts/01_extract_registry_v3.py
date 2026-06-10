#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests


DEFAULT_URL = "https://raw.githubusercontent.com/m-aryayi/Medical-Imaging-Datasets/main/README.md"


def clean_text(x):
    if x is None:
        return ""
    x = str(x)
    x = re.sub(r"<br\s*/?>", " ", x, flags=re.I)
    x = re.sub(r"<[^>]+>", " ", x)
    x = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r" \1 ", x)
    x = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", x)
    x = x.replace("&nbsp;", " ")
    x = x.replace("&amp;", "&")
    x = re.sub(r"\*+", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip(" -*|\t\r\n")


def extract_all_links(text):
    links = []

    # HTML links: <a href="URL"> LABEL </a>
    for m in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', text, flags=re.I):
        url = m.group(1).strip()
        label_html = m.group(2)
        label = clean_text(label_html)
        if url and not url.startswith("#"):
            links.append({"label": label, "url": url})

    # Markdown links: [label](url), excluding images
    for m in re.finditer(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", text):
        label = clean_text(m.group(1))
        url = m.group(2).strip()
        if url and not url.startswith("#") and not url.startswith("src/"):
            item = {"label": label, "url": url}
            if item not in links:
                links.append(item)

    return links


def infer_url_type(url, label=""):
    u = (url or "").lower()
    l = (label or "").lower()
    if any(x in u or x in l for x in [
        "paper", "arxiv", "pubmed", "doi.org", "nature.com", "springer",
        "ieee", "sciencedirect", "pmc.ncbi", "biorxiv", "medrxiv"
    ]):
        return "paper"
    if any(x in u or x in l for x in [
        "leaderboard", "challenge", "grand-challenge", "codalab",
        "kaggle.com/competitions"
    ]):
        return "leaderboard_or_challenge"
    if any(x in u or x in l for x in [
        "license", "licence", "creativecommons", "cc-by"
    ]):
        return "license"
    return "dataset_or_other"


def choose_primary_url(links):
    if not links:
        return ""
    typed = [(x, infer_url_type(x.get("url", ""), x.get("label", ""))) for x in links]
    for x, typ in typed:
        if typ == "dataset_or_other":
            return x.get("url", "")
    return links[0].get("url", "")


def keyword_extract(text):
    m = re.search(r"(?:keyword|keywords|keyboard)\s*[:：]\s*(.*)", text, flags=re.I)
    if not m:
        return ""
    val = m.group(1)
    # Stop if another HTML link line starts inside captured text.
    val = re.split(r"\s+-\s+<a\s+", val)[0]
    return clean_text(val)


def detect_modalities(text):
    t = text.lower()
    mapping = {
        "xray": ["x-ray", "xray", "radiograph", "cxr", "chest x ray", "chest x-ray", "chestx-ray"],
        "ct": ["ct", "computed tomography", "ct scan"],
        "mri": ["mri", "magnetic resonance", "t1", "t2", "flair", "mr image"],
        "ultrasound": ["ultrasound", "sonography"],
        "dermoscopy": ["dermoscopy", "dermoscopic", "skin lesion", "skin images"],
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


def detect_tasks(text):
    t = text.lower()
    mapping = {
        "classification": ["classification", "classify", "diagnosis", "diagnostic", "categorization", "categorisation"],
        "segmentation": ["segmentation", "segment", "mask", "segmented"],
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


def detect_domain(text, section_h2="", section_h3=""):
    t = " ".join([text, section_h2, section_h3]).lower()
    mapping = {
        "chest_xray": [
            "chest x-ray", "chest xray", "cxr", "chest radiograph",
            "chexpert", "mimic-cxr", "nih chest", "padchest",
            "vindr", "vinbigdata", "chestx-ray14", "chestxray14",
            "chestx-det", "chexphoto", "chexmask", "lungs"
        ],
        "skin_lesion": [
            "skin lesion", "dermoscopy", "dermoscopic", "isic",
            "melanoma", "ham10000", "pad-ufes", "skin cancer", "skin"
        ],
        "brain": ["brain", "brats", "glioma", "alzheimer", "stroke", "intracranial"],
        "retina": ["retina", "retinal", "fundus", "diabetic retinopathy"],
        "breast": ["breast", "mammography", "mammogram"],
        "lung": ["lung", "pulmonary", "lungs"],
        "liver": ["liver", "hepatic"],
        "colon": ["colon", "colonoscopy"],
    }
    found = []
    for key, pats in mapping.items():
        if any(p in t for p in pats):
            found.append(key)
    return ";".join(sorted(set(found)))


def detect_access(text, links):
    t = text.lower()
    all_urls = " ".join([x.get("url", "") for x in links]).lower()
    if "kaggle.com" in all_urls:
        return "kaggle_required"
    if "physionet.org" in all_urls:
        return "physionet_possible_credential_required"
    if "stanford" in all_urls or "aimi" in all_urls:
        return "possible_terms_required"
    if "isic" in all_urls:
        return "isic_public_or_login"
    if any(x in t for x in [
        "request access", "registration", "license agreement",
        "data use agreement", "restricted"
    ]):
        return "restricted_or_registration"
    return "unknown_or_open"


def parse_readme_blocks(md):
    records = []
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    current_h4 = ""

    current = None

    def flush():
        nonlocal current
        if not current:
            return

        raw_block = "\n".join(current["block_lines"])
        links = extract_all_links(raw_block)
        first_link = links[0] if links else {"label": current.get("dataset_name", ""), "url": ""}
        name = current.get("dataset_name") or clean_text(first_link.get("label", ""))

        # Remove image/icon pseudo names
        if not name or name.lower() in {"paper", "leaderboard", "licence", "license", "broken link"}:
            current = None
            return

        url_types = [infer_url_type(x.get("url", ""), x.get("label", "")) for x in links]
        desc = clean_text(raw_block)
        kw = keyword_extract(raw_block)
        text_blob = " ".join([name, desc, kw, current.get("section_h2", ""), current.get("section_h3", "")])

        records.append({
            "dataset_name": name,
            "section_h1": current.get("section_h1", ""),
            "section_h2": current.get("section_h2", ""),
            "section_h3": current.get("section_h3", ""),
            "section_h4": current.get("section_h4", ""),
            "description": desc,
            "keywords_raw": kw,
            "source_type": "html_bullet_block",
            "source_start_line": current.get("start_line", ""),
            "primary_url": choose_primary_url(links),
            "links_json": json.dumps(links, ensure_ascii=False),
            "num_links": len(links),
            "has_paper_link": int("paper" in url_types),
            "has_leaderboard_or_challenge": int("leaderboard_or_challenge" in url_types),
            "has_license_link": int("license" in url_types),
            "domain_inferred": detect_domain(text_blob, current.get("section_h2", ""), current.get("section_h3", "")),
            "modality_inferred": detect_modalities(text_blob),
            "task_inferred": detect_tasks(text_blob),
            "access_inferred": detect_access(text_blob, links),
            "source_block": raw_block,
        })

        current = None

    for i, raw in enumerate(md.splitlines(), start=1):
        line = raw.rstrip("\n")
        stripped = line.strip()

        h = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if h:
            flush()
            level = len(h.group(1))
            title = clean_text(h.group(2))
            if level == 1:
                current_h1 = title
            elif level == 2:
                current_h2 = title
            elif level == 3:
                current_h3 = title
            elif level >= 4:
                current_h4 = title
            continue

        # Dataset entry starts with '- <a href="..."> **Name**</a>'
        start = re.match(r'^\-\s*<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', stripped, flags=re.I)
        if start:
            flush()
            label = clean_text(start.group(2))
            current = {
                "dataset_name": label,
                "section_h1": current_h1,
                "section_h2": current_h2,
                "section_h3": current_h3,
                "section_h4": current_h4,
                "start_line": i,
                "block_lines": [stripped],
            }
            continue

        if current:
            # Append until the next dataset entry/header.
            current["block_lines"].append(stripped)

    flush()
    return records


def deduplicate(df):
    keys = []
    keep = []
    for _, row in df.iterrows():
        key = (
            clean_text(row.get("dataset_name", "")).lower(),
            clean_text(row.get("primary_url", "")).lower(),
        )
        if key in keys:
            keep.append(False)
        else:
            keys.append(key)
            keep.append(True)
    return df[keep].copy()


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

    records = parse_readme_blocks(md)
    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit("No dataset records extracted. Parser needs adjustment.")

    df = deduplicate(df)
    df.insert(0, "dataset_id", [f"D{i:05d}" for i in range(1, len(df) + 1)])

    raw_csv = out_dir / "datasets_raw_v3.csv"
    raw_json = out_dir / "datasets_raw_v3.json"
    df.to_csv(raw_csv, index=False)
    df.to_json(raw_json, orient="records", indent=2, force_ascii=False)

    focus_terms = (
        "chex|mimic|nih|padchest|vin|isic|ham10000|pad-ufes|"
        "chestx|cxr|melanoma|skin lesion|dermoscopy|chexphoto|chexmask"
    )
    focus = df[
        df["domain_inferred"].str.contains("chest_xray|skin_lesion", na=False, regex=True)
        | df["dataset_name"].str.lower().str.contains(focus_terms, regex=True)
        | df["description"].str.lower().str.contains(focus_terms, regex=True)
    ].copy()

    focus_csv = out_dir / "datasets_focus_chest_skin_v3.csv"
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
        "sample_focus_names": focus["dataset_name"].head(40).tolist(),
    }

    summary_path = out_dir / "registry_summary_v3.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== EXTRACTION V3 COMPLETE =====")
    print(f"Raw CSV: {raw_csv}")
    print(f"Raw JSON: {raw_json}")
    print(f"Focus CSV: {focus_csv}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
