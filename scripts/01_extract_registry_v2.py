#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests


DEFAULT_URL = "https://raw.githubusercontent.com/m-aryayi/Medical-Imaging-Datasets/main/README.md"


ICON_WORDS = {
    "paper", "leaderboard", "licence", "license", "broken link", "broken",
    "dataset", "code", "github"
}


def clean_text(x: str) -> str:
    if x is None:
        return ""
    x = str(x)
    x = re.sub(r"<[^>]+>", " ", x)
    x = x.replace("&nbsp;", " ")
    x = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r" \1 ", x)
    x = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip(" -*|\t\r\n")


def split_md_table_row(line: str):
    line = line.strip()
    if not line.startswith("|") or not line.endswith("|"):
        return []
    cells = [c.strip() for c in line.strip("|").split("|")]
    return cells


def is_separator_row(cells):
    if not cells:
        return False
    return all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells)


def extract_markdown_links(text: str):
    links = []

    # linked image: [![paper](src/paper.png)](real_url)
    for m in re.finditer(r"\[!\[([^\]]*)\]\([^)]+\)\]\(([^)]+)\)", text):
        label = clean_text(m.group(1))
        url = m.group(2).strip()
        if url and not url.startswith("#"):
            links.append({"label": label, "url": url})

    # normal markdown links, excluding image links
    for m in re.finditer(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", text):
        label = clean_text(m.group(1))
        url = m.group(2).strip()
        if not url or url.startswith("#"):
            continue
        # skip inner image asset links
        if url.startswith("src/") or url.endswith(".png") or url.endswith(".svg"):
            continue
        # skip duplicates
        item = {"label": label, "url": url}
        if item not in links:
            links.append(item)

    return links


def infer_url_type(url: str, label: str = "") -> str:
    u = (url or "").lower()
    l = (label or "").lower()
    if any(x in u or x in l for x in [
        "paper", "arxiv", "pubmed", "doi.org", "nature.com",
        "springer", "ieee", "sciencedirect", "pmc.ncbi", "biorxiv", "medrxiv"
    ]):
        return "paper"
    if any(x in u or x in l for x in [
        "leaderboard", "challenge", "grand-challenge",
        "kaggle.com/competitions", "codalab"
    ]):
        return "leaderboard_or_challenge"
    if any(x in u or x in l for x in [
        "license", "licence", "creativecommons", "cc-by", "mit-license"
    ]):
        return "license"
    return "dataset_or_other"


def keyword_extract(text: str):
    m = re.search(r"(?:keyword|keywords|keyboard)\s*[:：]\s*(.*)", text, flags=re.I)
    if not m:
        return ""
    return clean_text(m.group(1))


def detect_modalities(text: str):
    t = text.lower()
    mapping = {
        "xray": ["x-ray", "xray", "radiograph", "cxr", "chest x ray", "chest x-ray", "chestx-ray"],
        "ct": ["ct", "computed tomography"],
        "mri": ["mri", "magnetic resonance", "t1", "t2", "flair"],
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


def detect_tasks(text: str):
    t = text.lower()
    mapping = {
        "classification": ["classification", "classify", "diagnosis", "diagnostic", "categorization", "categorisation"],
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
            "vindr", "vinbigdata", "chestx-ray14", "chestxray14",
            "chexphoto", "openi"
        ],
        "skin_lesion": [
            "skin lesion", "dermoscopy", "dermoscopic", "isic",
            "melanoma", "ham10000", "pad-ufes", "skin cancer"
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
    if "isic" in all_urls:
        return "isic_public_or_login"
    if any(x in t for x in [
        "request access", "registration", "license agreement",
        "data use agreement", "restricted"
    ]):
        return "restricted_or_registration"
    return "unknown_or_open"


def choose_primary_url(links):
    typed = [(x, infer_url_type(x.get("url", ""), x.get("label", ""))) for x in links]
    for x, typ in typed:
        if typ == "dataset_or_other":
            return x.get("url", "")
    for x, typ in typed:
        if typ in {"leaderboard_or_challenge", "license", "paper"}:
            continue
    return links[0].get("url", "") if links else ""


def has_icon_only_name(name):
    n = clean_text(name).lower()
    return n in ICON_WORDS or n == "" or n.endswith(".png") or n.endswith(".svg")


def parse_table_rows(md: str):
    records = []
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    current_h4 = ""
    headers = []

    for raw in md.splitlines():
        line = raw.rstrip()

        h = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if h:
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

        cells = split_md_table_row(line)
        if not cells:
            continue
        if is_separator_row(cells):
            continue

        # header row
        lower_cells = [clean_text(c).lower() for c in cells]
        if any(x in " ".join(lower_cells) for x in ["dataset", "database", "link", "description", "task", "modality"]):
            # Keep header only if it looks like a header, not an actual dataset row.
            if not any(extract_markdown_links(c) for c in cells[:2]):
                headers = lower_cells
                continue

        row_text = " ".join(cells)
        row_links = extract_markdown_links(row_text)
        if not row_links and len(clean_text(row_text)) < 20:
            continue

        # dataset name from first plausible cell
        dataset_name = ""
        dataset_cell = ""
        for c in cells[:3]:
            visible = clean_text(c)
            if has_icon_only_name(visible):
                continue
            # Ignore cells that are just paper/license/leaderboard icons.
            if visible.lower() in ICON_WORDS:
                continue
            dataset_name = visible
            dataset_cell = c
            break

        if not dataset_name:
            # fallback to first non-icon link label
            for link in row_links:
                label = clean_text(link.get("label", ""))
                if not has_icon_only_name(label):
                    dataset_name = label
                    dataset_cell = label
                    break

        if not dataset_name or has_icon_only_name(dataset_name):
            continue

        # remove obvious non-dataset pseudo rows
        if dataset_name.lower() in {"paper", "leaderboard", "licence", "license", "broken link"}:
            continue

        links = row_links
        keywords = keyword_extract(row_text)
        desc = clean_text(row_text)

        text_blob = " ".join([dataset_name, desc, keywords, current_h2, current_h3, current_h4])
        url_types = [infer_url_type(x.get("url", ""), x.get("label", "")) for x in links]
        primary_url = choose_primary_url(links)

        rec = {
            "dataset_name": dataset_name,
            "section_h1": current_h1,
            "section_h2": current_h2,
            "section_h3": current_h3,
            "section_h4": current_h4,
            "description": desc,
            "keywords_raw": keywords,
            "source_type": "markdown_table",
            "source_line": line.strip(),
            "primary_url": primary_url,
            "links_json": json.dumps(links, ensure_ascii=False),
            "num_links": len(links),
            "has_paper_link": int("paper" in url_types),
            "has_leaderboard_or_challenge": int("leaderboard_or_challenge" in url_types),
            "has_license_link": int("license" in url_types),
            "domain_inferred": detect_domain(text_blob),
            "modality_inferred": detect_modalities(text_blob),
            "task_inferred": detect_tasks(text_blob),
            "access_inferred": detect_access(text_blob, links),
        }
        records.append(rec)

    return records


def parse_bullet_rows(md: str):
    records = []
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    current_h4 = ""

    for raw in md.splitlines():
        line = raw.strip()
        h = re.match(r"^(#{1,6})\s+(.*)$", line)
        if h:
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

        if not re.match(r"^[-*+]\s+", line):
            continue

        links = extract_markdown_links(line)
        if not links:
            continue

        name = ""
        for link in links:
            label = clean_text(link.get("label", ""))
            if not has_icon_only_name(label):
                name = label
                break

        if not name:
            visible = clean_text(line)
            if not has_icon_only_name(visible):
                name = visible[:120]

        if not name or has_icon_only_name(name):
            continue

        kw = keyword_extract(line)
        desc = clean_text(line)
        text_blob = " ".join([name, desc, kw, current_h2, current_h3, current_h4])
        url_types = [infer_url_type(x.get("url", ""), x.get("label", "")) for x in links]

        records.append({
            "dataset_name": name,
            "section_h1": current_h1,
            "section_h2": current_h2,
            "section_h3": current_h3,
            "section_h4": current_h4,
            "description": desc,
            "keywords_raw": kw,
            "source_type": "markdown_bullet",
            "source_line": line,
            "primary_url": choose_primary_url(links),
            "links_json": json.dumps(links, ensure_ascii=False),
            "num_links": len(links),
            "has_paper_link": int("paper" in url_types),
            "has_leaderboard_or_challenge": int("leaderboard_or_challenge" in url_types),
            "has_license_link": int("license" in url_types),
            "domain_inferred": detect_domain(text_blob),
            "modality_inferred": detect_modalities(text_blob),
            "task_inferred": detect_tasks(text_blob),
            "access_inferred": detect_access(text_blob, links),
        })

    return records


def deduplicate_records(records):
    seen = set()
    out = []
    for r in records:
        key = (
            clean_text(r.get("dataset_name", "")).lower(),
            clean_text(r.get("primary_url", "")).lower(),
            clean_text(r.get("section_h2", "")).lower(),
            clean_text(r.get("section_h3", "")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


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

    table_records = parse_table_rows(md)
    bullet_records = parse_bullet_rows(md)
    records = deduplicate_records(table_records + bullet_records)

    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit("No dataset records extracted. Parser needs adjustment.")

    # Remove icon/image pseudo-records.
    bad_names = {"paper", "leaderboard", "licence", "license", "broken link", "broken"}
    df = df[~df["dataset_name"].str.lower().isin(bad_names)].copy()
    df = df[~df["primary_url"].fillna("").str.startswith("src/")].copy()

    df.insert(0, "dataset_id", [f"D{i:05d}" for i in range(1, len(df) + 1)])
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").map(lambda x: clean_text(str(x)))

    raw_csv = out_dir / "datasets_raw_v2.csv"
    raw_json = out_dir / "datasets_raw_v2.json"
    df.to_csv(raw_csv, index=False)
    df.to_json(raw_json, orient="records", indent=2, force_ascii=False)

    focus_terms = "chex|mimic|nih|padchest|vin|isic|ham10000|pad-ufes|chestx|cxr|melanoma|skin lesion|dermoscopy"
    focus = df[
        df["domain_inferred"].str.contains("chest_xray|skin_lesion", na=False, regex=True)
        | df["dataset_name"].str.lower().str.contains(focus_terms, regex=True)
        | df["description"].str.lower().str.contains(focus_terms, regex=True)
    ].copy()

    focus_csv = out_dir / "datasets_focus_chest_skin_v2.csv"
    focus.to_csv(focus_csv, index=False)

    summary = {
        "total_records_extracted": int(len(df)),
        "table_records_before_dedup": int(len(table_records)),
        "bullet_records_before_dedup": int(len(bullet_records)),
        "focus_chest_skin_records": int(len(focus)),
        "records_with_primary_url": int((df["primary_url"] != "").sum()),
        "records_with_paper_link": int(df["has_paper_link"].sum()),
        "records_with_leaderboard_or_challenge": int(df["has_leaderboard_or_challenge"].sum()),
        "records_with_license_link": int(df["has_license_link"].sum()),
        "top_domains": df["domain_inferred"].value_counts().head(20).to_dict(),
        "top_modalities": df["modality_inferred"].value_counts().head(20).to_dict(),
        "top_tasks": df["task_inferred"].value_counts().head(20).to_dict(),
        "sample_focus_names": focus["dataset_name"].head(30).tolist(),
    }

    summary_path = out_dir / "registry_summary_v2.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== EXTRACTION V2 COMPLETE =====")
    print(f"Raw CSV: {raw_csv}")
    print(f"Raw JSON: {raw_json}")
    print(f"Focus CSV: {focus_csv}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
