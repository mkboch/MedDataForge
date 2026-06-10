#!/usr/bin/env python3
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


MANIFEST = Path("data/manifests/wave1a_no_kaggle_scriptable_candidates.csv")
OUT_DIR = Path("data/acquisition")
OUT_JSON = OUT_DIR / "probe_wave1a_sources.json"
OUT_CSV = OUT_DIR / "probe_wave1a_candidate_files.csv"


def safe_get(url, timeout=30, headers=None):
    h = {"User-Agent": "MedDataForge-Probe/0.1"}
    if headers:
        h.update(headers)
    try:
        r = requests.get(url, timeout=timeout, headers=h, allow_redirects=True)
        text_like = "text" in r.headers.get("content-type", "") or "json" in r.headers.get("content-type", "")
        js = None
        if "json" in r.headers.get("content-type", ""):
            try:
                js = r.json()
            except Exception:
                js = None
        return {
            "ok": True,
            "status_code": r.status_code,
            "url": url,
            "final_url": r.url,
            "content_type": r.headers.get("content-type", ""),
            "text": r.text if text_like else "",
            "json": js,
            "error": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "status_code": "",
            "url": url,
            "final_url": "",
            "content_type": "",
            "text": "",
            "json": None,
            "error": f"{type(e).__name__}: {e}",
        }


def add_file(files, dataset_name, source, label, url, kind="", size="", note=""):
    files.append({
        "dataset_name": dataset_name,
        "source": source,
        "label": str(label or "").strip(),
        "url": str(url or "").strip(),
        "kind": str(kind or "").strip(),
        "size": str(size or "").strip(),
        "note": str(note or "").strip(),
    })


def probe_huggingface(dataset_name, url, files):
    m = re.search(r"huggingface\.co/datasets/([^/]+/[^/?#]+)", url)
    if not m:
        return {"status": "could_not_parse_hf_repo"}

    repo_id = m.group(1)
    api = f"https://huggingface.co/api/datasets/{repo_id}"
    res = safe_get(api)

    out = {
        "repo_id": repo_id,
        "api_url": api,
        "status_code": res["status_code"],
        "ok": res["ok"],
        "error": res["error"],
    }

    if res["json"]:
        siblings = res["json"].get("siblings", [])
        out["num_siblings"] = len(siblings)
        for s in siblings[:500]:
            rfilename = s.get("rfilename", "")
            if rfilename:
                file_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rfilename}"
                kind = Path(rfilename).suffix.lower().strip(".")
                add_file(files, dataset_name, "huggingface", rfilename, file_url, kind=kind)
    return out


def probe_github(dataset_name, url, files):
    m = re.search(r"github\.com/([^/]+)/([^/?#]+)", url)
    if not m:
        return {"status": "could_not_parse_github_repo"}
    owner, repo = m.group(1), m.group(2).replace(".git", "")
    api = f"https://api.github.com/repos/{owner}/{repo}/contents"
    res = safe_get(api)

    out = {
        "repo": f"{owner}/{repo}",
        "api_url": api,
        "status_code": res["status_code"],
        "ok": res["ok"],
        "error": res["error"],
    }

    if isinstance(res["json"], list):
        out["num_root_items"] = len(res["json"])
        for item in res["json"]:
            name = item.get("name", "")
            download_url = item.get("download_url", "")
            html_url = item.get("html_url", "")
            kind = Path(name).suffix.lower().strip(".")
            if download_url:
                add_file(files, dataset_name, "github", name, download_url, kind=kind)
            elif html_url:
                add_file(files, dataset_name, "github", name, html_url, kind="directory_or_page")
    return out


def probe_dataverse_ham10000(dataset_name, url, files):
    m = re.search(r"persistentId=(doi:[^&#]+)", url)
    pid = m.group(1) if m else "doi:10.7910/DVN/DBW86T"
    api = f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={pid}"
    res = safe_get(api)

    out = {
        "persistent_id": pid,
        "api_url": api,
        "status_code": res["status_code"],
        "ok": res["ok"],
        "error": res["error"],
    }

    js = res["json"]
    if js and js.get("status") == "OK":
        files_list = js.get("data", {}).get("latestVersion", {}).get("files", [])
        out["num_files"] = len(files_list)
        for f in files_list:
            datafile = f.get("dataFile", {})
            fid = datafile.get("id", "")
            filename = datafile.get("filename", "")
            size = datafile.get("filesize", "")
            if fid:
                download_url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}"
                add_file(files, dataset_name, "harvard_dataverse", filename, download_url, kind=Path(filename).suffix.lower().strip("."), size=size)
    return out


def probe_mendeley(dataset_name, url, files):
    m = re.search(r"datasets/([^/]+)/(\d+)", url)
    if not m:
        return {"status": "could_not_parse_mendeley_dataset"}
    dsid, version = m.group(1), m.group(2)
    api = f"https://data.mendeley.com/api/datasets/{dsid}/versions/{version}"
    res = safe_get(api)

    out = {
        "dataset_id": dsid,
        "version": version,
        "api_url": api,
        "status_code": res["status_code"],
        "ok": res["ok"],
        "error": res["error"],
    }

    js = res["json"]
    if js:
        files_list = js.get("files", [])
        out["num_files"] = len(files_list)
        for f in files_list:
            filename = f.get("filename", "") or f.get("name", "")
            file_id = f.get("id", "")
            size = f.get("size", "")
            download_url = f.get("download_url", "") or f.get("downloadUrl", "")
            if not download_url and file_id:
                download_url = f"https://data.mendeley.com/api/datasets/{dsid}/versions/{version}/files/{file_id}"
            if download_url:
                add_file(files, dataset_name, "mendeley_data", filename, download_url, kind=Path(filename).suffix.lower().strip("."), size=size)
    return out


def probe_isic(dataset_name, url, files):
    res = safe_get(url)
    out = {
        "page_url": url,
        "status_code": res["status_code"],
        "ok": res["ok"],
        "error": res["error"],
    }

    html = res["text"] or ""
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a")
    out["num_links_on_page"] = len(anchors)

    for a in anchors:
        href = a.get("href", "")
        label = a.get_text(" ", strip=True)
        if not href:
            continue
        full = urljoin(url, href)
        low = full.lower()
        if any(x in low for x in [".zip", ".csv", ".tar", ".gz", "download"]):
            add_file(files, dataset_name, "isic_page", label or full.split("/")[-1], full, kind=Path(full).suffix.lower().strip("."), note="discovered_on_isic_challenge_page")

    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    probes = {}
    files = []

    for _, row in df.iterrows():
        name = row["dataset_name"]
        url = row["primary_url"]
        source = row["source_type_inferred"]
        print(f"===== Probing {name} [{source}] =====", flush=True)

        if source == "huggingface":
            probes[name] = probe_huggingface(name, url, files)
        elif source == "github":
            probes[name] = probe_github(name, url, files)
        elif source == "harvard_dataverse":
            probes[name] = probe_dataverse_ham10000(name, url, files)
        elif source == "mendeley_data":
            probes[name] = probe_mendeley(name, url, files)
        elif source == "isic":
            probes[name] = probe_isic(name, url, files)
        else:
            probes[name] = {"status": "no_probe_implemented", "source": source, "url": url}

    OUT_JSON.write_text(json.dumps(probes, indent=2), encoding="utf-8")
    fdf = pd.DataFrame(files)
    if not fdf.empty:
        fdf.to_csv(OUT_CSV, index=False)
    else:
        pd.DataFrame(columns=["dataset_name","source","label","url","kind","size","note"]).to_csv(OUT_CSV, index=False)

    print("===== WAVE 1A SOURCE PROBE COMPLETE =====")
    print(f"Probe JSON: {OUT_JSON}")
    print(f"Candidate files CSV: {OUT_CSV}")
    print(f"Candidate file rows: {len(files)}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ['text','json']} for k, v in probes.items()}, indent=2))

    if files:
        print("\n===== Candidate file preview =====")
        print(pd.DataFrame(files).head(100).to_string(index=False))


if __name__ == "__main__":
    main()
