#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path

import pandas as pd


IN_CSV = Path("data/registry/locked_domain_candidates_audited.csv")
OUT_DIR = Path("data/manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def disk_info(path):
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    try:
        total, used, free = shutil.disk_usage(str(p))
        gb = 1024 ** 3
        return {
            "path": str(path),
            "exists": True,
            "total_gb": round(total / gb, 2),
            "used_gb": round(used / gb, 2),
            "free_gb": round(free / gb, 2),
        }
    except Exception as e:
        return {"path": str(path), "exists": True, "error": str(e)}


def exists_nonempty(path):
    p = Path(path).expanduser()
    return p.exists() and p.stat().st_size > 0


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Missing audited candidates: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    # Wave 1 = realistic first acquisition candidates, but split by access type.
    wave1 = df[df["first_wave_priority"].eq("wave1_scriptable_or_easy")].copy()
    wave2 = df[df["first_wave_priority"].str.contains("wave2", na=False)].copy()

    # More conservative Wave 1A: no Kaggle, no manual broken URL.
    wave1a = wave1[
        wave1["download_feasibility"].isin([
            "likely_scriptable_or_api_download",
            "likely_scriptable_from_isic_challenge_archive",
        ])
    ].copy()

    # Kaggle candidates need ~/.kaggle/kaggle.json.
    wave1_kaggle = wave1[
        wave1["download_feasibility"].eq("scriptable_if_kaggle_api_configured")
    ].copy()

    # Main experimental candidate manifests.
    chest = df[df["locked_domain"].eq("chest_xray")].copy()
    skin = df[df["locked_domain"].eq("skin_lesion")].copy()

    wave1.to_csv(OUT_DIR / "wave1_all_candidates.csv", index=False)
    wave1a.to_csv(OUT_DIR / "wave1a_no_kaggle_scriptable_candidates.csv", index=False)
    wave1_kaggle.to_csv(OUT_DIR / "wave1b_kaggle_candidates.csv", index=False)
    wave2.to_csv(OUT_DIR / "wave2_core_gated_or_manual_candidates.csv", index=False)
    chest.to_csv(OUT_DIR / "all_chest_xray_candidates_audited.csv", index=False)
    skin.to_csv(OUT_DIR / "all_skin_lesion_candidates_audited.csv", index=False)

    env = {
        "cwd": os.getcwd(),
        "python": os.popen("which python").read().strip(),
        "kaggle_json_exists": exists_nonempty("~/.kaggle/kaggle.json"),
        "netrc_exists_for_physionet_possible": exists_nonempty("~/.netrc"),
        "disk_candidates": [
            disk_info("/home/manikm"),
            disk_info("/data"),
            disk_info("/space"),
            disk_info("/scratch"),
            disk_info("/tmp"),
        ],
        "manifest_counts": {
            "wave1_all": int(len(wave1)),
            "wave1a_no_kaggle_scriptable": int(len(wave1a)),
            "wave1b_kaggle": int(len(wave1_kaggle)),
            "wave2_core_gated_or_manual": int(len(wave2)),
            "chest_total": int(len(chest)),
            "skin_total": int(len(skin)),
        },
        "wave1a_names": wave1a["dataset_name"].tolist(),
        "wave1b_kaggle_names": wave1_kaggle["dataset_name"].tolist(),
        "wave2_names": wave2["dataset_name"].tolist(),
    }

    (OUT_DIR / "environment_and_manifest_summary.json").write_text(
        json.dumps(env, indent=2), encoding="utf-8"
    )

    print("===== MANIFEST PREPARATION COMPLETE =====")
    print(json.dumps(env, indent=2))

    show_cols = [
        "dataset_name", "locked_domain", "url_status", "source_type_inferred",
        "download_feasibility", "first_wave_priority", "primary_url"
    ]

    print("\n===== WAVE 1A: SCRIPTABLE WITHOUT KAGGLE/PHYSIONET =====")
    if len(wave1a):
        print(wave1a[show_cols].to_string(index=False))
    else:
        print("No Wave 1A candidates found.")

    print("\n===== WAVE 1B: KAGGLE CANDIDATES =====")
    if len(wave1_kaggle):
        print(wave1_kaggle[show_cols].to_string(index=False))
    else:
        print("No Kaggle candidates found.")

    print("\n===== WAVE 2: CORE BUT GATED/MANUAL =====")
    if len(wave2):
        print(wave2[show_cols].to_string(index=False))
    else:
        print("No Wave 2 candidates found.")


if __name__ == "__main__":
    main()
