#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PREDICTION_FILES = [
    {
        "protocol": "P3",
        "seed": 20260528,
        "file": "results/predictions/P3_seed20260528_predictions.csv",
    },
    {
        "protocol": "P3",
        "seed": 20260529,
        "file": "results/predictions/P3_seed20260529_predictions.csv",
    },
    {
        "protocol": "P3",
        "seed": 20260530,
        "file": "results/predictions/P3_seed20260530_predictions.csv",
    },
    {
        "protocol": "P4_shared7",
        "seed": 20260528,
        "file": "results/predictions/P4_shared7_seed20260528_predictions.csv",
    },
    {
        "protocol": "P4_shared7",
        "seed": 20260529,
        "file": "results/predictions/P4_shared7_seed20260529_predictions.csv",
    },
    {
        "protocol": "P4_shared7",
        "seed": 20260530,
        "file": "results/predictions/P4_shared7_seed20260530_predictions.csv",
    },
]

OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SEED = OUT_DIR / "bootstrap_p3_p4_seed_level_ci.csv"
OUT_AGG = OUT_DIR / "bootstrap_p3_p4_aggregate_ci.csv"
OUT_REPORT = OUT_DIR / "bootstrap_p3_p4_ci_report.md"

N_BOOT = 1000
BOOT_SEED = 20260529


def compute_metrics(df):
    y_true = df["true_label"].astype(str).to_numpy()
    y_pred = df["pred_label"].astype(str).to_numpy()

    labels = sorted(df["true_label"].astype(str).unique().tolist())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


def bootstrap_ci(df, rng, n_boot=N_BOOT):
    n = len(df)
    metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
    }

    idx = np.arange(n)

    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=n, replace=True)
        m = compute_metrics(df.iloc[sample_idx])
        for k, v in m.items():
            metrics[k].append(v)

    out = {}
    point = compute_metrics(df)
    for k, vals in metrics.items():
        arr = np.array(vals)
        out[f"{k}_point"] = float(point[k])
        out[f"{k}_ci_low"] = float(np.percentile(arr, 2.5))
        out[f"{k}_ci_high"] = float(np.percentile(arr, 97.5))
        out[f"{k}_boot_mean"] = float(arr.mean())
        out[f"{k}_boot_std"] = float(arr.std(ddof=1))
    return out


def main():
    rng = np.random.default_rng(BOOT_SEED)
    rows = []

    for item in PREDICTION_FILES:
        path = Path(item["file"])
        if not path.exists():
            raise SystemExit(f"Missing prediction file: {path}")

        df = pd.read_csv(path, low_memory=False)
        ci = bootstrap_ci(df, rng)

        rows.append({
            "protocol": item["protocol"],
            "seed": item["seed"],
            "prediction_file": str(path),
            "n": int(len(df)),
            **ci,
        })

    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(OUT_SEED, index=False)

    agg_rows = []
    for protocol, g in seed_df.groupby("protocol"):
        row = {
            "protocol": protocol,
            "n_runs": int(len(g)),
            "n_mean": float(g["n"].mean()),
        }
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            vals = g[f"{metric}_point"].to_numpy()
            row[f"{metric}_mean_across_seeds"] = float(vals.mean())
            row[f"{metric}_std_across_seeds"] = float(vals.std(ddof=1))
            row[f"{metric}_mean_seed_ci_low"] = float(g[f"{metric}_ci_low"].mean())
            row[f"{metric}_mean_seed_ci_high"] = float(g[f"{metric}_ci_high"].mean())
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_AGG, index=False)

    p3 = agg_df[agg_df["protocol"].eq("P3")].iloc[0]
    p4 = agg_df[agg_df["protocol"].eq("P4_shared7")].iloc[0]
    diff = p3["macro_f1_mean_across_seeds"] - p4["macro_f1_mean_across_seeds"]

    lines = []
    lines.append("# Bootstrap Confidence Intervals for P3 vs P4 Shared-7\n")
    lines.append("## Main result\n")
    lines.append(
        f"Across three seeds, P3 achieved macro F1 "
        f"{p3['macro_f1_mean_across_seeds']:.4f} ± {p3['macro_f1_std_across_seeds']:.4f}, "
        f"while P4 shared-7 achieved "
        f"{p4['macro_f1_mean_across_seeds']:.4f} ± {p4['macro_f1_std_across_seeds']:.4f}. "
        f"The mean macro-F1 difference was {diff:.4f}."
    )

    lines.append("\n## Aggregate bootstrap table\n")
    show_agg = agg_df.copy()
    for c in show_agg.columns:
        if c not in {"protocol", "n_runs"}:
            show_agg[c] = show_agg[c].map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    lines.append(show_agg.to_markdown(index=False))

    lines.append("\n\n## Seed-level bootstrap table\n")
    show_seed = seed_df[[
        "protocol", "seed", "n",
        "accuracy_point", "accuracy_ci_low", "accuracy_ci_high",
        "balanced_accuracy_point", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high",
        "macro_f1_point", "macro_f1_ci_low", "macro_f1_ci_high",
        "weighted_f1_point", "weighted_f1_ci_low", "weighted_f1_ci_high",
    ]].copy()
    for c in show_seed.columns:
        if c not in {"protocol", "seed", "n"}:
            show_seed[c] = show_seed[c].map(lambda x: f"{x:.4f}")
    lines.append(show_seed.to_markdown(index=False))

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("===== BOOTSTRAP CI COMPLETE =====")
    print(f"Seed CI: {OUT_SEED}")
    print(f"Aggregate CI: {OUT_AGG}")
    print(f"Report: {OUT_REPORT}")
    print("\n===== MAIN RESULT =====")
    print(f"P3 macro F1 mean ± seed SD: {p3['macro_f1_mean_across_seeds']:.4f} ± {p3['macro_f1_std_across_seeds']:.4f}")
    print(f"P4 shared7 macro F1 mean ± seed SD: {p4['macro_f1_mean_across_seeds']:.4f} ± {p4['macro_f1_std_across_seeds']:.4f}")
    print(f"Difference: {diff:.4f}")
    print("\n===== AGGREGATE =====")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
