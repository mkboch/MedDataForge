#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PRED_FILES = [
    {
        "protocol": "C2",
        "seed": 20260602,
        "metric_file": "results/chest_baselines/C2_resnet18_compiler_compatible_binary_pool_seed20260602/test_metrics.json",
    },
    {
        "protocol": "C2",
        "seed": 20260603,
        "metric_file": "results/chest_baselines/C2_resnet18_compiler_compatible_binary_pool_seed20260603/test_metrics.json",
    },
    {
        "protocol": "C2",
        "seed": 20260604,
        "metric_file": "results/chest_baselines/C2_resnet18_compiler_compatible_binary_pool_seed20260604/test_metrics.json",
    },
    {
        "protocol": "C3_on_C2test",
        "seed": 20260602,
        "metric_file": "results/chest_baselines/C3_resnet18_naive_binary_pool_all_abnormal_seed20260602/eval_on_C2_test_metrics.json",
    },
    {
        "protocol": "C3_on_C2test",
        "seed": 20260603,
        "metric_file": "results/chest_baselines/C3_resnet18_naive_binary_pool_all_abnormal_seed20260603/eval_on_C2_test_metrics.json",
    },
    {
        "protocol": "C3_on_C2test",
        "seed": 20260604,
        "metric_file": "results/chest_baselines/C3_resnet18_naive_binary_pool_all_abnormal_seed20260604/eval_on_C2_test_metrics.json",
    },
]

# For bootstrap from predictions, use confusion matrix reconstruction.
# This is valid for estimating metric CI from aggregate counts only approximately,
# because original per-sample order is not preserved. It reconstructs labels from
# the saved confusion matrix, then bootstraps the reconstructed sample set.
OUT_SEED = Path("results/tables/chest_c2_c3_bootstrap_seed_level_ci.csv")
OUT_AGG = Path("results/tables/chest_c2_c3_bootstrap_aggregate_ci.csv")
OUT_REPORT = Path("results/tables/chest_c2_c3_bootstrap_ci_report.md")

N_BOOT = 1000
BOOT_SEED = 20260602


def reconstruct_from_confusion(cm):
    y_true = []
    y_pred = []
    for i, row in enumerate(cm):
        for j, n in enumerate(row):
            y_true.extend([i] * int(n))
            y_pred.extend([j] * int(n))
    return np.array(y_true), np.array(y_pred)


def metrics_from_arrays(y_true, y_pred):
    labels = sorted(np.unique(y_true).tolist())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


def bootstrap_ci(y_true, y_pred, rng):
    n = len(y_true)
    idx = np.arange(n)

    point = metrics_from_arrays(y_true, y_pred)
    boot = {k: [] for k in point}

    for _ in range(N_BOOT):
        sample = rng.choice(idx, size=n, replace=True)
        m = metrics_from_arrays(y_true[sample], y_pred[sample])
        for k, v in m.items():
            boot[k].append(v)

    out = {}
    for k, vals in boot.items():
        vals = np.array(vals)
        out[f"{k}_point"] = float(point[k])
        out[f"{k}_ci_low"] = float(np.percentile(vals, 2.5))
        out[f"{k}_ci_high"] = float(np.percentile(vals, 97.5))
        out[f"{k}_boot_mean"] = float(vals.mean())
        out[f"{k}_boot_std"] = float(vals.std(ddof=1))
    return out


def main():
    OUT_SEED.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)

    rows = []
    for item in PRED_FILES:
        path = Path(item["metric_file"])
        if not path.exists():
            raise FileNotFoundError(path)

        metrics = json.loads(path.read_text())
        cm = metrics["confusion_matrix"]
        y_true, y_pred = reconstruct_from_confusion(cm)
        ci = bootstrap_ci(y_true, y_pred, rng)

        rows.append({
            "protocol": item["protocol"],
            "seed": item["seed"],
            "metric_file": str(path),
            "n": int(len(y_true)),
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

    c2 = agg_df[agg_df["protocol"].eq("C2")].iloc[0]
    c3 = agg_df[agg_df["protocol"].eq("C3_on_C2test")].iloc[0]
    diff = c2["macro_f1_mean_across_seeds"] - c3["macro_f1_mean_across_seeds"]

    lines = []
    lines.append("# Chest C2 vs C3 Bootstrap Confidence Intervals\n")
    lines.append("## Main result\n")
    lines.append(
        f"C2 achieved macro F1 {c2['macro_f1_mean_across_seeds']:.4f} ± {c2['macro_f1_std_across_seeds']:.4f} across seeds, "
        f"while C3 evaluated on the same C2 test set achieved {c3['macro_f1_mean_across_seeds']:.4f} ± {c3['macro_f1_std_across_seeds']:.4f}. "
        f"The mean macro-F1 difference was {diff:.4f}."
    )
    lines.append("\n## Note\n")
    lines.append(
        "Bootstrap samples were reconstructed from each run's saved confusion matrix. This gives a useful uncertainty estimate from aggregate prediction counts, "
        "but per-sample prediction CSVs would be preferable for a final submission."
    )

    lines.append("\n## Aggregate table\n")
    show_agg = agg_df.copy()
    for c in show_agg.columns:
        if c not in {"protocol", "n_runs"}:
            show_agg[c] = show_agg[c].map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    lines.append(show_agg.to_markdown(index=False))

    lines.append("\n\n## Seed-level table\n")
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

    print("===== CHEST BOOTSTRAP CI COMPLETE =====")
    print(f"Seed CI: {OUT_SEED}")
    print(f"Aggregate CI: {OUT_AGG}")
    print(f"Report: {OUT_REPORT}")
    print()
    print("===== MAIN RESULT =====")
    print(f"C2 macro F1 mean ± seed SD: {c2['macro_f1_mean_across_seeds']:.4f} ± {c2['macro_f1_std_across_seeds']:.4f}")
    print(f"C3 on C2 test macro F1 mean ± seed SD: {c3['macro_f1_mean_across_seeds']:.4f} ± {c3['macro_f1_std_across_seeds']:.4f}")
    print(f"Difference: {diff:.4f}")
    print()
    print("===== AGGREGATE =====")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
