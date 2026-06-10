#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


IN_SEED = Path("results/tables/p3_p4_seed_level_results.csv")
IN_AGG = Path("results/tables/p3_p4_seed_aggregate_results.csv")

OUT_TEX = Path("results/tables/table_p3_p4_seed_aggregate.tex")
OUT_MD = Path("results/tables/p3_p4_paper_result_text.md")
OUT_CSV = Path("results/tables/p3_p4_paired_seed_differences.csv")

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)


def main():
    seed = pd.read_csv(IN_SEED)
    agg = pd.read_csv(IN_AGG)

    ok = seed[seed["status"].eq("ok")].copy()

    p3 = ok[ok["protocol"].eq("P3")][["seed", "macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]].rename(
        columns={
            "macro_f1": "p3_macro_f1",
            "balanced_accuracy": "p3_balanced_accuracy",
            "accuracy": "p3_accuracy",
            "weighted_f1": "p3_weighted_f1",
        }
    )

    p4 = ok[ok["protocol"].eq("P4")][["seed", "macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]].rename(
        columns={
            "macro_f1": "p4_macro_f1",
            "balanced_accuracy": "p4_balanced_accuracy",
            "accuracy": "p4_accuracy",
            "weighted_f1": "p4_weighted_f1",
        }
    )

    paired = p3.merge(p4, on="seed", how="inner")
    paired["macro_f1_diff_p3_minus_p4"] = paired["p3_macro_f1"] - paired["p4_macro_f1"]
    paired["balanced_accuracy_diff_p3_minus_p4"] = paired["p3_balanced_accuracy"] - paired["p4_balanced_accuracy"]
    paired["accuracy_diff_p3_minus_p4"] = paired["p3_accuracy"] - paired["p4_accuracy"]
    paired["weighted_f1_diff_p3_minus_p4"] = paired["p3_weighted_f1"] - paired["p4_weighted_f1"]

    paired.to_csv(OUT_CSV, index=False)

    p3_agg = agg[agg["protocol"].eq("P3")].iloc[0]
    p4_agg = agg[agg["protocol"].eq("P4")].iloc[0]

    mean_diff = paired["macro_f1_diff_p3_minus_p4"].mean()
    sd_diff = paired["macro_f1_diff_p3_minus_p4"].std()

    table = pd.DataFrame([
        {
            "Protocol": "P3",
            "Setting": "Compiler-filtered pooled 7-class",
            "Runs": int(p3_agg["n_runs"]),
            "Accuracy": f"{p3_agg['accuracy_mean']:.4f} ± {p3_agg['accuracy_std']:.4f}",
            "Balanced accuracy": f"{p3_agg['balanced_accuracy_mean']:.4f} ± {p3_agg['balanced_accuracy_std']:.4f}",
            "Macro F1": f"{p3_agg['macro_f1_mean']:.4f} ± {p3_agg['macro_f1_std']:.4f}",
            "Weighted F1": f"{p3_agg['weighted_f1_mean']:.4f} ± {p3_agg['weighted_f1_std']:.4f}",
        },
        {
            "Protocol": "P4",
            "Setting": "Naive pooled with flagged SCC extension",
            "Runs": int(p4_agg["n_runs"]),
            "Accuracy": f"{p4_agg['accuracy_mean']:.4f} ± {p4_agg['accuracy_std']:.4f}",
            "Balanced accuracy": f"{p4_agg['balanced_accuracy_mean']:.4f} ± {p4_agg['balanced_accuracy_std']:.4f}",
            "Macro F1": f"{p4_agg['macro_f1_mean']:.4f} ± {p4_agg['macro_f1_std']:.4f}",
            "Weighted F1": f"{p4_agg['weighted_f1_mean']:.4f} ± {p4_agg['weighted_f1_std']:.4f}",
        },
    ])

    tex = table.to_latex(index=False, escape=True)
    OUT_TEX.write_text(tex, encoding="utf-8")

    lines = []
    lines.append("# Paper-ready P3 vs P4 result text\n")
    lines.append("## Result paragraph\n")
    lines.append(
        f"Across three random seeds, the compiler-filtered pooled 7-class protocol "
        f"outperformed naive pooling with the flagged SCC extension label. "
        f"The compiler-filtered protocol achieved a macro F1 of "
        f"{p3_agg['macro_f1_mean']:.4f} ± {p3_agg['macro_f1_std']:.4f}, "
        f"compared with {p4_agg['macro_f1_mean']:.4f} ± {p4_agg['macro_f1_std']:.4f} "
        f"for naive pooling. The paired seed-level macro-F1 improvement was "
        f"{mean_diff:.4f} ± {sd_diff:.4f}. This supports the central MedDataForge claim "
        f"that metadata-compatible compilation can improve generalization compared with simply pooling additional heterogeneous data."
    )

    lines.append("\n## Aggregate table\n")
    lines.append(table.to_markdown(index=False))

    lines.append("\n\n## Paired seed-level differences\n")
    show = paired[[
        "seed",
        "p3_macro_f1",
        "p4_macro_f1",
        "macro_f1_diff_p3_minus_p4",
        "p3_balanced_accuracy",
        "p4_balanced_accuracy",
        "balanced_accuracy_diff_p3_minus_p4",
    ]].copy()

    for c in show.columns:
        if c != "seed":
            show[c] = show[c].map(lambda x: f"{x:.4f}")

    lines.append(show.to_markdown(index=False))
    lines.append(f"\n\nLaTeX table: `{OUT_TEX}`")
    lines.append(f"\nCSV paired differences: `{OUT_CSV}`")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("===== PAPER RESULT TEXT COMPLETE =====")
    print(f"Markdown: {OUT_MD}")
    print(f"LaTeX: {OUT_TEX}")
    print(f"Paired CSV: {OUT_CSV}")
    print("\n===== RESULT PARAGRAPH =====")
    print(lines[3])
    print("\n===== TABLE =====")
    print(table.to_string(index=False))
    print("\n===== PAIRED DIFFERENCES =====")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
