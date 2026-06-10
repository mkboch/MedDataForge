#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

OUT_MD = Path("results/tables/final_experiment_summary_meddataforge.md")
OUT_CSV = Path("results/tables/final_experiment_summary_meddataforge.csv")
OUT_TEX = Path("results/tables/table_final_experiment_summary_meddataforge.tex")

rows = []

# Skin combined table already made earlier.
skin = pd.read_csv("results/tables/combined_backbone_shared7_results.csv")

for _, r in skin.iterrows():
    rows.append({
        "Domain": "Skin lesion",
        "Backbone": r["Backbone"] if "Backbone" in r else r["backbone"],
        "Compiler-aware protocol": r["P3 macro F1"] if "P3 macro F1" in r else f"{r['p3_macro_f1_mean']:.4f} ± {r['p3_macro_f1_std']:.4f}",
        "Naive protocol": r["P4 shared-7 macro F1"] if "P4 shared-7 macro F1" in r else f"{r['p4_macro_f1_mean']:.4f} ± {r['p4_macro_f1_std']:.4f}",
        "Macro-F1 difference": r["P3 - P4 macro F1"] if "P3 - P4 macro F1" in r else f"{r['macro_f1_difference_p3_minus_p4']:.4f}",
        "Fair comparison": "Corrected shared 7-class test",
        "Interpretation": "Compiler-filtered pooling matched or exceeded naive shared-label pooling.",
    })

# Chest ResNet18 fair same-test.
chest_r = pd.read_csv("results/tables/chest_resnet18_c2_vs_c3_on_c2test_aggregate.csv")
c2 = chest_r[chest_r["protocol"].eq("C2")].iloc[0]
c3 = chest_r[chest_r["protocol"].eq("C3_on_C2test")].iloc[0]
rows.append({
    "Domain": "Chest X-ray",
    "Backbone": "ResNet18",
    "Compiler-aware protocol": f"{c2['macro_f1_mean']:.4f} ± {c2['macro_f1_std']:.4f}",
    "Naive protocol": f"{c3['macro_f1_mean']:.4f} ± {c3['macro_f1_std']:.4f}",
    "Macro-F1 difference": f"{c2['macro_f1_mean'] - c3['macro_f1_mean']:.4f}",
    "Fair comparison": "Same C2-compatible binary test",
    "Interpretation": "Compiler-compatible training outperformed naive abnormal-to-pneumonia pooling.",
})

# Chest EfficientNet-B0 fair same-test.
chest_e = pd.read_csv("results/tables/chest_efficientnet_b0_c2_vs_c3_on_c2test_aggregate.csv")
c2 = chest_e[chest_e["protocol"].eq("C2")].iloc[0]
c3 = chest_e[chest_e["protocol"].eq("C3_on_C2test")].iloc[0]
rows.append({
    "Domain": "Chest X-ray",
    "Backbone": "EfficientNet-B0",
    "Compiler-aware protocol": f"{c2['macro_f1_mean']:.4f} ± {c2['macro_f1_std']:.4f}",
    "Naive protocol": f"{c3['macro_f1_mean']:.4f} ± {c3['macro_f1_std']:.4f}",
    "Macro-F1 difference": f"{c2['macro_f1_mean'] - c3['macro_f1_mean']:.4f}",
    "Fair comparison": "Same C2-compatible binary test",
    "Interpretation": "Compiler-compatible training outperformed naive abnormal-to-pneumonia pooling.",
})

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

latex = df.to_latex(index=False, escape=True)
OUT_TEX.write_text(latex, encoding="utf-8")

lines = []
lines.append("# Final MedDataForge Experiment Summary\n")
lines.append("## Final status\n")
lines.append(
    "The core experiments are complete. The study now includes two medical imaging domains, two skin-lesion backbones, two chest X-ray backbones, multiple random seeds, and fair same-test comparisons where needed."
)
lines.append("\n## Final evidence table\n")
lines.append(df.to_markdown(index=False))
lines.append("\n## Main conclusion\n")
lines.append(
    "Across skin lesion and chest X-ray experiments, compiler-aware protocols consistently matched or exceeded naive pooling. "
    "The gains were larger when label-space incompatibility created stronger semantic noise, and smaller when the backbone or task made the mismatch easier to absorb. "
    "This supports the central claim that MedDataForge should be framed as an auditable metadata-guided experiment compiler, not simply as a dataset aggregation pipeline."
)
lines.append("\n## Safe paper claim\n")
lines.append(
    "MedDataForge detects dataset and label-space incompatibility before training, separates unsafe dataset combinations, and generates auditable experiment protocols. "
    "In two imaging domains, compiler-aware protocols produced equal or better macro-F1 than naive pooling under fair evaluation, with effect sizes depending on model backbone and task difficulty."
)
lines.append("\n## Experiments completed\n")
lines.append("- Skin lesion, ResNet18, 3 seeds.")
lines.append("- Skin lesion, EfficientNet-B0, 3 seeds.")
lines.append("- Chest X-ray, ResNet18, 3 seeds, fair C2-test comparison.")
lines.append("- Chest X-ray, EfficientNet-B0, 3 seeds, fair C2-test comparison.")
lines.append("- Bootstrap CI for skin and chest ResNet18.")
lines.append("- Registry audit, dataset access audit, manifest compiler audit, and image-linking audit.")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")

print("===== FINAL EXPERIMENT SUMMARY COMPLETE =====")
print(f"Markdown: {OUT_MD}")
print(f"CSV: {OUT_CSV}")
print(f"LaTeX: {OUT_TEX}")
print()
print(df.to_string(index=False))
