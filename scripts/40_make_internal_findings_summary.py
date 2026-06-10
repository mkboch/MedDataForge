#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

OUT = Path("results/tables/internal_findings_summary_meddataforge.md")
OUT_CSV = Path("results/tables/internal_findings_summary_meddataforge.csv")

skin = pd.read_csv("results/tables/combined_backbone_shared7_results.csv")
chest = pd.read_csv("results/tables/chest_resnet18_c2_vs_c3_on_c2test_aggregate.csv")

rows = []

for _, r in skin.iterrows():
    rows.append({
        "domain": "Skin lesion",
        "backbone": r["backbone"],
        "comparison": "P3 compiler-filtered 7-class vs P4 naive shared-7 corrected",
        "compiler_setting": f"{r['p3_macro_f1_mean']:.4f} ± {r['p3_macro_f1_std']:.4f}",
        "naive_setting": f"{r['p4_macro_f1_mean']:.4f} ± {r['p4_macro_f1_std']:.4f}",
        "macro_f1_difference": f"{r['macro_f1_difference_p3_minus_p4']:.4f}",
        "interpretation": "Compiler-filtered pooling matched or exceeded naive shared-label pooling; effect was larger for ResNet18 and smaller for EfficientNet-B0.",
    })

c2 = chest[chest["protocol"].eq("C2")].iloc[0]
c3 = chest[chest["protocol"].eq("C3_on_C2test")].iloc[0]
rows.append({
    "domain": "Chest X-ray",
    "backbone": "ResNet18",
    "comparison": "C2 compatible binary pool vs C3 naive noisy pool on same C2 test",
    "compiler_setting": f"{c2['macro_f1_mean']:.4f} ± {c2['macro_f1_std']:.4f}",
    "naive_setting": f"{c3['macro_f1_mean']:.4f} ± {c3['macro_f1_std']:.4f}",
    "macro_f1_difference": f"{c2['macro_f1_mean'] - c3['macro_f1_mean']:.4f}",
    "interpretation": "Compiler-compatible training exceeded naive pooling that collapsed COVID-19 and lung opacity into pneumonia.",
})

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

lines = []
lines.append("# MedDataForge Internal Findings Summary\n")
lines.append("## Current target\n")
lines.append(
    "The current target is a methods/systems paper centered on MedDataForge as a metadata-guided experiment compiler. "
    "The key claim is not simply that harmonization improves performance, but that metadata-aware compilation can identify dataset incompatibility before training, reject unsafe pooling, and produce auditable experiment protocols."
)

lines.append("\n## Current empirical evidence\n")
lines.append(df.to_markdown(index=False))

lines.append("\n## Main interpretation\n")
lines.append(
    "Across two domains, the evidence supports the core compatibility hypothesis. In skin lesions, compiler-filtered pooling improved over naive shared-label pooling for ResNet18 and remained slightly better for EfficientNet-B0. "
    "In chest X-ray, the compiler separated binary pneumonia-compatible data from a naive pool that collapsed COVID-19 and lung opacity into pneumonia; on the same compatible test set, the compiler-compatible model performed better."
)

lines.append("\n## Important caveats\n")
lines.append(
    "The results are promising but should still be treated as internal evidence. The EfficientNet-B0 skin result shows only a small advantage after fair shared-label correction, so the paper should not overclaim universal large performance gains. "
    "The chest task has very high absolute scores, so its value is mainly as evidence of label-semantic auditing and unsafe-pooling detection, not as a difficult benchmark result."
)

lines.append("\n## Next internal fixes before drafting\n")
lines.append("1. Bootstrap confidence intervals for the chest C2 vs C3 fair comparison.")
lines.append("2. Optional: run EfficientNet-B0 on chest C2/C3 if we want model-independence in the second domain.")
lines.append("3. Improve registry license/access parsing.")
lines.append("4. Generate final reproducibility manifest: scripts, outputs, dataset counts, seeds, and hardware.")

OUT.write_text("\n".join(lines), encoding="utf-8")

print("===== INTERNAL FINDINGS SUMMARY COMPLETE =====")
print(f"Markdown: {OUT}")
print(f"CSV: {OUT_CSV}")
print()
print(df.to_string(index=False))
