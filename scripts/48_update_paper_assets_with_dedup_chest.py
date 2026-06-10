from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import shutil
import json

ROOT = Path("/home/manikm/meddataforge")
PA = ROOT / "paper_assets"

# ---------------------------------------------------------------------
# 1. Build updated final summary with deduplicated chest results.
# ---------------------------------------------------------------------
rows = [
    {
        "Domain": "Skin lesion",
        "Backbone": "EfficientNet-B0",
        "Compiler-aware protocol": "0.7540 ± 0.0157",
        "Naive protocol": "0.7440 ± 0.0147",
        "Macro-F1 difference": 0.0101,
        "Fair comparison": "Corrected shared seven-class test",
    },
    {
        "Domain": "Skin lesion",
        "Backbone": "ResNet18",
        "Compiler-aware protocol": "0.6961 ± 0.0251",
        "Naive protocol": "0.6226 ± 0.0274",
        "Macro-F1 difference": 0.0735,
        "Fair comparison": "Corrected shared seven-class test",
    },
    {
        "Domain": "Chest X-ray",
        "Backbone": "ResNet18",
        "Compiler-aware protocol": "0.9873 ± 0.0004",
        "Naive protocol": "0.9553 ± 0.0085",
        "Macro-F1 difference": 0.0320,
        "Fair comparison": "Deduplicated same C2-compatible binary test",
    },
    {
        "Domain": "Chest X-ray",
        "Backbone": "EfficientNet-B0",
        "Compiler-aware protocol": "0.9895 ± 0.0015",
        "Naive protocol": "0.9669 ± 0.0067",
        "Macro-F1 difference": 0.0226,
        "Fair comparison": "Deduplicated same C2-compatible binary test",
    },
]
final = pd.DataFrame(rows)

out_csv = ROOT / "results/tables/final_experiment_summary_meddataforge_dedup_chest.csv"
out_tex = ROOT / "results/tables/table_final_experiment_summary_meddataforge_dedup_chest.tex"
out_md = ROOT / "results/tables/final_experiment_summary_meddataforge_dedup_chest.md"

out_csv.parent.mkdir(parents=True, exist_ok=True)
final.to_csv(out_csv, index=False)

tex = r"""\begin{tabularx}{\textwidth}{l l c c c X}
\toprule
Domain & Backbone & Compiler-aware & Naive & Difference & Fair comparison \\
\midrule
Skin lesion & EfficientNet-B0 & $0.7540 \pm 0.0157$ & $0.7440 \pm 0.0147$ & $+0.0101$ & Corrected shared seven-class test \\
Skin lesion & ResNet18 & $0.6961 \pm 0.0251$ & $0.6226 \pm 0.0274$ & $+0.0735$ & Corrected shared seven-class test \\
Chest X-ray & ResNet18 & $0.9873 \pm 0.0004$ & $0.9553 \pm 0.0085$ & $+0.0320$ & Deduplicated same C2-compatible binary test \\
Chest X-ray & EfficientNet-B0 & $0.9895 \pm 0.0015$ & $0.9669 \pm 0.0067$ & $+0.0226$ & Deduplicated same C2-compatible binary test \\
\bottomrule
\end{tabularx}
"""
out_tex.write_text(tex, encoding="utf-8")

out_md.write_text(
    "# Final Experiment Summary with Deduplicated Chest Results\n\n"
    + final.to_markdown(index=False)
    + "\n\nChest X-ray rows use hash-level deduplicated C2/C3 splits with zero train/validation/test hash leakage.\n",
    encoding="utf-8",
)

# ---------------------------------------------------------------------
# 2. Updated final figure with deduplicated chest values.
# ---------------------------------------------------------------------
def mean(x): return float(str(x).split("±")[0].strip())
def sd(x): return float(str(x).split("±")[1].strip())

plot = final.copy()
plot["compiler_mean"] = plot["Compiler-aware protocol"].map(mean)
plot["compiler_sd"] = plot["Compiler-aware protocol"].map(sd)
plot["naive_mean"] = plot["Naive protocol"].map(mean)
plot["naive_sd"] = plot["Naive protocol"].map(sd)
plot["label"] = plot["Domain"] + "\n" + plot["Backbone"]

fig_path = ROOT / "results/figures/fig_final_cross_domain_macro_f1_dedup_chest.png"
x = list(range(len(plot)))
width = 0.35

plt.figure(figsize=(11, 5))
plt.bar([i - width/2 for i in x], plot["compiler_mean"], width, yerr=plot["compiler_sd"], capsize=5, label="Compiler-aware")
plt.bar([i + width/2 for i in x], plot["naive_mean"], width, yerr=plot["naive_sd"], capsize=5, label="Naive pooling")
plt.xticks(x, plot["label"])
plt.ylabel("Macro-F1, mean ± SD")
plt.ylim(0, 1.05)
plt.title("Compiler-aware protocols versus naive pooling")
plt.legend()
plt.tight_layout()
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=300)
plt.close()

# ---------------------------------------------------------------------
# 3. Copy dedup assets into paper_assets.
# ---------------------------------------------------------------------
(PA / "tables/main").mkdir(parents=True, exist_ok=True)
(PA / "tables/appendix").mkdir(parents=True, exist_ok=True)
(PA / "figures/main").mkdir(parents=True, exist_ok=True)
(PA / "reports").mkdir(parents=True, exist_ok=True)

copy_pairs = [
    (out_csv, PA / "tables/main/final_experiment_summary_meddataforge_dedup_chest.csv"),
    (out_tex, PA / "tables/main/table_final_experiment_summary_meddataforge_dedup_chest.tex"),
    (out_md, PA / "reports/final_experiment_summary_meddataforge_dedup_chest.md"),
    (fig_path, PA / "figures/main/fig_final_cross_domain_macro_f1_dedup_chest.png"),
    (ROOT / "results/figures/fig_chest_dedup_macro_f1.png", PA / "figures/main/fig_chest_dedup_macro_f1.png"),
    (ROOT / "results/tables/chest_dedup_c2_minus_c3_summary.csv", PA / "tables/main/chest_dedup_c2_minus_c3_summary.csv"),
    (ROOT / "results/tables/chest_dedup_aggregate_results.csv", PA / "tables/main/chest_dedup_aggregate_results.csv"),
    (ROOT / "results/tables/chest_dedup_seed_results.csv", PA / "tables/appendix/chest_dedup_seed_results.csv"),
    (ROOT / "results/tables/chest_dedup_split_audit_report.md", PA / "reports/chest_dedup_split_audit_report.md"),
    (ROOT / "results/tables/chest_dedup_result_report.md", PA / "reports/chest_dedup_result_report.md"),
    (ROOT / "results/tables/chest_dedup_split_audit.json", PA / "reports/chest_dedup_split_audit.json"),
]

for src, dst in copy_pairs:
    if src.exists():
        shutil.copy2(src, dst)
        print("COPIED:", src, "->", dst)
    else:
        print("MISSING:", src)

# ---------------------------------------------------------------------
# 4. Write a small README for what changed.
# ---------------------------------------------------------------------
readme = PA / "README_DEDUP_CHEST_UPDATE.md"
readme.write_text(
"""# Deduplicated Chest Update

Use these as the main paper chest results.

Main table:
- tables/main/table_final_experiment_summary_meddataforge_dedup_chest.tex

Main figure:
- figures/main/fig_final_cross_domain_macro_f1_dedup_chest.png

Deduplicated chest result figure:
- figures/main/fig_chest_dedup_macro_f1.png

Dedup audit:
- reports/chest_dedup_split_audit_report.md
- reports/chest_dedup_split_audit.json

Dedup audit summary:
- C2 input rows: 23,249
- C2 deduplicated rows: 17,353
- C2 duplicate rows removed: 5,896
- C3 input rows: 32,877
- C3 deduplicated rows: 26,935
- C3 duplicate rows removed: 5,942
- C2 train/val/test hash overlaps: 0
- C3 train/val/test hash overlaps: 0

Main deduplicated chest results:
- ResNet18 C2: 0.9873 ± 0.0004
- ResNet18 C3: 0.9553 ± 0.0085
- Difference: +0.0320
- EfficientNet-B0 C2: 0.9895 ± 0.0015
- EfficientNet-B0 C3: 0.9669 ± 0.0067
- Difference: +0.0226
""",
encoding="utf-8"
)

# ---------------------------------------------------------------------
# 5. Zip paper_assets.
# ---------------------------------------------------------------------
zip_path = ROOT / "paper_assets_meddataforge_updated_dedup_chest.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in PA.rglob("*"):
        if p.is_file():
            z.write(p, arcname=str(p.relative_to(ROOT)))

print("\n===== UPDATED PAPER ASSETS COMPLETE =====")
print("Updated zip:", zip_path)
print("Zip size MB:", round(zip_path.stat().st_size / (1024**2), 2))
print("Use this main table:", PA / "tables/main/table_final_experiment_summary_meddataforge_dedup_chest.tex")
print("Use this main figure:", PA / "figures/main/fig_final_cross_domain_macro_f1_dedup_chest.png")
