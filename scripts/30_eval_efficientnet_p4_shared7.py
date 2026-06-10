#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report


P4_RUNS = [
    {
        "seed": 20260528,
        "model_dir": Path("results/skin_baselines/P4_efficientnet_b0_naive_pool_flagged_8class_seed20260528"),
    },
    {
        "seed": 20260529,
        "model_dir": Path("results/skin_baselines/P4_efficientnet_b0_naive_pool_flagged_8class_seed20260529"),
    },
    {
        "seed": 20260530,
        "model_dir": Path("results/skin_baselines/P4_efficientnet_b0_naive_pool_flagged_8class_seed20260530"),
    },
]

P3_RUNS = [
    {
        "seed": 20260528,
        "metric_file": Path("results/skin_baselines/P3_efficientnet_b0_compiler_filtered_pool_7class_seed20260528/test_metrics.json"),
    },
    {
        "seed": 20260529,
        "metric_file": Path("results/skin_baselines/P3_efficientnet_b0_compiler_filtered_pool_7class_seed20260529/test_metrics.json"),
    },
    {
        "seed": 20260530,
        "metric_file": Path("results/skin_baselines/P3_efficientnet_b0_compiler_filtered_pool_7class_seed20260530/test_metrics.json"),
    },
]

SHARED7_CSV = Path("data/training_splits/shared7_eval/P4_naive_pool_shared7_test_only.csv")

SHARED7_LABELS = [
    "actinic_keratosis_or_intraepithelial_carcinoma",
    "basal_cell_carcinoma",
    "benign_keratosis_like_lesion",
    "dermatofibroma",
    "melanocytic_nevus",
    "melanoma",
    "vascular_lesion",
]

OUT_SEED = Path("results/tables/efficientnet_b0_p3_vs_p4_shared7_corrected_seed.csv")
OUT_AGG = Path("results/tables/efficientnet_b0_p3_vs_p4_shared7_corrected_aggregate.csv")
OUT_REPORT = Path("results/tables/efficientnet_b0_p3_vs_p4_shared7_corrected_report.md")
OUT_SEED.parent.mkdir(parents=True, exist_ok=True)


class SkinDataset(Dataset):
    def __init__(self, df, label_to_idx, transform):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        y = self.label_to_idx[row["canonical_label"]]
        img = self.transform(img)
        return img, y


def build_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


@torch.no_grad()
def evaluate_p4_shared7(model_dir):
    ckpt = torch.load(model_dir / "best_model.pt", map_location="cpu")
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label_raw = ckpt["idx_to_label"]
    idx_to_label = {int(k): v for k, v in idx_to_label_raw.items()} if isinstance(next(iter(idx_to_label_raw.keys())), str) else idx_to_label_raw

    df = pd.read_csv(SHARED7_CSV, low_memory=False)
    df = df[df["split"].eq("shared7_test")].copy()
    df = df[df["canonical_label"].isin(SHARED7_LABELS)].copy()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    loader = DataLoader(
        SkinDataset(df, label_to_idx, tf),
        batch_size=192,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_efficientnet_b0(len(label_to_idx))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_y = []
    all_pred = []

    for x, y in tqdm(loader, desc=f"eval {model_dir.name}"):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy().tolist()
        all_pred.extend(pred)
        all_y.extend(y.numpy().tolist())

    y_true_labels = [idx_to_label[int(i)] for i in all_y]
    y_pred_labels = [idx_to_label[int(i)] for i in all_pred]

    # Macro metrics are computed over the 7 true shared labels only.
    metrics = {
        "n": len(y_true_labels),
        "accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_labels, y_pred_labels)),
        "macro_f1": float(f1_score(y_true_labels, y_pred_labels, labels=SHARED7_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_labels, y_pred_labels, labels=SHARED7_LABELS, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true_labels,
            y_pred_labels,
            labels=SHARED7_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "predicted_labels": sorted(set(y_pred_labels)),
        "true_labels": sorted(set(y_true_labels)),
    }

    out_json = model_dir / "shared7_corrected_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, out_json


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    rows = []

    for run in P3_RUNS:
        metrics = load_json(run["metric_file"])
        rows.append({
            "backbone": "EfficientNet-B0",
            "protocol": "P3",
            "seed": run["seed"],
            "setting": "Compiler-filtered pooled 7-class",
            "metric_file": str(run["metric_file"]),
            "test_rows": metrics["n"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })

    for run in P4_RUNS:
        metrics, out_json = evaluate_p4_shared7(run["model_dir"])
        rows.append({
            "backbone": "EfficientNet-B0",
            "protocol": "P4_shared7_corrected",
            "seed": run["seed"],
            "setting": "Naive pooled model evaluated on shared 7-class subset",
            "metric_file": str(out_json),
            "test_rows": metrics["n"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_SEED, index=False)

    agg = (
        df.groupby(["backbone", "protocol", "setting"])
        .agg(
            n_runs=("macro_f1", "count"),
            test_rows_mean=("test_rows", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
        )
        .reset_index()
    )
    agg.to_csv(OUT_AGG, index=False)

    p3 = agg[agg["protocol"].eq("P3")].iloc[0]
    p4 = agg[agg["protocol"].eq("P4_shared7_corrected")].iloc[0]
    diff = p3["macro_f1_mean"] - p4["macro_f1_mean"]

    lines = []
    lines.append("# EfficientNet-B0 Corrected Shared-7 Comparison\n")
    lines.append("## Main result\n")
    lines.append(f"- **P3 macro F1**: {p3['macro_f1_mean']:.4f} ± {p3['macro_f1_std']:.4f}")
    lines.append(f"- **P4 shared-7 corrected macro F1**: {p4['macro_f1_mean']:.4f} ± {p4['macro_f1_std']:.4f}")
    lines.append(f"- **Difference, P3 - P4 shared-7 corrected**: {diff:.4f}")
    lines.append("\n## Aggregate table\n")
    show = agg.copy()
    for c in show.columns:
        if c.endswith("_mean") or c.endswith("_std"):
            show[c] = show[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    lines.append(show.to_markdown(index=False))
    lines.append("\n\n## Seed-level table\n")
    seed_show = df.copy()
    for c in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        seed_show[c] = seed_show[c].map(lambda x: f"{x:.4f}")
    lines.append(seed_show.to_markdown(index=False))

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("===== EFFICIENTNET SHARED-7 CORRECTED COMPLETE =====")
    print(f"Seed CSV: {OUT_SEED}")
    print(f"Aggregate CSV: {OUT_AGG}")
    print(f"Report: {OUT_REPORT}")
    print("\n===== AGGREGATE RESULTS =====")
    print(agg.to_string(index=False))
    print("\n===== MAIN DIFFERENCE =====")
    print(f"EfficientNet-B0 P3 - P4 shared-7 corrected macro F1 difference: {diff:.4f}")


if __name__ == "__main__":
    main()
