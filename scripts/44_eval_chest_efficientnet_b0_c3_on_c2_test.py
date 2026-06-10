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

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix


C2_SPLIT = Path("data/training_splits/chest/C2_compiler_compatible_binary_pool.csv")

C2_MODELS = [
    (20260602, Path("results/chest_baselines/C2_efficientnet_b0_compiler_compatible_binary_pool_seed20260602")),
    (20260603, Path("results/chest_baselines/C2_efficientnet_b0_compiler_compatible_binary_pool_seed20260603")),
    (20260604, Path("results/chest_baselines/C2_efficientnet_b0_compiler_compatible_binary_pool_seed20260604")),
]

C3_MODELS = [
    (20260602, Path("results/chest_baselines/C3_efficientnet_b0_naive_binary_pool_all_abnormal_seed20260602")),
    (20260603, Path("results/chest_baselines/C3_efficientnet_b0_naive_binary_pool_all_abnormal_seed20260603")),
    (20260604, Path("results/chest_baselines/C3_efficientnet_b0_naive_binary_pool_all_abnormal_seed20260604")),
]

OUT_SEED = Path("results/tables/chest_efficientnet_b0_c2_vs_c3_on_c2test_seed_results.csv")
OUT_AGG = Path("results/tables/chest_efficientnet_b0_c2_vs_c3_on_c2test_aggregate.csv")
OUT_REPORT = Path("results/tables/chest_efficientnet_b0_c2_vs_c3_on_c2test_report.md")


class ImageDataset(Dataset):
    def __init__(self, df, label_to_idx, transform):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        y = self.label_to_idx[row["canonical_label"]]
        return img, y


def build_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def eval_model(model_dir, split_df):
    ckpt = torch.load(model_dir / "best_model.pt", map_location="cpu")
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label_raw = ckpt["idx_to_label"]
    idx_to_label = {int(k): v for k, v in idx_to_label_raw.items()} if isinstance(next(iter(idx_to_label_raw.keys())), str) else idx_to_label_raw

    df = split_df[split_df["canonical_label"].isin(label_to_idx)].copy()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    loader = DataLoader(
        ImageDataset(df, label_to_idx, tf),
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

    y_true = []
    y_pred = []

    for x, y in tqdm(loader, desc=model_dir.name):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(y.numpy().tolist())

    labels = list(range(len(idx_to_label)))
    names = [idx_to_label[i] for i in labels]

    metrics = {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    return metrics


def main():
    OUT_SEED.parent.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(C2_SPLIT, low_memory=False)
    test_df = split_df[split_df["split"].eq("test")].copy()

    rows = []

    for seed, model_dir in C2_MODELS:
        metrics_path = model_dir / "test_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        rows.append({
            "protocol": "C2",
            "seed": seed,
            "backbone": "EfficientNet-B0",
            "setting": "Compiler-compatible model on C2 test",
            "metric_file": str(metrics_path),
            "test_rows": metrics["n"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })

    for seed, model_dir in C3_MODELS:
        metrics = eval_model(model_dir, test_df)
        out_json = model_dir / "eval_on_C2_test_metrics.json"
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        rows.append({
            "protocol": "C3_on_C2test",
            "seed": seed,
            "backbone": "EfficientNet-B0",
            "setting": "Naive noisy model evaluated on same C2 test",
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

    c2 = agg[agg["protocol"].eq("C2")].iloc[0]
    c3 = agg[agg["protocol"].eq("C3_on_C2test")].iloc[0]
    diff = c2["macro_f1_mean"] - c3["macro_f1_mean"]

    lines = []
    lines.append("# Chest EfficientNet-B0 Fair Test Comparison: C2 vs C3 on Same C2 Test Set\n")
    lines.append("## Main result\n")
    lines.append(f"- **C2 macro F1 on C2 test**: {c2['macro_f1_mean']:.4f} ± {c2['macro_f1_std']:.4f}")
    lines.append(f"- **C3 macro F1 on same C2 test**: {c3['macro_f1_mean']:.4f} ± {c3['macro_f1_std']:.4f}")
    lines.append(f"- **Difference, C2 - C3**: {diff:.4f}")
    lines.append("\n## Aggregate table\n")
    show = agg.copy()
    for c in show.columns:
        if c.endswith("_mean") or c.endswith("_std"):
            show[c] = show[c].map(lambda x: f"{x:.4f}")
    lines.append(show.to_markdown(index=False))
    lines.append("\n\n## Seed-level table\n")
    show_seed = df.copy()
    for c in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        show_seed[c] = show_seed[c].map(lambda x: f"{x:.4f}")
    lines.append(show_seed.to_markdown(index=False))

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("===== CHEST EFFICIENTNET-B0 FAIR C2 TEST COMPARISON COMPLETE =====")
    print(f"Seed CSV: {OUT_SEED}")
    print(f"Aggregate CSV: {OUT_AGG}")
    print(f"Report: {OUT_REPORT}")
    print()
    print("===== AGGREGATE =====")
    print(agg.to_string(index=False))
    print()
    print("===== MAIN DIFFERENCE =====")
    print(f"EfficientNet-B0 C2 - C3_on_C2test macro F1 difference: {diff:.4f}")


if __name__ == "__main__":
    main()
