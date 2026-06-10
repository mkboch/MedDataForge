#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SkinDataset(Dataset):
    def __init__(self, df, label_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = self.label_to_idx[row["canonical_label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(backbone: str, num_classes: int, pretrained: bool):
    backbone = backbone.lower()

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if backbone == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {backbone}")


def compute_class_weights(train_df, label_to_idx):
    counts = train_df["canonical_label"].value_counts()
    total = len(train_df)
    num_classes = len(label_to_idx)
    weights = []
    for label, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        c = counts.get(label, 1)
        weights.append(total / (num_classes * c))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, idx_to_label):
    model.eval()
    all_y = []
    all_pred = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())

    labels = list(range(len(idx_to_label)))
    return {
        "loss": total_loss / max(1, len(all_y)),
        "accuracy": float(accuracy_score(all_y, all_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(all_y, all_pred)),
        "macro_f1": float(f1_score(all_y, all_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_y, all_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            all_y,
            all_pred,
            labels=labels,
            target_names=[idx_to_label[i] for i in labels],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(all_y, all_pred, labels=labels).tolist(),
        "n": len(all_y),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20260528)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--pretrained", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split_csv, low_memory=False)
    df = df[df["image_path"].notna()].copy()
    df = df[df["canonical_label"].notna()].copy()

    required_splits = {"train", "val", "external_test", "test"}
    df = df[df["split"].isin(required_splits)].copy()

    df["image_exists"] = df["image_path"].map(lambda p: Path(str(p)).exists())
    missing = df[~df["image_exists"]]
    if len(missing):
        missing.to_csv(out_dir / "missing_images.csv", index=False)
        raise SystemExit(f"Missing images found: {len(missing)}")

    labels = sorted(df["canonical_label"].unique().tolist())
    label_to_idx = {x: i for i, x in enumerate(labels)}
    idx_to_label = {i: x for x, i in label_to_idx.items()}

    train_df = df[df["split"].eq("train")].copy()
    val_df = df[df["split"].eq("val")].copy()
    test_df = df[df["split"].isin(["external_test", "test"])].copy()

    print("===== DATA SUMMARY =====", flush=True)
    print(f"Backbone: {args.backbone}", flush=True)
    print(f"Total rows: {len(df)}", flush=True)
    print(f"Train rows: {len(train_df)}", flush=True)
    print(f"Val rows: {len(val_df)}", flush=True)
    print(f"Test rows: {len(test_df)}", flush=True)
    print(f"Labels: {labels}", flush=True)

    (out_dir / "label_to_idx.json").write_text(json.dumps(label_to_idx, indent=2), encoding="utf-8")

    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(SkinDataset(train_df, label_to_idx, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(SkinDataset(val_df, label_to_idx, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(SkinDataset(test_df, label_to_idx, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model = build_model(args.backbone, len(labels), pretrained=args.pretrained).to(device)

    class_weights = compute_class_weights(train_df, label_to_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0

        pbar = tqdm(train_loader, desc=f"{args.backbone} epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix(loss=running_loss / max(1, n_seen))

        val_metrics = evaluate(model, val_loader, device, idx_to_label)
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, n_seen),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "seconds": time.time() - t0,
        }
        history.append(epoch_result)

        print("===== EPOCH RESULT =====", flush=True)
        print(json.dumps(epoch_result, indent=2), flush=True)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, out_dir / "best_model.pt")
            (out_dir / "best_val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    print("===== LOADING BEST MODEL FOR TEST =====", flush=True)
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics = evaluate(model, val_loader, device, idx_to_label)
    test_metrics = evaluate(model, test_loader, device, idx_to_label)

    (out_dir / "final_val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    summary = {
        "split_csv": args.split_csv,
        "out_dir": str(out_dir),
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "labels": labels,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "best_val_macro_f1": best_val_macro_f1,
        "final_val": {
            "accuracy": val_metrics["accuracy"],
            "balanced_accuracy": val_metrics["balanced_accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
        },
        "test": {
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
    }

    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== TRAINING COMPLETE =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
