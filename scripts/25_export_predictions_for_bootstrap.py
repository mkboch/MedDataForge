#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


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
        y = self.label_to_idx[row["canonical_label"]]
        if self.transform:
            img = self.transform(img)
        return img, y, idx


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def export_predictions(model, loader, device, idx_to_label, df, out_csv):
    model.eval()
    rows = []

    for x, y, idx in tqdm(loader, desc="predict"):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).cpu().numpy().tolist()
        true = y.numpy().tolist()
        indices = idx.numpy().tolist()

        for ii, yy, pp in zip(indices, true, pred):
            r = df.iloc[ii].to_dict()
            r["true_idx"] = int(yy)
            r["pred_idx"] = int(pp)
            r["true_label"] = idx_to_label[int(yy)]
            r["pred_label"] = idx_to_label[int(pp)]
            r["correct"] = int(yy == pp)
            rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--eval_split", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    args = ap.parse_args()

    ckpt_path = Path(args.model_dir) / "best_model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label_raw = ckpt["idx_to_label"]
    idx_to_label = {int(k): v for k, v in idx_to_label_raw.items()} if isinstance(next(iter(idx_to_label_raw.keys())), str) else idx_to_label_raw

    df = pd.read_csv(args.split_csv, low_memory=False)
    df = df[df["split"].eq(args.eval_split)].copy()
    df = df[df["image_path"].notna()].copy()
    df = df[df["canonical_label"].isin(label_to_idx)].copy()

    if len(df) == 0:
        raise SystemExit(f"No rows found for eval split: {args.eval_split}")

    missing = df[~df["image_path"].map(lambda p: Path(str(p)).exists())]
    if len(missing):
        raise SystemExit(f"Missing image paths: {len(missing)}")

    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = SkinDataset(df, label_to_idx, tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(label_to_idx))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_df = export_predictions(model, loader, device, idx_to_label, df, out_csv)

    summary = {
        "model_dir": args.model_dir,
        "split_csv": args.split_csv,
        "eval_split": args.eval_split,
        "out_csv": str(out_csv),
        "rows": int(len(pred_df)),
        "accuracy": float(pred_df["correct"].mean()),
        "labels": sorted(pred_df["true_label"].unique().tolist()),
        "predicted_labels": sorted(pred_df["pred_label"].unique().tolist()),
    }

    out_json = out_csv.with_suffix(".summary.json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== PREDICTION EXPORT COMPLETE =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
