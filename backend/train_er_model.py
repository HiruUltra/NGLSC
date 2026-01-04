


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# train_er_model.py
# ======================================================
# MODEL ZOO TRAINING (Auto-pick best accuracy)
#
# Fixes:
# ✅ Remove ReduceLROnPlateau(verbose=...) for older torch versions
# ✅ Safe AMP handling for old torch / CPU
# ✅ Keeps stratified split + sampler + weighted loss + fine-tuning blocks
#
# Run:
#   Single model:
#     python train_er_model.py --data_dir data/er_images --backbone efficientnet_b0
#
#   Model zoo:
#     python train_er_model.py --data_dir data/er_images --model_zoo
# ======================================================

import argparse
import json
from datetime import datetime
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, models, transforms


# ----------------- Helpers -----------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def current_lr(optimizer):
    try:
        return float(optimizer.param_groups[0]["lr"])
    except Exception:
        return None

def save_table_as_png(df: pd.DataFrame, out_path: Path, title: str = ""):
    df_disp = df.copy()
    for col in df_disp.columns:
        if pd.api.types.is_numeric_dtype(df_disp[col]):
            df_disp[col] = df_disp[col].round(4)

    n_rows, n_cols = df_disp.shape
    fig_h = max(4, 0.4 * n_rows)
    fig_w = max(6, 0.8 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_disp.astype(str).values,
        colLabels=df_disp.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)

    if title:
        ax.set_title(title, pad=12, fontsize=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_confusion_matrix(cm, class_names, out_path: Path, title: str = "Confusion Matrix"):
    cm = np.asarray(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums, where=(row_sums != 0))
        cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0 if cm_norm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_training_curves(history: dict, out_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(epochs, history["train_loss"], label="Train Loss")
    ax[0].plot(epochs, history["val_loss"], label="Val Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(epochs, history["train_acc"], label="Train Acc")
    ax[1].plot(epochs, history["val_acc"], label="Val Acc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_leaderboard(df: pd.DataFrame, out_path: Path, title: str = "Model Zoo Leaderboard (Val Acc)"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["backbone"].astype(str), df["best_val_acc"].astype(float))
    ax.set_title(title)
    ax.set_xlabel("Backbone")
    ax.set_ylabel("Best Val Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def load_questions_mapping(path: Path) -> dict:
    if not path.exists():
        return {}
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    id_col = None
    text_col = None
    for c in df.columns:
        cl = c.lower()
        if cl == "question_id":
            id_col = c
        if cl in ["question_text", "question_tex", "question"]:
            text_col = c
    if id_col is None or text_col is None:
        return {}

    return {
        str(row[id_col]).strip(): str(row[text_col]).strip()
        for _, row in df.iterrows()
    }


# ----------------- Stratified Split + Dataloaders -----------------

def _manual_stratified_split(targets: np.ndarray, val_split: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)

    train_idx, val_idx = [], []
    for cls in np.unique(targets):
        idx = np.where(targets == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n >= 2:
            n_val = int(round(n * val_split))
            n_val = max(1, n_val)
            n_val = min(n - 1, n_val)
        else:
            n_val = 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)

def build_dataloaders(data_dir: Path, batch_size: int, val_split: float = 0.2):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(8),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    base_ds = datasets.ImageFolder(root=str(data_dir))
    targets = np.array(base_ds.targets)
    class_names = base_ds.classes

    train_idx, val_idx = _manual_stratified_split(targets, val_split=val_split, seed=42)

    train_ds = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
    val_ds   = datasets.ImageFolder(root=str(data_dir), transform=val_transform)

    train_subset = Subset(train_ds, train_idx)
    val_subset   = Subset(val_ds, val_idx)

    train_targets = targets[train_idx]
    counts = Counter(train_targets.tolist())
    class_weight_map = {c: 1.0 / counts[c] for c in counts}
    sample_weights = np.array([class_weight_map[t] for t in train_targets], dtype=np.float32)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("[info] Train class counts:", Counter(train_targets.tolist()))
    print("[info] Val class counts  :", Counter(targets[val_idx].tolist()))
    return train_loader, val_loader, class_names


# ----------------- Build Models (Backbones) -----------------

def build_model(backbone: str, num_classes: int):
    b = backbone.lower().strip()

    if b == "vgg16":
        try:
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.vgg16(pretrained=True)

        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.features[24:].parameters():
            p.requires_grad = True

        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

        return model, model.classifier.parameters(), model.features[24:].parameters()

    if b == "resnet50":
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            model = models.resnet50(pretrained=True)

        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        return model, model.fc.parameters(), model.layer4.parameters()

    if b == "efficientnet_b0":
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)

        for p in model.parameters():
            p.requires_grad = False
        for p in model.features[6:].parameters():
            p.requires_grad = True

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

        return model, model.classifier.parameters(), model.features[6:].parameters()

    if b == "mobilenet_v3_large":
        try:
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        except Exception:
            model = models.mobilenet_v3_large(pretrained=True)

        for p in model.parameters():
            p.requires_grad = False

        start = max(0, len(model.features) - 4)
        for p in model.features[start:].parameters():
            p.requires_grad = True

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

        return model, model.classifier.parameters(), model.features[start:].parameters()

    if b == "densenet121":
        try:
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.densenet121(pretrained=True)

        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model.features, "denseblock4"):
            for p in model.features.denseblock4.parameters():
                p.requires_grad = True

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

        ft_params = model.features.denseblock4.parameters() if hasattr(model.features, "denseblock4") else []
        return model, model.classifier.parameters(), ft_params

    raise ValueError(f"Unsupported backbone: {backbone}")


# ----------------- AMP Safe Helpers -----------------

def amp_available(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    # Older torch may not have amp
    return hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")

class NullAutocast:
    def __init__(self, enabled: bool): self.enabled = enabled
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False

class NullScaler:
    def __init__(self, enabled: bool): self.enabled = enabled
    def scale(self, loss): return loss
    def step(self, optimizer): optimizer.step()
    def update(self): return None


# ----------------- Train / Eval loops -----------------

def train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    use_amp_now = bool(use_amp and amp_available(device))
    autocast_ctx = torch.cuda.amp.autocast if use_amp_now else NullAutocast
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp_now else NullScaler(enabled=False)

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(enabled=use_amp_now):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, 1)
        running_loss += float(loss.item()) * inputs.size(0)
        running_correct += (preds == labels).sum().item()
        total += inputs.size(0)

    return running_loss / max(1, total), running_correct / max(1, total)

def eval_one_epoch(model, loader, criterion, device, use_amp: bool):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    use_amp_now = bool(use_amp and amp_available(device))
    autocast_ctx = torch.cuda.amp.autocast if use_amp_now else NullAutocast

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with autocast_ctx(enabled=use_amp_now):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += float(loss.item()) * inputs.size(0)
            running_correct += (preds == labels).sum().item()
            total += inputs.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return running_loss / max(1, total), running_correct / max(1, total), np.array(all_labels), np.array(all_preds)


# ----------------- Run One Backbone -----------------

def run_one_backbone(
    backbone: str,
    data_dir: Path,
    outdir: Path,
    questions_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    label_smoothing: float,
    use_amp: bool,
    device: torch.device,
):
    ensure_dir(outdir)

    train_loader, val_loader, class_names = build_dataloaders(data_dir, batch_size, val_split=val_split)
    num_classes = len(class_names)

    model, head_params, ft_params = build_model(backbone, num_classes)
    model = model.to(device)

    base_ds = datasets.ImageFolder(root=str(data_dir))
    targets = np.array(base_ds.targets)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    weights = (counts.sum() / (counts + 1e-6))
    weights = weights / weights.mean()
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_t, label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(
        [
            {"params": list(head_params), "lr": lr},
            {"params": list(ft_params), "lr": lr * 0.1},
        ],
        weight_decay=1e-4,
    )

    # ✅ FIX: remove verbose=True (older torch doesn't accept it)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state_dict = None
    best_val_labels = None
    best_val_preds = None

    for epoch in range(1, epochs + 1):
        print(f"\n[{backbone}] Epoch {epoch}/{epochs}  (lr={current_lr(optimizer)})")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp=use_amp)
        va_loss, va_acc, va_labels, va_preds = eval_one_epoch(model, val_loader, criterion, device, use_amp=use_amp)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"  Train - loss: {tr_loss:.4f}, acc: {tr_acc:.4f}")
        print(f"  Val   - loss: {va_loss:.4f}, acc: {va_acc:.4f}")

        prev_lr = current_lr(optimizer)
        scheduler.step(va_acc)
        new_lr = current_lr(optimizer)
        if prev_lr is not None and new_lr is not None and new_lr < prev_lr:
            print(f"  [scheduler] LR reduced: {prev_lr} -> {new_lr}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_labels = va_labels
            best_val_preds = va_preds

    if best_state_dict is None:
        best_state_dict = model.state_dict()
    else:
        model.load_state_dict(best_state_dict)

    model_path = outdir / "best_model.pth"
    torch.save(model.state_dict(), model_path)

    class_index = {i: name for i, name in enumerate(class_names)}
    with open(outdir / "class_index.json", "w", encoding="utf-8") as f:
        json.dump(class_index, f, indent=2)

    plot_training_curves(history, outdir / "training_curves.png")

    if best_val_labels is None or best_val_preds is None:
        _, _, best_val_labels, best_val_preds = eval_one_epoch(model, val_loader, criterion, device, use_amp=use_amp)

    target_names = [class_index[i] for i in sorted(class_index.keys())]
    report = classification_report(
        best_val_labels,
        best_val_preds,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    rep_df = pd.DataFrame(report).T
    rep_df.to_csv(outdir / "classification_report.csv")
    save_table_as_png(rep_df, outdir / "classification_report.png", title=f"Classification Report (Val - {backbone})")

    class_acc = {}
    for i, name in class_index.items():
        mask = (best_val_labels == i)
        class_acc[name] = np.nan if mask.sum() == 0 else (best_val_preds[mask] == i).mean()

    class_acc_df = pd.DataFrame({"class_name": list(class_acc.keys()), "accuracy": list(class_acc.values())})
    class_acc_df.to_csv(outdir / "class_accuracy.csv", index=False)
    save_table_as_png(class_acc_df, outdir / "class_accuracy.png", title=f"Per-Class Accuracy (Val - {backbone})")

    cm = confusion_matrix(best_val_labels, best_val_preds, labels=list(range(len(target_names))))
    plot_confusion_matrix(cm, target_names, outdir / "confusion_matrix.png", title=f"Confusion Matrix (Val - {backbone})")

    qmap = load_questions_mapping(questions_path) if questions_path.exists() else {}
    if qmap:
        with open(outdir / "question_map.json", "w", encoding="utf-8") as f:
            json.dump(qmap, f, indent=2, ensure_ascii=False)

    summary = {
        "backbone": backbone,
        "data_dir": str(data_dir),
        "num_classes": len(class_names),
        "classes": class_names,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_head": lr,
        "lr_finetune": lr * 0.1,
        "label_smoothing": label_smoothing,
        "best_val_acc": float(best_val_acc),
        "history": history,
        "saved_model": str(model_path),
        "device": str(device),
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ [{backbone}] done. Best val acc = {best_val_acc:.4f}  |  saved: {model_path}")
    return best_val_acc, model_path, summary


# ----------------- Main -----------------

def main():
    parser = argparse.ArgumentParser(description="Train ER/Flowchart classifier with Model Zoo (auto best).")
    parser.add_argument("--data_dir", type=str, default="data/er_images")
    parser.add_argument("--questions_path", type=str, default="data/er_questions.xlsx")
    parser.add_argument("--outdir", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--backbone", type=str, default="vgg16",
                        help="vgg16|resnet50|efficientnet_b0|mobilenet_v3_large|densenet121")
    parser.add_argument("--model_zoo", action="store_true",
                        help="If set, trains multiple backbones and selects the best.")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use mixed precision on CUDA (only if available).")

    args = parser.parse_args()
    set_seed(42)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir.resolve()}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = Path(args.outdir) if args.outdir else Path("outputs") / stamp
    ensure_dir(root_out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Output root: {root_out}")
    print(f"[info] Device: {device}")

    questions_path = Path(args.questions_path)

    zoo_list = ["efficientnet_b0", "resnet50", "mobilenet_v3_large", "densenet121", "vgg16"]

    results = []
    best = {"acc": -1.0, "backbone": None, "model_path": None, "summary": None}

    backbones_to_run = zoo_list if args.model_zoo else [args.backbone]

    for bb in backbones_to_run:
        bb_out = root_out / bb
        acc, model_path, summary = run_one_backbone(
            backbone=bb,
            data_dir=data_dir,
            outdir=bb_out,
            questions_path=questions_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
            label_smoothing=args.label_smoothing,
            use_amp=args.use_amp,
            device=device,
        )

        results.append({"backbone": bb, "best_val_acc": float(acc), "model_path": str(model_path)})

        if acc > best["acc"]:
            best["acc"] = float(acc)
            best["backbone"] = bb
            best["model_path"] = str(model_path)
            best["summary"] = summary

    lb_df = pd.DataFrame(results).sort_values("best_val_acc", ascending=False)
    lb_df.to_csv(root_out / "leaderboard.csv", index=False)
    save_table_as_png(lb_df, root_out / "leaderboard.png", title="Model Zoo Leaderboard (Val Acc)")
    plot_leaderboard(lb_df, root_out / "leaderboard_bar.png")

    best_dir = root_out / "BEST"
    ensure_dir(best_dir)

    best_state = torch.load(best["model_path"], map_location="cpu")
    torch.save(best_state, best_dir / "best_model.pth")

    with open(best_dir / "best_backbone.txt", "w", encoding="utf-8") as f:
        f.write(str(best["backbone"]))

    with open(best_dir / "summary_best.json", "w", encoding="utf-8") as f:
        json.dump(best["summary"], f, indent=2)

    print("\n==============================")
    print(f"🏆 BEST MODEL: {best['backbone']}  |  val_acc={best['acc']:.4f}")
    print(f"Saved best weights: {best_dir / 'best_model.pth'}")
    print("Leaderboard:", root_out / "leaderboard.csv")
    print("==============================\n")


if __name__ == "__main__":
    main()
