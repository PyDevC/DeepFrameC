import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
from config import Config
from model import DeepFakeDetector
from dataset import build_dataloaders


def train():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    loaders, train_df = build_dataloaders(cfg)   # single call — was called twice before
    model = DeepFakeDetector(
        backbone_name=cfg.BACKBONE,
        pretrained=cfg.PRETRAINED,
        dropout=cfg.DROPOUT,
        num_classes=cfg.NUM_CLASSES,
    ).to(device)
    n_real = (train_df["Label"] == "REAL").sum()
    n_fake = (train_df["Label"] == "FAKE").sum()
    total  = n_real + n_fake
    class_weights = torch.tensor([total/n_real, total/n_fake], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.LABEL_SMOOTHING)

    # Differential LR: backbone gets 10x lower LR than the classification head.
    # This preserves pretrained ImageNet features while training the head aggressively.
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.LR / 10},
        {"params": model.classifier.parameters(), "lr": cfg.LR},
    ], weight_decay=cfg.WEIGHT_DECAY)

    # 3-epoch linear warmup → cosine decay for the rest
    warmup_epochs = 3
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    scaler = torch.amp.grad_scaler.GradScaler(device)

    best_auc = 0.0
    epoch_bar = tqdm(range(cfg.EPOCHS), desc="Epochs", unit="epoch", position=0)

    for epoch in epoch_bar:
        # --- Train ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(
            loaders["train"],
            desc=f"  Train {epoch+1}/{cfg.EPOCHS}",
            leave=False,
            unit="batch",
            position=1,
        )

        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast_mode.autocast(device):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct/total:.4f}",
            )

        scheduler.step()

        # --- Validate ---
        auc = evaluate(model, loaders["val"], device, epoch, cfg.EPOCHS)

        avg_loss = train_loss / len(loaders["train"])
        train_acc = correct / total

        epoch_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{train_acc:.4f}",
            val_auc=f"{auc:.4f}",
        )

        tqdm.write(
            f"Epoch {epoch+1}/{cfg.EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val AUC: {auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"{cfg.CHECKPOINT_DIR}/best.pth")
            tqdm.write(f"  ✓ Saved best model (AUC: {best_auc:.4f})")

    tqdm.write(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")


def evaluate(model, loader, device, epoch=None, total_epochs=None):
    model.eval()
    all_probs, all_labels = [], []

    desc = f"  Val   {epoch+1}/{total_epochs}" if epoch is not None else "  Val"
    val_bar = tqdm(loader, desc=desc, leave=False, unit="batch", position=1)

    with torch.no_grad():
        for imgs, labels in val_bar:
            imgs = imgs.to(device)

            with torch.amp.autocast_mode.autocast(device):
                logits = model(imgs)

            # Cast to float32 before softmax — autocast may return float16
            logits = logits.float()

            # Replace nan/inf before softmax
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Final guard
    valid_mask = np.isfinite(all_probs)
    if valid_mask.sum() < len(all_probs):
        print(f"Warning: {(~valid_mask).sum()} NaN/inf probs dropped before AUC computation")
    all_probs  = all_probs[valid_mask]
    all_labels = all_labels[valid_mask]

    if len(np.unique(all_labels)) < 2:
        print("Warning: only one class in val batch — AUC undefined, returning 0.5")
        return 0.5

    return roc_auc_score(all_labels, all_probs)


if __name__ == "__main__":
    train()
