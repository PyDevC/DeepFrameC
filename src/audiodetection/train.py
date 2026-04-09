"""
train.py — Training loop for AudioDeepFakeDetector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import copy

from config import Config
from model import AudioDeepFakeDetector
from dataset import build_dataloaders

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        n_cls = logits.size(1)
        with torch.no_grad():
            smooth = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
            smooth = smooth * (1 - self.label_smoothing) + self.label_smoothing / n_cls

        log_p  = F.log_softmax(logits, dim=1)
        p      = log_p.exp()
        ce     = -(smooth * log_p).sum(dim=1)
        p_t       = (p * F.one_hot(targets, n_cls)).sum(dim=1)
        focal_w   = (1 - p_t) ** self.gamma
        loss = focal_w * ce

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss    = alpha_t * loss

        return loss.mean()

def mixup_batch(mels, labels, alpha: float = 0.4):
    if alpha <= 0:
        return mels, labels, labels, 1.0

    lam   = np.random.beta(alpha, alpha)
    B     = mels.size(0)
    idx   = torch.randperm(B, device=mels.device)
    mixed = lam * mels + (1 - lam) * mels[idx]
    return mixed, labels, labels[idx], lam

def mixup_criterion(criterion, logits, ya, yb, lam):
    return lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1 - self.decay)
        for s_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            s_b.copy_(m_b)

    def state_dict(self):
        return self.shadow.state_dict()

def train():
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders, train_df = build_dataloaders(cfg)

    model = AudioDeepFakeDetector(
        pretrained  = cfg.PRETRAINED,
        dropout     = cfg.DROPOUT,
        num_classes = cfg.NUM_CLASSES,
    ).to(device)

    n_real = (train_df["Label"] == "REAL").sum()
    n_fake = (train_df["Label"] == "FAKE").sum()
    total  = n_real + n_fake
    class_weights = torch.tensor(
        [total / max(n_real, 1), total / max(n_fake, 1)], dtype=torch.float
    ).to(device)
    
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=cfg.LABEL_SMOOTHING)
    ema = EMA(model, decay=cfg.EMA_DECAY)

    def set_backbone_grad(requires_grad: bool):
        for p in model.backbone.parameters():
            p.requires_grad = requires_grad

    set_backbone_grad(False)

    head_params = [
        {"params": model.classifier.parameters(), "lr": cfg.LR},
    ]
    optimizer = AdamW(head_params, weight_decay=cfg.WEIGHT_DECAY)

    steps_per_epoch   = len(loaders["train"])
    total_steps       = cfg.EPOCHS * steps_per_epoch

    scheduler = OneCycleLR(
        optimizer,
        max_lr        = cfg.LR,
        total_steps   = total_steps,
        pct_start     = cfg.WARMUP_EPOCHS / cfg.EPOCHS,
        anneal_strategy="cos",
        div_factor    = 10.0,
        final_div_factor=1e4,
    )

    scaler = torch.amp.grad_scaler.GradScaler(device.type)

    best_auc    = 0.0
    accum_steps = cfg.GRAD_ACCUM_STEPS

    epoch_bar = tqdm(range(cfg.EPOCHS), desc="Epochs", unit="epoch", position=0)

    for epoch in epoch_bar:
        if epoch == cfg.WARMUP_EPOCHS:
            print(f"\n  [Epoch {epoch+1}] Unfreezing backbone with LR={cfg.LR/10:.2e}")
            set_backbone_grad(True)
            optimizer.add_param_group({
                "params": model.backbone.parameters(),
                "lr"    : cfg.LR / 10,
            })

        model.train()
        train_loss = 0.0
        correct    = 0
        n_total    = 0

        train_bar = tqdm(
            loaders["train"],
            desc=f"  Train {epoch+1}/{cfg.EPOCHS}",
            leave=False, unit="batch", position=1,
        )

        optimizer.zero_grad()

        for step, (mels, labels) in enumerate(train_bar, 1):
            mels   = mels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            mels, ya, yb, lam = mixup_batch(mels, labels, alpha=cfg.MIXUP_ALPHA)

            with torch.amp.autocast_mode.autocast(device.type):
                logits = model(mels)
                loss   = mixup_criterion(criterion, logits, ya, yb, lam)
                loss   = loss / accum_steps

            scaler.scale(loss).backward()

            if step % accum_steps == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                ema.update(model)

            train_loss += loss.item() * accum_steps
            with torch.no_grad():
                preds    = logits.argmax(dim=1)
                correct  += (preds == labels).sum().item()
                n_total  += labels.size(0)

            train_bar.set_postfix(
                loss=f"{loss.item()*accum_steps:.4f}",
                acc =f"{correct/n_total:.4f}",
                lr  =f"{scheduler.get_last_lr()[0]:.2e}",
            )

        auc = evaluate(ema.shadow, loaders["val"], device, epoch, cfg.EPOCHS)

        avg_loss  = train_loss / len(loaders["train"])
        train_acc = correct / n_total

        epoch_bar.set_postfix(
            loss=f"{avg_loss:.4f}", acc=f"{train_acc:.4f}", val_auc=f"{auc:.4f}"
        )
        tqdm.write(
            f"Epoch {epoch+1}/{cfg.EPOCHS} | Loss {avg_loss:.4f} | "
            f"Acc {train_acc:.4f} | Val AUC {auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(ema.state_dict(), f"{cfg.CHECKPOINT_DIR}/best.pth")
            tqdm.write(f"  ✓ Saved best EMA model (AUC: {best_auc:.4f})")

        torch.save({
            "epoch"      : epoch,
            "model"      : model.state_dict(),
            "ema"        : ema.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "scheduler"  : scheduler.state_dict(),
            "best_auc"   : best_auc,
        }, f"{cfg.CHECKPOINT_DIR}/last.pth")

    tqdm.write(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")

def evaluate(model, loader, device, epoch=None, total_epochs=None):
    model.eval()
    all_probs, all_labels = [], []

    desc    = f"  Val {epoch+1}/{total_epochs}" if epoch is not None else "  Val"
    val_bar = tqdm(loader, desc=desc, leave=False, unit="batch", position=1)

    with torch.no_grad():
        for mels, labels in val_bar:
            mels = mels.to(device, non_blocking=True)
            with torch.amp.autocast_mode.autocast(device.type):
                logits = model(mels)
            logits = logits.float()
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    valid      = np.isfinite(all_probs)

    if valid.sum() < len(all_probs):
        print(f"Warning: {(~valid).sum()} NaN/inf probs dropped")
    all_probs  = all_probs[valid]
    all_labels = all_labels[valid]

    if len(np.unique(all_labels)) < 2:
        return 0.5

    return roc_auc_score(all_labels, all_probs)

if __name__ == "__main__":
    train()
