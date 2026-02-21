"""EfficientNet-B0 fine-tuning script for deepfake detection.

Usage:
    python train_efficientnet.py --data_dir /path/to/dataset --epochs 20 --output_dir weights/
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeepfakeDataset(Dataset):
    """Generic dataset for deepfake detection.

    Expects folder structure:
        data_dir/
            real/  (or authentic/, genuine/)
            fake/  (or deepfake/, manipulated/)
    """

    REAL_DIRS = {"real", "authentic", "genuine", "original"}
    FAKE_DIRS = {"fake", "deepfake", "manipulated", "synthetic", "generated"}
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        data_dir: str | Path,
        transform=None,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._collect_samples()
        self._split(split, val_ratio, seed)

    def _collect_samples(self) -> None:
        for subdir in self.data_dir.iterdir():
            if not subdir.is_dir():
                continue
            name = subdir.name.lower()
            if name in self.REAL_DIRS:
                label = 0
            elif name in self.FAKE_DIRS:
                label = 1
            else:
                continue
            for path in subdir.rglob("*"):
                if path.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((path, label))
        if not self.samples:
            raise ValueError(f"No images found in {self.data_dir}. "
                             f"Expected subdirs: {self.REAL_DIRS | self.FAKE_DIRS}")
        logger.info("Found %d samples (real=%d, fake=%d)",
                    len(self.samples),
                    sum(1 for _, l in self.samples if l == 0),
                    sum(1 for _, l in self.samples if l == 1))

    def _split(self, split: str, val_ratio: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.samples)).tolist()
        n_val = int(len(indices) * val_ratio)
        if split == "val":
            indices = indices[:n_val]
        else:
            indices = indices[n_val:]
        self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: list[int], labels: list[int], probs: list[float]) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
    f1 = f1_score(labels, preds, zero_division=0)
    return {"accuracy": round(acc, 4), "auc": round(auc, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class EfficientNetTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info("Device: %s", self.device)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_model(self) -> nn.Module:
        import timm
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=2,
        )
        if self.args.freeze_backbone:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            logger.info("Backbone frozen — fine-tuning classifier only")
        return model.to(self.device)

    def build_loaders(self):
        train_ds = DeepfakeDataset(self.args.data_dir, TRAIN_TRANSFORM, "train")
        val_ds = DeepfakeDataset(self.args.data_dir, VAL_TRANSFORM, "val")
        train_loader = DataLoader(
            train_ds, batch_size=self.args.batch_size,
            shuffle=True, num_workers=self.args.workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.args.batch_size * 2,
            shuffle=False, num_workers=self.args.workers, pin_memory=True,
        )
        return train_loader, val_loader

    def train(self) -> None:
        model = self.build_model()
        train_loader, val_loader = self.build_loaders()

        # Class weights for imbalanced datasets
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.epochs, eta_min=1e-6
        )

        best_auc = 0.0
        history: list[dict] = []

        for epoch in range(1, self.args.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            val_metrics = self._eval_epoch(model, val_loader)
            scheduler.step()

            elapsed = time.time() - t0
            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                **val_metrics,
                "lr": round(optimizer.param_groups[0]["lr"], 7),
                "time_s": round(elapsed, 1),
            }
            history.append(row)
            logger.info("Epoch %d/%d | loss=%.4f acc=%.4f auc=%.4f f1=%.4f lr=%.2e | %.1fs",
                        epoch, self.args.epochs, train_loss,
                        val_metrics["accuracy"], val_metrics["auc"], val_metrics["f1"],
                        optimizer.param_groups[0]["lr"], elapsed)

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                self._save(model, f"best_efficientnet_b0.pth")
                logger.info("  *** New best AUC=%.4f — saved ***", best_auc)

        self._save(model, "last_efficientnet_b0.pth")
        self._save_history(history)
        logger.info("Training complete. Best AUC=%.4f", best_auc)

    def _train_epoch(self, model, loader, criterion, optimizer) -> float:
        model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(imgs)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_epoch(self, model, loader) -> dict:
        import torch.nn.functional as F
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())
        return compute_metrics(all_preds, all_labels, all_probs)

    def _save(self, model: nn.Module, filename: str) -> None:
        path = self.output_dir / filename
        torch.save(model.state_dict(), path)
        logger.info("Saved: %s", path)

    def _save_history(self, history: list[dict]) -> None:
        import json
        path = self.output_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EfficientNet-B0 for deepfake detection")
    p.add_argument("--data_dir", required=True, help="Dataset root with real/ and fake/ subdirs")
    p.add_argument("--output_dir", default="weights/", help="Where to save checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze backbone, fine-tune only classifier")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    EfficientNetTrainer(args).train()
