"""CLIP ViT-L/14 fine-tuning script — LayerNorm-only (0.03% params).

Strategy:
  - Freeze entire CLIP except LayerNorm layers
  - Binary classification: real vs fake
  - Contrastive loss on image embeddings + BCE on probe head
  - Saves only LayerNorm state_dict (tiny file ~2MB)

Usage:
    python train_clip.py --data_dir /path/to/dataset --epochs 10 --output_dir weights/
"""

from __future__ import annotations

import argparse
import logging
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
# Dataset (same structure as EfficientNet trainer)
# ---------------------------------------------------------------------------

class ImageDataset(Dataset):
    REAL_DIRS = {"real", "authentic", "genuine", "original"}
    FAKE_DIRS = {"fake", "deepfake", "manipulated", "synthetic", "generated"}
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_dir: str | Path, split: str = "train",
                 val_ratio: float = 0.15, seed: int = 42) -> None:
        self.data_dir = Path(data_dir)
        self.samples: list[tuple[Path, int]] = []
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
            for p in subdir.rglob("*"):
                if p.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((p, label))
        if not self.samples:
            raise ValueError(f"No images in {data_dir}")
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.samples)).tolist()
        n_val = int(len(idx) * val_ratio)
        self.samples = [self.samples[i] for i in (idx[:n_val] if split == "val" else idx[n_val:])]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, label


# ---------------------------------------------------------------------------
# CLIP Trainer
# ---------------------------------------------------------------------------

class CLIPFineTuner:
    """Fine-tune only LayerNorm parameters of CLIP ViT-L/14."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_clip(self):
        from transformers import CLIPModel, CLIPProcessor
        model_name = "openai/clip-vit-large-patch14"
        logger.info("Loading %s …", model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)

        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only LayerNorm
        n_trainable = 0
        for name, param in model.named_parameters():
            if "layernorm" in name.lower() or "layer_norm" in name.lower():
                param.requires_grad = True
                n_trainable += param.numel()

        total = sum(p.numel() for p in model.parameters())
        logger.info("Trainable params: %d / %d (%.3f%%)",
                    n_trainable, total, 100 * n_trainable / total)
        return model.to(self.device), processor

    def build_loaders(self, processor):
        def collate(batch):
            imgs, labels = zip(*batch)
            inputs = processor(images=list(imgs), return_tensors="pt", padding=True)
            return inputs, torch.tensor(labels, dtype=torch.long)

        train_ds = ImageDataset(self.args.data_dir, "train")
        val_ds = ImageDataset(self.args.data_dir, "val")
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size,
                                  shuffle=True, collate_fn=collate,
                                  num_workers=self.args.workers)
        val_loader = DataLoader(val_ds, batch_size=self.args.batch_size,
                                shuffle=False, collate_fn=collate,
                                num_workers=self.args.workers)
        return train_loader, val_loader

    def train(self) -> None:
        model, processor = self.load_clip()

        # Lightweight probe head on top of CLIP image features (768-dim)
        probe = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        ).to(self.device)

        train_loader, val_loader = self.build_loaders(processor)
        criterion = nn.CrossEntropyLoss()
        params = (list(filter(lambda p: p.requires_grad, model.parameters()))
                  + list(probe.parameters()))
        optimizer = optim.AdamW(params, lr=self.args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.epochs, eta_min=1e-7
        )

        best_auc = 0.0
        for epoch in range(1, self.args.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(model, probe, train_loader, criterion, optimizer)
            val_metrics = self._eval_epoch(model, probe, val_loader)
            scheduler.step()

            logger.info("Epoch %d/%d | loss=%.4f acc=%.4f auc=%.4f | %.1fs",
                        epoch, self.args.epochs, train_loss,
                        val_metrics["accuracy"], val_metrics["auc"],
                        time.time() - t0)

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                self._save(model, probe)
                logger.info("  *** Best AUC=%.4f saved ***", best_auc)

        logger.info("Done. Best AUC=%.4f", best_auc)

    def _train_epoch(self, model, probe, loader, criterion, optimizer) -> float:
        model.train()
        probe.train()
        total_loss = 0.0
        for inputs, labels in loader:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)
            optimizer.zero_grad()
            features = model.get_image_features(**inputs)
            # L2 normalize
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_epoch(self, model, probe, loader) -> dict:
        import torch.nn.functional as F
        from sklearn.metrics import accuracy_score, roc_auc_score
        model.eval()
        probe.eval()
        all_preds, all_labels, all_probs = [], [], []
        for inputs, labels in loader:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = model.get_image_features(**inputs)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            logits = probe(features)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        return {"accuracy": round(acc, 4), "auc": round(auc, 4)}

    def _save(self, model, probe) -> None:
        # Save only LayerNorm weights (tiny ~2MB)
        ln_state = {k: v for k, v in model.state_dict().items()
                    if "layernorm" in k.lower() or "layer_norm" in k.lower()}
        torch.save(ln_state, self.output_dir / "clip_layernorm.pth")
        torch.save(probe.state_dict(), self.output_dir / "clip_probe_head.pth")
        logger.info("Saved: clip_layernorm.pth (%d tensors) + clip_probe_head.pth",
                    len(ln_state))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="weights/")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s")
    CLIPFineTuner(parse_args()).train()
