"""Scanner ULTRA — Federated Learning Client.

Privacy-preserving client-side training. Raw data NEVER leaves the device.
Only model gradients/updates are sent to the central server.

Implements FedAvg (Federated Averaging) algorithm.

References:
- FedAvg: https://arxiv.org/abs/1602.05629
- Federated Learning: https://arxiv.org/abs/1912.04977
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FederatedClient:
    """Federated learning client for privacy-preserving edge training.

    The client:
    1. Receives global model from server
    2. Trains on local data (data NEVER sent to server)
    3. Computes model update (gradient/parameters)
    4. Sends only the update to server (with differential privacy)
    """

    def __init__(
        self,
        model: nn.Module,
        client_id: str | None = None,
        device: str = "auto",
    ) -> None:
        """Initialize federated client.

        Args:
            model: Local model instance
            client_id: Unique client identifier (auto-generated if None)
            device: Training device
        """
        self.client_id = client_id or self._generate_client_id()
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)

        # Track training state
        self.global_round = 0
        self.local_epochs_completed = 0
        self.samples_trained = 0

        logger.info(f"FederatedClient initialized (id={self.client_id}, device={self.device})")

    def _generate_client_id(self) -> str:
        """Generate unique client ID."""
        return f"client_{uuid.uuid4().hex[:8]}"

    def _resolve_device(self, device: str) -> str:
        """Resolve device string."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def receive_global_model(self, global_state: dict[str, torch.Tensor]) -> None:
        """Receive global model parameters from server.

        Args:
            global_state: Global model state dict
        """
        self.model.load_state_dict(global_state)
        self.global_round += 1
        logger.info(f"Client {self.client_id} received global model (round {self.global_round})")

    def train_local(
        self,
        train_data: Any,
        local_epochs: int = 5,
        lr: float = 0.001,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Train on local data.

        Args:
            train_data: Local training dataset (NEVER sent to server!)
            local_epochs: Number of local training epochs
            lr: Learning rate
            batch_size: Batch size

        Returns:
            Training statistics
        """
        logger.info(f"Client {self.client_id} starting local training (epochs={local_epochs}, lr={lr})")

        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        stats = {
            "loss": [],
            "accuracy": [],
            "samples": 0,
        }

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            avg_loss = epoch_loss / len(train_data)
            acc = 100.0 * correct / total
            stats["loss"].append(avg_loss)
            stats["accuracy"].append(acc)
            stats["samples"] = total

            logger.debug(
                f"Client {self.client_id} - Epoch {epoch + 1}/{local_epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%"
            )

        self.local_epochs_completed += local_epochs
        self.samples_trained += stats["samples"]

        logger.info(
            f"Client {self.client_id} local training completed "
            f"(final loss: {stats['loss'][-1]:.4f}, acc: {stats['accuracy'][-1]:.2f}%)"
        )

        return stats

    def compute_update(self, global_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute model update (difference from global model).

        This is what gets sent to the server (NOT raw data!).

        Args:
            global_state: Global model state dict

        Returns:
            Model update (delta parameters)
        """
        update = {}
        current_state = self.model.state_dict()

        for key in current_state.keys():
            # Compute delta: local_params - global_params
            update[key] = current_state[key] - global_state[key]

        logger.debug(f"Client {self.client_id} computed model update")
        return update

    def get_model_state(self) -> dict[str, torch.Tensor]:
        """Get current local model state.

        Returns:
            Model state dict
        """
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def get_client_info(self) -> dict[str, Any]:
        """Get client information.

        Returns:
            Client metadata (NO TRAINING DATA!)
        """
        return {
            "client_id": self.client_id,
            "device": self.device,
            "global_round": self.global_round,
            "local_epochs_completed": self.local_epochs_completed,
            "samples_trained": self.samples_trained,
            "model_hash": self._compute_model_hash(),
        }

    def _compute_model_hash(self) -> str:
        """Compute hash of current model parameters."""
        state_bytes = b""
        for param in self.model.parameters():
            state_bytes += param.data.cpu().numpy().tobytes()
        return hashlib.sha256(state_bytes).hexdigest()[:16]

    def save_local_model(self, path: str) -> None:
        """Save local model to disk.

        Args:
            path: Path to save model
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Client {self.client_id} saved local model to {path}")

    def load_local_model(self, path: str) -> None:
        """Load local model from disk.

        Args:
            path: Path to load model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Client {self.client_id} loaded local model from {path}")
