"""Scanner ULTRA — Federated Learning Server.

Central server for federated learning. Aggregates model updates from clients
using FedAvg (Federated Averaging) algorithm.

Server NEVER sees raw training data - only model updates.

References:
- FedAvg: https://arxiv.org/abs/1602.05629
- Secure Aggregation: https://arxiv.org/abs/1611.04482
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FederatedServer:
    """Federated learning server for secure model aggregation.

    The server:
    1. Maintains global model
    2. Distributes global model to clients
    3. Receives model updates from clients
    4. Aggregates updates using FedAvg
    5. Updates global model
    """

    def __init__(
        self,
        model: nn.Module,
        aggregation_strategy: str = "fedavg",
        min_clients: int = 2,
        device: str = "cpu",
    ) -> None:
        """Initialize federated server.

        Args:
            model: Global model instance
            aggregation_strategy: Aggregation method (fedavg, fedprox)
            min_clients: Minimum number of clients required for aggregation
            device: Server device
        """
        self.model = model.to(device)
        self.device = device
        self.aggregation_strategy = aggregation_strategy
        self.min_clients = min_clients

        # Track rounds and clients
        self.global_round = 0
        self.client_updates: dict[str, dict[str, torch.Tensor]] = {}
        self.client_weights: dict[str, float] = {}  # Sample weights for weighted averaging
        self.round_history: list[dict[str, Any]] = []

        logger.info(f"FederatedServer initialized (strategy={aggregation_strategy}, min_clients={min_clients})")

    def get_global_model(self) -> dict[str, torch.Tensor]:
        """Get current global model state.

        Returns:
            Global model state dict
        """
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def receive_update(
        self,
        client_id: str,
        update: dict[str, torch.Tensor],
        num_samples: int,
    ) -> None:
        """Receive model update from a client.

        Args:
            client_id: Client identifier
            update: Model update (delta parameters)
            num_samples: Number of samples client trained on
        """
        self.client_updates[client_id] = update
        self.client_weights[client_id] = float(num_samples)

        logger.info(
            f"Received update from {client_id} "
            f"({num_samples} samples, {len(self.client_updates)}/{self.min_clients} clients)"
        )

    def aggregate_updates(self) -> bool:
        """Aggregate client updates and update global model.

        Uses FedAvg: weighted average of client updates based on sample counts.

        Returns:
            True if aggregation successful, False if insufficient clients
        """
        if len(self.client_updates) < self.min_clients:
            logger.warning(f"Insufficient clients for aggregation ({len(self.client_updates)}/{self.min_clients})")
            return False

        logger.info(f"Aggregating updates from {len(self.client_updates)} clients (round {self.global_round + 1})")

        if self.aggregation_strategy == "fedavg":
            self._aggregate_fedavg()
        elif self.aggregation_strategy == "fedprox":
            self._aggregate_fedprox()
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")

        # Record round history
        self._record_round()

        # Clear updates for next round
        self.client_updates.clear()
        self.client_weights.clear()
        self.global_round += 1

        logger.info(f"Global model updated (round {self.global_round})")
        return True

    def _aggregate_fedavg(self) -> None:
        """Aggregate using FedAvg (weighted average by sample count)."""
        global_state = self.model.state_dict()

        # Compute total samples
        total_samples = sum(self.client_weights.values())

        # Weighted average of updates
        aggregated_update: dict[str, torch.Tensor] = {}

        for client_id, update in self.client_updates.items():
            weight = self.client_weights[client_id] / total_samples

            for key, delta in update.items():
                delta = delta.to(self.device)

                if key not in aggregated_update:
                    aggregated_update[key] = weight * delta
                else:
                    aggregated_update[key] += weight * delta

        # Apply aggregated update to global model
        for key in global_state.keys():
            if key in aggregated_update:
                global_state[key] += aggregated_update[key]

        self.model.load_state_dict(global_state)

    def _aggregate_fedprox(self) -> None:
        """Aggregate using FedProx (FedAvg with proximal term).

        FedProx adds a proximal term to handle non-IID data better.
        For now, we use FedAvg as the base implementation.
        """
        # FedProx implementation similar to FedAvg
        # with additional proximal term regularization
        # For simplicity, using FedAvg here
        self._aggregate_fedavg()

    def _record_round(self) -> None:
        """Record round statistics."""
        round_stats = {
            "round": self.global_round + 1,
            "num_clients": len(self.client_updates),
            "total_samples": sum(self.client_weights.values()),
            "participating_clients": list(self.client_updates.keys()),
        }
        self.round_history.append(round_stats)

    def get_server_info(self) -> dict[str, Any]:
        """Get server information.

        Returns:
            Server metadata
        """
        return {
            "global_round": self.global_round,
            "aggregation_strategy": self.aggregation_strategy,
            "min_clients": self.min_clients,
            "pending_updates": len(self.client_updates),
            "total_rounds": len(self.round_history),
            "device": self.device,
        }

    def get_round_history(self) -> list[dict[str, Any]]:
        """Get aggregation round history.

        Returns:
            List of round statistics
        """
        return self.round_history

    def save_global_model(self, path: str) -> None:
        """Save global model to disk.

        Args:
            path: Path to save model
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Global model saved to {path} (round {self.global_round})")

    def load_global_model(self, path: str) -> None:
        """Load global model from disk.

        Args:
            path: Path to load model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Global model loaded from {path}")

    def evaluate_global_model(self, test_data: Any) -> dict[str, float]:
        """Evaluate global model on test data.

        Args:
            test_data: Test dataset

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        loss = 0.0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in test_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss += criterion(outputs, labels).item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = loss / len(test_data)

        logger.info(f"Global model evaluation (round {self.global_round}) - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

        return {
            "accuracy": round(accuracy, 2),
            "loss": round(avg_loss, 4),
            "samples": total,
        }

    def reset_round(self) -> None:
        """Reset current round (clear pending updates)."""
        self.client_updates.clear()
        self.client_weights.clear()
        logger.info("Round reset - cleared pending updates")
