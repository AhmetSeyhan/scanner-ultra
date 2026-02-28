"""Scanner ULTRA — Continual Learning Engine.

Learn new deepfake techniques without forgetting old ones.
Implements Elastic Weight Consolidation (EWC) + Experience Replay.

Solves the "catastrophic forgetting" problem in neural networks.

References:
- EWC: https://arxiv.org/abs/1612.00796
- Experience Replay: https://www.nature.com/articles/nature14236
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContinualLearner:
    """Continual learning with EWC and Experience Replay.

    Enables model to learn new tasks without forgetting previous tasks.
    Critical for adapting to new deepfake techniques over time.
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        replay_buffer_size: int = 5000,
        replay_batch_fraction: float = 0.2,
        device: str = "auto",
    ) -> None:
        """Initialize continual learner.

        Args:
            model: Model to train continually
            ewc_lambda: EWC regularization strength (higher = more protection)
            replay_buffer_size: Max samples in replay buffer
            replay_batch_fraction: Fraction of batch from replay (0.0-1.0)
            device: Training device
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.replay_buffer_size = replay_buffer_size
        self.replay_batch_fraction = replay_batch_fraction
        self.device = self._resolve_device(device)

        # Move model to device
        self.model.to(self.device)

        # EWC: Fisher Information Matrix and optimal parameters
        self.fisher_dict: dict[str, torch.Tensor] = {}
        self.optimal_params: dict[str, torch.Tensor] = {}
        self.ewc_initialized = False

        # Experience Replay: buffer of previous samples
        self.replay_buffer: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=replay_buffer_size)

        # Task tracking
        self.num_tasks_learned = 0

        logger.info(
            f"ContinualLearner initialized "
            f"(λ={ewc_lambda}, buffer_size={replay_buffer_size}, "
            f"replay_fraction={replay_batch_fraction})"
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve device string."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def compute_fisher_information(
        self,
        dataloader: Any,
        num_samples: int = 1000,
    ) -> None:
        """Compute Fisher Information Matrix for current task.

        The Fisher matrix captures parameter importance for the current task.
        Parameters with high Fisher values are protected during future learning.

        Args:
            dataloader: DataLoader for current task
            num_samples: Number of samples to estimate Fisher matrix
        """
        logger.info(f"Computing Fisher Information Matrix ({num_samples} samples)")

        self.model.eval()
        self.fisher_dict = {
            n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if p.requires_grad
        }

        samples_processed = 0

        for inputs, labels in dataloader:
            if samples_processed >= num_samples:
                break

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass to get gradients
            self.model.zero_grad()
            loss.backward()

            # Accumulate squared gradients (Fisher approximation)
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher_dict[n] += p.grad.data**2 / num_samples

            samples_processed += inputs.size(0)

        # Save current parameters as optimal for this task
        self.optimal_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

        self.ewc_initialized = True
        logger.info("Fisher Information Matrix computed")

    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss.

        Penalizes changes to important parameters (high Fisher values).

        Returns:
            EWC loss term
        """
        if not self.ewc_initialized:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for n, p in self.model.named_parameters():
            if n in self.fisher_dict:
                # Loss = λ * Σ F_i * (θ_i - θ*_i)^2
                # F_i: Fisher importance
                # θ_i: current param
                # θ*_i: optimal param from previous task
                loss += (self.fisher_dict[n] * (p - self.optimal_params[n]) ** 2).sum()

        return self.ewc_lambda * loss

    def add_to_replay_buffer(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Add samples to experience replay buffer.

        Args:
            inputs: Input batch
            labels: Label batch
        """
        # Add each sample individually
        for i in range(inputs.size(0)):
            self.replay_buffer.append(
                (
                    inputs[i].cpu().clone(),
                    labels[i].cpu().clone(),
                )
            )

    def sample_from_replay(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample batch from replay buffer.

        Args:
            batch_size: Number of samples to retrieve

        Returns:
            (inputs, labels) tuple or None if buffer empty
        """
        if len(self.replay_buffer) == 0:
            return None

        # Sample random indices
        indices = np.random.choice(
            len(self.replay_buffer),
            size=min(batch_size, len(self.replay_buffer)),
            replace=False,
        )

        # Gather samples
        inputs_list = []
        labels_list = []

        for idx in indices:
            inp, lbl = self.replay_buffer[idx]
            inputs_list.append(inp)
            labels_list.append(lbl)

        inputs = torch.stack(inputs_list).to(self.device)
        labels = torch.stack(labels_list).to(self.device)

        return inputs, labels

    def train_with_replay(
        self,
        dataloader: Any,
        epochs: int = 10,
        lr: float = 0.001,
    ) -> dict[str, Any]:
        """Train with experience replay and EWC regularization.

        Args:
            dataloader: DataLoader for new task
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Training statistics
        """
        logger.info(f"Training with continual learning (epochs={epochs}, lr={lr}, task={self.num_tasks_learned + 1})")

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        stats = {
            "loss": [],
            "task_loss": [],
            "ewc_loss": [],
            "accuracy": [],
        }

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_task_loss = 0.0
            epoch_ewc_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Determine replay batch size
                replay_size = int(inputs.size(0) * self.replay_batch_fraction)

                # Mix current batch with replay samples
                if replay_size > 0:
                    replay_batch = self.sample_from_replay(replay_size)
                    if replay_batch is not None:
                        replay_inputs, replay_labels = replay_batch
                        inputs = torch.cat([inputs, replay_inputs], dim=0)
                        labels = torch.cat([labels, replay_labels], dim=0)

                # Add current samples to replay buffer
                self.add_to_replay_buffer(inputs, labels)

                # Forward pass
                outputs = self.model(inputs)
                task_loss = F.cross_entropy(outputs, labels)

                # Add EWC regularization
                ewc_loss_term = self.ewc_loss()
                total_loss = task_loss + ewc_loss_term

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Stats
                epoch_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_ewc_loss += ewc_loss_term.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Epoch stats
            avg_loss = epoch_loss / len(dataloader)
            avg_task_loss = epoch_task_loss / len(dataloader)
            avg_ewc_loss = epoch_ewc_loss / len(dataloader)
            acc = 100.0 * correct / total

            stats["loss"].append(avg_loss)
            stats["task_loss"].append(avg_task_loss)
            stats["ewc_loss"].append(avg_ewc_loss)
            stats["accuracy"].append(acc)

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {avg_loss:.4f} (task: {avg_task_loss:.4f}, "
                    f"EWC: {avg_ewc_loss:.4f}), Acc: {acc:.2f}%"
                )

        logger.info(f"Continual learning completed (final acc: {stats['accuracy'][-1]:.2f}%)")

        return stats

    def consolidate_task(self, dataloader: Any, num_samples: int = 1000) -> None:
        """Consolidate current task (compute Fisher, update optimal params).

        Call this after training on a new task to protect it from forgetting.

        Args:
            dataloader: DataLoader for task to consolidate
            num_samples: Samples for Fisher computation
        """
        logger.info(f"Consolidating task {self.num_tasks_learned + 1}")

        self.compute_fisher_information(dataloader, num_samples)
        self.num_tasks_learned += 1

        logger.info(
            f"Task consolidated "
            f"(total tasks: {self.num_tasks_learned}, "
            f"replay buffer: {len(self.replay_buffer)} samples)"
        )

    def get_learner_info(self) -> dict[str, Any]:
        """Get continual learner information.

        Returns:
            Learner metadata
        """
        return {
            "num_tasks_learned": self.num_tasks_learned,
            "ewc_initialized": self.ewc_initialized,
            "ewc_lambda": self.ewc_lambda,
            "replay_buffer_size": self.replay_buffer_size,
            "replay_buffer_count": len(self.replay_buffer),
            "replay_batch_fraction": self.replay_batch_fraction,
            "device": self.device,
        }

    def save_state(self, path: str) -> None:
        """Save continual learner state.

        Args:
            path: Path to save state
        """
        state = {
            "model_state": self.model.state_dict(),
            "fisher_dict": self.fisher_dict,
            "optimal_params": self.optimal_params,
            "replay_buffer": list(self.replay_buffer),
            "num_tasks_learned": self.num_tasks_learned,
            "ewc_initialized": self.ewc_initialized,
        }
        torch.save(state, path)
        logger.info(f"Continual learner state saved to {path}")

    def load_state(self, path: str) -> None:
        """Load continual learner state.

        Args:
            path: Path to load state from
        """
        state = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state["model_state"])
        self.fisher_dict = state["fisher_dict"]
        self.optimal_params = state["optimal_params"]
        self.replay_buffer = deque(state["replay_buffer"], maxlen=self.replay_buffer_size)
        self.num_tasks_learned = state["num_tasks_learned"]
        self.ewc_initialized = state["ewc_initialized"]

        logger.info(f"Continual learner state loaded from {path} ({self.num_tasks_learned} tasks learned)")
