"""PyTorch training and evaluation engine for CS2 C0 neural models.

Implements:
1. C0Trainer:
   - AdamW optimizer with CosineAnnealingLR scheduling.
   - Pairwise anti-symmetric loss formulation (symmetric binary cross entropy):
     L = 0.5 * (BCE(logits, y) + BCE(-logits, 1 - y))
     canceling any floating-point roundoff asymmetries.
   - Dynamic batch ingestion supporting dictionary and tuple collated batches.
   - Gradient clipping and NaN-safe parameter updates.
   - Comprehensive validation evaluation with BCE loss, Brier score, and accuracy.
   - Epoch training loop with early stopping patience and best-checkpoint saving/loading.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

__all__ = ["C0Trainer"]


class C0Trainer:
    """PyTorch training engine for Counter-Strike 2 C0 neural models.

    Integrates:
    - AdamW optimizer and Cosine Annealing learning rate schedule.
    - Symmetrized binary cross-entropy loss matching the model's anti-symmetric scoring head.
    - Gradient clipping to prevent gradient explosion.
    - Metrics computation: validation loss, Brier score, and classification accuracy.
    - Early stopping with best checkpoint persistence.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device | str | None = None,
        scheduler: _LRScheduler | Any | None = None,
        symmetric_loss: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize C0Trainer.

        Args:
            model: PyTorch neural model (e.g. CS2C0Model).
            optimizer: Optional pre-configured optimizer. If None, AdamW is initialized.
            lr: Learning rate for AdamW (default 1e-3).
            weight_decay: Weight decay for AdamW (default 1e-4).
            device: Optional PyTorch device or device string. Defaults to CUDA if available, else CPU.
            scheduler: Optional learning rate scheduler. If None, CosineAnnealingLR is initialized.
            symmetric_loss: Whether to use symmetrized BCE loss enforcing strict anti-symmetry.
            **kwargs: Extra parameters (e.g. T_max, eta_min for CosineAnnealingLR).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.symmetric_loss = bool(symmetric_loss)

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        self.criterion = nn.BCEWithLogitsLoss()

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            t_max = kwargs.get("T_max", kwargs.get("t_max", 20))
            eta_min = kwargs.get("eta_min", 1e-6)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(t_max)),
                eta_min=float(eta_min),
            )

    def _unpack_batch(
        self,
        batch: Any,
        require_label: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Extract history, mask, and labels from arbitrary DataLoader batches.

        Supports:
        - dict with keys 'history', optional 'mask', and 'label'/'labels'/'target'/'y'
        - tuple/list of (history, mask, label)
        - tuple/list of (history, label)
        - tuple/list of (dict_input, label)
        - bare history tensor (when require_label is False)

        Args:
            batch: Collated batch from DataLoader.
            require_label: If True, raises KeyError when no target label is found.

        Returns:
            Tuple of (history_tensor, mask_tensor_or_None, labels_tensor_or_None).
        """
        history: torch.Tensor
        mask: torch.Tensor | None = None
        labels: torch.Tensor | None = None

        if isinstance(batch, Mapping):
            if "history" not in batch:
                raise KeyError(f"Expected 'history' key in batch mapping, found: {list(batch.keys())}")
            history = batch["history"].to(self.device)
            if "mask" in batch and batch["mask"] is not None:
                mask = batch["mask"].to(self.device)

            label_candidate = None
            for key in ("label", "labels", "target", "targets", "y"):
                if key in batch and batch[key] is not None:
                    label_candidate = batch[key]
                    break

            if label_candidate is not None:
                labels = label_candidate.to(self.device, dtype=torch.float32)
            elif require_label:
                raise KeyError("Batch mapping must contain one of: 'label', 'labels', 'target', 'targets', 'y'")

        elif isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                h, m, y = batch
                history = h.to(self.device)
                mask = m.to(self.device) if m is not None else None
                labels = y.to(self.device, dtype=torch.float32) if y is not None else None
            elif len(batch) == 2:
                first, second = batch
                if isinstance(first, Mapping):
                    history = first["history"].to(self.device)
                    mask = first["mask"].to(self.device) if "mask" in first and first["mask"] is not None else None
                    labels = second.to(self.device, dtype=torch.float32) if second is not None else None
                else:
                    history = first.to(self.device)
                    mask = None
                    labels = second.to(self.device, dtype=torch.float32) if second is not None else None
            elif len(batch) == 1:
                return self._unpack_batch(batch[0], require_label=require_label)
            else:
                raise ValueError(f"Unexpected tuple/list batch length: {len(batch)}")
            if require_label and labels is None:
                raise ValueError("Target label tensor is required but was None in batch.")

        elif isinstance(batch, torch.Tensor):
            history = batch.to(self.device)
            if require_label:
                raise ValueError("Received bare Tensor batch with require_label=True.")
        else:
            raise TypeError(f"Unsupported batch container type: {type(batch)}")

        return history, mask, labels

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute (optionally symmetrized) binary cross-entropy loss with logits.

        For CS2 pairwise match forecasting, the anti-symmetric scoring head
        guarantees logit(A, B) = -logit(B, A). The symmetric loss evaluates:
            L = 0.5 * (BCE(logits, y) + BCE(-logits, 1 - y))
        which guarantees exact gradient anti-symmetry and cancels any floating-point
        asymmetry around zero.

        Args:
            logits: Predicted logits from model, shape (B, 1) or (B,).
            labels: Ground truth binary targets in {0, 1}.

        Returns:
            Scalar loss tensor.
        """
        targets = labels.reshape_as(logits).float()
        if self.symmetric_loss:
            loss_direct = self.criterion(logits, targets)
            loss_flipped = self.criterion(-logits, 1.0 - targets)
            return 0.5 * (loss_direct + loss_flipped)
        return self.criterion(logits, targets)

    def train_epoch(
        self,
        train_loader: Iterable[Any],
        max_grad_norm: float = 1.0,
    ) -> dict[str, float]:
        """Train the model for a single epoch.

        Args:
            train_loader: Iterable yielding training batches.
            max_grad_norm: Maximum gradient norm for clipping (default 1.0).

        Returns:
            Dictionary with 'train_loss' and 'train_accuracy'.
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for batch in train_loader:
            history, mask, labels = self._unpack_batch(batch, require_label=True)
            if labels is None:
                raise ValueError("Batch did not contain target labels for training.")

            batch_size = labels.shape[0] if labels.ndim > 0 else 1

            self.optimizer.zero_grad()

            if hasattr(self.model, "compute_logits"):
                logits = self.model.compute_logits(history, mask)
            else:
                out = self.model(history, mask)
                if isinstance(out, tuple):
                    logits = out[1] if len(out) > 1 else out[0]
                else:
                    logits = out

            loss = self.compute_loss(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered in training: {loss.item()}")

            loss.backward()

            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_grad_norm
                )

            self.optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                targets = (labels.reshape_as(logits) >= 0.5).float()
                correct = (preds == targets).float().sum().item()

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

        if total_samples == 0:
            return {"train_loss": 0.0, "train_accuracy": 0.0}

        mean_loss = float(total_loss / total_samples)
        mean_acc = float(total_correct / total_samples)

        return {
            "train_loss": mean_loss,
            "train_accuracy": mean_acc,
            "loss": mean_loss,
            "accuracy": mean_acc,
        }

    def evaluate(self, eval_loader: Iterable[Any]) -> dict[str, float]:
        """Evaluate the model on an evaluation or test loader.

        Computes:
        - BCE validation loss
        - Brier score: (1/N) * sum((p_i - y_i)^2)
        - Classification accuracy: (1/N) * sum(I[p_i >= 0.5] == y_i)

        Args:
            eval_loader: Iterable yielding validation batches.

        Returns:
            Dictionary with 'val_loss', 'val_brier', and 'val_accuracy'.
        """
        self.model.eval()
        total_loss = 0.0
        total_brier = 0.0
        total_correct = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                history, mask, labels = self._unpack_batch(batch, require_label=True)
                if labels is None:
                    raise ValueError("Batch did not contain target labels for evaluation.")

                batch_size = labels.shape[0] if labels.ndim > 0 else 1

                if hasattr(self.model, "compute_logits"):
                    logits = self.model.compute_logits(history, mask)
                else:
                    out = self.model(history, mask)
                    if isinstance(out, tuple):
                        logits = out[1] if len(out) > 1 else out[0]
                    else:
                        logits = out

                loss = self.compute_loss(logits, labels)

                probs = torch.sigmoid(logits)
                targets = labels.reshape_as(logits).float()
                brier = ((probs - targets) ** 2).sum().item()

                preds = (probs >= 0.5).float()
                target_bool = (targets >= 0.5).float()
                correct = (preds == target_bool).float().sum().item()

                total_loss += loss.item() * batch_size
                total_brier += brier
                total_correct += correct
                total_samples += batch_size

        if total_samples == 0:
            return {
                "val_loss": 0.0,
                "val_brier": 0.0,
                "val_accuracy": 0.0,
                "loss": 0.0,
                "brier": 0.0,
                "brier_score": 0.0,
                "accuracy": 0.0,
            }

        val_loss = float(total_loss / total_samples)
        val_brier = float(total_brier / total_samples)
        val_accuracy = float(total_correct / total_samples)

        return {
            "val_loss": val_loss,
            "val_brier": val_brier,
            "val_accuracy": val_accuracy,
            "loss": val_loss,
            "brier": val_brier,
            "brier_score": val_brier,
            "accuracy": val_accuracy,
        }

    def fit(
        self,
        train_loader: Iterable[Any],
        val_loader: Iterable[Any] | None = None,
        epochs: int = 20,
        patience: int = 5,
        checkpoint_path: str | Path | None = None,
        max_grad_norm: float = 1.0,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train model across epochs with validation tracking and early stopping.

        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            epochs: Total number of training epochs (default 20).
            patience: Early stopping patience epochs on validation loss (default 5).
            checkpoint_path: Optional path to save best model checkpoint weights.
            max_grad_norm: Maximum gradient norm for clipping.
            verbose: If True, prints epoch progress.

        Returns:
            Dictionary containing lists of 'train_loss', 'val_loss', 'val_brier', 'val_accuracy'.
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_brier": [],
            "val_accuracy": [],
        }

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_model_weights: dict[str, torch.Tensor] | None = None

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, max_grad_norm=max_grad_norm)
            history["train_loss"].append(train_metrics["train_loss"])
            history["train_accuracy"].append(train_metrics["train_accuracy"])

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                current_loss = val_metrics["val_loss"]
                history["val_loss"].append(current_loss)
                history["val_brier"].append(val_metrics["val_brier"])
                history["val_accuracy"].append(val_metrics["val_accuracy"])

                if verbose:
                    print(
                        f"Epoch {epoch:02d}/{epochs:02d} - "
                        f"train_loss: {train_metrics['train_loss']:.4f} - "
                        f"train_acc: {train_metrics['train_accuracy']:.4f} - "
                        f"val_loss: {current_loss:.4f} - "
                        f"val_brier: {val_metrics['val_brier']:.4f} - "
                        f"val_acc: {val_metrics['val_accuracy']:.4f}"
                    )

                if current_loss < best_val_loss:
                    best_val_loss = current_loss
                    best_epoch = epoch
                    patience_counter = 0
                    best_model_weights = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    if checkpoint_path is not None:
                        self.save_checkpoint(
                            checkpoint_path,
                            epoch=epoch,
                            val_loss=best_val_loss,
                            metrics=val_metrics,
                        )
                else:
                    patience_counter += 1
                    if patience is not None and patience > 0 and patience_counter >= patience:
                        if verbose:
                            print(
                                f"Early stopping at epoch {epoch}: "
                                f"no val_loss improvement for {patience} epochs "
                                f"(best val_loss: {best_val_loss:.4f} at epoch {best_epoch})."
                            )
                        break
            else:
                current_loss = train_metrics["train_loss"]
                if verbose:
                    print(
                        f"Epoch {epoch:02d}/{epochs:02d} - "
                        f"train_loss: {current_loss:.4f} - "
                        f"train_acc: {train_metrics['train_accuracy']:.4f}"
                    )
                if current_loss < best_val_loss:
                    best_val_loss = current_loss
                    best_epoch = epoch
                    best_model_weights = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    if checkpoint_path is not None:
                        self.save_checkpoint(
                            checkpoint_path,
                            epoch=epoch,
                            val_loss=best_val_loss,
                        )

            if self.scheduler is not None:
                self.scheduler.step()

        # Restore model to best weights observed
        if best_model_weights is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_model_weights.items()}
            )

        return history

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        epoch: int | None = None,
        val_loss: float | None = None,
        **extra_metadata: Any,
    ) -> Path:
        """Save model checkpoint weights and trainer state.

        Args:
            checkpoint_path: Target path on filesystem.
            epoch: Optional current epoch number.
            val_loss: Optional best validation loss.
            extra_metadata: Additional metadata to serialize.

        Returns:
            Path to saved file.
        """
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "val_loss": val_loss,
            "metadata": extra_metadata,
        }
        torch.save(state, path)
        return path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load saved checkpoint weights into self.model.

        Args:
            checkpoint_path: Path to checkpoint file.
            strict: Whether to strictly enforce state_dict matching.

        Returns:
            Loaded checkpoint dictionary.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            if self.optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception:
                    pass
            if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception:
                    pass
            return dict(checkpoint)
        elif isinstance(checkpoint, Mapping):
            self.model.load_state_dict(checkpoint, strict=strict)
            return {"model_state_dict": checkpoint}
        else:
            raise ValueError(f"Unrecognized checkpoint object type: {type(checkpoint)}")

    def predict_proba(self, loader: Iterable[Any]) -> torch.Tensor:
        """Compute win probabilities across an evaluation loader.

        Args:
            loader: Iterable yielding match batches.

        Returns:
            Tensor of probabilities of shape (N, 1) in [0, 1].
        """
        self.model.eval()
        probabilities: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                history, mask, _ = self._unpack_batch(batch, require_label=False)
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(history, mask)
                elif hasattr(self.model, "compute_logits"):
                    logits = self.model.compute_logits(history, mask)
                    probs = torch.sigmoid(logits)
                else:
                    out = self.model(history, mask)
                    probs = torch.sigmoid(out[1] if isinstance(out, tuple) else out)

                probabilities.append(probs.cpu())

        if not probabilities:
            return torch.empty((0, 1), dtype=torch.float32)

        return torch.cat(probabilities, dim=0)
