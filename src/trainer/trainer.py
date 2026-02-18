"""Three-view trainer (GIN + Brain3DCNN + Brain4DCNN)."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from ..models import Brain3DCNN, Brain4DCNN, FusionModel, GINModel
from ..utils.info_bottleneck import three_view_ib_loss


class Trainer:
    """Trainer aligned with MvHo-IB pipeline, extended to 3 views."""

    def __init__(self, config: Dict, device: torch.device) -> None:
        self.config = config
        self.device = device

        self.train_cfg = config["training"]
        self.ib_cfg = config["information_bottleneck"]
        self.ablation_cfg = config.get("ablation", {})

        ds_name = config["dataset_name"]
        self.num_classes = int(config["datasets"][ds_name]["num_classes"])

        self.use_gin = self.ablation_cfg.get("use_gin", True)
        self.use_3d = self.ablation_cfg.get("use_brain3dcnn", True)
        self.use_4d = self.ablation_cfg.get("use_brain4dcnn", True)

        self.gin_model: Optional[GINModel] = None
        self.cnn3d_model: Optional[Brain3DCNN] = None
        self.cnn4d_model: Optional[Brain4DCNN] = None
        self.fusion_model: Optional[FusionModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.CrossEntropyLoss()

    def setup_models(self, sample_graph_batch, sample_x3d: torch.Tensor, sample_x4d: Optional[torch.Tensor]) -> None:
        fusion_input_size = 0

        if self.use_gin:
            self.gin_model = GINModel(
                num_features=sample_graph_batch.x.shape[1],
                embedding_dim=self.train_cfg["gin_embedding_dim"],
                hidden_dims=self.train_cfg["gin_hidden_dims"],
                dropout_rate=self.train_cfg["dropout_rate"],
            ).to(self.device)
            fusion_input_size += int(self.train_cfg["gin_embedding_dim"])

        if self.use_3d:
            self.cnn3d_model = Brain3DCNN(
                example_tensor=sample_x3d,
                embedding_dim=self.train_cfg["cnn3d_embedding_dim"],
                channels=tuple(self.train_cfg["cnn3d_channels"]),
                dropout_rate=self.train_cfg["dropout_rate"],
            ).to(self.device)
            fusion_input_size += int(self.train_cfg["cnn3d_embedding_dim"])

        if self.use_4d and sample_x4d is not None:
            self.cnn4d_model = Brain4DCNN(
                example_4d=sample_x4d,
                embedding_dim=self.train_cfg["cnn4d_embedding_dim"],
                channels=tuple(self.train_cfg["cnn4d_channels"]),
                dropout_rate=self.train_cfg["dropout_rate"],
            ).to(self.device)
            fusion_input_size += int(self.train_cfg["cnn4d_embedding_dim"])

        if fusion_input_size <= 0:
            raise ValueError("No active view encoder")

        self.fusion_model = FusionModel(
            input_size=fusion_input_size,
            num_classes=self.num_classes,
            dropout_rate=self.train_cfg["dropout_rate"],
        ).to(self.device)

        params = list(self.fusion_model.parameters())
        for m in [self.gin_model, self.cnn3d_model, self.cnn4d_model]:
            if m is not None:
                params.extend(m.parameters())
        self.optimizer = optim.Adam(
            params,
            lr=float(self.train_cfg["learning_rate"]),
            weight_decay=float(self.train_cfg["weight_decay"]),
        )
        logging.info("Models initialized")

    def _forward_batch(self, graph_batch, x3d, x4d):
        z_gin = self.gin_model(graph_batch) if self.gin_model is not None else None
        z_3d = self.cnn3d_model(x3d) if self.cnn3d_model is not None else None
        z_4d = self.cnn4d_model(x4d) if (self.cnn4d_model is not None and x4d is not None) else None
        fused, logits = self.fusion_model(z_gin, z_3d, z_4d)
        return z_gin, z_3d, z_4d, fused, logits

    def _run_epoch(self, loader, train: bool) -> Tuple[float, float]:
        if self.fusion_model is None:
            raise RuntimeError("call setup_models first")

        for m in [self.gin_model, self.cnn3d_model, self.cnn4d_model, self.fusion_model]:
            if m is None:
                continue
            m.train(mode=train)

        total_loss = 0.0
        y_true, y_pred = [], []

        for graph_batch, x3d, x4d, labels in loader:
            graph_batch = graph_batch.to(self.device)
            x3d = x3d.unsqueeze(1).to(self.device)
            x4d = x4d.unsqueeze(1).to(self.device) if x4d is not None else None
            labels = labels.to(self.device)

            if train:
                self.optimizer.zero_grad()

            z_gin, z_3d, z_4d, _, logits = self._forward_batch(graph_batch, x3d, x4d)
            loss = self.criterion(logits, labels)

            if self.ib_cfg.get("use_ib", True):
                loss = loss + three_view_ib_loss(
                    z_gin, z_3d, z_4d,
                    beta_gin=float(self.ib_cfg.get("beta_gin", 0.0)),
                    beta_3d=float(self.ib_cfg.get("beta_3d", 0.0)),
                    beta_4d=float(self.ib_cfg.get("beta_4d", 0.0)),
                    beta_mutual=float(self.ib_cfg.get("beta_mutual", 0.0)),
                    sigma=float(self.ib_cfg.get("sigma", 5.0)),
                    alpha=float(self.ib_cfg.get("alpha", 1.01)),
                )

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += float(loss.item())
            pred = torch.argmax(logits, dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

        avg_loss = total_loss / max(1, len(loader))
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        return avg_loss, acc

    def train_model(self, train_loader, val_loader):
        best_val = float("inf")
        patience = int(self.train_cfg.get("patience", 30))
        bad_epochs = 0
        epochs = int(self.train_cfg["num_epochs"])

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            logging.info(
                "Epoch %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch, epochs, train_loss, train_acc, val_loss, val_acc,
            )

            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    logging.info("Early stopping at epoch %d", epoch)
                    break

    def test_model(self, test_loader) -> Dict[str, float]:
        loss, acc = self._run_epoch(test_loader, train=False)
        return {"test_loss": loss, "test_accuracy": acc}

    def save_results(self, results: Dict[str, float]) -> None:
        logging.info("Test results: %s", results)
