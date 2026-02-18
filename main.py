#!/usr/bin/env python3
"""Main entry for MvHo-IB++ training pipeline.

Project layout:
- config.yaml
- src/data, src/models, src/trainer, src/utils

Model setup:
- three-view pipeline (GIN + 3D CNN + 4D CNN)
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add src directory for imports
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from src.data import create_data_loaders, load_precomputed_data
from src.trainer import Trainer
from src.utils import load_config, validate_config


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MvHo-IB++ training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()


def setup_device(gpu_id: int = None) -> torch.device:
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    setup_logging()

    config = load_config(args.config)
    validate_config(config)

    seed = int(config.get("experiment", {}).get("seed", 42))
    setup_seed(seed)

    device = setup_device(args.gpu)
    logging.info("Using device: %s", device)

    train_split, val_split, test_split = load_precomputed_data(config)
    batch_size = int(config["training"]["batch_size"])
    train_loader, val_loader, test_loader = create_data_loaders(
        train_split, val_split, test_split, batch_size=batch_size
    )

    trainer = Trainer(config, device)

    sample_graph, sample_x3d, sample_x4d, _ = next(iter(train_loader))
    sample_x3d = sample_x3d.unsqueeze(1)
    sample_x4d = sample_x4d.unsqueeze(1) if sample_x4d is not None else None
    trainer.setup_models(sample_graph.to(device), sample_x3d.to(device), sample_x4d.to(device) if sample_x4d is not None else None)

    trainer.train_model(train_loader, val_loader)
    results = trainer.test_model(test_loader)
    trainer.save_results(results)


if __name__ == "__main__":
    main()
