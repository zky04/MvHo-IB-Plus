"""Configuration utilities for loading, validating, and saving YAML settings."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Project root (src/utils -> project root).
ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to a config file. Defaults to project-root `config.yaml`.

    Returns:
        Parsed configuration dictionary. Returns an empty dict on failure.
    """
    if config_path is None:
        config_path = ROOT / "config.yaml"
    config_file = Path(config_path)
    if not config_file.exists():
        logging.warning(f"Config file not found: {config_path}")
        return {}
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded config: {config_path}")
        return config if config is not None else {}
    except yaml.YAMLError as e:
        logging.error(f"YAML parse error: {e}")
        return {}


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration with support for the current and legacy layouts."""
    if not config:
        raise ValueError("empty config")

    if "dataset_name" in config:
        required_sections = ["dataset_name", "datasets", "training", "information_bottleneck"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"missing required section: {section}")
        dataset_name = config["dataset_name"]
        if dataset_name not in config["datasets"]:
            raise ValueError(f"dataset '{dataset_name}' not found in datasets")
        ds = config["datasets"][dataset_name]
        for key in ["x1_path", "x2_path", "num_classes", "num_regions"]:
            if key not in ds:
                raise ValueError(f"missing dataset field: {key}")
    else:
        for section in ["dataset", "compute", "model", "training"]:
            if section not in config:
                logging.warning(f"missing legacy section: {section}")

    logging.info("configuration validation passed")
    return True


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Write configuration to a YAML file."""
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with open(save_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
    logging.info(f"Saved config: {save_path}")


def update_config_paths(
    config: Dict[str, Any],
    base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve relative paths in config to absolute paths."""
    base = Path(base_path) if base_path else ROOT
    datasets = config.get("datasets", {})
    for name, ds in datasets.items():
        if not isinstance(ds, dict):
            continue
        for key in ["x1_path", "x2_path", "x3_path", "data_path", "root"]:
            if key in ds and ds[key]:
                p = Path(ds[key])
                if not p.is_absolute():
                    config["datasets"][name][key] = str((base / p).resolve())
    dataset = config.get("dataset", {})
    if isinstance(dataset, dict):
        for key in ["path", "root", "data_dir"]:
            if key in dataset and dataset[key]:
                p = Path(dataset[key])
                if not p.is_absolute():
                    config["dataset"][key] = str((base / p).resolve())
    return config
