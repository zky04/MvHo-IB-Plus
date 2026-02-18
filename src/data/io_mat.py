"""Load time-series matrices and subject file lists from `.mat` files."""

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def load_timecourses_from_mat(mat_path: Path) -> np.ndarray:
    """Load time-series data from a `.mat` file and return shape `(T, N)`."""
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("Scipy is required. Install with: pip install scipy")
    data = sio.loadmat(str(mat_path))
    for key in ["TC", "timecourse", "timeseries", "data", "ts"]:
        if key not in data:
            continue
        arr = data[key]
        if not hasattr(arr, "shape"):
            try:
                arr = arr.flatten()
                if len(arr) > 0 and hasattr(arr[0], "shape"):
                    arr = np.asarray(arr[0])
                else:
                    continue
            except Exception:
                continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 2:
            T, N = arr.shape
            if N > T:
                arr = arr.T
            return arr
        if arr.ndim == 3:
            arr = arr[0]
            if arr.shape[0] < arr.shape[1]:
                arr = arr.T
            return arr
    best = None
    for k, v in data.items():
        if k.startswith("_"):
            continue
        try:
            a = np.asarray(v, dtype=np.float64)
            if a.ndim == 2 and a.size > 0:
                if best is None or a.size > best.size:
                    best = a
        except Exception:
            continue
    if best is not None:
        if best.shape[0] < best.shape[1]:
            best = best.T
        return best
    raise ValueError(f"Unable to parse `(T, N)` time series from {mat_path}. Available keys: {list(data.keys())}")


def get_subject_mat_files(dataset_dir: Path) -> List[Tuple[int, Path]]:
    """Return `[(subject_index, path), ...]` sorted by subject index."""
    if not dataset_dir.is_dir():
        return []
    out = []
    for f in dataset_dir.iterdir():
        if f.suffix.lower() != ".mat":
            continue
        m = re.match(r"sub(\d+)", f.stem, re.I)
        if m:
            out.append((int(m.group(1)), f))
    out.sort(key=lambda x: x[0])
    return out


def normalize_timecourses(x: np.ndarray) -> torch.Tensor:
    """Normalize `(T, N)` data and return a float32 tensor with shape `(T, N)`."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0) + 1e-8
    x = (x - mean) / std
    return torch.tensor(x, dtype=torch.float32)
