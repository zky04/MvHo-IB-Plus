"""Fast Gaussian-approximation O-information computation.

This module uses covariance-based closed-form expressions and symmetry-aware
triplet enumeration for efficient 3D tensor construction.
"""

import torch
import numpy as np
import itertools
import time
from typing import Tuple, List, Dict
from dataclasses import dataclass
import warnings


@dataclass
class GaussianApproxConfig:
    """Runtime configuration for Gaussian approximation."""
    device: torch.device = None
    regularization: float = 1e-6
    verbose: bool = True

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PaperGaussianOInfoCalculator:
    """Gaussian O-information calculator with symmetric tensor assembly."""

    def __init__(self, config: GaussianApproxConfig = None):
        if config is None:
            config = GaussianApproxConfig()
        self.config = config
        self.device = config.device

    def total_correlation_gaussian(self, time_courses: torch.Tensor) -> torch.Tensor:
        """Total correlation under Gaussian assumption: TC^N = -ln(|Sigma|)/2."""
        cov_matrix = torch.cov(time_courses.T)
        n = cov_matrix.shape[0]
        reg_cov = cov_matrix + self.config.regularization * torch.eye(n, device=self.device)
        try:
            log_det = torch.logdet(reg_cov)
            if torch.isnan(log_det) or torch.isinf(log_det):
                eigenvals = torch.linalg.eigvals(reg_cov).real
                eigenvals = torch.clamp(eigenvals, min=1e-8)
                log_det = torch.sum(torch.log(eigenvals))
        except Exception:
            eigenvals = torch.linalg.eigvals(reg_cov).real
            eigenvals = torch.clamp(eigenvals, min=1e-8)
            log_det = torch.sum(torch.log(eigenvals))
        return -0.5 * log_det

    def calculate_o_information_triplet(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> float:
        """Triplet O-information: Omega_3^N = -TC(X,Y,Z) + TC(X,Y) + TC(X,Z) + TC(Y,Z)."""
        try:
            x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
            if x.numel() < 3 or y.numel() < 3 or z.numel() < 3:
                return 0.0
            valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z)
            if valid.sum() < 3:
                return 0.0
            x, y, z = x[valid], y[valid], z[valid]
            if torch.std(x) < 1e-8 or torch.std(y) < 1e-8 or torch.std(z) < 1e-8:
                return 0.0
            xyz = torch.stack([x, y, z], dim=1)
            xy = torch.stack([x, y], dim=1)
            xz = torch.stack([x, z], dim=1)
            yz = torch.stack([y, z], dim=1)
            tc_xyz = self.total_correlation_gaussian(xyz)
            tc_xy = self.total_correlation_gaussian(xy)
            tc_xz = self.total_correlation_gaussian(xz)
            tc_yz = self.total_correlation_gaussian(yz)
            o_info = -tc_xyz + tc_xy + tc_xz + tc_yz
            return float(o_info.item())
        except Exception as e:
            if self.config.verbose:
                warnings.warn(f"O-info triplet failed: {e}")
            return 0.0

    def compute_3d_tensor_symmetric(self, time_courses: torch.Tensor,
                                    verbose: bool = None) -> Tuple[torch.Tensor, Dict]:
        """Compute a 3D O-information tensor from unique triplets and symmetric fill."""
        verbose = verbose if verbose is not None else self.config.verbose
        time_courses = time_courses.to(self.device)
        num_regions = time_courses.shape[1]
        tensor_3d = torch.zeros(num_regions, num_regions, num_regions, device=self.device)
        unique_triplets = [(i, j, k) for i in range(num_regions)
                           for j in range(i, num_regions) for k in range(j, num_regions)]
        total = len(unique_triplets)
        start = time.time()
        for idx, (i, j, k) in enumerate(unique_triplets):
            x = time_courses[:, i]
            y = time_courses[:, j]
            z = time_courses[:, k]
            o_info = self.calculate_o_information_triplet(x, y, z)
            for pi, pj, pk in itertools.permutations([i, j, k]):
                tensor_3d[pi, pj, pk] = o_info
            if verbose and (idx % max(1, total // 50) == 0 or idx < 5):
                print(f"[Gauss] {idx+1}/{total} ({100*(idx+1)/total:.1f}%) "
                      f"triplet({i},{j},{k}) O={o_info:.4f} elapsed={time.time()-start:.1f}s")
        elapsed = time.time() - start
        metadata = {
            'computation_time': elapsed,
            'total_triplets': total,
            'method': 'gaussian_symmetric',
            'num_regions': num_regions,
            'tensor_shape': tuple(tensor_3d.shape),
        }
        if verbose:
            print(f"Gauss 3D tensor done in {elapsed:.2f}s, shape {tensor_3d.shape}")
        return tensor_3d.cpu(), metadata
