"""3D/4D O-information computation utilities (gauss/random)."""

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch

from .compute.random import EfficientOInfoCalculator


@dataclass
class ComputeConfig:
    method: str = "gauss"  # gauss | random
    alpha: float = 1.01
    num_random_vectors: int = 50
    regularization: float = 1e-6


def _gaussian_entropy(vars_2d: torch.Tensor, reg: float = 1e-6) -> torch.Tensor:
    """Entropy surrogate 0.5*logdet(cov), constants omitted (cancel in O-info)."""
    cov = torch.cov(vars_2d.T)
    n = cov.shape[0]
    cov = cov + reg * torch.eye(n, device=cov.device, dtype=cov.dtype)
    sign, logabsdet = torch.linalg.slogdet(cov)
    if sign <= 0:
        evals = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
        logabsdet = torch.sum(torch.log(evals))
    return 0.5 * logabsdet


def _gaussian_oinfo(vars_list: List[torch.Tensor], reg: float = 1e-6) -> torch.Tensor:
    k = len(vars_list)
    x = torch.stack(vars_list, dim=1)
    h_all = _gaussian_entropy(x, reg=reg)
    h_single = torch.stack([_gaussian_entropy(v.view(-1, 1), reg=reg) for v in vars_list]).sum()

    h_rest_sum = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    for i in range(k):
        rest = [vars_list[j] for j in range(k) if j != i]
        h_rest_sum = h_rest_sum + _gaussian_entropy(torch.stack(rest, dim=1), reg=reg)

    tc = h_single - h_all
    dtc = h_rest_sum - (k - 1) * h_all
    return tc - dtc


def _random_oinfo(vars_list: List[torch.Tensor], alpha: float, calc: EfficientOInfoCalculator) -> torch.Tensor:
    k = len(vars_list)
    sigmas = [0.8] * k
    h_all = calc.efficient_joint_entropy(vars_list, sigmas, alpha)
    h_single = torch.stack([calc.efficient_renyi_entropy(v, 0.8, alpha) for v in vars_list]).sum()

    h_rest_sum = torch.tensor(0.0, dtype=h_all.dtype, device=h_all.device)
    for i in range(k):
        rest_vars = [vars_list[j] for j in range(k) if j != i]
        rest_sigmas = [0.8] * (k - 1)
        h_rest_sum = h_rest_sum + calc.efficient_joint_entropy(rest_vars, rest_sigmas, alpha)

    tc = h_single - h_all
    dtc = h_rest_sum - (k - 1) * h_all
    return tc - dtc


def _fill_permutations(tensor: torch.Tensor, indices: Tuple[int, ...], value: float) -> None:
    for p in set(itertools.permutations(indices)):
        tensor[p] = value


def _tuple_generator(num_nodes: int, order: int) -> Iterable[Tuple[int, ...]]:
    # Distinct tuples only, then permutation-fill to symmetric tensor.
    return itertools.combinations(range(num_nodes), order)


def compute_oinfo_tensor(
    time_courses: torch.Tensor,
    order: int,
    config: ComputeConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """Compute order-k symmetric O-information tensor from (T, C)."""
    if order not in (3, 4):
        raise ValueError("order must be 3 or 4")

    tc = time_courses.to(device)
    _, c = tc.shape
    shape = (c,) * order
    out = torch.zeros(shape, dtype=torch.float32, device=device)

    tuples = list(_tuple_generator(c, order))
    if config.method == "random":
        calc = EfficientOInfoCalculator(
            num_random_vectors=config.num_random_vectors,
            alpha=config.alpha,
            device=device,
        )
    else:
        calc = None

    for idx in tuples:
        vars_list = [tc[:, i] for i in idx]
        if config.method == "gauss":
            o = _gaussian_oinfo(vars_list, reg=config.regularization)
        elif config.method == "random":
            o = _random_oinfo(vars_list, alpha=config.alpha, calc=calc)
        else:
            raise ValueError("method must be gauss or random")
        _fill_permutations(out, idx, float(o.item()))

    meta = {
        "method": config.method,
        "order": order,
        "num_nodes": c,
        "num_unique_tuples": len(tuples),
    }
    return out.cpu(), meta
