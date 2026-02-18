"""Efficient O-information computation with randomized trace estimation.

The implementation reduces matrix-function trace costs via stochastic estimators
and computes only unique triplets before symmetric tensor filling.
"""

import torch
import itertools
from typing import Tuple, Optional, List
import time


class EfficientOInfoCalculator:
    """High-performance O-information calculator using stochastic trace estimation."""

    def __init__(self, num_random_vectors: int = 50, alpha: float = 1.01, device: Optional[torch.device] = None,
                 use_lanczos: bool = True, lanczos_steps: int = 30):
        self.s = num_random_vectors
        self.alpha = alpha
        self.device = device or torch.device('cpu')
        self.use_lanczos = use_lanczos
        self.lanczos_steps = lanczos_steps

    def generate_random_vectors(self, n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(self.s, n, dtype=dtype, device=self.device)

    def efficient_gram_matrix(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        x = x.view(-1, 1)
        distances_sq = (x - x.t()) ** 2
        gram_matrix = torch.exp(-distances_sq / sigma)
        trace_val = torch.trace(gram_matrix)
        if trace_val > 1e-8:
            gram_matrix = gram_matrix / trace_val
        return gram_matrix

    def randomized_trace_estimation(self, G: torch.Tensor, alpha: float) -> torch.Tensor:
        n = G.shape[0]
        random_vectors = self.generate_random_vectors(n, dtype=G.dtype)
        if abs(alpha - round(alpha)) < 1e-6:
            G_alpha = torch.matrix_power(G, int(round(alpha)))
            trace_estimates = [random_vectors[i].view(-1, 1).t() @ G_alpha @ random_vectors[i].view(-1, 1) for i in range(self.s)]
            return torch.tensor(sum(t.item() for t in trace_estimates) / self.s, dtype=G.dtype, device=self.device)
        if self.use_lanczos:
            m = max(5, min(self.lanczos_steps, n))
            trace_estimates = [self._lanczos_gauss_quadrature(G, random_vectors[i], alpha, m) for i in range(self.s)]
            return torch.tensor(sum(trace_estimates) / self.s, dtype=G.dtype, device=self.device)
        eigenvals, eigenvecs = torch.linalg.eigh(G)
        eigenvals = torch.clamp(eigenvals, min=1e-12)
        G_alpha = eigenvecs @ torch.diag(eigenvals ** alpha) @ eigenvecs.t()
        trace_estimates = [random_vectors[i].view(-1, 1).t() @ G_alpha @ random_vectors[i].view(-1, 1) for i in range(self.s)]
        return torch.tensor(sum(t.item() for t in trace_estimates) / self.s, dtype=G.dtype, device=self.device)

    def _lanczos_gauss_quadrature(self, G: torch.Tensor, g: torch.Tensor, alpha: float, m: int) -> float:
        g = g.to(self.device)
        n, m = G.shape[0], int(min(max(1, m), G.shape[0]))
        g_norm = torch.norm(g)
        if g_norm.item() < 1e-20:
            return 0.0
        q_prev, q = torch.zeros_like(g), (g / g_norm).clone()
        beta_prev = torch.tensor(0.0, dtype=G.dtype, device=self.device)
        alphas_list, betas_list = [], []
        for k in range(m):
            w = G @ q - beta_prev * q_prev
            alpha_k = torch.dot(q, w)
            w = w - alpha_k * q
            beta_k = torch.norm(w)
            alphas_list.append(alpha_k)
            if k < m - 1:
                betas_list.append(beta_k)
            if beta_k.item() < 1e-20:
                m = k + 1
                break
            q_prev, q = q, (w / beta_k)
            beta_prev = beta_k
        T = torch.zeros((m, m), dtype=G.dtype, device=self.device)
        for i in range(m):
            T[i, i] = alphas_list[i]
            if i < m - 1:
                T[i, i+1] = T[i+1, i] = betas_list[i]
        evals, evecs = torch.linalg.eigh(T)
        evals = torch.clamp(evals, min=1e-12)
        e1 = torch.zeros((m,), dtype=G.dtype, device=self.device)
        e1[0] = 1.0
        coeffs = evecs.t() @ e1
        quad_approx = torch.sum((coeffs ** 2) * (evals ** alpha))
        return (g_norm ** 2 * quad_approx).item()

    def efficient_renyi_entropy(self, x: torch.Tensor, sigma: float, alpha: float) -> torch.Tensor:
        G = self.efficient_gram_matrix(x, sigma)
        trace_G_alpha = self.randomized_trace_estimation(G, alpha)
        if abs(alpha - 1.0) < 1e-6:
            eigenvals = torch.linalg.eigh(G)[0].clamp(min=1e-12)
            return -torch.sum(eigenvals * torch.log2(eigenvals))
        return (1.0 / (1.0 - alpha)) * torch.log2(torch.clamp(trace_G_alpha, min=1e-12))

    def efficient_joint_entropy(self, vars_list: list, sigmas: list, alpha: float) -> torch.Tensor:
        joint_gram = self.efficient_gram_matrix(vars_list[0], sigmas[0])
        for var, sigma in zip(vars_list[1:], sigmas[1:]):
            joint_gram = joint_gram * self.efficient_gram_matrix(var, sigma)
        trace_val = torch.trace(joint_gram)
        if trace_val > 1e-8:
            joint_gram = joint_gram / trace_val
        trace_joint_alpha = self.randomized_trace_estimation(joint_gram, alpha)
        if abs(alpha - 1.0) < 1e-6:
            eigenvals = torch.linalg.eigh(joint_gram)[0].clamp(min=1e-12)
            return -torch.sum(eigenvals * torch.log2(eigenvals))
        return (1.0 / (1.0 - alpha)) * torch.log2(torch.clamp(trace_joint_alpha, min=1e-12))

    def efficient_TC(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                     s_x: float, s_y: float, s_z: float, alpha: float) -> torch.Tensor:
        Hx = self.efficient_renyi_entropy(x, s_x, alpha)
        Hy = self.efficient_renyi_entropy(y, s_y, alpha)
        Hz = self.efficient_renyi_entropy(z, s_z, alpha)
        Hxyz = self.efficient_joint_entropy([x, y, z], [s_x, s_y, s_z], alpha)
        return Hx + Hy + Hz - Hxyz

    def efficient_conditional_entropy(self, index: int, vars_list: list, sigmas: list, alpha: float) -> torch.Tensor:
        joint_ent_all = self.efficient_joint_entropy(vars_list, sigmas, alpha)
        vars_without = vars_list[:index] + vars_list[index+1:]
        sigmas_without = sigmas[:index] + sigmas[index+1:]
        joint_ent_without = self.efficient_joint_entropy(vars_without, sigmas_without, alpha)
        return joint_ent_all - joint_ent_without

    def efficient_DTC(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                      s_x: float, s_y: float, s_z: float, alpha: float) -> torch.Tensor:
        vars_list, sigmas = [x, y, z], [s_x, s_y, s_z]
        HX = self.efficient_joint_entropy(vars_list, sigmas, alpha)
        cond_ents = [self.efficient_conditional_entropy(i, vars_list, sigmas, alpha) for i in range(3)]
        return HX - sum(cond_ents)

    def calculate_O_information_efficient(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                                         s_x: float, s_y: float, s_z: float, alpha: float) -> torch.Tensor:
        x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
        TC_value = self.efficient_TC(x, y, z, s_x, s_y, s_z, alpha)
        DTC_value = self.efficient_DTC(x, y, z, s_x, s_y, s_z, alpha)
        return TC_value - DTC_value


def generate_symmetric_triplets(n_nodes: int) -> List[Tuple[int, int, int]]:
    return [(i, j, k) for i in range(n_nodes) for j in range(i + 1, n_nodes) for k in range(j + 1, n_nodes)]


def fill_symmetric_tensor(O_tensor: torch.Tensor, i: int, j: int, k: int, value: float):
    for perm in itertools.permutations([i, j, k]):
        O_tensor[perm[0], perm[1], perm[2]] = value


def calculate_O_tensor_symmetric(time_courses_tensor: torch.Tensor,
                                 calculator: EfficientOInfoCalculator,
                                 s_x: float = 0.8, s_y: float = 0.8, s_z: float = 0.8,
                                 alpha: float = 1.01, progress_callback=None) -> torch.Tensor:
    """Compute a full 3D O-information tensor using symmetry-aware triplet traversal."""
    time_steps, num_nodes = time_courses_tensor.shape
    O_tensor = torch.zeros(num_nodes, num_nodes, num_nodes, device=calculator.device)
    unique_triplets = generate_symmetric_triplets(num_nodes)
    total_triplets = len(unique_triplets)
    start_time = time.time()
    for idx, (i, j, k) in enumerate(unique_triplets):
        x = time_courses_tensor[:, i]
        y = time_courses_tensor[:, j]
        z = time_courses_tensor[:, k]
        try:
            o_value = calculator.calculate_O_information_efficient(x, y, z, s_x, s_y, s_z, alpha).item()
            fill_symmetric_tensor(O_tensor, i, j, k, o_value)
        except Exception as e:
            print(f"Warning: triplet ({i},{j},{k}) failed: {e}")
            fill_symmetric_tensor(O_tensor, i, j, k, 0.0)
        if progress_callback and idx % max(1, total_triplets // 100) == 0:
            progress_callback((idx + 1) / total_triplets * 100, time.time() - start_time,
                             (time.time() - start_time) / (idx + 1) * (total_triplets - idx - 1))
    print(f"Random O-tensor done in {time.time() - start_time:.1f}s")
    return O_tensor
