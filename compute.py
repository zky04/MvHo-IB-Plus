#!/usr/bin/env python3
"""Batch compute for methodology-aligned multi-view representations.

Views:
- V1: Renyi-MI graph with top-30% sparsification
- V2: 3D O-information tensor
- V3: 4D O-information tensor
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_subject_mat_files, load_timecourses_from_mat, normalize_timecourses
from src.data.graph_builder import build_renyi_graph_from_timeseries
from src.utils.oinfo_compute import ComputeConfig, compute_oinfo_tensor

DATASET_NAMES = ["abide", "adni", "mdd", "ucla"]


def compute_subject(
    mat_path: Path,
    out_path: Path,
    subject_id: int,
    label: int,
    method: str,
    order: str,
    device: torch.device,
    alpha: float,
    num_random_vectors: int,
    regularization: float,
    build_graph: bool,
) -> None:
    tc = load_timecourses_from_mat(mat_path)
    tc_t = normalize_timecourses(tc).to(device)

    payload = {
        "subject_id": subject_id,
        "source_file": mat_path.name,
        "method": method,
    }

    if build_graph:
        v1_sparse, graph = build_renyi_graph_from_timeseries(
            tc_t.cpu(),
            label=label,
            sample_id=subject_id,
            alpha=alpha,
            top_ratio=0.3,
        )
        payload["v1_sparse"] = v1_sparse
        payload["graph"] = graph

    cfg = ComputeConfig(
        method=method,
        alpha=alpha,
        num_random_vectors=num_random_vectors,
        regularization=regularization,
    )

    if order in ("3", "both"):
        o3, meta3 = compute_oinfo_tensor(tc_t, order=3, config=cfg, device=device)
        payload["o3_tensor"] = o3
        payload["meta3"] = meta3

    if order in ("4", "both"):
        o4, meta4 = compute_oinfo_tensor(tc_t, order=4, config=cfg, device=device)
        payload["o4_tensor"] = o4
        payload["meta4"] = meta4

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def run_dataset(args, dataset_name: str) -> None:
    dataset_dir = ROOT / "dataset" / dataset_name
    out_dir = ROOT / "computed" / dataset_name
    if not dataset_dir.is_dir():
        print(f"[{dataset_name}] skip: missing directory {dataset_dir}")
        return

    subjects = get_subject_mat_files(dataset_dir)
    if not subjects:
        print(f"[{dataset_name}] skip: no sub*.mat in {dataset_dir}")
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[{dataset_name}] subjects={len(subjects)} method={args.method} order={args.order} device={device}")

    for sid, mat_path in subjects:
        out_path = out_dir / f"sub{sid}.pt"
        if out_path.exists() and not args.overwrite:
            print(f"  skip existing {out_path.name}")
            continue
        try:
            compute_subject(
                mat_path=mat_path,
                out_path=out_path,
                subject_id=sid,
                label=0,
                method=args.method,
                order=args.order,
                device=device,
                alpha=args.alpha,
                num_random_vectors=args.random_vectors,
                regularization=args.regularization,
                build_graph=args.build_graph,
            )
            print(f"  saved {out_path.name}")
        except Exception as e:
            print(f"  failed sub{sid}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MvHo-IB++ multi-view precompute")
    parser.add_argument("--method", type=str, required=True, choices=["gauss", "random"]) 
    parser.add_argument("--dataset", type=str, default="all", help="abide|adni|mdd|ucla|all")
    parser.add_argument("--order", type=str, default="3", choices=["3", "4", "both"]) 
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--alpha", type=float, default=1.01)
    parser.add_argument("--random-vectors", type=int, default=50)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--build-graph", action="store_true", help="build view-1 Renyi graph")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    names = DATASET_NAMES if args.dataset == "all" else [args.dataset]
    for n in names:
        if n not in DATASET_NAMES:
            raise ValueError(f"unknown dataset: {n}")
        run_dataset(args, n)


if __name__ == "__main__":
    main()
