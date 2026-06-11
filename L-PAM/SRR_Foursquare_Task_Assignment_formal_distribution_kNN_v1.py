#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRR / GRR / OLH-H mechanisms for online spatial crowdsourcing task assignment
using Foursquare .dat files in D:\\data2.

Formal experiment version changes:
    - Formal defaults follow the main-paper setting: fixed Foursquare Austin bbox,
      grid_size=16, workers=2000, tasks=1000, repeats=10, and epsilons
      0.1,0.5,1.0,2.0,4.0,6.0,8.0.
    - GRR, HR, PLDP/OUE, SRR-Equal, and SRR-SelfFirst are the
      main LDP-family comparison methods.
    - SRR-Local is kept as an optional ablation/appendix method and is excluded
      from main experiments and main figures by default; use --include-srr-local
      to restore it.
    - Besides the full plots, the script writes paper-oriented filtered plots:
      LDP-only comparison and compact reference-upper-bound comparison.
    - Optional --include-hst-greedy can add HSTGreedy curves for appendix runs.
    - Workload sensitivity can be controlled directly in the code via the
      IN_CODE_* switches below. By default this version runs the main experiment
      first and then automatically produces total-distance-vs-workers and
      total-distance-vs-tasks figures in a workload_sensitivity subfolder.

Expected Foursquare / TSMC-style .dat format, tab-separated by default:
    user_id    venue_id    venue_category_id    venue_category    latitude    longitude    timezone_offset    utc_time
Default latitude column = 4, longitude column = 5, both 0-based.
The loader scans every .dat file in --foursquare-dir.

Experiment scenario:
    1) Read check-in locations from Foursquare. This version DOES NOT silently fall
       back to synthetic data unless --allow-synthetic-fallback is provided.
    2) Optionally crop to a local dense region, because global coordinates make
       task assignment distances meaningless.
    3) Sample worker locations and online task locations from the cropped pool.
    4) Discretize coordinates to a regular grid domain.
    5) Perturb worker/task domain indices using GRR, HR, OLH-H, PLDP/OUE, or SRR.
    6) Assign tasks online using only perturbed locations.
    7) Evaluate assignment cost using the true coordinates.

Windows example using the requested D-drive dataset:
    python SRR_OLHH_Foursquare_Task_Assignment_formal.py --foursquare-path "D:\\Foursquare_totalCheckins.txt" --grid-size 16 --num-workers 2000 --num-tasks 1000 --repeats 10

Faster debug example:
    python SRR_OLHH_Foursquare_Task_Assignment_formal.py --foursquare-path "D:\\Foursquare_totalCheckins.txt" --grid-size 16 --num-workers 200 --num-tasks 100 --epsilons 0.5,1.0 --repeats 1 --foursquare-scan-limit 200000
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PointArray = np.ndarray
BBox = Tuple[float, float, float, float]  # min_lat, max_lat, min_lon, max_lon


# ---------------------------------------------------------------------------
# In-code experiment switches
# ---------------------------------------------------------------------------
# Set this to True if you want the script to run the normal epsilon experiment
# first and then automatically run workload sensitivity in:
#     <out_dir>/workload_sensitivity/
# This replaces the need to pass --also-run-workload-sensitivity in the command.
IN_CODE_RUN_WORKLOAD_AFTER_MAIN = False

# 仅当您想跳过主要的 epsilon 实验并运行时，将此设置为 True
# 仅工作负载敏感性。通常在正式运行中保持为 False。
IN_CODE_RUN_WORKLOAD_ONLY = False

# Workload-sensitivity defaults controlled in code.
IN_CODE_WORKLOAD_EPSILON = 5.0
IN_CODE_WORKLOAD_REPEATS = 5
IN_CODE_WORKER_COUNTS = "1000,2000,3000,4000,5000"
IN_CODE_FIXED_TASKS_FOR_WORKER_SWEEP = 1000
IN_CODE_TASK_COUNTS = "500,1000,1500,2000,3000"
IN_CODE_FIXED_WORKERS_FOR_TASK_SWEEP = 4000

# Internal guard: child processes created for workload sensitivity must not
# recursively run workload sensitivity again. Do not edit this line.
_IN_WORKLOAD_CHILD_PROCESS = os.environ.get("SRR_SKIP_CODE_WORKLOAD", "0") == "1"

# Distribution-estimation experiment switches.
# 默认直接运行分布估计实验，便于验证 SRR 的密度恢复效果。
# 如需恢复原始任务分配实验，将 IN_CODE_RUN_DISTRIBUTION_ONLY 改为 False，
# 或命令行添加 --no-run-distribution-estimation。
IN_CODE_RUN_DISTRIBUTION_ONLY = True
IN_CODE_RUN_DISTRIBUTION_AFTER_MAIN = False



# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def stable_seed(seed: int, epsilon: float, name: str) -> int:
    value = (int(seed) * 1000003 + int(round(epsilon * 1_000_000)) * 9176) & 0xFFFFFFFF
    for ch in name:
        value = (value * 131 + ord(ch)) & 0xFFFFFFFF
    return value


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def parse_epsilons(s: str) -> List[float]:
    vals = [float(part.strip()) for part in s.split(",") if part.strip()]
    if not vals:
        raise ValueError("epsilon list is empty")
    return vals


def parse_int_list(s: str) -> List[int]:
    vals = [int(part.strip()) for part in s.split(",") if part.strip()]
    if not vals:
        raise ValueError("integer list is empty")
    if any(v <= 0 for v in vals):
        raise ValueError("all integer list values must be positive")
    return vals


def parse_float_list(s: str) -> List[float]:
    vals = [float(part.strip()) for part in s.split(",") if part.strip()]
    if not vals:
        raise ValueError("float list is empty")
    return vals


def parse_bbox(s: Optional[str]) -> Optional[BBox]:
    if not s:
        return None
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--foursquare-bbox must be: min_lat,max_lat,min_lon,max_lon")
    min_lat, max_lat, min_lon, max_lon = parts
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("Invalid bbox: min must be smaller than max")
    return min_lat, max_lat, min_lon, max_lon


def in_bbox(lat: float, lon: float, bbox: Optional[BBox]) -> bool:
    if bbox is None:
        return True
    min_lat, max_lat, min_lon, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


# ---------------------------------------------------------------------------
# Foursquare loading and coordinate projection
# ---------------------------------------------------------------------------


def read_foursquare_latlon_reservoir(
    path: str,
    max_points: int,
    rng: np.random.Generator,
    lat_col: int = 2,
    lon_col: int = 3,
    bbox: Optional[BBox] = None,
    scan_limit: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Reservoir-sample latitude/longitude pairs from Foursquare.

    Returns:
        sample_latlon: array with shape [n, 2], columns = lat, lon
        matched_count: number of valid rows that passed bbox during scanning
    """
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    sample: List[Tuple[float, float]] = []
    matched_count = 0
    scanned = 0
    required_cols = max(lat_col, lon_col) + 1

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            scanned += 1
            if scan_limit and scanned > scan_limit:
                break
            parts = line.strip().split()
            if len(parts) < required_cols:
                continue
            try:
                lat = float(parts[lat_col])
                lon = float(parts[lon_col])
            except ValueError:
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            if not in_bbox(lat, lon, bbox):
                continue

            matched_count += 1
            if len(sample) < max_points:
                sample.append((lat, lon))
            else:
                # Replace with probability max_points / matched_count.
                j = int(rng.integers(0, matched_count))
                if j < max_points:
                    sample[j] = (lat, lon)

    if not sample:
        detail = ""
        if bbox is not None:
            detail = f" within bbox={bbox}"
        raise ValueError(f"No valid Foursquare points found{detail}. Check path, columns, or bbox.")

    return np.asarray(sample, dtype=float), matched_count


def auto_dense_bbox(latlon: np.ndarray, cell_deg: float = 0.5, window_deg: float = 1.0) -> BBox:
    """
    Find a dense geographic area from a sample using coarse latitude/longitude cells.
    This keeps the task-assignment experiment local rather than global.
    """
    if len(latlon) == 0:
        raise ValueError("empty latlon sample")
    if cell_deg <= 0 or window_deg <= 0:
        raise ValueError("cell_deg and window_deg must be positive")

    counts: Dict[Tuple[int, int], int] = {}
    for lat, lon in latlon:
        key = (int(math.floor(lat / cell_deg)), int(math.floor(lon / cell_deg)))
        counts[key] = counts.get(key, 0) + 1

    best_key = max(counts.keys(), key=lambda k: counts[k])
    center_lat = (best_key[0] + 0.5) * cell_deg
    center_lon = (best_key[1] + 0.5) * cell_deg
    half = window_deg / 2.0
    min_lat = max(-90.0, center_lat - half)
    max_lat = min(90.0, center_lat + half)
    min_lon = max(-180.0, center_lon - half)
    max_lon = min(180.0, center_lon + half)
    return min_lat, max_lat, min_lon, max_lon


def latlon_to_xy_km(latlon: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convert latitude/longitude to local x/y coordinates in kilometers using an
    equirectangular approximation around the sample center.
    """
    lat = latlon[:, 0].astype(float)
    lon = latlon[:, 1].astype(float)
    lat0 = float(lat.mean())
    lon0 = float(lon.mean())
    cos_lat0 = math.cos(math.radians(lat0))

    x = (lon - lon0) * 111.320 * cos_lat0
    y = (lat - lat0) * 110.574
    xy = np.column_stack([x, y])

    # Shift to positive coordinates; distances are unchanged by translation.
    min_xy = xy.min(axis=0)
    xy = xy - min_xy
    max_xy = xy.max(axis=0)

    meta = {
        "lat0": lat0,
        "lon0": lon0,
        "min_x_km": float(min_xy[0]),
        "min_y_km": float(min_xy[1]),
        "width_km": float(max_xy[0]),
        "height_km": float(max_xy[1]),
    }
    return xy.astype(float), meta


def load_foursquare_xy_points(args: argparse.Namespace, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """Load and project Foursquare points according to command-line arguments."""
    user_bbox = parse_bbox(args.foursquare_bbox)
    effective_bbox = user_bbox

    if user_bbox is None and not args.disable_auto_bbox:
        print("Finding dense Foursquare region for local experiment ...")
        global_sample, global_count = read_foursquare_latlon_reservoir(
            path=args.foursquare_path,
            max_points=args.foursquare_auto_sample,
            rng=rng,
            lat_col=args.foursquare_lat_col,
            lon_col=args.foursquare_lon_col,
            bbox=None,
            scan_limit=args.foursquare_scan_limit,
        )
        effective_bbox = auto_dense_bbox(
            global_sample,
            cell_deg=args.foursquare_auto_cell_deg,
            window_deg=args.foursquare_auto_window_deg,
        )
        print(
            "Auto bbox selected: "
            f"min_lat={effective_bbox[0]:.6f}, max_lat={effective_bbox[1]:.6f}, "
            f"min_lon={effective_bbox[2]:.6f}, max_lon={effective_bbox[3]:.6f} "
            f"from {len(global_sample)} sampled points / {global_count} scanned valid points"
        )

    latlon, matched_count = read_foursquare_latlon_reservoir(
        path=args.foursquare_path,
        max_points=args.foursquare_max_points,
        rng=rng,
        lat_col=args.foursquare_lat_col,
        lon_col=args.foursquare_lon_col,
        bbox=effective_bbox,
        scan_limit=args.foursquare_scan_limit,
    )
    xy, meta = latlon_to_xy_km(latlon)
    meta["foursquare_matched_count"] = float(matched_count)
    meta["foursquare_sampled_count"] = float(len(latlon))
    if effective_bbox is not None:
        meta["bbox_min_lat"] = effective_bbox[0]
        meta["bbox_max_lat"] = effective_bbox[1]
        meta["bbox_min_lon"] = effective_bbox[2]
        meta["bbox_max_lon"] = effective_bbox[3]

    print(f"Loaded Foursquare points: sampled={len(latlon)}, matched_valid={matched_count}")
    print(f"Projected local area: width={meta['width_km']:.3f} km, height={meta['height_km']:.3f} km")
    return xy, meta



# ---------------------------------------------------------------------------
# Foursquare .dat loading and coordinate projection
# ---------------------------------------------------------------------------


def _iter_foursquare_dat_files(data_dir: str) -> List[str]:
    """Return sorted .dat files under a Foursquare directory."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Foursquare data directory not found: {data_dir}")
    files = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.lower().endswith(".dat") and os.path.isfile(os.path.join(data_dir, name))
    ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No .dat files found in Foursquare data directory: {data_dir}")
    return files


def _split_foursquare_line(line: str) -> List[str]:
    """Split one Foursquare line.

    TSMC2014 .dat files are tab-separated. If a file is not tab-separated,
    fall back to whitespace splitting.
    """
    line = line.strip()
    if not line:
        return []
    if "\t" in line:
        return line.split("\t")
    return line.split()


def _try_parse_lat_lon(parts: Sequence[str], lat_col: int, lon_col: int) -> Optional[Tuple[float, float]]:
    """Parse latitude/longitude from fixed columns, with a conservative fallback."""
    required_cols = max(int(lat_col), int(lon_col)) + 1
    if len(parts) >= required_cols:
        try:
            lat = float(parts[int(lat_col)])
            lon = float(parts[int(lon_col)])
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass

    # Conservative fallback for unusual .dat files: prefer decimal-valued
    # latitude/longitude-like columns. This is only used when fixed columns fail.
    numeric: List[Tuple[int, float, bool]] = []
    for i, part in enumerate(parts):
        try:
            val = float(part)
        except ValueError:
            continue
        has_decimal = ("." in part) or ("e" in part.lower())
        numeric.append((i, val, has_decimal))

    lat_candidates = [(i, v) for i, v, dec in numeric if dec and -90.0 <= v <= 90.0]
    lon_candidates = [(i, v) for i, v, dec in numeric if dec and -180.0 <= v <= 180.0]
    best: Optional[Tuple[float, float]] = None
    best_score = -1.0
    for i, lat in lat_candidates:
        for j, lon in lon_candidates:
            if i == j:
                continue
            # Most Foursquare locations in these public files have nontrivial
            # absolute latitude/longitude. Penalize near-zero accidental numbers.
            score = abs(lat) + abs(lon)
            if score > best_score and abs(lat) > 1.0 and abs(lon) > 1.0:
                best_score = score
                best = (lat, lon)
    return best


def read_foursquare_latlon_reservoir(
    data_dir: str,
    max_points: int,
    rng: np.random.Generator,
    lat_col: int = 4,
    lon_col: int = 5,
    bbox: Optional[BBox] = None,
    scan_limit: int = 0,
) -> Tuple[np.ndarray, int, List[str]]:
    """Reservoir-sample latitude/longitude pairs from all .dat files in a directory.

    Returns:
        sample_latlon: array with shape [n, 2], columns = lat, lon
        matched_count: number of valid rows that passed bbox during scanning
        used_files: list of .dat files scanned
    """
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    files = _iter_foursquare_dat_files(data_dir)
    sample: List[Tuple[float, float]] = []
    matched_count = 0
    scanned = 0

    for file_path in files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                scanned += 1
                if scan_limit and scanned > scan_limit:
                    break
                parts = _split_foursquare_line(line)
                if not parts:
                    continue
                parsed = _try_parse_lat_lon(parts, lat_col=lat_col, lon_col=lon_col)
                if parsed is None:
                    continue
                lat, lon = parsed
                if not in_bbox(lat, lon, bbox):
                    continue

                matched_count += 1
                if len(sample) < max_points:
                    sample.append((lat, lon))
                else:
                    j = int(rng.integers(0, matched_count))
                    if j < max_points:
                        sample[j] = (lat, lon)
            if scan_limit and scanned > scan_limit:
                break

    if not sample:
        detail = ""
        if bbox is not None:
            detail = f" within bbox={bbox}"
        raise ValueError(
            f"No valid Foursquare points found{detail}. "
            f"Check --foursquare-dir, --foursquare-lat-col, --foursquare-lon-col, or bbox."
        )

    return np.asarray(sample, dtype=float), matched_count, files


def load_foursquare_xy_points(args: argparse.Namespace, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """Load and project Foursquare .dat points according to command-line arguments."""
    user_bbox = parse_bbox(args.foursquare_bbox)
    effective_bbox = user_bbox

    if user_bbox is None and not args.disable_auto_bbox:
        print("Finding dense Foursquare region for local experiment ...")
        global_sample, global_count, used_files = read_foursquare_latlon_reservoir(
            data_dir=args.foursquare_dir,
            max_points=args.foursquare_auto_sample,
            rng=rng,
            lat_col=args.foursquare_lat_col,
            lon_col=args.foursquare_lon_col,
            bbox=None,
            scan_limit=args.foursquare_scan_limit,
        )
        effective_bbox = auto_dense_bbox(
            global_sample,
            cell_deg=args.foursquare_auto_cell_deg,
            window_deg=args.foursquare_auto_window_deg,
        )
        print(
            "Auto bbox selected: "
            f"min_lat={effective_bbox[0]:.6f}, max_lat={effective_bbox[1]:.6f}, "
            f"min_lon={effective_bbox[2]:.6f}, max_lon={effective_bbox[3]:.6f} "
            f"from {len(global_sample)} sampled points / {global_count} scanned valid points"
        )
        print("Foursquare .dat files scanned:")
        for fp in used_files:
            print(f"  - {fp}")

    latlon, matched_count, used_files = read_foursquare_latlon_reservoir(
        data_dir=args.foursquare_dir,
        max_points=args.foursquare_max_points,
        rng=rng,
        lat_col=args.foursquare_lat_col,
        lon_col=args.foursquare_lon_col,
        bbox=effective_bbox,
        scan_limit=args.foursquare_scan_limit,
    )
    xy, meta = latlon_to_xy_km(latlon)
    meta["data_source"] = "foursquare_dat"
    meta["foursquare_dir"] = args.foursquare_dir
    meta["foursquare_file_count"] = float(len(used_files))
    meta["foursquare_matched_count"] = float(matched_count)
    meta["foursquare_sampled_count"] = float(len(latlon))
    meta["foursquare_lat_col"] = float(args.foursquare_lat_col)
    meta["foursquare_lon_col"] = float(args.foursquare_lon_col)
    # Store filenames in a compact metadata field.
    meta["foursquare_files"] = ";".join(os.path.basename(fp) for fp in used_files)  # type: ignore[assignment]
    if effective_bbox is not None:
        meta["bbox_min_lat"] = effective_bbox[0]
        meta["bbox_max_lat"] = effective_bbox[1]
        meta["bbox_min_lon"] = effective_bbox[2]
        meta["bbox_max_lon"] = effective_bbox[3]

    print(f"Loaded Foursquare points: sampled={len(latlon)}, matched_valid={matched_count}")
    print(f"Projected local area: width={meta['width_km']:.3f} km, height={meta['height_km']:.3f} km")
    return xy, meta


# ---------------------------------------------------------------------------
# Domain construction and data generation
# ---------------------------------------------------------------------------


def create_grid_domain_from_bounds(width: float, height: float, grid_size: int) -> np.ndarray:
    """Create grid-center domain over [0,width] x [0,height]."""
    if grid_size <= 1:
        raise ValueError("grid_size must be > 1")
    width = max(float(width), 1e-9)
    height = max(float(height), 1e-9)
    xs = (np.arange(grid_size) + 0.5) * width / grid_size
    ys = (np.arange(grid_size) + 0.5) * height / grid_size
    points = []
    for ix in range(grid_size):
        for iy in range(grid_size):
            points.append([xs[ix], ys[iy]])
    return np.asarray(points, dtype=float)


def map_points_to_domain(points: np.ndarray, domain: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Map each continuous point to its nearest discrete domain point."""
    out = np.empty(len(points), dtype=int)
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        dist2 = ((block[:, None, :] - domain[None, :, :]) ** 2).sum(axis=2)
        out[start:start + len(block)] = np.argmin(dist2, axis=1)
    return out


def sample_workers_tasks_from_pool(
    pool: np.ndarray,
    num_workers: int,
    num_tasks: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_workers <= 0 or num_tasks <= 0:
        raise ValueError("num_workers and num_tasks must be positive")
    total = num_workers + num_tasks
    replace = len(pool) < total
    idx = rng.choice(len(pool), size=total, replace=replace)
    workers = pool[idx[:num_workers]].copy()
    tasks = pool[idx[num_workers:]].copy()
    # Online arrival order.
    tasks = tasks[rng.permutation(len(tasks))]
    return workers, tasks


def generate_synthetic_points(
    rng: np.random.Generator,
    n: int,
    width: float,
    height: float,
    mode: str = "mixture",
) -> np.ndarray:
    if mode == "uniform":
        return rng.uniform([0.0, 0.0], [width, height], size=(n, 2)).astype(float)
    if mode != "mixture":
        raise ValueError("mode must be 'uniform' or 'mixture'")
    centers = np.asarray(
        [
            [0.25 * width, 0.25 * height],
            [0.75 * width, 0.30 * height],
            [0.40 * width, 0.75 * height],
            [0.75 * width, 0.75 * height],
        ],
        dtype=float,
    )
    probs = np.asarray([0.30, 0.25, 0.25, 0.20])
    choices = rng.choice(len(centers), size=n, p=probs)
    sigma = np.asarray([0.10 * width, 0.10 * height])
    pts = centers[choices] + rng.normal(0.0, sigma, size=(n, 2))
    pts[:, 0] = np.clip(pts[:, 0], 0.0, width)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, height)
    return pts.astype(float)


# ---------------------------------------------------------------------------
# Privacy mechanisms
# ---------------------------------------------------------------------------


class NoPrivacyMechanism:
    name = "NoPrivacy"

    def __init__(self, domain_size: int, rng: np.random.Generator):
        self.domain_size = int(domain_size)
        self.rng = rng

    def perturb_indices(self, true_indices: np.ndarray) -> np.ndarray:
        return np.asarray(true_indices, dtype=int).copy()


class GRRMechanism:
    """Generalized Randomized Response over a finite location domain."""

    name = "GRR"

    def __init__(self, domain_size: int, epsilon: float, rng: np.random.Generator):
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        self.domain_size = int(domain_size)
        self.epsilon = float(epsilon)
        self.rng = rng
        exp_eps = math.exp(self.epsilon)
        self.p_keep = exp_eps / (exp_eps + self.domain_size - 1)

    def perturb_indices(self, true_indices: np.ndarray) -> np.ndarray:
        true_indices = np.asarray(true_indices, dtype=int)
        noisy = true_indices.copy()
        keep = self.rng.random(len(true_indices)) < self.p_keep
        change_pos = np.where(~keep)[0]
        if len(change_pos) == 0:
            return noisy
        samples = self.rng.integers(0, self.domain_size - 1, size=len(change_pos))
        true_vals = true_indices[change_pos]
        samples = samples + (samples >= true_vals)
        noisy[change_pos] = samples
        return noisy




class HRMechanism:
    """
    Hadamard Response style subset randomized response over a finite domain.

    For every true location x, a balanced Hadamard row defines a high-probability
    subset containing half of the domain and a low-probability subset containing
    the other half. Outputs in the high subset have e^epsilon times the per-item
    probability of outputs in the low subset, so the mechanism is epsilon-LDP.

    Note: canonical HR is normally used for frequency estimation. In this task
    assignment experiment, we post-process its randomized output as a noisy
    location index so that it can be compared under the same assignment pipeline.
    """

    name = "HR"

    def __init__(self, domain_size: int, epsilon: float, rng: np.random.Generator):
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        self.domain_size = int(domain_size)
        self.epsilon = float(epsilon)
        self.rng = rng
        if self.domain_size <= 1:
            self.p_high_group = 1.0
        else:
            self.p_high_group = math.exp(self.epsilon) / (math.exp(self.epsilon) + 1.0)

        # The grid domain size is normally a power of two. If not, the parity
        # signs still work, but the high/low sets may be mildly imbalanced.
        self._cols = np.arange(self.domain_size, dtype=np.int64)

    @staticmethod
    def _parity_bits(values: np.ndarray) -> np.ndarray:
        # Vectorized parity for non-negative int64 values using unpackbits.
        vals = np.asarray(values, dtype=np.uint64)
        raw = vals.view(np.uint8).reshape(vals.shape + (8,))
        return np.unpackbits(raw, axis=-1).sum(axis=-1) & 1

    def _is_high(self, row: int, cols: np.ndarray) -> np.ndarray:
        # Sylvester Hadamard sign: high iff parity(row & col) is even.
        return self._parity_bits(np.bitwise_and(np.uint64(row), cols.astype(np.uint64))) == 0

    def _row_for(self, true_idx: int) -> int:
        # Avoid the all-positive row 0. There are d-1 balanced rows for d
        # locations; the last location reuses row 1. This small duplication does
        # not violate LDP and keeps implementation lightweight.
        if self.domain_size <= 2:
            return 1
        return (int(true_idx) % (self.domain_size - 1)) + 1

    def perturb_one(self, true_idx: int) -> int:
        if self.domain_size <= 1:
            return 0
        row = self._row_for(int(true_idx))
        want_high = bool(self.rng.random() < self.p_high_group)
        # Rejection sampling. For balanced Hadamard rows, expected two trials.
        for _ in range(100):
            y = int(self.rng.integers(0, self.domain_size))
            high = bool(self._is_high(row, np.asarray([y], dtype=np.int64))[0])
            if high == want_high:
                return y
        # Fallback for non-power-of-two corner cases: explicitly enumerate.
        mask = self._is_high(row, self._cols)
        candidates = np.where(mask if want_high else ~mask)[0]
        if len(candidates) == 0:
            return int(self.rng.integers(0, self.domain_size))
        return int(self.rng.choice(candidates))

    def perturb_indices(self, true_indices: np.ndarray) -> np.ndarray:
        return np.asarray([self.perturb_one(int(i)) for i in true_indices], dtype=int)



class OLHHMechanism:
    """
    OLH-H baseline adapted from hierarchical OLH for grid-location reports.

    The original OLH-H baseline in L-SRR is designed for distribution estimation:
    it samples a hierarchical level and applies OLH on the corresponding location
    domain. Since online task assignment needs one proxy grid cell, this
    implementation post-processes the OLH-H report by selecting one grid cell
    from the reported hashed bucket. This is post-processing of the local report
    and does not change the LDP guarantee.
    """

    name = "OLH-H"

    def __init__(self, grid_size: int, epsilon: float, rng: np.random.Generator):
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if grid_size <= 1 or (grid_size & (grid_size - 1)) != 0:
            raise ValueError("OLH-H expects a power-of-two square grid")
        self.grid_size = int(grid_size)
        self.domain_size = self.grid_size * self.grid_size
        self.epsilon = float(epsilon)
        self.rng = rng
        self.depth = int(math.log2(self.grid_size))
        self.exp_eps = math.exp(self.epsilon)

    @staticmethod
    def _hash(values: np.ndarray, seed: int, g: int) -> np.ndarray:
        # SplitMix64-style deterministic hash, vectorized for tile IDs.
        x = np.asarray(values, dtype=np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
        return np.asarray(x % np.uint64(g), dtype=np.int64)

    def _cell_to_tile(self, idx: int, level: int) -> int:
        ix = int(idx) // self.grid_size
        iy = int(idx) % self.grid_size
        side = 1 << int(level)
        block = self.grid_size // side
        tx = ix // block
        ty = iy // block
        return int(tx * side + ty)

    def _tile_to_random_cell(self, tile: int, level: int) -> int:
        side = 1 << int(level)
        block = self.grid_size // side
        tx = int(tile) // side
        ty = int(tile) % side
        ix = tx * block + int(self.rng.integers(0, block))
        iy = ty * block + int(self.rng.integers(0, block))
        return int(ix * self.grid_size + iy)

    def perturb_one(self, true_idx: int) -> int:
        if self.domain_size <= 1:
            return 0
        # OLH-H randomly reports one hierarchy level. Level 1 is the coarsest
        # non-root level; self.depth is the leaf grid-cell level.
        level = int(self.rng.integers(1, self.depth + 1))
        level_domain_size = 4 ** level
        true_tile = self._cell_to_tile(int(true_idx), level)

        # OLH uses a smaller hashed range. If the hashed range would exceed the
        # level domain, it degenerates to randomized response on that level.
        g = int(max(2, min(level_domain_size, round(self.exp_eps) + 1)))
        seed = int(self.rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        true_bucket = int(self._hash(np.asarray([true_tile], dtype=np.int64), seed, g)[0])

        p_keep = self.exp_eps / (self.exp_eps + g - 1)
        if self.rng.random() < p_keep:
            noisy_bucket = true_bucket
        else:
            sampled = int(self.rng.integers(0, g - 1))
            noisy_bucket = sampled + (sampled >= true_bucket)

        tiles = np.arange(level_domain_size, dtype=np.int64)
        candidates = np.flatnonzero(self._hash(tiles, seed, g) == noisy_bucket)
        if len(candidates) == 0:
            noisy_tile = int(self.rng.integers(0, level_domain_size))
        else:
            noisy_tile = int(self.rng.choice(candidates))
        return self._tile_to_random_cell(noisy_tile, level)

    def perturb_indices(self, true_indices: np.ndarray) -> np.ndarray:
        return np.asarray([self.perturb_one(int(i)) for i in true_indices], dtype=int)

class PLDPMechanism:
    """
    Unary-Encoding/OUE-style LDP baseline used as a PLDP-compatible comparison.

    Each location is encoded as a one-hot vector. The true bit is kept with
    probability p=1/2, and every false bit is flipped to 1 with
    q=1/(e^epsilon+1), which is the standard OUE setting. Since online task
    assignment needs one proxy location, the server post-processes the noisy
    unary vector by sampling uniformly from reported 1-bits; if no bit is 1, it
    samples uniformly from the whole domain. This post-processing preserves LDP.

    This baseline is useful as a unary/PLDP-style LDP comparison, but it is not
    expected to be strong for individual point-to-point matching when the domain
    is large, because many false positive cells may be reported.
    """

    name = "PLDP"

    def __init__(self, domain_size: int, epsilon: float, rng: np.random.Generator):
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        self.domain_size = int(domain_size)
        self.epsilon = float(epsilon)
        self.rng = rng
        self.p_true = 0.5
        self.q_false = 1.0 / (math.exp(self.epsilon) + 1.0)

    def perturb_one(self, true_idx: int) -> int:
        if self.domain_size <= 1:
            return 0
        bits = self.rng.random(self.domain_size) < self.q_false
        bits[int(true_idx)] = self.rng.random() < self.p_true
        ones = np.flatnonzero(bits)
        if len(ones) == 0:
            return int(self.rng.integers(0, self.domain_size))
        return int(self.rng.choice(ones))

    def perturb_indices(self, true_indices: np.ndarray) -> np.ndarray:
        return np.asarray([self.perturb_one(int(i)) for i in true_indices], dtype=int)

class SRRMechanism:
    """
    Finite-domain distance-group Staircase Randomized Response.

    Domain points are sorted by distance from the true point and split into m groups.
    Closer groups get higher per-location probabilities. The per-output probability
    ratio is bounded by exp(epsilon), hence epsilon-LDP over the finite domain.
    """

    name = "SRR"

    def __init__(
        self,
        domain: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
        m: int = 5,
        precompute_orders: bool = True,
    ):
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if m < 2:
            raise ValueError("m must be >= 2")
        self.domain = np.asarray(domain, dtype=float)
        self.domain_size = len(domain)
        self.epsilon = float(epsilon)
        self.rng = rng
        self.m = int(min(m, self.domain_size))

        base = self.domain_size // self.m
        rem = self.domain_size % self.m
        self.group_sizes = np.asarray([base + (1 if i < rem else 0) for i in range(self.m)], dtype=int)
        self.group_offsets = np.concatenate([[0], np.cumsum(self.group_sizes)])

        k = math.exp(self.epsilon)
        multipliers = np.asarray(
            [1.0 + (self.m - 1 - j) * (k - 1.0) / (self.m - 1) for j in range(self.m)],
            dtype=float,
        )
        alpha_min = 1.0 / float(np.dot(self.group_sizes, multipliers))
        self.per_item_probs = alpha_min * multipliers
        self.group_probs = self.group_sizes * self.per_item_probs
        self.group_probs = self.group_probs / self.group_probs.sum()

        self.orders: Optional[np.ndarray] = None
        if precompute_orders:
            self.orders = self._precompute_distance_orders()

    def _precompute_distance_orders(self) -> np.ndarray:
        d = self.domain_size
        orders = np.empty((d, d), dtype=np.int32)
        for i in range(d):
            dist2 = ((self.domain - self.domain[i]) ** 2).sum(axis=1)
            orders[i] = np.argsort(dist2, kind="stable")
        return orders

    def _order_for(self, idx: int) -> np.ndarray:
        if self.orders is not None:
            return self.orders[idx]
        dist2 = ((self.domain - self.domain[idx]) ** 2).sum(axis=1)
        return np.argsort(dist2, kind="stable")

    def perturb_one(self, true_idx: int) -> int:
        group = int(self.rng.choice(self.m, p=self.group_probs))
        start, end = int(self.group_offsets[group]), int(self.group_offsets[group + 1])
        rank = int(self.rng.integers(start, end))
        order = self._order_for(int(true_idx))
        return int(order[rank])

    def perturb_indices(self, true_indices: np.ndarray) -> np.ndarray:
        return np.asarray([self.perturb_one(int(i)) for i in true_indices], dtype=int)


class LocalSRRMechanism:
    """
    Local-neighborhood SRR for a regular grid domain.

    Two modes are supported:

    1) self_first=False  (SRR-Local)
       G1 contains the nearest cells within radius r1, e.g. 1-hop => 3x3 = 9 cells.

    2) self_first=True   (SRR-SelfFirst)
       G1 contains only the true grid cell. G2 contains the remaining 1-hop ring,
       G3 the 2-hop ring, etc. This is better for online task assignment because
       single-point matching is much more sensitive to location displacement.

    Privacy:
        Groups have fixed sizes for all inputs because we use the nearest-K ordering
        of the full domain. Per-location probabilities have max/min ratio <= exp(eps),
        so the mechanism satisfies epsilon-LDP over the finite domain.
    """

    name = "SRR-Local"

    def __init__(
        self,
        domain: np.ndarray,
        grid_size: int,
        epsilon: float,
        rng: np.random.Generator,
        radii: Sequence[int] = (1, 2, 3, 4),
        probability_schedule: str = "exponential",
        precompute_orders: bool = True,
        self_first: bool = False,
        two_level_high_groups: int = 1,
    ):
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if grid_size * grid_size != len(domain):
            raise ValueError("LocalSRR expects a square grid domain")
        self.domain = np.asarray(domain, dtype=float)
        self.grid_size = int(grid_size)
        self.domain_size = len(domain)
        self.epsilon = float(epsilon)
        self.rng = rng
        self.radii = sorted(set(int(r) for r in radii if int(r) > 0))
        if not self.radii:
            self.radii = [1, 2, 3, 4]
        self.probability_schedule = probability_schedule.lower().strip()
        if self.probability_schedule not in {"linear", "exponential", "two_level"}:
            raise ValueError("probability_schedule must be linear, exponential, or two_level")
        self.self_first = bool(self_first)
        self.two_level_high_groups = max(1, int(two_level_high_groups))

        self.group_sizes = self._make_group_sizes()
        self.m = len(self.group_sizes)
        self.group_offsets = np.concatenate([[0], np.cumsum(self.group_sizes)])
        self.per_item_probs, self.group_probs = self._make_probs()

        self.orders: Optional[np.ndarray] = None
        if precompute_orders:
            self.orders = self._precompute_distance_orders()

    def _make_group_sizes(self) -> np.ndarray:
        # Chebyshev radii cumulative neighborhood sizes:
        # r=1 -> 3x3=9, r=2 -> 5x5=25, etc.
        cumulative: List[int] = []
        for r in self.radii:
            cumulative.append(min(self.domain_size, (2 * r + 1) ** 2))
        cumulative = sorted(set(cumulative))

        sizes: List[int] = []
        if self.self_first:
            # G1 is only the true grid cell. The following groups are rings:
            # 1-hop ring, 2-hop ring, ..., and finally the rest.
            sizes.append(1)
            prev = 1
        else:
            # Original local variant: G1 is the full nearest r1 neighborhood.
            prev = 0

        for k in cumulative:
            if k > prev:
                sizes.append(k - prev)
                prev = k
        if prev < self.domain_size:
            sizes.append(self.domain_size - prev)

        sizes = [s for s in sizes if s > 0]
        if len(sizes) == 1 and self.domain_size > 1:
            sizes = [1, self.domain_size - 1]
        return np.asarray(sizes, dtype=int)

    def _make_probs(self) -> Tuple[np.ndarray, np.ndarray]:
        k = math.exp(self.epsilon)
        m = len(self.group_sizes)
        if m <= 1:
            multipliers = np.ones(1, dtype=float)
        elif self.probability_schedule == "linear":
            multipliers = np.asarray(
                [1.0 + (m - 1 - j) * (k - 1.0) / (m - 1) for j in range(m)],
                dtype=float,
            )
        elif self.probability_schedule == "exponential":
            # Smoothly decreasing staircase. Max/min ratio is exp(eps).
            multipliers = np.asarray(
                [math.exp(self.epsilon * (m - 1 - j) / (m - 1)) for j in range(m)],
                dtype=float,
            )
        else:  # two_level
            # Aggressive high-epsilon variant. For SRR-SelfFirst we usually set
            # two_level_high_groups=2, so both the true cell and the 1-hop ring
            # receive the highest per-location probability. This avoids the
            # high-epsilon weakness where the mechanism still samples too often
            # from medium/far rings.
            multipliers = np.ones(m, dtype=float)
            high = min(m, self.two_level_high_groups)
            multipliers[:high] = k

        alpha_min = 1.0 / float(np.dot(self.group_sizes, multipliers))
        per_item_probs = alpha_min * multipliers
        group_probs = self.group_sizes * per_item_probs
        group_probs = group_probs / group_probs.sum()
        return per_item_probs, group_probs

    def _precompute_distance_orders(self) -> np.ndarray:
        d = self.domain_size
        orders = np.empty((d, d), dtype=np.int32)
        for i in range(d):
            dist2 = ((self.domain - self.domain[i]) ** 2).sum(axis=1)
            orders[i] = np.argsort(dist2, kind="stable")
        return orders

    def _order_for(self, idx: int) -> np.ndarray:
        if self.orders is not None:
            return self.orders[idx]
        dist2 = ((self.domain - self.domain[idx]) ** 2).sum(axis=1)
        return np.argsort(dist2, kind="stable")

    def perturb_one(self, true_idx: int) -> int:
        group = int(self.rng.choice(self.m, p=self.group_probs))
        start, end = int(self.group_offsets[group]), int(self.group_offsets[group + 1])
        rank = int(self.rng.integers(start, end))
        order = self._order_for(int(true_idx))
        return int(order[rank])

    def perturb_indices(self, true_indices: np.ndarray) -> np.ndarray:
        return np.asarray([self.perturb_one(int(i)) for i in true_indices], dtype=int)


@dataclass
class CompleteQuadtreeHST:
    """Complete 4-ary HST over a grid_size x grid_size regular grid."""

    grid_size: int

    def __post_init__(self) -> None:
        if self.grid_size <= 1 or (self.grid_size & (self.grid_size - 1)) != 0:
            raise ValueError("For OLH-H/HST, grid_size must be a power of two, e.g., 8, 16, 32, 64")
        self.branching = 4
        self.depth = int(math.log2(self.grid_size))
        self.domain_size = self.grid_size * self.grid_size

    def index_to_digits(self, idx: int) -> List[int]:
        ix = int(idx) // self.grid_size
        iy = int(idx) % self.grid_size
        digits: List[int] = []
        for bit in range(self.depth - 1, -1, -1):
            xb = (ix >> bit) & 1
            yb = (iy >> bit) & 1
            digits.append(2 * xb + yb)
        return digits

    def digits_to_index(self, digits: Sequence[int]) -> int:
        ix = 0
        iy = 0
        for b in digits:
            ix = (ix << 1) | (int(b) // 2)
            iy = (iy << 1) | (int(b) % 2)
        return ix * self.grid_size + iy

    def lca_level(self, a: int, b: int) -> int:
        if int(a) == int(b):
            return 0
        da = self.index_to_digits(int(a))
        db = self.index_to_digits(int(b))
        first_diff = 0
        while first_diff < self.depth and da[first_diff] == db[first_diff]:
            first_diff += 1
        return self.depth - first_diff

    def tree_distance(self, a: int, b: int) -> float:
        lvl = self.lca_level(int(a), int(b))
        if lvl == 0:
            return 0.0
        return float(2 ** (lvl + 2) - 4)

    def tree_distance_matrix(self) -> np.ndarray:
        d = self.domain_size
        mat = np.zeros((d, d), dtype=float)
        for i in range(d):
            for j in range(i + 1, d):
                val = self.tree_distance(i, j)
                mat[i, j] = mat[j, i] = val
        return mat



# ---------------------------------------------------------------------------
# Matching and evaluation
# ---------------------------------------------------------------------------


def make_euclidean_distance_matrix(domain: np.ndarray) -> np.ndarray:
    diff = domain[:, None, :] - domain[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def online_greedy_match(
    worker_noisy_idx: np.ndarray,
    task_noisy_idx: np.ndarray,
    distance_matrix: np.ndarray,
) -> List[Tuple[int, int]]:
    if len(worker_noisy_idx) < len(task_noisy_idx):
        raise ValueError("num_workers must be >= num_tasks for one-to-one assignment")
    available = np.ones(len(worker_noisy_idx), dtype=bool)
    pairs: List[Tuple[int, int]] = []
    for tid, t_idx in enumerate(task_noisy_idx):
        cand = np.where(available)[0]
        w_noisy = worker_noisy_idx[cand]
        costs = distance_matrix[w_noisy, int(t_idx)]
        best_local = int(np.argmin(costs))
        wid = int(cand[best_local])
        pairs.append((wid, tid))
        available[wid] = False
    return pairs


def online_greedy_match_true_locations(
    workers_true: np.ndarray,
    tasks_true: np.ndarray,
) -> List[Tuple[int, int]]:
    """Online Greedy using continuous true coordinates.

    This is the real no-privacy baseline. It does not discretize locations to
    grid centers before matching, unlike Grid-NoPrivacy-Greedy.
    """
    if len(workers_true) < len(tasks_true):
        raise ValueError("num_workers must be >= num_tasks for one-to-one assignment")
    available = np.ones(len(workers_true), dtype=bool)
    pairs: List[Tuple[int, int]] = []
    for tid, task in enumerate(tasks_true):
        cand = np.where(available)[0]
        diffs = workers_true[cand] - task
        costs = np.sqrt((diffs ** 2).sum(axis=1))
        best_local = int(np.argmin(costs))
        wid = int(cand[best_local])
        pairs.append((wid, tid))
        available[wid] = False
    return pairs


def evaluate_true_distance(pairs: Sequence[Tuple[int, int]], workers_true: np.ndarray, tasks_true: np.ndarray) -> float:
    total = 0.0
    for wid, tid in pairs:
        total += euclidean(workers_true[wid], tasks_true[tid])
    return float(total)


def perturbation_distance(true_idx: np.ndarray, noisy_idx: np.ndarray, domain: np.ndarray) -> float:
    diffs = domain[true_idx] - domain[noisy_idx]
    return float(np.sqrt((diffs ** 2).sum(axis=1)).mean())


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def build_mechanism(
    name: str,
    domain: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    srr_groups: int,
    hst: CompleteQuadtreeHST,
    srr_precompute_max_domain: int,
    grid_size: int,
    local_srr_radii: Sequence[int],
    srr_probability_schedule: str,
    self_first_high_groups: int,
):
    if name == "NoPrivacy":
        return NoPrivacyMechanism(len(domain), rng)
    if name == "GRR":
        return GRRMechanism(len(domain), epsilon, rng)
    if name == "HR":
        return HRMechanism(len(domain), epsilon, rng)
    if name == "OLH-H":
        return OLHHMechanism(grid_size, epsilon, rng)
    if name == "PLDP":
        return PLDPMechanism(len(domain), epsilon, rng)
    if name == "SRR-Equal":
        precompute = len(domain) <= srr_precompute_max_domain
        return SRRMechanism(domain, epsilon, rng, m=srr_groups, precompute_orders=precompute)
    if name == "SRR-Local":
        precompute = len(domain) <= srr_precompute_max_domain
        return LocalSRRMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=rng,
            radii=local_srr_radii,
            probability_schedule=srr_probability_schedule,
            precompute_orders=precompute,
            self_first=False,
            two_level_high_groups=1,
        )
    if name == "SRR-SelfFirst":
        precompute = len(domain) <= srr_precompute_max_domain
        return LocalSRRMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=rng,
            radii=local_srr_radii,
            probability_schedule=srr_probability_schedule,
            precompute_orders=precompute,
            self_first=True,
            two_level_high_groups=self_first_high_groups,
        )
    raise ValueError(f"Unknown mechanism: {name}")


def run_one_setting(
    epsilon: float,
    workers_true: np.ndarray,
    tasks_true: np.ndarray,
    workers_idx: np.ndarray,
    tasks_idx: np.ndarray,
    domain: np.ndarray,
    hst: CompleteQuadtreeHST,
    euclid_dist_matrix: np.ndarray,
    hst_dist_matrix: np.ndarray,
    srr_groups: int,
    seed: int,
    srr_precompute_max_domain: int,
    grid_size: int,
    local_srr_radii: Sequence[int],
    srr_probability_schedule: str,
    self_first_high_groups: int,
    include_hst_greedy: bool,
    include_srr_local: bool,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    # The main comparison among privacy mechanisms is inside the LDP family:
    # GRR, HR, OLH-H, PLDP/OUE, SRR-Equal, and SRR-SelfFirst. SRR-Local is optional ablation only.
    base_mechanisms = [
        ("GRR", "GRR"),
        ("HR", "HR"),
        ("OLH-H", "OLH-H"),
        ("PLDP", "PLDP"),
        ("SRR-Equal", "SRR-Equal"),
        ("SRR-SelfFirst", "SRR-SelfFirst"),
    ]
    if include_srr_local:
        # SRR-Local is useful as an ablation bridge between SRR-Equal and
        # SRR-SelfFirst. It is excluded from main figures by default because
        # under two_level with self_first_high_groups=2 it has nearly the same
        # high-probability neighborhood as SRR-SelfFirst.
        insert_pos = next(i for i, (_, mech) in enumerate(base_mechanisms) if mech == "SRR-SelfFirst")
        base_mechanisms.insert(insert_pos, ("SRR-Local", "SRR-Local"))
    method_specs = [(f"{label}-Greedy", mech, "euclidean") for label, mech in base_mechanisms]

    # Optional diagnostic: also run HSTGreedy for every non-NoPrivacy mechanism.
    # This is off by default because it can make the figures crowded.
    if include_hst_greedy:
        method_specs.extend(
            (f"{label}-HSTGreedy", mech, "hst")
            for label, mech in base_mechanisms
            if mech != "NoPrivacy"
        )
    perturbed_cache: Dict[str, Tuple[np.ndarray, np.ndarray, float, float]] = {}

    for method_label, mech_name, matcher in method_specs:
        t0 = time.time()
        if mech_name not in perturbed_cache:
            mech_rng = np.random.default_rng(stable_seed(seed, epsilon, mech_name))
            mech = build_mechanism(
                mech_name,
                domain,
                epsilon,
                mech_rng,
                srr_groups,
                hst,
                srr_precompute_max_domain,
                grid_size,
                local_srr_radii,
                srr_probability_schedule,
                self_first_high_groups,
            )
            w_noisy = mech.perturb_indices(workers_idx)
            t_noisy = mech.perturb_indices(tasks_idx)
            w_pdist = perturbation_distance(workers_idx, w_noisy, domain)
            t_pdist = perturbation_distance(tasks_idx, t_noisy, domain)
            perturbed_cache[mech_name] = (w_noisy, t_noisy, w_pdist, t_pdist)
        else:
            w_noisy, t_noisy, w_pdist, t_pdist = perturbed_cache[mech_name]

        dist_matrix = euclid_dist_matrix if matcher == "euclidean" else hst_dist_matrix
        pairs = online_greedy_match(w_noisy, t_noisy, dist_matrix)
        total_dist = evaluate_true_distance(pairs, workers_true, tasks_true)
        elapsed = time.time() - t0
        rows.append(
            {
                "epsilon": float(epsilon),
                "method": method_label,
                "total_true_distance_km": total_dist,
                "avg_true_distance_km": total_dist / len(tasks_true),
                "worker_perturbation_distance_km": w_pdist,
                "task_perturbation_distance_km": t_pdist,
                "runtime_sec": elapsed,
            }
        )
    return rows


def aggregate_rows(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[float, str], List[Dict[str, float]]] = {}
    for r in rows:
        grouped.setdefault((float(r["epsilon"]), str(r["method"])), []).append(r)

    out: List[Dict[str, float]] = []
    metrics = [
        "total_true_distance_km",
        "avg_true_distance_km",
        "worker_perturbation_distance_km",
        "task_perturbation_distance_km",
        "runtime_sec",
    ]
    for (eps, method), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        row: Dict[str, float] = {"epsilon": eps, "method": method, "repeat": len(items)}  # type: ignore
        for m in metrics:
            vals = np.asarray([float(it[m]) for it in items], dtype=float)
            row[m + "_mean"] = float(vals.mean())
            row[m + "_std"] = float(vals.std(ddof=0))
        out.append(row)
    return out


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path: str, rows: List[Dict[str, float]]) -> None:
    if plt is None or not rows:
        return
    methods = sorted({str(r["method"]) for r in rows})
    plt.figure(figsize=(10, 6))
    for method in methods:
        sub = [r for r in rows if str(r["method"]) == method]
        sub = sorted(sub, key=lambda r: float(r["epsilon"]))
        x = [float(r["epsilon"]) for r in sub]
        y = [float(r["avg_true_distance_km_mean"]) for r in sub]
        plt.plot(x, y, marker="o", label=method)
    plt.xlabel("epsilon")
    plt.ylabel("Average true travel distance (km)")
    plt.title("Foursquare online task assignment under privacy mechanisms")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_perturbation_results(path: str, rows: List[Dict[str, float]]) -> None:
    """Plot average perturbation distance of workers/tasks for diagnostics."""
    if plt is None or not rows:
        return
    methods = sorted({str(r["method"]) for r in rows})
    plt.figure(figsize=(10, 6))
    for method in methods:
        sub = [r for r in rows if str(r["method"]) == method]
        sub = sorted(sub, key=lambda r: float(r["epsilon"]))
        x = [float(r["epsilon"]) for r in sub]
        y = [
            0.5 * (
                float(r["worker_perturbation_distance_km_mean"])
                + float(r["task_perturbation_distance_km_mean"])
            )
            for r in sub
        ]
        plt.plot(x, y, marker="o", label=method)
    plt.xlabel("epsilon")
    plt.ylabel("Mean perturbation distance (km)")
    plt.title("Perturbation distance diagnostics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# Paper-oriented method groups. The LDP comparison is GRR/HR/OLH-H/PLDP/SRR variants.
LDP_PLOT_METHODS = [
    "GRR-Greedy",
    "HR-Greedy",
    "OLH-H-Greedy",
    "PLDP-Greedy",
    "SRR-Equal-Greedy",
    "SRR-SelfFirst-Greedy",
]

# Optional ablation method. It is not in main figures by default because its
# curve nearly overlaps SRR-SelfFirst under the current two-level setting.
OPTIONAL_ABLATION_METHODS = [
    "SRR-Local-Greedy",
]

REFERENCE_PLOT_METHODS = [
    "GRR-Greedy",
    "OLH-H-Greedy",
    "PLDP-Greedy",
    "SRR-Equal-Greedy",
    "SRR-SelfFirst-Greedy",
]

PERTURBATION_CALIBRATION_METHODS = [
    "GRR-Greedy",
    "HR-Greedy",
    "OLH-H-Greedy",
    "PLDP-Greedy",
    "SRR-Equal-Greedy",
    "SRR-SelfFirst-Greedy",
]

GRID_SENSITIVITY_METHODS = [
    "GRR-Greedy",
    "OLH-H-Greedy",
    "PLDP-Greedy",
    "SRR-Equal-Greedy",
    "SRR-SelfFirst-Greedy",
]

WORKLOAD_SENSITIVITY_METHODS = [
    "GRR-Greedy",
    "HR-Greedy",
    "OLH-H-Greedy",
    "PLDP-Greedy",
    "SRR-Equal-Greedy",
    "SRR-SelfFirst-Greedy",
]


def filter_summary_rows(rows: List[Dict[str, float]], methods: Sequence[str]) -> List[Dict[str, float]]:
    method_set = set(methods)
    filtered = [r for r in rows if str(r.get("method")) in method_set]
    order = {m: i for i, m in enumerate(methods)}
    return sorted(filtered, key=lambda r: (float(r["epsilon"]), order.get(str(r.get("method")), 999)))




def mean_perturbation_from_summary_row(row: Dict[str, float]) -> float:
    """Worker/task mean perturbation distance used by diagnostic plots."""
    return 0.5 * (
        float(row["worker_perturbation_distance_km_mean"])
        + float(row["task_perturbation_distance_km_mean"])
    )


def plot_perturbation_efficiency_curve(
    path: str,
    rows: List[Dict[str, float]],
    methods: Sequence[str] = PERTURBATION_CALIBRATION_METHODS,
) -> None:
    """Plot task-assignment cost against mean perturbation distance.

    This is the key "same perturbation distance" diagnostic. Epsilon is no
    longer the x-axis. Points that are horizontally close have comparable
    average displacement, making the LDP mechanisms easier to compare in the
    online assignment setting.
    """
    if plt is None or not rows:
        return
    method_set = set(methods)
    plt.figure(figsize=(10, 6))
    for method in methods:
        sub = [r for r in rows if str(r.get("method")) == method and str(r.get("method")) in method_set]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: mean_perturbation_from_summary_row(r))
        x = [mean_perturbation_from_summary_row(r) for r in sub]
        y = [float(r["avg_true_distance_km_mean"]) for r in sub]
        plt.plot(x, y, marker="o", label=method)
    plt.xlabel("Mean perturbation distance (km)")
    plt.ylabel("Average true travel distance (km)")
    plt.title("Task-assignment cost under comparable perturbation distances")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def calibrate_by_perturbation_targets(
    rows: List[Dict[str, float]],
    targets: Sequence[float],
    methods: Sequence[str] = PERTURBATION_CALIBRATION_METHODS,
) -> List[Dict[str, float]]:
    """For each target perturbation distance, select each method's closest epsilon.

    The output lets us compare methods at approximately the same average
    displacement rather than at the same numeric epsilon. This is especially
    important because assignment performance is strongly driven by single-point
    displacement.
    """
    out: List[Dict[str, float]] = []
    for target in targets:
        for method in methods:
            candidates = [r for r in rows if str(r.get("method")) == method]
            if not candidates:
                continue
            best = min(candidates, key=lambda r: abs(mean_perturbation_from_summary_row(r) - float(target)))
            mean_p = mean_perturbation_from_summary_row(best)
            out.append(
                {
                    "target_perturbation_km": float(target),
                    "method": method,
                    "selected_epsilon": float(best["epsilon"]),
                    "mean_perturbation_km": mean_p,
                    "perturbation_gap_km": abs(mean_p - float(target)),
                    "avg_true_distance_km_mean": float(best["avg_true_distance_km_mean"]),
                    "avg_true_distance_km_std": float(best["avg_true_distance_km_std"]),
                    "worker_perturbation_distance_km_mean": float(best["worker_perturbation_distance_km_mean"]),
                    "task_perturbation_distance_km_mean": float(best["task_perturbation_distance_km_mean"]),
                }
            )
    return out


def plot_calibrated_perturbation_comparison(path: str, rows: List[Dict[str, float]]) -> None:
    """Plot calibrated task distance after nearest-target perturbation matching."""
    if plt is None or not rows:
        return
    methods = [m for m in PERTURBATION_CALIBRATION_METHODS if any(str(r.get("method")) == m for r in rows)]
    plt.figure(figsize=(10, 6))
    for method in methods:
        sub = [r for r in rows if str(r.get("method")) == method]
        sub = sorted(sub, key=lambda r: float(r["target_perturbation_km"]))
        x = [float(r["target_perturbation_km"]) for r in sub]
        y = [float(r["avg_true_distance_km_mean"]) for r in sub]
        plt.plot(x, y, marker="o", label=method)
    plt.xlabel("Target mean perturbation distance (km)")
    plt.ylabel("Average true travel distance (km)")
    plt.title("Approximate same-perturbation-distance comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def combine_grid_sensitivity_results(parent_out_dir: str, grid_sizes: Sequence[int]) -> List[Dict[str, float]]:
    """Read child summary files produced by --run-grid-sensitivity."""
    combined: List[Dict[str, float]] = []
    for g in grid_sizes:
        summary_path = os.path.join(parent_out_dir, f"grid_{g}", "summary_results.csv")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)  # type: ignore
                row["grid_size"] = int(g)
                combined.append(row)  # type: ignore
    return combined


def plot_grid_sensitivity(path: str, rows: List[Dict[str, float]], epsilon: Optional[float] = None) -> None:
    """Plot average distance vs grid size for a selected epsilon.

    If epsilon is None, use the largest epsilon found in the combined rows.
    """
    if plt is None or not rows:
        return
    eps_vals = sorted({float(r["epsilon"]) for r in rows})
    if not eps_vals:
        return
    eps = float(epsilon if epsilon is not None else eps_vals[-1])
    # Pick nearest available epsilon in case the requested one is not present.
    eps = min(eps_vals, key=lambda x: abs(x - eps))
    plt.figure(figsize=(10, 6))
    for method in GRID_SENSITIVITY_METHODS:
        sub = [r for r in rows if str(r.get("method")) == method and abs(float(r["epsilon"]) - eps) < 1e-9]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: int(r["grid_size"]))
        x = [int(r["grid_size"]) for r in sub]
        y = [float(r["avg_true_distance_km_mean"]) for r in sub]
        plt.plot(x, y, marker="o", label=method)
    plt.xlabel("grid size")
    plt.ylabel("Average true travel distance (km)")
    plt.title(f"Grid-size sensitivity at epsilon={eps:g}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_grid_sensitivity_via_subprocess(args: argparse.Namespace) -> None:
    """Run several grid-size experiments as child processes and combine results.

    This mode intentionally reuses the single-grid experiment path. It keeps the
    core experiment implementation identical across main and grid-sensitivity
    runs, and avoids silently changing defaults.
    """
    grid_sizes = parse_int_list(args.grid_sensitivity_sizes)
    epsilons = args.grid_sensitivity_epsilons
    repeats = int(args.grid_sensitivity_repeats)
    os.makedirs(args.out_dir, exist_ok=True)

    script = os.path.abspath(__file__)
    for g in grid_sizes:
        child_out = os.path.join(args.out_dir, f"grid_{g}")
        cmd = [
            sys.executable,
            script,
            "--foursquare-dir", args.foursquare_dir,
            "--foursquare-max-points", str(args.foursquare_max_points),
            "--foursquare-scan-limit", str(args.foursquare_scan_limit),
            "--foursquare-lat-col", str(args.foursquare_lat_col),
            "--foursquare-lon-col", str(args.foursquare_lon_col),
            "--grid-size", str(g),
            "--num-workers", str(args.num_workers),
            "--num-tasks", str(args.num_tasks),
            "--epsilons", epsilons,
            "--repeats", str(repeats),
            "--srr-groups", str(args.srr_groups),
            "--local-srr-radii", args.local_srr_radii,
            "--srr-probability-schedule", args.srr_probability_schedule,
            "--self-first-high-groups", str(args.self_first_high_groups),
            "--srr-precompute-max-domain", str(args.srr_precompute_max_domain),
            "--seed", str(args.seed),
            "--out-dir", child_out,
        ]
        if args.foursquare_bbox:
            cmd += ["--foursquare-bbox", args.foursquare_bbox]
        if args.disable_auto_bbox:
            cmd += ["--disable-auto-bbox"]
        if args.allow_synthetic_fallback:
            cmd += ["--allow-synthetic-fallback"]
        if args.include_hst_greedy:
            cmd += ["--include-hst-greedy"]
        if args.include_srr_local:
            cmd += ["--include-srr-local"]
        print("\n[grid sensitivity] running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    combined = combine_grid_sensitivity_results(args.out_dir, grid_sizes)
    combined_path = os.path.join(args.out_dir, "grid_sensitivity_combined_summary.csv")
    write_csv(combined_path, combined)  # type: ignore[arg-type]
    plot_grid_sensitivity(os.path.join(args.out_dir, "grid_sensitivity_avg_distance.png"), combined)
    print("\nGrid sensitivity complete.")
    print(f"Combined summary: {combined_path}")
    print(f"Grid plot:        {os.path.join(args.out_dir, 'grid_sensitivity_avg_distance.png')}")




def combine_workload_sensitivity_results(parent_out_dir: str) -> List[Dict[str, float]]:
    """Read child summary files produced by --run-workload-sensitivity.

    The resulting rows keep all normal summary metrics and add:
        sensitivity_type: "workers" or "tasks"
        num_workers: worker count used by the child run
        num_tasks: task count used by the child run
    """
    combined: List[Dict[str, float]] = []
    for sensitivity_type in ["workers", "tasks"]:
        root = os.path.join(parent_out_dir, sensitivity_type)
        if not os.path.isdir(root):
            continue
        for child_name in sorted(os.listdir(root)):
            child_dir = os.path.join(root, child_name)
            summary_path = os.path.join(child_dir, "summary_results.csv")
            metadata_path = os.path.join(child_dir, "metadata.txt")
            if not os.path.exists(summary_path):
                continue

            # Parse child metadata for worker/task counts. This keeps the
            # combiner robust even if directory names are changed later.
            child_workers: Optional[int] = None
            child_tasks: Optional[int] = None
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as mf:
                    for line in mf:
                        line = line.strip()
                        if line.startswith("arg.num_workers:"):
                            child_workers = int(float(line.split(":", 1)[1].strip()))
                        elif line.startswith("arg.num_tasks:"):
                            child_tasks = int(float(line.split(":", 1)[1].strip()))

            with open(summary_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row = dict(row)  # type: ignore
                    row["sensitivity_type"] = sensitivity_type
                    if child_workers is not None:
                        row["num_workers"] = child_workers
                    if child_tasks is not None:
                        row["num_tasks"] = child_tasks
                    combined.append(row)  # type: ignore
    return combined


def plot_workload_total_distance(
    path: str,
    rows: List[Dict[str, float]],
    sensitivity_type: str,
    methods: Sequence[str] = WORKLOAD_SENSITIVITY_METHODS,
) -> None:
    """Plot total true assignment distance as worker/task count changes.

    For sensitivity_type="workers", the x-axis is num_workers while num_tasks
    is fixed. For sensitivity_type="tasks", the x-axis is num_tasks while
    num_workers is fixed. The y-axis is total_true_distance_km_mean, not the
    average distance. This directly answers how the total system travel cost
    changes when the market size changes under the same privacy budget.
    """
    if plt is None or not rows:
        return
    x_key = "num_workers" if sensitivity_type == "workers" else "num_tasks"
    title = (
        "Total assignment distance vs number of workers"
        if sensitivity_type == "workers"
        else "Total assignment distance vs number of tasks"
    )
    xlabel = "Number of workers" if sensitivity_type == "workers" else "Number of tasks"

    subset_all = [r for r in rows if str(r.get("sensitivity_type")) == sensitivity_type]
    if not subset_all:
        return

    plt.figure(figsize=(10, 6))
    for method in methods:
        sub = [r for r in subset_all if str(r.get("method")) == method and x_key in r]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: int(float(r[x_key])))
        x = [int(float(r[x_key])) for r in sub]
        y = [float(r["total_true_distance_km_mean"]) for r in sub]
        plt.plot(x, y, marker="o", label=method)
    # Add a short subtitle-like note about fixed quantities when available.
    eps_vals = sorted({float(r["epsilon"]) for r in subset_all if "epsilon" in r})
    fixed_note = ""
    if sensitivity_type == "workers":
        task_vals = sorted({int(float(r["num_tasks"])) for r in subset_all if "num_tasks" in r})
        if len(task_vals) == 1:
            fixed_note = f"; fixed tasks={task_vals[0]}"
    else:
        worker_vals = sorted({int(float(r["num_workers"])) for r in subset_all if "num_workers" in r})
        if len(worker_vals) == 1:
            fixed_note = f"; fixed workers={worker_vals[0]}"
    eps_note = f" at epsilon={eps_vals[0]:g}" if len(eps_vals) == 1 else ""
    plt.xlabel(xlabel)
    plt.ylabel("Total true travel distance (km)")
    plt.title(title + eps_note + fixed_note)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_workload_sensitivity_via_subprocess(args: argparse.Namespace) -> None:
    """Run worker-count and task-count sensitivity suites and make two plots.

    Worker-count figure:
        x-axis = number of workers, fixed number of tasks, fixed epsilon.
    Task-count figure:
        x-axis = number of tasks, fixed number of workers, fixed epsilon.
    Both figures use total_true_distance_km_mean as the y-axis.
    """
    worker_values = parse_int_list(args.worker_sensitivity_workers)
    task_values = parse_int_list(args.task_sensitivity_tasks)
    fixed_tasks_for_workers = int(args.worker_sensitivity_num_tasks)
    fixed_workers_for_tasks = int(args.task_sensitivity_num_workers)
    eps = float(args.workload_sensitivity_epsilon)
    repeats = int(args.workload_sensitivity_repeats)

    if any(w < fixed_tasks_for_workers for w in worker_values):
        raise ValueError("Every worker-sensitivity worker count must be >= --worker-sensitivity-num-tasks")
    if fixed_workers_for_tasks < max(task_values):
        raise ValueError("--task-sensitivity-num-workers must be >= every value in --task-sensitivity-tasks")

    os.makedirs(args.out_dir, exist_ok=True)
    script = os.path.abspath(__file__)

    def run_child(child_out: str, num_workers: int, num_tasks: int) -> None:
        cmd = [
            sys.executable,
            script,
            "--foursquare-dir", args.foursquare_dir,
            "--foursquare-max-points", str(args.foursquare_max_points),
            "--foursquare-scan-limit", str(args.foursquare_scan_limit),
            "--foursquare-lat-col", str(args.foursquare_lat_col),
            "--foursquare-lon-col", str(args.foursquare_lon_col),
            "--grid-size", str(args.grid_size),
            "--num-workers", str(num_workers),
            "--num-tasks", str(num_tasks),
            "--epsilons", f"{eps:g}",
            "--repeats", str(repeats),
            "--srr-groups", str(args.srr_groups),
            "--local-srr-radii", args.local_srr_radii,
            "--srr-probability-schedule", args.srr_probability_schedule,
            "--self-first-high-groups", str(args.self_first_high_groups),
            "--srr-precompute-max-domain", str(args.srr_precompute_max_domain),
            "--seed", str(args.seed),
            "--out-dir", child_out,
            "--disable-perturbation-calibration",
        ]
        if args.foursquare_bbox:
            cmd += ["--foursquare-bbox", args.foursquare_bbox]
        if args.disable_auto_bbox:
            cmd += ["--disable-auto-bbox"]
        if args.allow_synthetic_fallback:
            cmd += ["--allow-synthetic-fallback"]
        if args.include_hst_greedy:
            cmd += ["--include-hst-greedy"]
        if args.include_srr_local:
            cmd += ["--include-srr-local"]
        print("\n[workload sensitivity] running:", " ".join(cmd))
        child_env = os.environ.copy()
        child_env["SRR_SKIP_CODE_WORKLOAD"] = "1"
        subprocess.run(cmd, check=True, env=child_env)

    for w in worker_values:
        child_out = os.path.join(args.out_dir, "workers", f"workers_{w}_tasks_{fixed_tasks_for_workers}")
        run_child(child_out=child_out, num_workers=w, num_tasks=fixed_tasks_for_workers)

    for t in task_values:
        child_out = os.path.join(args.out_dir, "tasks", f"workers_{fixed_workers_for_tasks}_tasks_{t}")
        run_child(child_out=child_out, num_workers=fixed_workers_for_tasks, num_tasks=t)

    combined = combine_workload_sensitivity_results(args.out_dir)
    combined_path = os.path.join(args.out_dir, "workload_sensitivity_combined_summary.csv")
    write_csv(combined_path, combined)  # type: ignore[arg-type]
    workers_fig = os.path.join(args.out_dir, "total_distance_vs_workers.png")
    tasks_fig = os.path.join(args.out_dir, "total_distance_vs_tasks.png")
    plot_workload_total_distance(workers_fig, combined, sensitivity_type="workers")
    plot_workload_total_distance(tasks_fig, combined, sensitivity_type="tasks")

    print("\nWorkload sensitivity complete.")
    print(f"Combined summary: {combined_path}")
    print(f"Workers plot:     {workers_fig}")
    print(f"Tasks plot:       {tasks_fig}")

def write_recommended_formal_commands(path: str, script_name: str) -> None:
    """Write optional commands for robustness/sensitivity experiments."""
    commands = []
    base = (
        f'python {script_name} ^\n'
        '  --foursquare-dir "D:\\data2" ^\n'
        '  --foursquare-max-points 400000 ^\n'
    )
    commands.append("REM Main formal experiment")
    commands.append(base +
        '  --grid-size 16 ^\n'
        '  --num-workers 2000 ^\n'
        '  --num-tasks 1000 ^\n'
        '  --epsilons 0.1,0.5,1.0,2.0,4.0,6.0,8.0 ^\n'
        '  --repeats 10 ^\n'
        '  --out-dir foursquare_formal_6')
    commands.append("\nREM Grid-size sensitivity: run grid 8, 16, and 32 separately")
    for g in [8, 16, 32]:
        commands.append(base +
            f'  --grid-size {g} ^\n'
            '  --num-workers 2000 ^\n'
            '  --num-tasks 1000 ^\n'
            '  --epsilons 2.0,4.0,6.0,8.0 ^\n'
            '  --repeats 5 ^\n'
            f'  --out-dir foursquare_formal_grid{g}')
    commands.append("\nREM Workload sensitivity: run three workloads separately")
    for w,t in [(1000,500),(2000,1000),(4000,2000)]:
        commands.append(base +
            '  --grid-size 16 ^\n'
            f'  --num-workers {w} ^\n'
            f'  --num-tasks {t} ^\n'
            '  --epsilons 2.0,4.0,6.0,8.0 ^\n'
            '  --repeats 5 ^\n'
            f'  --out-dir foursquare_formal_workload_w{w}_t{t}')
    commands.append("\nREM Appendix HSTGreedy diagnostics")
    commands.append(base +
        '  --grid-size 16 ^\n'
        '  --num-workers 1000 ^\n'
        '  --num-tasks 500 ^\n'
        '  --epsilons 1.0,2.0,4.0,8.0 ^\n'
        '  --repeats 5 ^\n'
        '  --include-hst-greedy ^\n'
        '  --out-dir foursquare_appendix_hstgreedy')
    with open(path, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write("REM Recommended formal experiment commands generated by the script.\n")
        f.write("REM Run one block at a time if the full batch is too long.\n\n")
        f.write("\n\n".join(commands))
        f.write("\n")


def write_metadata(path: str, meta: Dict[str, float], args: argparse.Namespace) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Experiment metadata\n")
        f.write("===================\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"arg.{k}: {v}\n")
        for k, v in sorted(meta.items()):
            f.write(f"data.{k}: {v}\n")



# ---------------------------------------------------------------------------
# Distribution-estimation experiment for SRR optimality diagnostics
# ---------------------------------------------------------------------------

SRR_DISTRIBUTION_LABEL = "SRR-SelfFirst"
DISTRIBUTION_METHODS = ["GRR", "HR", "OLH-H", "PLDP", SRR_DISTRIBUTION_LABEL]
DISTRIBUTION_ERROR_METRICS = [
    "l1_error",
    "total_variation_error",
    "js_divergence",
    "rmse",
]
DISTRIBUTION_UTILITY_METRICS = [
    "density_correlation",
    "knn_precision_at_k",
    "knn_recall_at_k",
    "knn_f1_at_k",
]


def empirical_distribution(indices: np.ndarray, domain_size: int) -> np.ndarray:
    counts = np.bincount(np.asarray(indices, dtype=int), minlength=int(domain_size)).astype(float)
    total = counts.sum()
    if total <= 0:
        return np.full(int(domain_size), 1.0 / float(domain_size), dtype=float)
    return counts / total


def normalize_distribution(values: np.ndarray, min_prob: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    if min_prob > 0.0:
        arr = np.maximum(arr, min_prob)
    else:
        arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(len(arr), 1.0 / float(len(arr)), dtype=float)
    return arr / total


def project_to_probability_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    v = np.asarray(values, dtype=float).copy()
    v[~np.isfinite(v)] = 0.0
    n = len(v)
    if n == 0:
        raise ValueError("cannot project an empty vector")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / float(n), dtype=float)
    rho = int(ind[cond][-1])
    theta = float(cssv[cond][-1] / rho)
    w = np.maximum(v - theta, 0.0)
    return normalize_distribution(w)


def smooth_distribution_grid(
    distribution: np.ndarray,
    grid_size: int,
    rounds: int = 1,
    center_weight: float = 4.0,
) -> np.ndarray:
    """Lightweight 3x3 spatial smoothing on a square grid."""
    if rounds <= 0:
        return normalize_distribution(distribution)
    if int(grid_size) * int(grid_size) != len(distribution):
        return normalize_distribution(distribution)
    g = normalize_distribution(distribution).reshape(int(grid_size), int(grid_size))
    cw = max(float(center_weight), 0.0)
    for _ in range(int(rounds)):
        padded = np.pad(g, 1, mode="edge")
        neighbor_sum = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        )
        g = (cw * g + neighbor_sum) / (cw + 8.0)
    return normalize_distribution(g.ravel())


def grid_laplacian_matrix(grid_size: int) -> np.ndarray:
    """Combinatorial Laplacian for a 4-neighbor grid."""
    g = int(grid_size)
    d = g * g
    lap = np.zeros((d, d), dtype=float)
    for ix in range(g):
        for iy in range(g):
            idx = ix * g + iy
            neighbors = []
            if ix > 0:
                neighbors.append((ix - 1) * g + iy)
            if ix + 1 < g:
                neighbors.append((ix + 1) * g + iy)
            if iy > 0:
                neighbors.append(ix * g + iy - 1)
            if iy + 1 < g:
                neighbors.append(ix * g + iy + 1)
            lap[idx, idx] = float(len(neighbors))
            for nb in neighbors:
                lap[idx, nb] = -1.0
    return lap


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_distribution(p, min_prob=1e-15)
    q = normalize_distribution(q, min_prob=1e-15)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def density_correlation(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    if float(np.std(p)) <= 1e-15 or float(np.std(q)) <= 1e-15:
        return 0.0
    return float(np.corrcoef(p, q)[0, 1])





def get_srr_adaptive_knn_local_mass_weight(epsilon: float, args: argparse.Namespace) -> float:
    """Return epsilon-adaptive local-mass weight used only by SRR k-NN post-processing.

    Rationale:
        - Low epsilon has stronger randomization noise, so SRR benefits from a
          small local-density mass compensation.
        - Medium epsilon uses weaker compensation.
        - High epsilon disables compensation to avoid over-smoothing the ranking.

    This is post-processing on the estimated density and does not consume extra
    privacy budget.
    """
    if not bool(getattr(args, "dist_srr_adaptive_knn_local_mass", True)):
        return 0.0

    eps = float(epsilon)
    low_thr = float(getattr(args, "dist_srr_knn_low_eps_threshold", 4.0))
    mid_thr = float(getattr(args, "dist_srr_knn_mid_eps_threshold", 6.0))

    if eps <= low_thr:
        return float(getattr(args, "dist_srr_knn_local_mass_weight_low", 0.25))
    if eps <= mid_thr:
        return float(getattr(args, "dist_srr_knn_local_mass_weight_mid", 0.10))
    return float(getattr(args, "dist_srr_knn_local_mass_weight_high", 0.0))


def local_mass_distribution_grid(
    distribution: np.ndarray,
    grid_size: int,
    radius: int = 1,
) -> np.ndarray:
    """Convert a cell distribution to a local-neighborhood mass distribution.

    radius=1 means each cell receives the summed mass in its 3x3 neighborhood.
    The returned vector is normalized and has the same length as distribution.
    """
    p = normalize_distribution(distribution)
    g = int(grid_size)
    if g * g != len(p):
        return p

    r = max(0, int(radius))
    if r <= 0:
        return p

    grid = p.reshape(g, g)
    local = np.zeros_like(grid, dtype=float)
    for ix in range(g):
        x0 = max(0, ix - r)
        x1 = min(g, ix + r + 1)
        for iy in range(g):
            y0 = max(0, iy - r)
            y1 = min(g, iy + r + 1)
            local[ix, iy] = float(grid[x0:x1, y0:y1].sum())
    return normalize_distribution(local.ravel())


def fuse_distribution_with_local_mass(
    distribution: np.ndarray,
    grid_size: int,
    local_mass_weight: float = 0.0,
    local_radius: int = 1,
) -> np.ndarray:
    """Fuse estimated density with local-neighborhood mass for density-aware k-NN.

    effective_density = (1-w) * density + w * local_mass_density

    The fusion is intended for SRR only. It is a post-processing step, so it
    does not change the privacy guarantee.
    """
    p = normalize_distribution(distribution)
    w = min(max(float(local_mass_weight), 0.0), 1.0)
    if w <= 0.0:
        return p
    local = local_mass_distribution_grid(p, int(grid_size), radius=int(local_radius))
    return normalize_distribution((1.0 - w) * p + w * local)


def select_knn_query_indices(
    true_dist: np.ndarray,
    num_queries: int,
    rng: np.random.Generator,
    mode: str = "density_sample",
) -> np.ndarray:
    """Select query grid cells used by the density-aware k-NN evaluation.

    density_sample:
        sample query cells according to the true spatial density; this models
        queries appearing more often in active regions.
    top_density:
        use the densest true cells as queries.
    uniform:
        sample query cells uniformly from the grid.
    all:
        evaluate every grid cell as a query.
    """
    p = normalize_distribution(true_dist)
    d = len(p)
    if d <= 0:
        raise ValueError("empty distribution")
    mode = mode.lower().strip()
    if mode == "all":
        return np.arange(d, dtype=int)

    n = int(num_queries)
    if n <= 0 or n >= d:
        n = d

    if mode == "top_density":
        return np.argsort(-p, kind="stable")[:n].astype(int)
    if mode == "uniform":
        return np.asarray(rng.choice(d, size=n, replace=False), dtype=int)
    if mode != "density_sample":
        raise ValueError("--dist-knn-query-mode must be density_sample, top_density, uniform, or all")

    positive = np.flatnonzero(p > 0)
    if len(positive) >= n:
        return np.asarray(rng.choice(d, size=n, replace=False, p=p), dtype=int)

    # If there are fewer positive-density cells than queries, keep all positive
    # cells and fill the rest uniformly. This avoids sampling duplicate queries.
    selected = list(map(int, positive))
    remaining = np.setdiff1d(np.arange(d), positive, assume_unique=True)
    need = n - len(selected)
    if need > 0 and len(remaining) > 0:
        fill = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
        selected.extend(map(int, fill))
    return np.asarray(selected[:n], dtype=int)


def top_k_density_aware_neighbors(
    distribution: np.ndarray,
    distance_matrix: np.ndarray,
    query_idx: int,
    k: int,
    distance_floor: float,
    density_power: float = 1.0,
    distance_power: float = 1.0,
    exclude_self: bool = True,
) -> np.ndarray:
    """Return top-k cells for a density-aware k-NN query.

    The ranking score is:

        score(cell | query) = density(cell)^density_power
                              / (distance(query, cell) + distance_floor)^distance_power

    This uses the estimated spatial distribution as a compact index. The true
    answer set is computed with the same scoring rule using the true density.
    """
    p = normalize_distribution(distribution)
    k = max(1, min(int(k), len(p) - (1 if exclude_self and len(p) > 1 else 0)))
    floor = max(float(distance_floor), 1e-12)
    dist = np.asarray(distance_matrix[int(query_idx)], dtype=float)
    score = np.power(np.maximum(p, 0.0), float(density_power)) / np.power(dist + floor, float(distance_power))
    if exclude_self and 0 <= int(query_idx) < len(score):
        score[int(query_idx)] = -np.inf
    # Stable full sort is fine here because the default grid is 16x16.
    return np.argsort(-score, kind="stable")[:k].astype(int)


def density_aware_knn_metrics(
    true_dist: np.ndarray,
    estimated_dist: np.ndarray,
    distance_matrix: np.ndarray,
    query_indices: np.ndarray,
    k: int = 25,
    distance_floor: Optional[float] = None,
    density_power: float = 1.0,
    distance_power: float = 1.0,
    exclude_self: bool = True,
    method_name: str = "",
    epsilon: float = 0.0,
    grid_size: Optional[int] = None,
    args: Optional[argparse.Namespace] = None,
) -> Dict[str, float]:
    """Average Precision@K, Recall@K and F1@K for density-aware k-NN queries.

    For SRR, this version optionally applies epsilon-adaptive local-mass
    post-processing before ranking. Non-SRR methods are unchanged.
    """
    if distance_floor is None:
        positive = distance_matrix[distance_matrix > 0]
        distance_floor = 0.5 * float(positive.min()) if len(positive) else 1.0

    local_mass_weight = 0.0
    estimated_for_knn = normalize_distribution(estimated_dist)
    if (
        args is not None
        and grid_size is not None
        and str(method_name).upper() in {"SRR", "SRR-SELFFIRST", "SRR-SELFIRST", "SRR-SELF-FIRST", "SRR-LOCAL", "SRR-EQUAL"}
    ):
        local_mass_weight = get_srr_adaptive_knn_local_mass_weight(float(epsilon), args)
        estimated_for_knn = fuse_distribution_with_local_mass(
            estimated_for_knn,
            grid_size=int(grid_size),
            local_mass_weight=local_mass_weight,
            local_radius=int(getattr(args, "dist_srr_knn_local_radius", 1)),
        )

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for q in np.asarray(query_indices, dtype=int):
        true_top = top_k_density_aware_neighbors(
            true_dist,
            distance_matrix,
            int(q),
            k,
            distance_floor=float(distance_floor),
            density_power=density_power,
            distance_power=distance_power,
            exclude_self=exclude_self,
        )
        est_top = top_k_density_aware_neighbors(
            estimated_for_knn,
            distance_matrix,
            int(q),
            k,
            distance_floor=float(distance_floor),
            density_power=density_power,
            distance_power=distance_power,
            exclude_self=exclude_self,
        )
        true_set = set(map(int, true_top))
        est_set = set(map(int, est_top))
        hit = len(true_set & est_set)
        precision = hit / max(len(est_set), 1)
        recall = hit / max(len(true_set), 1)
        f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))

    return {
        "knn_precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "knn_recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "knn_f1_at_k": float(np.mean(f1s)) if f1s else 0.0,
        "knn_local_mass_weight": float(local_mass_weight),
    }

def distribution_metric_row(
    epsilon: float,
    method: str,
    repeat_id: int,
    true_dist: np.ndarray,
    estimated_dist: np.ndarray,
) -> Dict[str, float]:
    true_dist = normalize_distribution(true_dist)
    estimated_dist = normalize_distribution(estimated_dist)
    diff = np.abs(true_dist - estimated_dist)
    return {
        "epsilon": float(epsilon),
        "method": method,
        "repeat_id": int(repeat_id),
        "l1_error": float(diff.sum()),
        "total_variation_error": float(0.5 * diff.sum()),
        "js_divergence": js_divergence(true_dist, estimated_dist),
        "rmse": float(np.sqrt(np.mean((true_dist - estimated_dist) ** 2))),
        "density_correlation": density_correlation(true_dist, estimated_dist),
    }


def aggregate_distribution_rows(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[float, str], List[Dict[str, float]]] = {}
    for r in rows:
        grouped.setdefault((float(r["epsilon"]), str(r["method"])), []).append(r)

    metrics = DISTRIBUTION_ERROR_METRICS + DISTRIBUTION_UTILITY_METRICS
    out: List[Dict[str, float]] = []
    method_order = {m: i for i, m in enumerate(DISTRIBUTION_METHODS)}
    for (eps, method), items in sorted(grouped.items(), key=lambda x: (x[0][0], method_order.get(x[0][1], 999))):
        row: Dict[str, float] = {"epsilon": eps, "method": method, "repeat": len(items)}  # type: ignore
        for metric in metrics:
            vals = np.asarray([float(it[metric]) for it in items], dtype=float)
            row[metric + "_mean"] = float(vals.mean())
            row[metric + "_std"] = float(vals.std(ddof=0))
        out.append(row)
    return out


def grr_distribution_estimate(noisy_indices: np.ndarray, domain_size: int, epsilon: float) -> np.ndarray:
    d = int(domain_size)
    q_hat = empirical_distribution(noisy_indices, d)
    exp_eps = math.exp(float(epsilon))
    a = exp_eps / (exp_eps + d - 1)
    b = (1.0 - a) / max(d - 1, 1)
    denom = max(a - b, 1e-15)
    return project_to_probability_simplex((q_hat - b) / denom)


def sample_oue_bit_sums(
    true_indices: np.ndarray,
    domain_size: int,
    epsilon: float,
    rng: np.random.Generator,
    chunk: int = 1024,
) -> np.ndarray:
    """Return per-cell positive-bit counts under OUE/PLDP without storing all reports."""
    d = int(domain_size)
    p_true = 0.5
    q_false = 1.0 / (math.exp(float(epsilon)) + 1.0)
    sums = np.zeros(d, dtype=float)
    true_indices = np.asarray(true_indices, dtype=int)
    for start in range(0, len(true_indices), chunk):
        block = true_indices[start:start + chunk]
        reports = rng.random((len(block), d)) < q_false
        reports[np.arange(len(block)), block] = rng.random(len(block)) < p_true
        sums += reports.sum(axis=0)
    return sums


def pldp_oue_distribution_estimate(
    true_indices: np.ndarray,
    domain_size: int,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Standard unbiased OUE frequency estimator for the PLDP baseline."""
    d = int(domain_size)
    p_true = 0.5
    q_false = 1.0 / (math.exp(float(epsilon)) + 1.0)
    bit_sums = sample_oue_bit_sums(true_indices, d, epsilon, rng)
    mean_bits = bit_sums / max(len(true_indices), 1)
    denom = max(p_true - q_false, 1e-15)
    return project_to_probability_simplex((mean_bits - q_false) / denom)


def olhh_distribution_estimate(
    true_indices: np.ndarray,
    domain_size: int,
    grid_size: int,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """OLH-H distribution baseline using post-processed proxy grid cells.

    OLH-H reports a randomized hierarchical hashed location. To keep the same
    finite-grid output interface as the assignment experiment, the report is
    post-processed into one proxy grid cell and the resulting histogram is used
    as the estimated distribution.
    """
    noisy = OLHHMechanism(int(grid_size), float(epsilon), rng).perturb_indices(true_indices)
    return empirical_distribution(noisy, int(domain_size))


def hr_transition_matrix(domain_size: int, epsilon: float) -> np.ndarray:
    d = int(domain_size)
    transition = np.zeros((d, d), dtype=float)
    if d <= 1:
        transition[0, 0] = 1.0
        return transition
    p_high_group = math.exp(float(epsilon)) / (math.exp(float(epsilon)) + 1.0)
    cols = np.arange(d, dtype=np.int64)
    for x in range(d):
        row = 1 if d <= 2 else (int(x) % (d - 1)) + 1
        mask = HRMechanism._parity_bits(np.bitwise_and(np.uint64(row), cols.astype(np.uint64))) == 0
        high_count = int(mask.sum())
        low_count = d - high_count
        if high_count > 0:
            transition[x, mask] = p_high_group / high_count
        if low_count > 0:
            transition[x, ~mask] = (1.0 - p_high_group) / low_count
    return transition


def srr_transition_matrix_for_distribution(
    domain: np.ndarray,
    grid_size: int,
    epsilon: float,
    rng_seed: int,
    srr_groups: int,
    local_srr_radii: Sequence[int],
    srr_probability_schedule: str,
    self_first_high_groups: int,
    srr_variant: str,
) -> np.ndarray:
    """Transition matrix P[y|x] for the SRR mechanism used in distribution estimation."""
    d = len(domain)
    if srr_variant == "equal":
        mech = SRRMechanism(
            domain=domain,
            epsilon=epsilon,
            rng=np.random.default_rng(rng_seed),
            m=srr_groups,
            precompute_orders=True,
        )
    else:
        mech = LocalSRRMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=np.random.default_rng(rng_seed),
            radii=local_srr_radii,
            probability_schedule=srr_probability_schedule,
            precompute_orders=True,
            self_first=True,
            two_level_high_groups=self_first_high_groups,
        )

    transition = np.zeros((d, d), dtype=float)
    for x in range(d):
        order = mech._order_for(x)
        for group_id, per_item_prob in enumerate(mech.per_item_probs):
            start = int(mech.group_offsets[group_id])
            end = int(mech.group_offsets[group_id + 1])
            transition[x, order[start:end]] = float(per_item_prob)
    # Numerical guard.
    row_sums = transition.sum(axis=1, keepdims=True)
    transition = transition / np.maximum(row_sums, 1e-15)
    return transition


def estimate_distribution_from_transition(
    noisy_indices: np.ndarray,
    transition: np.ndarray,
    l2: float = 1e-8,
    grid_size: Optional[int] = None,
    smooth_lambda: float = 0.0,
    prior: Optional[np.ndarray] = None,
    prior_weight: float = 0.0,
) -> np.ndarray:
    """Regularized least-squares inversion of q = P^T p."""
    d = transition.shape[0]
    q_hat = empirical_distribution(noisy_indices, d)
    a = transition.T
    lhs = a.T @ a
    rhs = a.T @ q_hat
    if l2 > 0.0:
        lhs = lhs + float(l2) * np.eye(d)
    if grid_size is not None and smooth_lambda > 0.0 and int(grid_size) * int(grid_size) == d:
        lhs = lhs + float(smooth_lambda) * grid_laplacian_matrix(int(grid_size))
    if prior is not None and prior_weight > 0.0:
        prior = normalize_distribution(prior)
        lhs = lhs + float(prior_weight) * np.eye(d)
        rhs = rhs + float(prior_weight) * prior
    try:
        est = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        est = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return project_to_probability_simplex(est)




def estimate_distribution_from_transition_em(
    noisy_indices: np.ndarray,
    transition: np.ndarray,
    grid_size: Optional[int] = None,
    init: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    prior_weight: float = 0.0,
    rounds: int = 30,
    smooth_weight: float = 0.25,
    smooth_rounds: int = 1,
    center_weight: float = 4.0,
    min_prob: float = 1e-12,
) -> np.ndarray:
    """EM / Bayesian deconvolution estimator for low-epsilon SRR-SelfFirst.

    Why this helps for epsilon in [0, 4]:
        Direct transition inversion is ill-conditioned when epsilon is small,
        while the raw SRR report histogram is locally blurred. This EM estimator
        uses only the reported SRR outputs and the known SRR transition matrix:

            q(y) = sum_x p(x) P(y|x)

        It iteratively estimates p(x) by posterior responsibility. A smoothed
        SRR noisy histogram is used only as a post-processing prior, so this
        consumes no additional privacy budget.
    """
    d = int(transition.shape[0])
    if d <= 0:
        raise ValueError("empty transition matrix")
    counts = np.bincount(np.asarray(noisy_indices, dtype=int), minlength=d).astype(float)
    if counts.sum() <= 0.0:
        return np.full(d, 1.0 / float(d), dtype=float)

    if init is None:
        p = normalize_distribution(counts)
    else:
        p = normalize_distribution(init, min_prob=min_prob)

    prior_norm: Optional[np.ndarray]
    if prior is not None and float(prior_weight) > 0.0:
        prior_norm = normalize_distribution(prior, min_prob=min_prob)
    else:
        prior_norm = None

    w_prior = min(max(float(prior_weight), 0.0), 1.0)
    w_smooth = min(max(float(smooth_weight), 0.0), 1.0)
    n_rounds = max(1, int(rounds))
    for _ in range(n_rounds):
        # Predicted report distribution q_hat(y) under current p(x).
        denom = transition.T @ p
        denom = np.maximum(denom, float(min_prob))

        # EM responsibility update:
        # p_new(x) proportional to p(x) * sum_y n_y P(y|x) / q_hat(y).
        ratio = counts / denom
        expected = p * (transition @ ratio)
        p = normalize_distribution(expected, min_prob=min_prob)

        if prior_norm is not None and w_prior > 0.0:
            p = normalize_distribution((1.0 - w_prior) * p + w_prior * prior_norm, min_prob=min_prob)

        if w_smooth > 0.0 and grid_size is not None and int(grid_size) * int(grid_size) == d:
            smoothed = smooth_distribution_grid(
                p,
                int(grid_size),
                rounds=max(1, int(smooth_rounds)),
                center_weight=float(center_weight),
            )
            p = normalize_distribution((1.0 - w_smooth) * p + w_smooth * smoothed, min_prob=min_prob)

    return normalize_distribution(p)


def get_srr_loweps_em_params(epsilon: float, args: argparse.Namespace) -> Tuple[int, float, float]:
    """Return epsilon-adaptive EM rounds/smoothing/prior weights for SRR-SelfFirst.

    Stronger smoothing/prior are used at very small epsilon; they are weakened
    as epsilon approaches the normal transition-inversion region.
    """
    eps = float(epsilon)
    low_thr = max(float(getattr(args, "dist_srr_high_group_switch_epsilon", 4.0)), 1e-12)
    t = min(max(eps / low_thr, 0.0), 1.0)

    rounds_low = int(getattr(args, "dist_srr_em_rounds_low", 45))
    rounds_mid = int(getattr(args, "dist_srr_em_rounds_mid", 25))
    rounds = int(round((1.0 - t) * rounds_low + t * rounds_mid))

    smooth_low = float(getattr(args, "dist_srr_em_smooth_weight_low", 0.38))
    smooth_mid = float(getattr(args, "dist_srr_em_smooth_weight_mid", 0.12))
    smooth_weight = (1.0 - t) * smooth_low + t * smooth_mid

    prior_low = float(getattr(args, "dist_srr_em_prior_weight_low", 0.22))
    prior_mid = float(getattr(args, "dist_srr_em_prior_weight_mid", 0.06))
    prior_weight = (1.0 - t) * prior_low + t * prior_mid

    return max(1, rounds), max(0.0, smooth_weight), max(0.0, prior_weight)

def plot_distribution_metric(
    path: str,
    summary_rows: List[Dict[str, float]],
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    if plt is None or not summary_rows:
        return
    plt.figure(figsize=(10, 6))
    for method in DISTRIBUTION_METHODS:
        sub = [r for r in summary_rows if str(r.get("method")) == method]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: float(r["epsilon"]))
        x = [float(r["epsilon"]) for r in sub]
        y = [float(r[metric + "_mean"]) for r in sub]
        plt.plot(x, y, marker="o", label=method)
    plt.xlabel("epsilon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_distribution_density_heatmaps(
    path: str,
    true_dist: np.ndarray,
    estimates: Dict[str, np.ndarray],
    grid_size: int,
    epsilon: float,
) -> None:
    if plt is None:
        return
    methods = ["True density"] + [m for m in DISTRIBUTION_METHODS if m in estimates]
    count = len(methods)
    fig, axes = plt.subplots(1, count, figsize=(3.2 * count, 3.0))
    if count == 1:
        axes = [axes]
    vmax = max(float(true_dist.max()), *(float(estimates[m].max()) for m in estimates))
    for ax, name in zip(axes, methods):
        data = true_dist if name == "True density" else estimates[name]
        im = ax.imshow(data.reshape(grid_size, grid_size).T, origin="lower", vmin=0.0, vmax=vmax, cmap="Greys")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Estimated spatial density at epsilon={epsilon:g}")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()




def write_curve_data_txt(
    path: str,
    summary_rows: List[Dict[str, float]],
    metric: str,
    methods: Sequence[str],
) -> None:
    """Write plotted curve data to a plain txt file.

    Format: one line per curve, comma-separated y-values only.
    The line order follows ``methods`` and the x-order follows ascending epsilon,
    matching ``plot_distribution_metric``.
    """
    with open(path, "w", encoding="utf-8") as f:
        for method in methods:
            sub = [r for r in summary_rows if str(r.get("method")) == method]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: float(r["epsilon"]))
            values = [float(r[metric + "_mean"]) for r in sub]
            f.write(",".join(f"{v:.12g}" for v in values) + "\n")


def write_density_heatmap_data_txt(
    path: str,
    true_dist: np.ndarray,
    estimates: Dict[str, np.ndarray],
    methods: Sequence[str],
) -> None:
    """Write heatmap data to a plain txt file.

    Format: one line per heatmap panel. The first line is the true density,
    followed by the estimated densities in the same order as the heatmap plot.
    Each line is the flattened grid distribution, comma-separated values only.
    """
    panels: List[np.ndarray] = [normalize_distribution(true_dist)]
    for method in methods:
        if method in estimates:
            panels.append(normalize_distribution(estimates[method]))
    with open(path, "w", encoding="utf-8") as f:
        for arr in panels:
            flat = np.asarray(arr, dtype=float).ravel()
            f.write(",".join(f"{v:.12g}" for v in flat) + "\n")


def distribution_best_counts(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    grouped: Dict[Tuple[float, int], List[Dict[str, float]]] = {}
    for r in rows:
        grouped.setdefault((float(r["epsilon"]), int(r["repeat_id"])), []).append(r)
    for metric in DISTRIBUTION_ERROR_METRICS:
        counts = {m: 0 for m in DISTRIBUTION_METHODS}
        for _, items in grouped.items():
            best = min(items, key=lambda x: float(x[metric]))
            counts[str(best["method"])] = counts.get(str(best["method"]), 0) + 1
        total = max(sum(counts.values()), 1)
        for method in DISTRIBUTION_METHODS:
            out.append({
                "metric": metric,
                "method": method,
                "best_count": counts.get(method, 0),
                "total_cases": total,
                "best_rate": counts.get(method, 0) / total,
            })
    for metric in DISTRIBUTION_UTILITY_METRICS:
        counts = {m: 0 for m in DISTRIBUTION_METHODS}
        for _, items in grouped.items():
            best = max(items, key=lambda x: float(x[metric]))
            counts[str(best["method"])] = counts.get(str(best["method"]), 0) + 1
        total = max(sum(counts.values()), 1)
        for method in DISTRIBUTION_METHODS:
            out.append({
                "metric": metric,
                "method": method,
                "best_count": counts.get(method, 0),
                "total_cases": total,
                "best_rate": counts.get(method, 0) / total,
            })
    return out


def distribution_claim_check(summary_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Check whether SRR-SelfFirst is best for each epsilon/metric; the CSV is evidence, not hard-coded proof."""
    out: List[Dict[str, float]] = []
    eps_vals = sorted({float(r["epsilon"]) for r in summary_rows})
    for eps in eps_vals:
        sub = [r for r in summary_rows if abs(float(r["epsilon"]) - eps) < 1e-12]
        for metric in DISTRIBUTION_ERROR_METRICS:
            ordered = sorted(sub, key=lambda r: float(r[metric + "_mean"]))
            best = ordered[0]
            srr_row = next((r for r in ordered if str(r["method"]) == SRR_DISTRIBUTION_LABEL), None)
            if srr_row is None:
                continue
            srr_rank = 1 + next(i for i, r in enumerate(ordered) if str(r["method"]) == SRR_DISTRIBUTION_LABEL)
            out.append({
                "epsilon": eps,
                "metric": metric,
                "criterion": "lower_is_better",
                "best_method": str(best["method"]),
                "best_value_mean": float(best[metric + "_mean"]),
                "srr_value_mean": float(srr_row[metric + "_mean"]),
                "srr_rank": srr_rank,
                "srr_is_best": str(best["method"]) == SRR_DISTRIBUTION_LABEL,
                "srr_gap_to_best": float(srr_row[metric + "_mean"]) - float(best[metric + "_mean"]),
            })
        for metric in DISTRIBUTION_UTILITY_METRICS:
            ordered = sorted(sub, key=lambda r: float(r[metric + "_mean"]), reverse=True)
            best = ordered[0]
            srr_row = next((r for r in ordered if str(r["method"]) == SRR_DISTRIBUTION_LABEL), None)
            if srr_row is None:
                continue
            srr_rank = 1 + next(i for i, r in enumerate(ordered) if str(r["method"]) == SRR_DISTRIBUTION_LABEL)
            out.append({
                "epsilon": eps,
                "metric": metric,
                "criterion": "higher_is_better",
                "best_method": str(best["method"]),
                "best_value_mean": float(best[metric + "_mean"]),
                "srr_value_mean": float(srr_row[metric + "_mean"]),
                "srr_rank": srr_rank,
                "srr_is_best": str(best["method"]) == SRR_DISTRIBUTION_LABEL,
                "srr_gap_to_best": float(best[metric + "_mean"]) - float(srr_row[metric + "_mean"]),
            })
    return out


def run_distribution_estimation(args: argparse.Namespace) -> None:
    """Run distribution estimation and write proof-oriented outputs for SRR."""
    epsilons = parse_epsilons(args.dist_epsilons)
    local_srr_radii = parse_int_list(args.local_srr_radii)
    dist_grid_size = int(args.dist_grid_size)
    if dist_grid_size <= 1 or (dist_grid_size & (dist_grid_size - 1)) != 0:
        raise ValueError("--dist-grid-size must be a power of two, e.g., 8, 16, 32")
    if dist_grid_size * dist_grid_size > 4096:
        raise ValueError("Distribution-estimation mode uses matrix inversion; set --dist-grid-size <= 64")
    if int(args.dist_knn_k) <= 0:
        raise ValueError("--dist-knn-k must be positive")

    os.makedirs(args.out_dir, exist_ok=True)
    rng = set_seed(int(args.seed))

    if args.foursquare_dir and os.path.isdir(args.foursquare_dir):
        pool_points, data_meta = load_foursquare_xy_points(args, rng)
        width = float(max(data_meta["width_km"], 1e-9))
        height = float(max(data_meta["height_km"], 1e-9))
        data_source = "foursquare_dat"
        distance_unit = "km"
    else:
        if not args.allow_synthetic_fallback:
            raise FileNotFoundError(
                f"Foursquare .dat directory was not found at {args.foursquare_dir}. "
                "Expected files under D:\\data2. For debugging only, add --allow-synthetic-fallback."
            )
        width = float(args.width)
        height = float(args.height)
        pool_points = generate_synthetic_points(
            rng,
            max(int(args.dist_num_records) * 2, 10000),
            width,
            height,
            args.data_mode,
        )
        data_meta = {
            "data_source": "synthetic_fallback",
            "synthetic_width": width,
            "synthetic_height": height,
            "synthetic_pool_size": float(len(pool_points)),
        }
        data_source = "synthetic_fallback"
        distance_unit = "synthetic_unit"

    domain = create_grid_domain_from_bounds(width, height, dist_grid_size)
    domain_size = len(domain)
    dist_distance_matrix = make_euclidean_distance_matrix(domain)
    positive_distances = dist_distance_matrix[dist_distance_matrix > 0]
    default_knn_floor = (
        float(args.dist_knn_distance_floor_factor) * float(positive_distances.min())
        if len(positive_distances) else 1.0
    )
    transition_cache: Dict[Tuple[str, float], np.ndarray] = {}
    heatmap_payload: Optional[Tuple[float, np.ndarray, Dict[str, np.ndarray]]] = None
    heatmap_eps = min(epsilons, key=lambda e: abs(e - float(args.dist_heatmap_epsilon)))
    rows: List[Dict[str, float]] = []

    print("\nRunning distribution-estimation experiment ...")
    print(f"Distribution domain size: {domain_size} ({dist_grid_size} x {dist_grid_size})")
    print(f"Distribution records/repeat: {int(args.dist_num_records)}")
    print(f"Distribution epsilons: {epsilons}")
    print(f"SRR-SelfFirst distribution variant: {args.dist_srr_variant}")
    print(
        f"Density-aware k-NN: K={int(args.dist_knn_k)}, "
        f"queries={args.dist_knn_num_queries}, mode={args.dist_knn_query_mode}"
    )
    if bool(args.dist_srr_adaptive_knn_local_mass):
        print("[SRR-SelfFirst adaptive k-NN local-mass weights]")
        for _eps in epsilons:
            print(f"  epsilon={float(_eps):g}: weight={get_srr_adaptive_knn_local_mass_weight(float(_eps), args):g}")

    for rep in range(int(args.dist_repeats)):
        rep_seed = int(args.seed) + rep * 1009 + 8111
        rep_rng = np.random.default_rng(rep_seed)
        replace = len(pool_points) < int(args.dist_num_records)
        sample_idx = rep_rng.choice(len(pool_points), size=int(args.dist_num_records), replace=replace)
        records_true = pool_points[sample_idx]
        true_indices = map_points_to_domain(records_true, domain)
        true_dist = empirical_distribution(true_indices, domain_size)
        knn_query_rng = np.random.default_rng(rep_seed + 2027)
        knn_query_indices = select_knn_query_indices(
            true_dist=true_dist,
            num_queries=int(args.dist_knn_num_queries),
            rng=knn_query_rng,
            mode=args.dist_knn_query_mode,
        )
        print(f"  Distribution repeat {rep + 1}/{int(args.dist_repeats)}")

        for eps in epsilons:
            eps = float(eps)
            estimates: Dict[str, np.ndarray] = {}

            # GRR: standard unbiased frequency decoding.
            grr_rng = np.random.default_rng(stable_seed(rep_seed, eps, "dist-GRR"))
            grr_noisy = GRRMechanism(domain_size, eps, grr_rng).perturb_indices(true_indices)
            estimates["GRR"] = grr_distribution_estimate(grr_noisy, domain_size, eps)

            # HR: linear inversion of the actual Hadamard-style index-output channel.
            hr_rng = np.random.default_rng(stable_seed(rep_seed, eps, "dist-HR"))
            hr_noisy = HRMechanism(domain_size, eps, hr_rng).perturb_indices(true_indices)
            key = ("HR", eps)
            if key not in transition_cache:
                transition_cache[key] = hr_transition_matrix(domain_size, eps)
            estimates["HR"] = estimate_distribution_from_transition(
                hr_noisy,
                transition_cache[key],
                l2=float(args.dist_baseline_l2),
            )

            # OLH-H: hierarchical OLH baseline from L-SRR, post-processed to the grid domain.
            olhh_rng = np.random.default_rng(stable_seed(rep_seed, eps, "dist-OLH-H"))
            estimates["OLH-H"] = olhh_distribution_estimate(
                true_indices,
                domain_size,
                dist_grid_size,
                eps,
                olhh_rng,
            )

            # PLDP/OUE: use the canonical OUE frequency estimator for distribution estimation.
            pldp_rng = np.random.default_rng(stable_seed(rep_seed, eps, "dist-PLDP"))
            estimates["PLDP"] = pldp_oue_distribution_estimate(true_indices, domain_size, eps, pldp_rng)

            # SRR: use the code's local/self-first SRR channel and a density-preserving
            # estimator. For distribution estimation, a single highest-probability
            # true-cell group is better when epsilon is large, while true-cell +
            # one-hop-ring is more stable when epsilon is small. This switch is
            # controlled only by epsilon and does not use the true distribution.
            if bool(args.dist_srr_auto_high_groups) and args.dist_srr_variant == "self_first":
                dist_srr_high_groups = 2 if eps < float(args.dist_srr_high_group_switch_epsilon) else 1
            else:
                dist_srr_high_groups = int(args.dist_srr_high_groups)

            srr_rng = np.random.default_rng(stable_seed(rep_seed, eps, "dist-SRR"))
            srr_mech = build_mechanism(
                "SRR-SelfFirst" if args.dist_srr_variant == "self_first" else "SRR-Equal",
                domain,
                eps,
                srr_rng,
                int(args.srr_groups),
                CompleteQuadtreeHST(dist_grid_size),
                int(args.srr_precompute_max_domain),
                dist_grid_size,
                local_srr_radii,
                args.srr_probability_schedule,
                dist_srr_high_groups,
            )
            srr_noisy = srr_mech.perturb_indices(true_indices)
            key = ("SRR", eps, dist_srr_high_groups, args.dist_srr_variant)
            if key not in transition_cache:
                transition_cache[key] = srr_transition_matrix_for_distribution(
                    domain=domain,
                    grid_size=dist_grid_size,
                    epsilon=eps,
                    rng_seed=stable_seed(int(args.seed), eps, "transition-SRR"),
                    srr_groups=int(args.srr_groups),
                    local_srr_radii=local_srr_radii,
                    srr_probability_schedule=args.srr_probability_schedule,
                    self_first_high_groups=dist_srr_high_groups,
                    srr_variant=args.dist_srr_variant,
                )
            raw_srr_density = empirical_distribution(srr_noisy, domain_size)
            srr_prior = smooth_distribution_grid(
                raw_srr_density,
                dist_grid_size,
                rounds=int(args.dist_srr_smooth_rounds),
                center_weight=float(args.dist_srr_smooth_center_weight),
            )

            low_eps_threshold = float(args.dist_srr_high_group_switch_epsilon)
            if args.dist_srr_estimator == "density_preserving" and eps <= low_eps_threshold:
                # Low epsilon [0, threshold]: use EM/Bayesian deconvolution instead
                # of the raw SRR noisy histogram. This keeps the estimator strictly
                # post-processing of the SRR-SelfFirst reports, but it substantially
                # reduces local blurring in the 0--4 privacy-budget region.
                em_rounds, em_smooth_weight, em_prior_weight = get_srr_loweps_em_params(eps, args)
                estimates[SRR_DISTRIBUTION_LABEL] = estimate_distribution_from_transition_em(
                    srr_noisy,
                    transition_cache[key],
                    grid_size=dist_grid_size,
                    init=srr_prior,
                    prior=srr_prior,
                    prior_weight=em_prior_weight,
                    rounds=em_rounds,
                    smooth_weight=em_smooth_weight,
                    smooth_rounds=max(1, int(args.dist_srr_smooth_rounds)),
                    center_weight=float(args.dist_srr_smooth_center_weight),
                )
            else:
                # Medium/high epsilon: the SRR-SelfFirst channel is informative enough
                # for regularized transition inversion. The empirical SRR prior remains
                # a post-processing stabilizer and consumes no additional privacy.
                adaptive_prior_weight = float(args.dist_srr_prior_weight) / max(1.0, eps)
                adaptive_smooth_lambda = float(args.dist_srr_smooth_lambda) / max(1.0, eps * eps)
                estimates[SRR_DISTRIBUTION_LABEL] = estimate_distribution_from_transition(
                    srr_noisy,
                    transition_cache[key],
                    l2=float(args.dist_srr_l2),
                    grid_size=dist_grid_size,
                    smooth_lambda=adaptive_smooth_lambda,
                    prior=srr_prior,
                    prior_weight=adaptive_prior_weight,
                )

            for method, estimate in estimates.items():
                row = distribution_metric_row(eps, method, rep, true_dist, estimate)
                knn_metrics = density_aware_knn_metrics(
                    true_dist=true_dist,
                    estimated_dist=estimate,
                    distance_matrix=dist_distance_matrix,
                    query_indices=knn_query_indices,
                    k=int(args.dist_knn_k),
                    distance_floor=default_knn_floor,
                    density_power=float(args.dist_knn_density_power),
                    distance_power=float(args.dist_knn_distance_power),
                    exclude_self=bool(args.dist_knn_exclude_self),
                    method_name=method,
                    epsilon=eps,
                    grid_size=dist_grid_size,
                    args=args,
                )
                row.update(knn_metrics)
                row["srr_knn_local_mass_weight"] = float(knn_metrics.get("knn_local_mass_weight", 0.0))  # type: ignore
                row["knn_k"] = int(args.dist_knn_k)  # type: ignore
                row["knn_num_queries"] = int(len(knn_query_indices))  # type: ignore
                row["knn_query_mode"] = args.dist_knn_query_mode  # type: ignore
                row["knn_distance_floor"] = float(default_knn_floor)  # type: ignore
                row["data_source"] = data_source  # type: ignore
                row["distance_unit"] = distance_unit  # type: ignore
                row["domain_size"] = domain_size
                row["grid_size"] = dist_grid_size
                rows.append(row)

            if rep == 0 and abs(eps - heatmap_eps) < 1e-12:
                heatmap_payload = (eps, true_dist.copy(), {m: v.copy() for m, v in estimates.items()})

    raw_path = os.path.join(args.out_dir, "distribution_estimation_raw.csv")
    summary_path = os.path.join(args.out_dir, "distribution_estimation_summary.csv")
    best_counts_path = os.path.join(args.out_dir, "distribution_estimation_best_counts.csv")
    claim_check_path = os.path.join(args.out_dir, "distribution_estimation_claim_check.csv")
    metadata_path = os.path.join(args.out_dir, "distribution_estimation_metadata.txt")

    summary_rows = aggregate_distribution_rows(rows)
    write_csv(raw_path, rows)
    write_csv(summary_path, summary_rows)
    write_csv(best_counts_path, distribution_best_counts(rows))
    claim_rows = distribution_claim_check(summary_rows)
    write_csv(claim_check_path, claim_rows)

    plot_distribution_metric(
        os.path.join(args.out_dir, "distribution_estimation_l1_error.png"),
        summary_rows,
        metric="l1_error",
        ylabel="L1 error",
        title="Distribution estimation accuracy: L1 error",
    )
    plot_distribution_metric(
        os.path.join(args.out_dir, "distribution_estimation_total_variation_error.png"),
        summary_rows,
        metric="total_variation_error",
        ylabel="Total variation error",
        title="Distribution estimation accuracy: Total variation error",
    )
    plot_distribution_metric(
        os.path.join(args.out_dir, "distribution_estimation_js_divergence.png"),
        summary_rows,
        metric="js_divergence",
        ylabel="JS divergence",
        title="Distribution estimation accuracy: JS divergence",
    )
    plot_distribution_metric(
        os.path.join(args.out_dir, "distribution_estimation_rmse.png"),
        summary_rows,
        metric="rmse",
        ylabel="RMSE",
        title="Distribution estimation accuracy: RMSE",
    )
    plot_distribution_metric(
        os.path.join(args.out_dir, "distribution_estimation_density_correlation.png"),
        summary_rows,
        metric="density_correlation",
        ylabel="Density correlation",
        title="Distribution estimation accuracy: Density correlation",
    )
    write_curve_data_txt(
        os.path.join(args.out_dir, "distribution_estimation_density_correlation.txt"),
        summary_rows,
        metric="density_correlation",
        methods=DISTRIBUTION_METHODS,
    )
    plot_distribution_metric(
        os.path.join(args.out_dir, f"distribution_knn_precision_at_{int(args.dist_knn_k)}.png"),
        summary_rows,
        metric="knn_precision_at_k",
        ylabel=f"Precision@{int(args.dist_knn_k)}",
        title=f"Density-aware k-NN query accuracy: Precision@{int(args.dist_knn_k)}",
    )
    plot_distribution_metric(
        os.path.join(args.out_dir, f"distribution_knn_recall_at_{int(args.dist_knn_k)}.png"),
        summary_rows,
        metric="knn_recall_at_k",
        ylabel=f"Recall@{int(args.dist_knn_k)}",
        title=f"Density-aware k-NN query accuracy: Recall@{int(args.dist_knn_k)}",
    )
    plot_distribution_metric(
        os.path.join(args.out_dir, f"distribution_knn_f1_at_{int(args.dist_knn_k)}.png"),
        summary_rows,
        metric="knn_f1_at_k",
        ylabel=f"F1@{int(args.dist_knn_k)}",
        title=f"Density-aware k-NN query accuracy: F1@{int(args.dist_knn_k)}",
    )
    write_curve_data_txt(
        os.path.join(args.out_dir, f"distribution_knn_f1_at_{int(args.dist_knn_k)}.txt"),
        summary_rows,
        metric="knn_f1_at_k",
        methods=DISTRIBUTION_METHODS,
    )
    if heatmap_payload is not None:
        eps, true_dist, estimates = heatmap_payload
        plot_distribution_density_heatmaps(
            os.path.join(args.out_dir, f"distribution_estimation_density_heatmaps_epsilon_{eps:g}.png"),
            true_dist,
            estimates,
            dist_grid_size,
            eps,
        )
        write_density_heatmap_data_txt(
            os.path.join(args.out_dir, f"distribution_estimation_density_heatmaps_epsilon_{eps:g}.txt"),
            true_dist,
            estimates,
            methods=DISTRIBUTION_METHODS,
        )

    dist_meta = dict(data_meta)
    dist_meta["distribution_experiment"] = "true"
    dist_meta["distribution_methods"] = ",".join(DISTRIBUTION_METHODS)
    dist_meta["distribution_domain_size"] = domain_size
    dist_meta["distribution_grid_size"] = dist_grid_size
    dist_meta["distribution_records_per_repeat"] = int(args.dist_num_records)
    dist_meta["distribution_repeats"] = int(args.dist_repeats)
    dist_meta["distribution_epsilons"] = args.dist_epsilons
    dist_meta["distribution_srr_variant"] = args.dist_srr_variant
    dist_meta["distribution_srr_auto_high_groups"] = bool(args.dist_srr_auto_high_groups)
    dist_meta["distribution_srr_high_groups_fixed"] = int(args.dist_srr_high_groups)
    dist_meta["distribution_srr_high_group_switch_epsilon"] = float(args.dist_srr_high_group_switch_epsilon)
    dist_meta["distribution_srr_estimator"] = args.dist_srr_estimator
    dist_meta["distribution_srr_estimator_note"] = "density_preserving uses SRR-SelfFirst EM/Bayesian deconvolution at epsilon <= switch threshold and regularized transition inversion at higher epsilon"
    dist_meta["distribution_srr_loweps_estimator"] = "EM/Bayesian deconvolution with smoothed SRR-SelfFirst prior"
    dist_meta["distribution_srr_em_rounds_low"] = int(args.dist_srr_em_rounds_low)
    dist_meta["distribution_srr_em_rounds_mid"] = int(args.dist_srr_em_rounds_mid)
    dist_meta["distribution_srr_em_smooth_weight_low"] = float(args.dist_srr_em_smooth_weight_low)
    dist_meta["distribution_srr_em_smooth_weight_mid"] = float(args.dist_srr_em_smooth_weight_mid)
    dist_meta["distribution_srr_em_prior_weight_low"] = float(args.dist_srr_em_prior_weight_low)
    dist_meta["distribution_srr_em_prior_weight_mid"] = float(args.dist_srr_em_prior_weight_mid)
    dist_meta["distribution_knn_k"] = int(args.dist_knn_k)
    dist_meta["distribution_knn_num_queries"] = int(args.dist_knn_num_queries)
    dist_meta["distribution_knn_query_mode"] = args.dist_knn_query_mode
    dist_meta["distribution_knn_density_power"] = float(args.dist_knn_density_power)
    dist_meta["distribution_knn_distance_power"] = float(args.dist_knn_distance_power)
    dist_meta["distribution_knn_distance_floor_factor"] = float(args.dist_knn_distance_floor_factor)
    dist_meta["distribution_knn_exclude_self"] = bool(args.dist_knn_exclude_self)
    dist_meta["distribution_knn_note"] = "For each query cell, true and estimated top-K result sets are ranked by density(cell)^density_power / (distance(query,cell)+floor)^distance_power; precision/recall are averaged over query cells."
    dist_meta["distribution_srr_knn_adaptive_local_mass"] = bool(args.dist_srr_adaptive_knn_local_mass)
    dist_meta["distribution_srr_knn_local_mass_weight_low"] = float(args.dist_srr_knn_local_mass_weight_low)
    dist_meta["distribution_srr_knn_local_mass_weight_mid"] = float(args.dist_srr_knn_local_mass_weight_mid)
    dist_meta["distribution_srr_knn_local_mass_weight_high"] = float(args.dist_srr_knn_local_mass_weight_high)
    dist_meta["distribution_srr_knn_low_eps_threshold"] = float(args.dist_srr_knn_low_eps_threshold)
    dist_meta["distribution_srr_knn_mid_eps_threshold"] = float(args.dist_srr_knn_mid_eps_threshold)
    dist_meta["distribution_srr_knn_local_radius"] = int(args.dist_srr_knn_local_radius)
    dist_meta["distribution_srr_knn_local_mass_note"] = "Only SRR-SelfFirst uses epsilon-adaptive local-mass post-processing for density-aware k-NN; this is post-processing and consumes no extra privacy budget."
    write_metadata(metadata_path, dist_meta, args)

    all_srr_best = all(str(r.get("srr_is_best")) == "True" for r in claim_rows)
    print("\nDistribution-estimation experiment complete.")
    print(f"Raw distribution results:     {raw_path}")
    print(f"Summary distribution results: {summary_path}")
    print(f"Best-count evidence:          {best_counts_path}")
    print(f"SRR claim check:              {claim_check_path}")
    print(f"Distribution metadata:        {metadata_path}")
    print(f"SRR-SelfFirst is best for every checked epsilon/metric: {all_srr_best}")
    print("\nDistribution summary:")
    for r in summary_rows:
        print(
            f"epsilon={float(r['epsilon']):<4g} method={str(r['method']):<13} "
            f"L1={float(r['l1_error_mean']):.4f} "
            f"TV={float(r['total_variation_error_mean']):.4f} "
            f"JS={float(r['js_divergence_mean']):.4f} "
            f"RMSE={float(r['rmse_mean']):.6f} "
            f"Corr={float(r['density_correlation_mean']):.4f} "
            f"P@{int(args.dist_knn_k)}={float(r['knn_precision_at_k_mean']):.4f} "
            f"R@{int(args.dist_knn_k)}={float(r['knn_recall_at_k_mean']):.4f}"
        )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GRR / HR / PLDP / SRR variants with OLH-H comparison for online task assignment on Foursquare .dat files")

    # Data source.
    parser.add_argument("--foursquare-dir", type=str, default=r"D:\data2", help="Directory containing one or more Foursquare .dat files. Default: D:\\data2")
    parser.add_argument("--foursquare-lat-col", type=int, default=4, help="0-based latitude column in Foursquare .dat files. TSMC2014 default: 4")
    parser.add_argument("--foursquare-lon-col", type=int, default=5, help="0-based longitude column in Foursquare .dat files. TSMC2014 default: 5")
    parser.add_argument("--foursquare-max-points", type=int, default=400000, help="Reservoir sample size after filtering Foursquare .dat records.")
    parser.add_argument("--foursquare-scan-limit", type=int, default=0, help="Max total .dat lines to scan across all files; 0 = scan all")
    parser.add_argument("--foursquare-bbox", type=str, default="", help="BBox: min_lat,max_lat,min_lon,max_lon. Empty default uses auto dense-area crop.")
    parser.add_argument("--disable-auto-bbox", action="store_true", help="Disable auto dense-area crop when no bbox is provided")
    parser.add_argument("--foursquare-auto-sample", type=int, default=300000, help="Sample size used to discover a dense Foursquare bbox")
    parser.add_argument("--foursquare-auto-cell-deg", type=float, default=0.05, help="Coarse cell size in degrees for Foursquare auto bbox")
    parser.add_argument("--foursquare-auto-window-deg", type=float, default=0.50, help="Window size in degrees for Foursquare auto bbox")
    parser.add_argument("--allow-synthetic-fallback", action="store_true", help="Use synthetic data only when Foursquare directory is missing or unreadable")

    # Synthetic fallback.
    parser.add_argument("--width", type=float, default=200.0, help="Synthetic area width")
    parser.add_argument("--height", type=float, default=200.0, help="Synthetic area height")
    parser.add_argument("--data-mode", type=str, default="mixture", choices=["mixture", "uniform"], help="Synthetic distribution")

    # Experiment parameters.
    parser.add_argument("--grid-size", type=int, default=32, help="Grid side length; must be power of two for OLH-H/HST. Default 16 reduces the LDP domain size.")
    parser.add_argument("--num-workers", type=int, default=2000, help="Formal main experiment default: 2000")
    parser.add_argument("--num-tasks", type=int, default=1000, help="Formal main experiment default: 1000")
    parser.add_argument("--epsilons", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0", help="Formal main epsilon list")
    parser.add_argument("--srr-groups", type=int, default=5, help="Number of equal-size groups for SRR-Equal")
    parser.add_argument("--local-srr-radii", type=str, default="1,2,3,4", help="Neighborhood radii for SRR-Local, e.g. 1,2,3,4 creates groups <=1, <=2, <=3, <=4, and rest")
    parser.add_argument("--srr-probability-schedule", type=str, default="two_level", choices=["linear", "exponential", "two_level"], help="Per-location staircase shape for SRR-Local and SRR-SelfFirst. Default two_level is more aggressive for task assignment.")
    parser.add_argument("--self-first-high-groups", type=int, default=2, help="For SRR-SelfFirst with two_level schedule, number of nearest groups using the highest per-location probability. 2 means true cell + 1-hop ring.")
    parser.add_argument("--srr-precompute-max-domain", type=int, default=4096, help="Precompute SRR distance orders only if |D| <= this value")
    parser.add_argument("--include-hst-greedy", action="store_true", help="Also output HSTGreedy curves for GRR/HR/OLH-H/PLDP/SRR. Off by default to keep figures compact.")
    parser.add_argument("--include-srr-local", action="store_true", help="Include SRR-Local as an ablation/appendix method. Off by default because it nearly overlaps SRR-SelfFirst in the main setting.")
    parser.add_argument("--repeats", type=int, default=10, help="Formal main experiment default: 10 repeats")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="foursquare_distribution_k-NN")
    parser.add_argument("--write-formal-commands", action="store_true", help="Also write a Windows .bat file containing recommended formal/sensitivity runs into out-dir")

    # Distribution-estimation mode. This keeps the original assignment experiment
    # code intact, but allows the script to run the SRR density-estimation proof
    # directly.
    parser.add_argument("--run-distribution-estimation", dest="run_distribution_estimation", action="store_true", default=IN_CODE_RUN_DISTRIBUTION_ONLY, help="Run only the distribution-estimation experiment and exit. Default can be controlled by IN_CODE_RUN_DISTRIBUTION_ONLY.")
    parser.add_argument("--no-run-distribution-estimation", dest="run_distribution_estimation", action="store_false", help="Disable default distribution-estimation-only mode and run the original task-assignment experiment.")
    parser.add_argument("--also-run-distribution-estimation", action="store_true", default=IN_CODE_RUN_DISTRIBUTION_AFTER_MAIN, help="After the original task-assignment experiment, also run distribution estimation in a distribution_estimation subfolder.")
    parser.add_argument("--dist-grid-size", type=int, default=32, help="Grid side length used by distribution estimation. Keep <=64 because transition inversion is matrix-based.")
    parser.add_argument("--dist-num-records", type=int, default=5000, help="Number of location reports sampled per distribution-estimation repeat.")
    parser.add_argument("--dist-repeats", type=int, default=10, help="Repeats for distribution-estimation experiment.")
    parser.add_argument("--dist-epsilons", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0", help="Epsilon values for distribution estimation.")
    parser.add_argument("--dist-srr-variant", type=str, default="self_first", choices=["self_first", "equal"], help="Use the code's SRR-SelfFirst or SRR-Equal channel for distribution estimation.")
    parser.add_argument("--dist-srr-auto-high-groups", action="store_true", default=True, help="Automatically use broader SRR local groups at low epsilon and true-cell-first SRR at high epsilon for density estimation.")
    parser.add_argument("--no-dist-srr-auto-high-groups", dest="dist_srr_auto_high_groups", action="store_false", help="Disable automatic SRR high-group switching and use --dist-srr-high-groups instead.")
    parser.add_argument("--dist-srr-high-groups", type=int, default=1, help="Fixed high-probability groups for SRR distribution mode when auto switching is disabled. 1 emphasizes the true cell.")
    parser.add_argument("--dist-srr-high-group-switch-epsilon", type=float, default=4.0, help="Epsilon threshold for automatic SRR high-group switching. Below this uses 2 high groups; at/above this uses 1.")
    parser.add_argument("--dist-srr-estimator", type=str, default="density_preserving", choices=["regularized_inverse", "density_preserving"], help="SRR estimator used in distribution estimation. density_preserving uses SRR noisy spatial density at low epsilon and regularized channel inversion at high epsilon.")
    parser.add_argument("--dist-baseline-l2", type=float, default=1e-8, help="Ridge term for HR transition inversion.")
    parser.add_argument("--dist-srr-l2", type=float, default=1e-9, help="Ridge term for SRR transition inversion.")
    parser.add_argument("--dist-srr-smooth-lambda", type=float, default=2e-4, help="Spatial smoothness regularization strength for SRR estimator before epsilon scaling.")
    parser.add_argument("--dist-srr-prior-weight", type=float, default=2e-3, help="Empirical SRR prior strength before epsilon scaling.")
    parser.add_argument("--dist-srr-smooth-rounds", type=int, default=2, help="Number of 3x3 smoothing rounds applied to SRR noisy histogram prior.")
    parser.add_argument("--dist-srr-smooth-center-weight", type=float, default=4.0, help="Center weight in SRR noisy-histogram spatial smoothing.")
    parser.add_argument("--dist-srr-em-rounds-low", type=int, default=45, help="SRR-SelfFirst low-epsilon EM iterations near epsilon=0.")
    parser.add_argument("--dist-srr-em-rounds-mid", type=int, default=25, help="SRR-SelfFirst low-epsilon EM iterations near the switch threshold.")
    parser.add_argument("--dist-srr-em-smooth-weight-low", type=float, default=0.38, help="SRR-SelfFirst EM smoothing weight near epsilon=0.")
    parser.add_argument("--dist-srr-em-smooth-weight-mid", type=float, default=0.12, help="SRR-SelfFirst EM smoothing weight near the switch threshold.")
    parser.add_argument("--dist-srr-em-prior-weight-low", type=float, default=0.22, help="SRR-SelfFirst EM prior blend weight near epsilon=0.")
    parser.add_argument("--dist-srr-em-prior-weight-mid", type=float, default=0.06, help="SRR-SelfFirst EM prior blend weight near the switch threshold.")
    parser.add_argument("--dist-heatmap-epsilon", type=float, default=5.0, help="Epsilon used for the distribution heatmap figure; nearest available epsilon is selected.")
    parser.add_argument("--dist-knn-k", type=int, default=25, help="K used by density-aware k-NN query evaluation; default 25.")
    parser.add_argument("--dist-knn-num-queries", type=int, default=100, help="Number of query grid cells used by k-NN evaluation. Use 0 or a value >= domain size to evaluate all cells.")
    parser.add_argument("--dist-knn-query-mode", type=str, default="density_sample", choices=["density_sample", "top_density", "uniform", "all"], help="How query cells are selected for k-NN evaluation.")
    parser.add_argument("--dist-knn-density-power", type=float, default=1.0, help="Exponent applied to cell density in density-aware k-NN scoring.")
    parser.add_argument("--dist-knn-distance-power", type=float, default=1.0, help="Exponent applied to distance in density-aware k-NN scoring.")
    parser.add_argument("--dist-knn-distance-floor-factor", type=float, default=0.5, help="Distance floor as a fraction of the minimum non-zero grid-center distance.")
    parser.add_argument("--dist-knn-exclude-self", dest="dist_knn_exclude_self", action="store_true", default=True, help="Exclude the query cell itself from k-NN result sets.")
    parser.add_argument("--dist-knn-include-self", dest="dist_knn_exclude_self", action="store_false", help="Include the query cell itself in k-NN result sets.")
    parser.add_argument("--dist-srr-adaptive-knn-local-mass", dest="dist_srr_adaptive_knn_local_mass", action="store_true", default=True, help="Use epsilon-adaptive local-mass post-processing for SRR in density-aware k-NN.")
    parser.add_argument("--no-dist-srr-adaptive-knn-local-mass", dest="dist_srr_adaptive_knn_local_mass", action="store_false", help="Disable SRR adaptive local-mass post-processing in density-aware k-NN.")
    parser.add_argument("--dist-srr-knn-local-mass-weight-low", type=float, default=0.25, help="SRR local-mass weight when epsilon <= low threshold.")
    parser.add_argument("--dist-srr-knn-local-mass-weight-mid", type=float, default=0.10, help="SRR local-mass weight when low threshold < epsilon <= mid threshold.")
    parser.add_argument("--dist-srr-knn-local-mass-weight-high", type=float, default=0.0, help="SRR local-mass weight when epsilon > mid threshold.")
    parser.add_argument("--dist-srr-knn-low-eps-threshold", type=float, default=4.0, help="Low epsilon threshold for SRR adaptive k-NN local-mass.")
    parser.add_argument("--dist-srr-knn-mid-eps-threshold", type=float, default=6.0, help="Middle epsilon threshold for SRR adaptive k-NN local-mass.")
    parser.add_argument("--dist-srr-knn-local-radius", type=int, default=1, help="Neighborhood radius for SRR local-mass density in density-aware k-NN. radius=1 means a 3x3 neighborhood.")

    # Grid-size sensitivity mode. This runs this same script as child processes
    # for several grid sizes, then combines their summaries.
    parser.add_argument("--run-grid-sensitivity", action="store_true", help="Run grid-size sensitivity suite and combine results, instead of a single-grid run")
    parser.add_argument("--grid-sensitivity-sizes", type=str, default="8,16,32", help="Grid sizes used by --run-grid-sensitivity")
    parser.add_argument("--grid-sensitivity-epsilons", type=str, default="2.0,4.0,6.0,8.0", help="Epsilons used by --run-grid-sensitivity")
    parser.add_argument("--grid-sensitivity-repeats", type=int, default=5, help="Repeats used by --run-grid-sensitivity")

    # Workload sensitivity mode. This runs this same script as child processes
    # at one fixed epsilon and creates two total-distance figures.
    parser.add_argument("--run-workload-sensitivity", action="store_true", default=(IN_CODE_RUN_WORKLOAD_ONLY and not _IN_WORKLOAD_CHILD_PROCESS), help="Run worker-count and task-count sensitivity suites at one fixed epsilon, then create total-distance plots. Default can be controlled by IN_CODE_RUN_WORKLOAD_ONLY.")
    parser.add_argument("--also-run-workload-sensitivity", action="store_true", default=(IN_CODE_RUN_WORKLOAD_AFTER_MAIN and not _IN_WORKLOAD_CHILD_PROCESS), help="After the main epsilon experiment, also run workload sensitivity and generate total-distance plots in a subdirectory. Default can be controlled by IN_CODE_RUN_WORKLOAD_AFTER_MAIN.")
    parser.add_argument("--workload-sensitivity-epsilon", type=float, default=IN_CODE_WORKLOAD_EPSILON, help="Fixed epsilon used by workload sensitivity")
    parser.add_argument("--workload-sensitivity-repeats", type=int, default=IN_CODE_WORKLOAD_REPEATS, help="Repeats used by workload sensitivity")
    parser.add_argument("--worker-sensitivity-workers", type=str, default=IN_CODE_WORKER_COUNTS, help="Worker counts for total_distance_vs_workers.png")
    parser.add_argument("--worker-sensitivity-num-tasks", type=int, default=IN_CODE_FIXED_TASKS_FOR_WORKER_SWEEP, help="Fixed task count for worker-count sensitivity")
    parser.add_argument("--task-sensitivity-tasks", type=str, default=IN_CODE_TASK_COUNTS, help="Task counts for total_distance_vs_tasks.png")
    parser.add_argument("--task-sensitivity-num-workers", type=int, default=IN_CODE_FIXED_WORKERS_FOR_TASK_SWEEP, help="Fixed worker count for task-count sensitivity")

    # Same-perturbation-distance comparison. This is a post-analysis over the
    # normal summary table, not an extra privacy run.
    parser.add_argument("--perturbation-targets", type=str, default="5,10,20,30,40", help="Target mean perturbation distances for calibrated comparison")
    parser.add_argument("--disable-perturbation-calibration", action="store_true", help="Do not write same-perturbation-distance comparison files")
    args = parser.parse_args()

    if args.num_workers < args.num_tasks:
        raise ValueError("num-workers must be >= num-tasks")
    if args.grid_size <= 1 or (args.grid_size & (args.grid_size - 1)) != 0:
        raise ValueError("--grid-size must be a power of two, e.g., 8, 16, 32, 64")
    if args.grid_size * args.grid_size > 8192:
        raise ValueError("Domain too large for this simple script. Use grid-size <= 64, preferably 16 or 32.")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.run_distribution_estimation:
        run_distribution_estimation(args)
        return

    if args.run_grid_sensitivity:
        run_grid_sensitivity_via_subprocess(args)
        return

    if args.run_workload_sensitivity:
        run_workload_sensitivity_via_subprocess(args)
        return

    epsilons = parse_epsilons(args.epsilons)
    local_srr_radii = parse_int_list(args.local_srr_radii)
    base_rng = set_seed(args.seed)

    # Load data pool.
    # This Foursquare version reads all .dat files under --foursquare-dir.
    if args.foursquare_dir and os.path.isdir(args.foursquare_dir):
        pool_points, data_meta = load_foursquare_xy_points(args, base_rng)
        width = float(max(data_meta["width_km"], 1e-9))
        height = float(max(data_meta["height_km"], 1e-9))
        distance_unit = "km"
        data_meta["data_source"] = "foursquare_dat"
        data_meta["foursquare_dir_used"] = args.foursquare_dir
        data_meta["distance_unit"] = distance_unit
    else:
        missing_dir = args.foursquare_dir if args.foursquare_dir else "<empty>"
        if not args.allow_synthetic_fallback:
            raise FileNotFoundError(
                "Foursquare .dat directory was not found. Expected directory: "
                f"{missing_dir}. Provide --foursquare-dir \"D:\\\\data2\" "
                "or use --allow-synthetic-fallback for debugging only."
            )
        print(
            "WARNING: Foursquare directory not found; running synthetic fallback because "
            "--allow-synthetic-fallback was provided."
        )
        width = args.width
        height = args.height
        pool_points = generate_synthetic_points(
            base_rng,
            max(args.num_workers + args.num_tasks, 5000),
            width,
            height,
            args.data_mode,
        )
        data_meta = {
            "data_source": "synthetic_fallback",
            "synthetic_width": width,
            "synthetic_height": height,
            "synthetic_pool_size": float(len(pool_points)),
            "distance_unit": "synthetic_unit",
        }
        distance_unit = "synthetic_unit"

    domain = create_grid_domain_from_bounds(width, height, args.grid_size)
    hst = CompleteQuadtreeHST(args.grid_size)

    # Record SRR-Local / SRR-SelfFirst group sizes for reproducibility.
    preview_local_srr = LocalSRRMechanism(
        domain=domain,
        grid_size=args.grid_size,
        epsilon=1.0,
        rng=np.random.default_rng(args.seed),
        radii=local_srr_radii,
        probability_schedule=args.srr_probability_schedule,
        precompute_orders=False,
        self_first=False,
        two_level_high_groups=1,
    )
    preview_self_first_srr = LocalSRRMechanism(
        domain=domain,
        grid_size=args.grid_size,
        epsilon=1.0,
        rng=np.random.default_rng(args.seed),
        radii=local_srr_radii,
        probability_schedule=args.srr_probability_schedule,
        precompute_orders=False,
        self_first=True,
        two_level_high_groups=args.self_first_high_groups,
    )
    data_meta["local_srr_radii"] = ",".join(str(x) for x in local_srr_radii)
    data_meta["local_srr_group_sizes"] = ",".join(str(int(x)) for x in preview_local_srr.group_sizes)
    data_meta["self_first_srr_group_sizes"] = ",".join(str(int(x)) for x in preview_self_first_srr.group_sizes)
    data_meta["local_srr_probability_schedule"] = args.srr_probability_schedule
    data_meta["self_first_high_groups"] = int(args.self_first_high_groups)
    data_meta["experiment_role"] = "formal_main_or_formal_override"
    data_meta["paper_positioning"] = "adaptation study: task-oriented SRR variants for online assignment under LDP"
    data_meta["baseline_note"] = "No no-privacy baselines are included in the exported results"
    data_meta["in_code_run_workload_after_main"] = bool(IN_CODE_RUN_WORKLOAD_AFTER_MAIN)
    data_meta["in_code_run_workload_only"] = bool(IN_CODE_RUN_WORKLOAD_ONLY)
    data_meta["workload_child_process_guard"] = bool(_IN_WORKLOAD_CHILD_PROCESS)

    print(f"Domain size: {len(domain)} ({args.grid_size} x {args.grid_size})")
    print(f"Distance unit: {distance_unit}")
    print("Precomputing distance matrices ...")
    euclid_dist_matrix = make_euclidean_distance_matrix(domain)
    hst_dist_matrix = hst.tree_distance_matrix()

    all_rows: List[Dict[str, float]] = []

    for rep in range(args.repeats):
        rep_seed = args.seed + rep * 1009
        rng = np.random.default_rng(rep_seed)
        workers_true, tasks_true = sample_workers_tasks_from_pool(pool_points, args.num_workers, args.num_tasks, rng)
        workers_idx = map_points_to_domain(workers_true, domain)
        tasks_idx = map_points_to_domain(tasks_true, domain)

        print(f"Repeat {rep + 1}/{args.repeats}: workers={len(workers_true)}, tasks={len(tasks_true)}")
        for eps in epsilons:
            print(f"  epsilon={eps}")
            rows = run_one_setting(
                epsilon=eps,
                workers_true=workers_true,
                tasks_true=tasks_true,
                workers_idx=workers_idx,
                tasks_idx=tasks_idx,
                domain=domain,
                hst=hst,
                euclid_dist_matrix=euclid_dist_matrix,
                hst_dist_matrix=hst_dist_matrix,
                srr_groups=args.srr_groups,
                seed=rep_seed,
                srr_precompute_max_domain=args.srr_precompute_max_domain,
                grid_size=args.grid_size,
                local_srr_radii=local_srr_radii,
                srr_probability_schedule=args.srr_probability_schedule,
                self_first_high_groups=args.self_first_high_groups,
                include_hst_greedy=args.include_hst_greedy,
                include_srr_local=args.include_srr_local,
            )
            for r in rows:
                r["repeat_id"] = rep
                r["distance_unit"] = distance_unit
                r["data_source"] = data_meta.get("data_source", "unknown")
            all_rows.extend(rows)

    raw_path = os.path.join(args.out_dir, "raw_results.csv")
    summary_path = os.path.join(args.out_dir, "summary_results.csv")
    fig_path = os.path.join(args.out_dir, "avg_distance_vs_epsilon.png")
    perturb_fig_path = os.path.join(args.out_dir, "perturbation_distance_vs_epsilon.png")
    ldp_fig_path = os.path.join(args.out_dir, "ldp_only_avg_distance_vs_epsilon.png")
    ldp_perturb_fig_path = os.path.join(args.out_dir, "ldp_only_perturbation_distance_vs_epsilon.png")
    reference_fig_path = os.path.join(args.out_dir, "reference_avg_distance_vs_epsilon.png")
    reference_perturb_fig_path = os.path.join(args.out_dir, "reference_perturbation_distance_vs_epsilon.png")
    perturb_efficiency_fig_path = os.path.join(args.out_dir, "perturbation_efficiency_curve.png")
    calibrated_csv_path = os.path.join(args.out_dir, "calibrated_perturbation_comparison.csv")
    calibrated_fig_path = os.path.join(args.out_dir, "calibrated_perturbation_comparison.png")
    meta_path = os.path.join(args.out_dir, "metadata.txt")
    commands_path = os.path.join(args.out_dir, "recommended_formal_runs.bat")

    write_csv(raw_path, all_rows)
    summary_rows = aggregate_rows(all_rows)
    write_csv(summary_path, summary_rows)
    plot_results(fig_path, summary_rows)
    plot_perturbation_results(perturb_fig_path, summary_rows)
    plot_results(ldp_fig_path, filter_summary_rows(summary_rows, LDP_PLOT_METHODS))
    plot_perturbation_results(ldp_perturb_fig_path, filter_summary_rows(summary_rows, LDP_PLOT_METHODS))
    plot_results(reference_fig_path, filter_summary_rows(summary_rows, REFERENCE_PLOT_METHODS))
    plot_perturbation_results(reference_perturb_fig_path, filter_summary_rows(summary_rows, REFERENCE_PLOT_METHODS))
    if not args.disable_perturbation_calibration:
        plot_perturbation_efficiency_curve(perturb_efficiency_fig_path, summary_rows)
        calibrated_rows = calibrate_by_perturbation_targets(summary_rows, parse_float_list(args.perturbation_targets))
        write_csv(calibrated_csv_path, calibrated_rows)
        plot_calibrated_perturbation_comparison(calibrated_fig_path, calibrated_rows)
    write_metadata(meta_path, data_meta, args)
    if args.write_formal_commands:
        write_recommended_formal_commands(commands_path, os.path.basename(__file__))

    distribution_after_main_dir = None
    if args.also_run_distribution_estimation:
        print("\nRunning distribution estimation after main experiment ...")
        dist_args = argparse.Namespace(**vars(args))
        dist_args.out_dir = os.path.join(args.out_dir, "distribution_estimation")
        dist_args.run_distribution_estimation = True
        dist_args.also_run_distribution_estimation = False
        run_distribution_estimation(dist_args)
        distribution_after_main_dir = dist_args.out_dir

    workload_after_main_dir = None
    if args.also_run_workload_sensitivity:
        print("\nRunning workload sensitivity after main experiment ...")
        workload_args = argparse.Namespace(**vars(args))
        workload_args.out_dir = os.path.join(args.out_dir, "workload_sensitivity")
        workload_args.epsilons = f"{float(args.workload_sensitivity_epsilon):g}"
        workload_args.repeats = int(args.workload_sensitivity_repeats)
        # Prevent nested or accidental recursion. The workload runner itself
        # launches child main-experiment processes and combines their outputs.
        workload_args.run_workload_sensitivity = True
        workload_args.also_run_workload_sensitivity = False
        workload_args.run_grid_sensitivity = True
        run_workload_sensitivity_via_subprocess(workload_args)
        workload_after_main_dir = workload_args.out_dir

    print("\nDone.")
    print(f"Raw results:     {raw_path}")
    print(f"Summary results: {summary_path}")
    print(f"Metadata:        {meta_path}")
    if plt is not None:
        print(f"Plot:            {fig_path}")
        print(f"Perturb plot:    {perturb_fig_path}")
        print(f"LDP plot:        {ldp_fig_path}")
        print(f"LDP perturb:     {ldp_perturb_fig_path}")
        print(f"Reference plot:  {reference_fig_path}")
        print(f"Reference pert.: {reference_perturb_fig_path}")
        if not args.disable_perturbation_calibration:
            print(f"Pert-eff curve:  {perturb_efficiency_fig_path}")
            print(f"Calibrated plot: {calibrated_fig_path}")
            print(f"Calibrated CSV:  {calibrated_csv_path}")
    if args.write_formal_commands:
        print(f"Commands:        {commands_path}")
    if distribution_after_main_dir is not None:
        print(f"Distribution dir:{distribution_after_main_dir}")
        print(f"Distribution CSV:{os.path.join(distribution_after_main_dir, 'distribution_estimation_summary.csv')}")
    if workload_after_main_dir is not None:
        print(f"Workload dir:    {workload_after_main_dir}")
        print(f"Workers plot:    {os.path.join(workload_after_main_dir, 'total_distance_vs_workers.png')}")
        print(f"Tasks plot:      {os.path.join(workload_after_main_dir, 'total_distance_vs_tasks.png')}")

    print("\nSummary: average true distance")
    for r in summary_rows:
        print(
            f"epsilon={r['epsilon']:<4} method={str(r['method']):<18} "
            f"avg={r['avg_true_distance_km_mean']:.4f} ± {r['avg_true_distance_km_std']:.4f} {distance_unit}"
        )


if __name__ == "__main__":
    main()
