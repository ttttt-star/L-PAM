#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

IN_CODE_RUN_WORKLOAD_AFTER_MAIN = False

IN_CODE_RUN_WORKLOAD_ONLY = False

IN_CODE_WORKLOAD_EPSILON = 5.0
IN_CODE_WORKLOAD_REPEATS = 5
IN_CODE_WORKER_COUNTS = "1000,2000,3000,4000,5000"
IN_CODE_FIXED_TASKS_FOR_WORKER_SWEEP = 1000
IN_CODE_TASK_COUNTS = "500,1000,1500,2000,3000"
IN_CODE_FIXED_WORKERS_FOR_TASK_SWEEP = 4000

_IN_WORKLOAD_CHILD_PROCESS = os.environ.get("L-PAM_SKIP_CODE_WORKLOAD", "0") == "1"

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
        raise ValueError("bbox must be: min_lat,max_lat,min_lon,max_lon")
    min_lat, max_lat, min_lon, max_lon = parts
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("Invalid bbox: min must be smaller than max")
    return min_lat, max_lat, min_lon, max_lon


def in_bbox(lat: float, lon: float, bbox: Optional[BBox]) -> bool:
    if bbox is None:
        return True
    min_lat, max_lat, min_lon, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

def read_gowalla_latlon_reservoir(
    path: str,
    max_points: int,
    rng: np.random.Generator,
    lat_col: int = 2,
    lon_col: int = 3,
    bbox: Optional[BBox] = None,
    scan_limit: int = 0,
) -> Tuple[np.ndarray, int]:
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
        raise ValueError(f"No valid Geolife points found{detail}. Check path, columns, or bbox.")

    return np.asarray(sample, dtype=float), matched_count


def auto_dense_bbox(latlon: np.ndarray, cell_deg: float = 0.5, window_deg: float = 1.0) -> BBox:

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


def load_gowalla_xy_points(args: argparse.Namespace, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """Load and project Geolife points according to command-line arguments."""
    user_bbox = parse_bbox(args.gowalla_bbox)
    effective_bbox = user_bbox

    if user_bbox is None and not args.disable_auto_bbox:
        print("Finding dense Gowalla region for local experiment ...")
        global_sample, global_count = read_gowalla_latlon_reservoir(
            path=args.gowalla_path,
            max_points=args.gowalla_auto_sample,
            rng=rng,
            lat_col=args.gowalla_lat_col,
            lon_col=args.gowalla_lon_col,
            bbox=None,
            scan_limit=args.gowalla_scan_limit,
        )
        effective_bbox = auto_dense_bbox(
            global_sample,
            cell_deg=args.gowalla_auto_cell_deg,
            window_deg=args.gowalla_auto_window_deg,
        )
        print(
            "Auto bbox selected: "
            f"min_lat={effective_bbox[0]:.6f}, max_lat={effective_bbox[1]:.6f}, "
            f"min_lon={effective_bbox[2]:.6f}, max_lon={effective_bbox[3]:.6f} "
            f"from {len(global_sample)} sampled points / {global_count} scanned valid points"
        )

    latlon, matched_count = read_gowalla_latlon_reservoir(
        path=args.gowalla_path,
        max_points=args.gowalla_max_points,
        rng=rng,
        lat_col=args.gowalla_lat_col,
        lon_col=args.gowalla_lon_col,
        bbox=effective_bbox,
        scan_limit=args.gowalla_scan_limit,
    )
    xy, meta = latlon_to_xy_km(latlon)
    meta["gowalla_matched_count"] = float(matched_count)
    meta["gowalla_sampled_count"] = float(len(latlon))
    if effective_bbox is not None:
        meta["bbox_min_lat"] = effective_bbox[0]
        meta["bbox_max_lat"] = effective_bbox[1]
        meta["bbox_min_lon"] = effective_bbox[2]
        meta["bbox_max_lon"] = effective_bbox[3]

    print(f"Loaded Geolife points: sampled={len(latlon)}, matched_valid={matched_count}")
    print(f"Projected local area: width={meta['width_km']:.3f} km, height={meta['height_km']:.3f} km")
    return xy, meta

def iter_geolife_plt_files(root: str, max_files: int = 0) -> Iterable[str]:
    """Yield Geolife .plt files recursively from a dataset root or a single file."""
    if os.path.isfile(root):
        if root.lower().endswith(".plt"):
            yield root
        return
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.lower().endswith(".plt"):
                continue
            yield os.path.join(dirpath, name)
            count += 1
            if max_files and count >= max_files:
                return


def read_geolife_latlon_reservoir(
    root: str,
    max_points: int,
    rng: np.random.Generator,
    bbox: Optional[BBox] = None,
    scan_limit: int = 0,
    max_files: int = 0,
) -> Tuple[np.ndarray, int, int, int]:

    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if not os.path.exists(root):
        raise FileNotFoundError(root)

    sample: List[Tuple[float, float]] = []
    matched_count = 0
    files_read = 0
    point_rows_scanned = 0

    for plt_path in iter_geolife_plt_files(root, max_files=max_files):
        files_read += 1
        try:
            with open(plt_path, "r", encoding="utf-8", errors="ignore") as f:
                # Geolife PLT lines 1..6 are headers and can be ignored.
                for _ in range(6):
                    next(f, None)
                for line in f:
                    if scan_limit and point_rows_scanned >= scan_limit:
                        break
                    point_rows_scanned += 1
                    parts = line.strip().split(",")
                    if len(parts) < 2:
                        continue
                    try:
                        lat = float(parts[0])
                        lon = float(parts[1])
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
                        j = int(rng.integers(0, matched_count))
                        if j < max_points:
                            sample[j] = (lat, lon)
        except OSError:
            continue
        if scan_limit and point_rows_scanned >= scan_limit:
            break

    if not sample:
        detail = ""
        if bbox is not None:
            detail = f" within bbox={bbox}"
        raise ValueError(
            f"No valid Geolife points found{detail}. Check --geolife-root, bbox, or whether .plt files exist."
        )
    return np.asarray(sample, dtype=float), matched_count, files_read, point_rows_scanned


def load_geolife_xy_points(args: argparse.Namespace, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """Load and project Geolife PLT points according to command-line arguments."""
    user_bbox = parse_bbox(args.geolife_bbox)
    effective_bbox = user_bbox

    if user_bbox is None and not args.disable_auto_bbox:
        print("Finding dense Geolife region for local experiment ...")
        global_sample, global_count, global_files, global_rows = read_geolife_latlon_reservoir(
            root=args.geolife_root,
            max_points=args.geolife_auto_sample,
            rng=rng,
            bbox=None,
            scan_limit=args.geolife_scan_limit,
            max_files=args.geolife_max_files,
        )
        effective_bbox = auto_dense_bbox(
            global_sample,
            cell_deg=args.geolife_auto_cell_deg,
            window_deg=args.geolife_auto_window_deg,
        )
        print(
            "Auto bbox selected: "
            f"min_lat={effective_bbox[0]:.6f}, max_lat={effective_bbox[1]:.6f}, "
            f"min_lon={effective_bbox[2]:.6f}, max_lon={effective_bbox[3]:.6f} "
            f"from {len(global_sample)} sampled points / {global_count} matched valid points "
            f"/ {global_files} files / {global_rows} point rows"
        )

    latlon, matched_count, files_read, point_rows_scanned = read_geolife_latlon_reservoir(
        root=args.geolife_root,
        max_points=args.geolife_max_points,
        rng=rng,
        bbox=effective_bbox,
        scan_limit=args.geolife_scan_limit,
        max_files=args.geolife_max_files,
    )
    xy, meta = latlon_to_xy_km(latlon)
    meta["geolife_matched_count"] = float(matched_count)
    meta["geolife_sampled_count"] = float(len(latlon))
    meta["geolife_files_read"] = float(files_read)
    meta["geolife_point_rows_scanned"] = float(point_rows_scanned)
    if effective_bbox is not None:
        meta["bbox_min_lat"] = effective_bbox[0]
        meta["bbox_max_lat"] = effective_bbox[1]
        meta["bbox_min_lon"] = effective_bbox[2]
        meta["bbox_max_lon"] = effective_bbox[3]

    print(
        f"Loaded Geolife points: sampled={len(latlon)}, matched_valid={matched_count}, "
        f"files_read={files_read}, point_rows_scanned={point_rows_scanned}"
    )
    print(f"Projected local area: width={meta['width_km']:.3f} km, height={meta['height_km']:.3f} km")
    return xy, meta

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

    name = "OLH"

    def __init__(self, grid_size: int, epsilon: float, rng: np.random.Generator):
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if grid_size <= 1 or (grid_size & (grid_size - 1)) != 0:
            raise ValueError("OLH expects a power-of-two square grid")
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
        # OLH randomly reports one hierarchy level. Level 1 is the coarsest
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

class OUEMechanism:

    name = "OUE"

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

class PAMMechanism:

    name = "L-PAM"

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


class LocalPAMMechanism:

    name = "L-PAM-Local"

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
            raise ValueError("LocalL-PAM expects a square grid domain")
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
            # Aggressive high-epsilon variant. For L-PAM we usually set
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
            raise ValueError("For OLH/HST, grid_size must be a power of two, e.g., 8, 16, 32, 64")
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

def build_mechanism(
    name: str,
    domain: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    hst: CompleteQuadtreeHST,
    pam_precompute_max_domain: int,
    grid_size: int,
    local_pam_radii: Sequence[int],
    pam_probability_schedule: str,
    self_first_high_groups: int,
):
    if name == "NoPrivacy":
        return NoPrivacyMechanism(len(domain), rng)
    if name == "GRR":
        return GRRMechanism(len(domain), epsilon, rng)
    if name == "HR":
        return HRMechanism(len(domain), epsilon, rng)
    if name == "OLH":
        return OLHHMechanism(grid_size, epsilon, rng)
    if name == "OUE":
        return OUEMechanism(len(domain), epsilon, rng)
    if name == "L-PAM-Local":
        precompute = len(domain) <= pam_precompute_max_domain
        return LocalPAMMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=rng,
            radii=local_pam_radii,
            probability_schedule=pam_probability_schedule,
            precompute_orders=precompute,
            self_first=False,
            two_level_high_groups=1,
        )
    if name == "L-PAM":
        precompute = len(domain) <= pam_precompute_max_domain
        return LocalPAMMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=rng,
            radii=local_pam_radii,
            probability_schedule=pam_probability_schedule,
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
    seed: int,
    pam_precompute_max_domain: int,
    grid_size: int,
    local_pam_radii: Sequence[int],
    pam_probability_schedule: str,
    self_first_high_groups: int,
    include_hst_greedy: bool,
    include_pam_local: bool,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    # Main comparison among privacy mechanisms is inside the LDP family:
    # GRR, HR, OLH, OUE/OUE, and L-PAM. L-PAM-Local is optional ablation only.
    base_mechanisms = [
        ("GRR", "GRR"),
        ("HR", "HR"),
        ("OLH", "OLH"),
        ("OUE", "OUE"),
        ("L-PAM", "L-PAM"),
    ]
    if include_pam_local:
        # L-PAM-Local is useful as an ablation bridge toward
        # L-PAM. It is excluded from main figures by default because
        # under two_level with self_first_high_groups=2 it has nearly the same
        # high-probability neighborhood as L-PAM.
        insert_pos = next(i for i, (_, mech) in enumerate(base_mechanisms) if mech == "L-PAM")
        base_mechanisms.insert(insert_pos, ("L-PAM-Local", "L-PAM-Local"))
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
                hst,
                pam_precompute_max_domain,
                grid_size,
                local_pam_radii,
                pam_probability_schedule,
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




def _format_txt_number(value: float) -> str:
    return f"{float(value):.10g}"


def write_curve_txt(path: str, rows: List[Dict[str, float]], x_key: str, y_key: str, methods: Optional[Sequence[str]] = None) -> None:

    if not rows:
        return
    if methods is None:
        method_order = sorted({str(r.get("method")) for r in rows})
    else:
        method_order = [m for m in methods if any(str(r.get("method")) == m for r in rows)]
    with open(path, "w", encoding="utf-8") as f:
        for method in method_order:
            sub = [r for r in rows if str(r.get("method")) == method and x_key in r and y_key in r]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: float(r[x_key]))
            values = [_format_txt_number(float(r[y_key])) for r in sub]
            f.write(",".join(values) + "\n")


def write_perturbation_curve_txt(path: str, rows: List[Dict[str, float]], methods: Optional[Sequence[str]] = None) -> None:
    """Write perturbation-distance curves as one comma-separated line per method."""
    if not rows:
        return
    if methods is None:
        method_order = sorted({str(r.get("method")) for r in rows})
    else:
        method_order = [m for m in methods if any(str(r.get("method")) == m for r in rows)]
    with open(path, "w", encoding="utf-8") as f:
        for method in method_order:
            sub = [r for r in rows if str(r.get("method")) == method]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: float(r["epsilon"]))
            values = [
                _format_txt_number(
                    0.5 * (
                        float(r["worker_perturbation_distance_km_mean"])
                        + float(r["task_perturbation_distance_km_mean"])
                    )
                )
                for r in sub
            ]
            f.write(",".join(values) + "\n")

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
    plt.title("Geolife online task assignment under privacy mechanisms")
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


# Paper-oriented method groups. The LDP comparison is GRR/HR/OLH/OUE/L-PAM variants.
LDP_PLOT_METHODS = [
    "GRR-Greedy",
    "HR-Greedy",
    "OLH-Greedy",
    "OUE-Greedy",
    "L-PAM-Greedy",
]

# Optional ablation method. It is not in main figures by default because its
# curve nearly overlaps L-PAM under the current two-level setting.
OPTIONAL_ABLATION_METHODS = [
    "L-PAM-Local-Greedy",
]

REFERENCE_PLOT_METHODS = [
    "GRR-Greedy",
    "OLH-Greedy",
    "OUE-Greedy",
    "L-PAM-Greedy",
]

PERTURBATION_CALIBRATION_METHODS = [
    "GRR-Greedy",
    "HR-Greedy",
    "OLH-Greedy",
    "OUE-Greedy",
    "L-PAM-Greedy",
]

GRID_SENSITIVITY_METHODS = [
    "GRR-Greedy",
    "OLH-Greedy",
    "OUE-Greedy",
    "L-PAM-Greedy",
]

WORKLOAD_SENSITIVITY_METHODS = [
    "GRR-Greedy",
    "HR-Greedy",
    "OLH-Greedy",
    "OUE-Greedy",
    "L-PAM-Greedy",
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
            "--geolife-root", args.geolife_root,
            "--geolife-max-points", str(args.geolife_max_points),
            "--geolife-scan-limit", str(args.geolife_scan_limit),
            "--geolife-max-files", str(args.geolife_max_files),
            "--grid-size", str(g),
            "--num-workers", str(args.num_workers),
            "--num-tasks", str(args.num_tasks),
            "--epsilons", epsilons,
            "--repeats", str(repeats),
            "--local-pam-radii", args.local_pam_radii,
            "--pam-probability-schedule", args.pam_probability_schedule,
            "--self-first-high-groups", str(args.self_first_high_groups),
            "--pam-precompute-max-domain", str(args.pam_precompute_max_domain),
            "--seed", str(args.seed),
            "--out-dir", child_out,
        ]
        if args.geolife_bbox:
            cmd += ["--geolife-bbox", args.geolife_bbox]
        if args.disable_auto_bbox:
            cmd += ["--disable-auto-bbox"]
        if args.allow_synthetic_fallback:
            cmd += ["--allow-synthetic-fallback"]
        if args.include_hst_greedy:
            cmd += ["--include-hst-greedy"]
        if args.include_pam_local:
            cmd += ["--include-pam-local"]
        print("\n[grid sensitivity] running:", " ".join(cmd))
        child_env = os.environ.copy()
        child_env["L-PAM_SKIP_CODE_WORKLOAD"] = "1"
        subprocess.run(cmd, check=True, env=child_env)

    combined = combine_grid_sensitivity_results(args.out_dir, grid_sizes)
    combined_path = os.path.join(args.out_dir, "grid_sensitivity_combined_summary.csv")
    write_csv(combined_path, combined)  # type: ignore[arg-type]
    plot_grid_sensitivity(os.path.join(args.out_dir, "grid_sensitivity_avg_distance.png"), combined)
    print("\nGrid sensitivity complete.")
    print(f"Combined summary: {combined_path}")
    print(f"Grid plot:        {os.path.join(args.out_dir, 'grid_sensitivity_avg_distance.png')}")




def combine_workload_sensitivity_results(parent_out_dir: str) -> List[Dict[str, float]]:

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
            "--geolife-root", args.geolife_root,
            "--geolife-max-points", str(args.geolife_max_points),
            "--geolife-scan-limit", str(args.geolife_scan_limit),
            "--geolife-max-files", str(args.geolife_max_files),
            "--grid-size", str(args.grid_size),
            "--num-workers", str(num_workers),
            "--num-tasks", str(num_tasks),
            "--epsilons", f"{eps:g}",
            "--repeats", str(repeats),
            "--local-pam-radii", args.local_pam_radii,
            "--pam-probability-schedule", args.pam_probability_schedule,
            "--self-first-high-groups", str(args.self_first_high_groups),
            "--pam-precompute-max-domain", str(args.pam_precompute_max_domain),
            "--seed", str(args.seed),
            "--out-dir", child_out,
            "--disable-perturbation-calibration",
        ]
        if args.geolife_bbox:
            cmd += ["--geolife-bbox", args.geolife_bbox]
        if args.disable_auto_bbox:
            cmd += ["--disable-auto-bbox"]
        if args.allow_synthetic_fallback:
            cmd += ["--allow-synthetic-fallback"]
        if args.include_hst_greedy:
            cmd += ["--include-hst-greedy"]
        if args.include_pam_local:
            cmd += ["--include-pam-local"]
        print("\n[workload sensitivity] running:", " ".join(cmd))
        child_env = os.environ.copy()
        child_env["L-PAM_SKIP_CODE_WORKLOAD"] = "1"
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
    workers_txt = os.path.join(args.out_dir, "total_distance_vs_workers.txt")
    tasks_txt = os.path.join(args.out_dir, "total_distance_vs_tasks.txt")
    plot_workload_total_distance(workers_fig, combined, sensitivity_type="workers")
    plot_workload_total_distance(tasks_fig, combined, sensitivity_type="tasks")
    write_curve_txt(
        workers_txt,
        [r for r in combined if str(r.get("sensitivity_type")) == "workers"],
        "num_workers",
        "total_true_distance_km_mean",
        WORKLOAD_SENSITIVITY_METHODS,
    )
    write_curve_txt(
        tasks_txt,
        [r for r in combined if str(r.get("sensitivity_type")) == "tasks"],
        "num_tasks",
        "total_true_distance_km_mean",
        WORKLOAD_SENSITIVITY_METHODS,
    )

    print("\nWorkload sensitivity complete.")
    print(f"Combined summary: {combined_path}")
    print(f"Workers plot:     {workers_fig}")
    print(f"Workers txt:      {workers_txt}")
    print(f"Tasks plot:       {tasks_fig}")
    print(f"Tasks txt:        {tasks_txt}")

def write_recommended_formal_commands(path: str, script_name: str) -> None:
    """Write optional commands for robustness/sensitivity experiments."""
    commands = []
    base = (
        f'python {script_name} ^\n'
        '  --geolife-root "Your file location" ^\n'
        '  --geolife-bbox "39.75,40.10,116.15,116.60" ^\n'
        '  --geolife-max-points 400000 ^\n'
    )
    commands.append("REM Main formal Geolife experiment")
    commands.append(base +
        '  --grid-size 64 ^\n'
        '  --num-workers 2000 ^\n'
        '  --num-tasks 1000 ^\n'
        '  --epsilons 0.1,0.5,1.0,2.0,4.0,6.0,8.0 ^\n'
        '  --repeats 10 ^\n'
        '  --out-dir geolife_formal_main_g64_w2000_t1000_r10')
    commands.append("\nREM Grid-size sensitivity: run grid 8, 16, and 32")
    commands.append(base +
        '  --run-grid-sensitivity ^\n'
        '  --grid-sensitivity-sizes 8,16,32 ^\n'
        '  --grid-sensitivity-epsilons 2.0,4.0,6.0,8.0 ^\n'
        '  --grid-sensitivity-repeats 5 ^\n'
        '  --out-dir geolife_grid_sensitivity')
    commands.append("\nREM Main experiment plus workload sensitivity")
    commands.append(base +
        '  --grid-size 64 ^\n'
        '  --out-dir geolife_formal_with_workload')
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

def main() -> None:
    parser = argparse.ArgumentParser(description="GRR / HR / OLH / OUE / L-PAM variants for online task assignment on Geolife Trajectories 1.3")

    # Data source.
    parser.add_argument(
        "--geolife-root",
        type=str,
        default=r"Your file location",
        help="Path to Geolife Trajectories 1.3 dataset root. Usually the Data folder containing user folders and .plt files.",
    )
    parser.add_argument(
        "--geolife-max-points",
        type=int,
        default=400000,
        help="Reservoir sample size after filtering Geolife points.",
    )
    parser.add_argument(
        "--geolife-scan-limit",
        type=int,
        default=0,
        help="Max Geolife point rows to scan after PLT headers; 0 = scan all.",
    )
    parser.add_argument(
        "--geolife-max-files",
        type=int,
        default=0,
        help="Max number of .plt files to read; 0 = read all files.",
    )
    parser.add_argument(
        "--geolife-bbox",
        type=str,
        default="39.75,40.10,116.15,116.60",
        help="BBox: min_lat,max_lat,min_lon,max_lon. Default crops a dense Beijing area.",
    )
    parser.add_argument("--disable-auto-bbox", action="store_true", help="Disable auto dense-area crop when no bbox is provided")
    parser.add_argument("--geolife-auto-sample", type=int, default=300000, help="Sample size used to discover a dense bbox")
    parser.add_argument("--geolife-auto-cell-deg", type=float, default=0.05, help="Coarse cell size in degrees for auto bbox")
    parser.add_argument("--geolife-auto-window-deg", type=float, default=0.35, help="Window size in degrees for auto bbox")
    parser.add_argument("--allow-synthetic-fallback", action="store_true", help="Use synthetic data only when Geolife path is missing or unreadable")

    # Synthetic fallback.
    parser.add_argument("--width", type=float, default=200.0, help="Synthetic area width")
    parser.add_argument("--height", type=float, default=200.0, help="Synthetic area height")
    parser.add_argument("--data-mode", type=str, default="mixture", choices=["mixture", "uniform"], help="Synthetic distribution")

    # Experiment parameters.
    parser.add_argument("--grid-size", type=int, default=128, help="Grid side length; must be power of two for OLH/HST. Default 64 matches the latest formal Geolife/Gowalla configuration.")
    parser.add_argument("--num-workers", type=int, default=2000, help="Formal main experiment default: 2000")
    parser.add_argument("--num-tasks", type=int, default=1000, help="Formal main experiment default: 1000")
    parser.add_argument("--epsilons", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0", help="Formal main epsilon list")
    parser.add_argument("--local-pam-radii", type=str, default="1,2,3,4", help="Neighborhood radii for L-PAM-Local, e.g. 1,2,3,4 creates groups <=1, <=2, <=3, <=4, and rest")
    parser.add_argument("--pam-probability-schedule", type=str, default="two_level", choices=["linear", "exponential", "two_level"], help="Per-location staircase shape for L-PAM-Local and L-PAM. Default two_level is more aggressive for task assignment.")
    parser.add_argument("--self-first-high-groups", type=int, default=2, help="For L-PAM with two_level schedule, number of nearest groups using the highest per-location probability. 2 means true cell + 1-hop ring.")
    parser.add_argument("--pam-precompute-max-domain", type=int, default=4096, help="Precompute L-PAM distance orders only if |D| <= this value")
    parser.add_argument("--include-hst-greedy", action="store_true", help="Also output HSTGreedy curves for GRR/HR/OUE/L-PAM/OLH. Off by default to keep figures compact.")
    parser.add_argument("--include-pam-local", action="store_true", help="Include L-PAM-Local as an ablation/appendix method. Off by default because it nearly overlaps L-PAM in the main setting.")
    parser.add_argument("--repeats", type=int, default=10, help="Formal main experiment default: 10 repeats")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="geolife_formal")
    parser.add_argument("--write-formal-commands", action="store_true", help="Also write a Windows .bat file containing recommended formal/sensitivity runs into out-dir")

    # Grid-size sensitivity mode. This runs this same script as child processes
    # for several grid sizes, then combines their summaries.
    parser.add_argument("--run-grid-sensitivity", action="store_true", help="Run grid-size sensitivity suite and combine results, instead of a single-grid run")
    parser.add_argument("--grid-sensitivity-sizes", type=str, default="8,16,32,64,128", help="Grid sizes used by --run-grid-sensitivity")
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
        raise ValueError("--grid-size must be a power of two, e.g., 8, 16,32, 64 ,128")
    if args.grid_size * args.grid_size > 32768:
        raise ValueError("Domain too large for this simple script. Use grid-size <= 128")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.run_grid_sensitivity:
        run_grid_sensitivity_via_subprocess(args)
        return

    if args.run_workload_sensitivity:
        run_workload_sensitivity_via_subprocess(args)
        return

    epsilons = parse_epsilons(args.epsilons)
    local_pam_radii = parse_int_list(args.local_pam_radii)
    base_rng = set_seed(args.seed)

    # Load data pool.
    # Geolife formal runs should use real PLT trajectory points. The script does
    # not silently run synthetic data unless --allow-synthetic-fallback is given.
    if args.geolife_root and os.path.exists(args.geolife_root):
        pool_points, data_meta = load_geolife_xy_points(args, base_rng)
        width = float(max(data_meta["width_km"], 1e-9))
        height = float(max(data_meta["height_km"], 1e-9))
        distance_unit = "km"
        data_meta["data_source"] = "geolife"
        data_meta["geolife_root_used"] = args.geolife_root
        data_meta["distance_unit"] = distance_unit
    else:
        missing_path = args.geolife_root if args.geolife_root else "<empty>"
        if not args.allow_synthetic_fallback:
            raise FileNotFoundError(
                "Geolife dataset was not found. Expected a dataset root such as "
                f"{missing_path}. Provide --geolife-root \"Your file location"
                "or use --allow-synthetic-fallback for debugging only."
            )
        print(
            "WARNING: Geolife files not found; running synthetic fallback because "
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

    # Record L-PAM-Local / L-PAM group sizes for reproducibility.
    preview_local_pam = LocalPAMMechanism(
        domain=domain,
        grid_size=args.grid_size,
        epsilon=1.0,
        rng=np.random.default_rng(args.seed),
        radii=local_pam_radii,
        probability_schedule=args.pam_probability_schedule,
        precompute_orders=False,
        self_first=False,
        two_level_high_groups=1,
    )
    preview_self_first_pam = LocalPAMMechanism(
        domain=domain,
        grid_size=args.grid_size,
        epsilon=1.0,
        rng=np.random.default_rng(args.seed),
        radii=local_pam_radii,
        probability_schedule=args.pam_probability_schedule,
        precompute_orders=False,
        self_first=True,
        two_level_high_groups=args.self_first_high_groups,
    )
    data_meta["local_pam_radii"] = ",".join(str(x) for x in local_pam_radii)
    data_meta["local_pam_group_sizes"] = ",".join(str(int(x)) for x in preview_local_pam.group_sizes)
    data_meta["self_first_pam_group_sizes"] = ",".join(str(int(x)) for x in preview_self_first_pam.group_sizes)
    data_meta["local_pam_probability_schedule"] = args.pam_probability_schedule
    data_meta["self_first_high_groups"] = int(args.self_first_high_groups)
    data_meta["experiment_role"] = "formal_main_or_formal_override"
    data_meta["paper_positioning"] = "adaptation study: task-oriented L-PAM variants for online assignment under LDP"
    data_meta["olh_role"] = "hierarchical OLH baseline adapted to output one proxy grid cell for online assignment"
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
                seed=rep_seed,
                pam_precompute_max_domain=args.pam_precompute_max_domain,
                grid_size=args.grid_size,
                local_pam_radii=local_pam_radii,
                pam_probability_schedule=args.pam_probability_schedule,
                self_first_high_groups=args.self_first_high_groups,
                include_hst_greedy=args.include_hst_greedy,
                include_pam_local=args.include_pam_local,
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
    avg_txt_path = os.path.join(args.out_dir, "avg_distance_vs_epsilon.txt")
    perturb_txt_path = os.path.join(args.out_dir, "perturbation_distance_vs_epsilon.txt")
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
    write_curve_txt(avg_txt_path, summary_rows, "epsilon", "avg_true_distance_km_mean")
    write_perturbation_curve_txt(perturb_txt_path, summary_rows)
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
        workload_args.run_grid_sensitivity = False
        run_workload_sensitivity_via_subprocess(workload_args)
        workload_after_main_dir = workload_args.out_dir

    print("\nDone.")
    print(f"Raw results:     {raw_path}")
    print(f"Summary results: {summary_path}")
    print(f"Metadata:        {meta_path}")
    if plt is not None:
        print(f"Plot:            {fig_path}")
        print(f"Avg txt:         {avg_txt_path}")
        print(f"Perturb plot:    {perturb_fig_path}")
        print(f"Perturb txt:     {perturb_txt_path}")
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
    if workload_after_main_dir is not None:
        print(f"Workload dir:    {workload_after_main_dir}")
        print(f"Workers plot:    {os.path.join(workload_after_main_dir, 'total_distance_vs_workers.png')}")
        print(f"Workers txt:     {os.path.join(workload_after_main_dir, 'total_distance_vs_workers.txt')}")
        print(f"Tasks plot:      {os.path.join(workload_after_main_dir, 'total_distance_vs_tasks.png')}")
        print(f"Tasks txt:       {os.path.join(workload_after_main_dir, 'total_distance_vs_tasks.txt')}")

    print("\nSummary: average true distance")
    for r in summary_rows:
        print(
            f"epsilon={r['epsilon']:<4} method={str(r['method']):<18} "
            f"avg={r['avg_true_distance_km_mean']:.4f} ± {r['avg_true_distance_km_std']:.4f} {distance_unit}"
        )


if __name__ == "__main__":
    main()
