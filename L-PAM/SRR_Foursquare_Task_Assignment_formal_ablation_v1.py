#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foursquare-user-dataset formal experiment for SRR adaptation in online spatial task assignment.

This script keeps the same experimental framework used in the Gowalla, Geolife,
and synthetic formal experiments, but replaces the spatial point pool with the
real Foursquare user dataset files that you have already extracted to D:\\data2.

Default formal configuration:
    grid_size = 64
    num_workers = 2000
    num_tasks = 1000
    repeats = 10
    epsilons = 0.1,0.5,1.0,2.0,4.0,6.0,8.0,10.0
    workload sensitivity epsilon = 8.0

Expected input folder:
    D:\\data2

The loader searches the folder recursively for files whose names contain
"venues" and "checkins". It first reads venue latitude/longitude coordinates,
then uses check-in counts to sample locations proportional to venue activity.
If the check-in file cannot be parsed, it falls back to uniform venue sampling
and prints a warning.

Run example:
    python SRR_OLHH_Foursquare_Task_Assignment_formal_v1.py

Optional examples:
    python SRR_OLHH_Foursquare_Task_Assignment_formal_v1.py --foursquare-root "D:\\data2" --out-dir foursquare_formal
    python SRR_OLHH_Foursquare_Task_Assignment_formal_v1.py --foursquare-bbox "40.55,40.95,-74.10,-73.70" --out-dir foursquare_nyc_formal
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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

# Set this to True only if you want to skip the main epsilon experiment and run
# workload sensitivity alone. Usually keep this False for formal runs.
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


# ---------------------------------------------------------------------------
# Ablation experiment switches
# ---------------------------------------------------------------------------
# Set this to True if you want the script to run the ablation experiment by
# default without passing --run-ablation. Usually keep False and enable it by
# command line so the original main experiment is not accidentally replaced.
IN_CODE_RUN_ABLATION_ONLY = True

# Set this to True if you want the script to run the normal main experiment
# first and then run ablation in <out_dir>/ablation/.
IN_CODE_RUN_ABLATION_AFTER_MAIN = False

# Default SRR ablation variants. These variants decompose SRR-SelfFirst into
# grouping, self-first, one-hop-high, and probability-schedule components.
DEFAULT_ABLATION_VARIANTS = (
    "Full-SRR-SelfFirst,"
    "SRR-SelfFirst-Adaptive,"
    "NoSelfFirst,"
    "SelfCellOnly,"
    "NoTwoLevel,"
    "LinearSchedule"
)


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


# ---------------------------------------------------------------------------
# Gowalla loading and coordinate projection
# ---------------------------------------------------------------------------


def read_gowalla_latlon_reservoir(
    path: str,
    max_points: int,
    rng: np.random.Generator,
    lat_col: int = 2,
    lon_col: int = 3,
    bbox: Optional[BBox] = None,
    scan_limit: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Reservoir-sample latitude/longitude pairs from Gowalla.

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
        raise ValueError(f"No valid Geolife points found{detail}. Check path, columns, or bbox.")

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


# ---------------------------------------------------------------------------
# Geolife loading
# ---------------------------------------------------------------------------


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
    """Reservoir-sample latitude/longitude pairs from Geolife PLT files.

    Returns:
        sample_latlon: array with shape [n, 2], columns = lat, lon
        matched_count: number of valid points that passed bbox
        files_read: number of PLT files opened
        point_rows_scanned: number of point rows scanned after PLT headers
    """
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



# ---------------------------------------------------------------------------
# Foursquare loading
# ---------------------------------------------------------------------------

_ID_KEYS = {
    "id", "venueid", "venue", "venue_id", "vid", "poi", "poiid", "poi_id",
    "locationid", "location_id", "businessid", "business_id"
}
_LAT_KEYS = {"lat", "latitude", "y"}
_LON_KEYS = {"lon", "lng", "longitude", "x"}
_CHECKIN_VENUE_KEYS = {
    "venueid", "venue", "venue_id", "vid", "poi", "poiid", "poi_id",
    "locationid", "location_id", "businessid", "business_id"
}
_TEXT_EXTS = {"", ".txt", ".dat", ".csv", ".tsv", ".json", ".jsonl"}
_SKIP_EXTS = {".7z", ".zip", ".rar", ".gz", ".bz2", ".xz", ".png", ".jpg", ".jpeg", ".pdf"}


def _norm_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _clean_id(value: object) -> str:
    s = str(value).strip().strip('"').strip("'").strip()
    return s


def _split_loose(line: str) -> List[str]:
    text = line.strip().lstrip("\ufeff")
    if not text:
        return []
    if "\t" in text:
        return [x.strip() for x in text.split("\t")]
    if "," in text:
        try:
            return [x.strip() for x in next(csv.reader([text]))]
        except Exception:
            return [x.strip() for x in text.split(",")]
    if ";" in text:
        return [x.strip() for x in text.split(";")]
    return text.split()


def _is_probable_header(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    keys = {_norm_col_name(t) for t in tokens}
    header_hits = len(keys & (_ID_KEYS | _LAT_KEYS | _LON_KEYS | _CHECKIN_VENUE_KEYS))
    alpha_hits = sum(any(ch.isalpha() for ch in str(t)) for t in tokens)
    return header_hits >= 1 and alpha_hits >= 1


def _float_or_none(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        # Strip common wrappers that appear in raw dumps.
        v = float(str(x).strip().strip('"').strip("'"))
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _valid_lat_lon(lat: Optional[float], lon: Optional[float]) -> bool:
    return lat is not None and lon is not None and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _find_col(header_map: Dict[str, int], keys: set) -> Optional[int]:
    for k in keys:
        if k in header_map:
            return header_map[k]
    # relaxed contains matching, e.g. venueId, venue_id_str, location.lat
    for col, idx in header_map.items():
        for k in keys:
            if k and (col == k or col.endswith(k) or k in col):
                return idx
    return None


def _find_dataset_file(root: str, explicit_path: str, keywords: Sequence[str]) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(explicit_path)
        return explicit_path
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Foursquare root folder not found: {root}")

    candidates: List[Tuple[int, str]] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("."):
                continue
            lower = name.lower()
            base, ext = os.path.splitext(lower)
            if ext in _SKIP_EXTS:
                continue
            if ext not in _TEXT_EXTS:
                # Keep no-extension or common text files only; this avoids binary cache files.
                continue
            if not any(k.lower() in lower for k in keywords):
                continue
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            if size <= 0:
                continue
            score = 0
            if base in [k.lower() for k in keywords]:
                score += 100
            if lower.startswith(tuple(k.lower() for k in keywords)):
                score += 50
            if ext in {".txt", ".dat", ".csv", ".tsv"}:
                score += 20
            # Prefer larger files if all else equal.
            score += min(int(size // 1024), 5000)
            candidates.append((score, path))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find a file containing {keywords} under {root}. "
            "Use --foursquare-venue-file or --foursquare-checkin-file explicitly."
        )
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def _json_extract_latlon(obj: object) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if not isinstance(obj, dict):
        return None, None, None
    flat: Dict[str, object] = {}

    def visit(prefix: str, val: object) -> None:
        if isinstance(val, dict):
            for kk, vv in val.items():
                visit(f"{prefix}.{kk}" if prefix else str(kk), vv)
        else:
            flat[_norm_col_name(prefix)] = val

    visit("", obj)
    vid = None
    for key in _ID_KEYS:
        if key in flat:
            vid = _clean_id(flat[key])
            break
    lat = None
    lon = None
    for key in _LAT_KEYS:
        if key in flat:
            lat = _float_or_none(flat[key])
            break
    for key in _LON_KEYS:
        if key in flat:
            lon = _float_or_none(flat[key])
            break
    return vid, lat, lon


def _parse_venue_tokens(
    tokens: Sequence[str],
    header_map: Optional[Dict[str, int]],
    venue_id_col: int,
    lat_col: int,
    lon_col: int,
    fallback_id: str,
) -> Optional[Tuple[str, float, float]]:
    if len(tokens) < 2:
        return None

    vid: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    if venue_id_col >= 0 and venue_id_col < len(tokens):
        vid = _clean_id(tokens[venue_id_col])
    if lat_col >= 0 and lon_col >= 0 and lat_col < len(tokens) and lon_col < len(tokens):
        lat = _float_or_none(tokens[lat_col])
        lon = _float_or_none(tokens[lon_col])

    if header_map is not None:
        if vid is None:
            i = _find_col(header_map, _ID_KEYS)
            if i is not None and i < len(tokens):
                vid = _clean_id(tokens[i])
        if lat is None or lon is None:
            i_lat = _find_col(header_map, _LAT_KEYS)
            i_lon = _find_col(header_map, _LON_KEYS)
            if i_lat is not None and i_lon is not None and i_lat < len(tokens) and i_lon < len(tokens):
                lat = _float_or_none(tokens[i_lat])
                lon = _float_or_none(tokens[i_lon])

    # Auto-detect latitude/longitude from adjacent numeric columns.
    if not _valid_lat_lon(lat, lon):
        best: Optional[Tuple[int, float, float]] = None
        for i in range(0, len(tokens) - 1):
            a = _float_or_none(tokens[i])
            b = _float_or_none(tokens[i + 1])
            if _valid_lat_lon(a, b):
                # Prefer coordinate pairs after the ID/name columns; this avoids using user IDs as coordinates.
                penalty = 0 if i >= 1 else 10
                score = penalty + i
                if best is None or score < best[0]:
                    best = (score, float(a), float(b))
        if best is not None:
            _, lat, lon = best

    if vid is None:
        # The first token is the usual venue identifier in these raw dumps.
        vid = _clean_id(tokens[0]) if tokens else fallback_id
    if not vid:
        vid = fallback_id
    if not _valid_lat_lon(lat, lon):
        return None
    return vid, float(lat), float(lon)


def read_foursquare_venues(
    path: str,
    venue_id_col: int = -1,
    lat_col: int = -1,
    lon_col: int = -1,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    """Read venue coordinates from the extracted Foursquare venues file.

    The loader is deliberately format-tolerant because different mirrors of
    foursquare-user-dataset-master may decompress to .txt, .dat, .csv, or JSONL.
    """
    venues: Dict[str, Tuple[float, float]] = {}
    header_map: Optional[Dict[str, int]] = None
    scanned = 0
    parsed = 0
    duplicate = 0

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        first_data_line = True
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            scanned += 1
            if text.startswith("{"):
                try:
                    obj = json.loads(text)
                    vid, lat, lon = _json_extract_latlon(obj)
                    if vid and _valid_lat_lon(lat, lon):
                        if vid in venues:
                            duplicate += 1
                        venues[vid] = (float(lat), float(lon))
                        parsed += 1
                    continue
                except Exception:
                    pass

            tokens = _split_loose(text)
            if first_data_line and _is_probable_header(tokens):
                header_map = {_norm_col_name(t): i for i, t in enumerate(tokens)}
                first_data_line = False
                continue
            first_data_line = False

            rec = _parse_venue_tokens(tokens, header_map, venue_id_col, lat_col, lon_col, fallback_id=f"line_{line_no}")
            if rec is None:
                continue
            vid, lat, lon = rec
            if vid in venues:
                duplicate += 1
            venues[vid] = (lat, lon)
            parsed += 1

    if not venues:
        raise ValueError(
            f"No venue coordinates parsed from {path}. Try setting "
            "--foursquare-venue-id-col, --foursquare-venue-lat-col and --foursquare-venue-lon-col."
        )
    meta = {
        "foursquare_venue_rows_scanned": float(scanned),
        "foursquare_venue_rows_parsed": float(parsed),
        "foursquare_venue_duplicates": float(duplicate),
        "foursquare_unique_venues": float(len(venues)),
    }
    return venues, meta


def _parse_checkin_venue_id(
    tokens: Sequence[str],
    header_map: Optional[Dict[str, int]],
    venue_col: int,
    valid_ids: set,
) -> Optional[str]:
    if not tokens:
        return None
    if venue_col >= 0 and venue_col < len(tokens):
        vid = _clean_id(tokens[venue_col])
        return vid if vid in valid_ids else None
    if header_map is not None:
        i = _find_col(header_map, _CHECKIN_VENUE_KEYS)
        if i is not None and i < len(tokens):
            vid = _clean_id(tokens[i])
            return vid if vid in valid_ids else None

    # Common check-in layout is user_id, venue_id, timestamp,... .
    for idx in [1, 0, 2, 3]:
        if idx < len(tokens):
            vid = _clean_id(tokens[idx])
            if vid in valid_ids:
                return vid
    # Last fallback: scan all tokens and use the first one that matches a venue id.
    for tok in tokens:
        vid = _clean_id(tok)
        if vid in valid_ids:
            return vid
    return None


def read_foursquare_checkin_counts(
    path: str,
    valid_ids: set,
    venue_col: int = -1,
    scan_limit: int = 0,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Read check-in rows and count visits per venue ID."""
    counts: Dict[str, int] = {}
    header_map: Optional[Dict[str, int]] = None
    scanned = 0
    matched = 0

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        first_data_line = True
        for line in f:
            if scan_limit and scanned >= scan_limit:
                break
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            scanned += 1
            if text.startswith("{"):
                try:
                    obj = json.loads(text)
                    vid = None
                    if isinstance(obj, dict):
                        # Try common direct and nested keys.
                        flat: Dict[str, object] = {}
                        def visit(prefix: str, val: object) -> None:
                            if isinstance(val, dict):
                                for kk, vv in val.items():
                                    visit(f"{prefix}.{kk}" if prefix else str(kk), vv)
                            else:
                                flat[_norm_col_name(prefix)] = val
                        visit("", obj)
                        for key in _CHECKIN_VENUE_KEYS:
                            if key in flat:
                                candidate = _clean_id(flat[key])
                                if candidate in valid_ids:
                                    vid = candidate
                                    break
                    if vid is not None:
                        counts[vid] = counts.get(vid, 0) + 1
                        matched += 1
                    continue
                except Exception:
                    pass

            tokens = _split_loose(text)
            if first_data_line and _is_probable_header(tokens):
                header_map = {_norm_col_name(t): i for i, t in enumerate(tokens)}
                first_data_line = False
                continue
            first_data_line = False

            vid = _parse_checkin_venue_id(tokens, header_map, venue_col, valid_ids)
            if vid is None:
                continue
            counts[vid] = counts.get(vid, 0) + 1
            matched += 1

    meta = {
        "foursquare_checkin_rows_scanned": float(scanned),
        "foursquare_checkin_rows_matched": float(matched),
        "foursquare_checkin_weighted_venues": float(len(counts)),
    }
    return counts, meta


def _weighted_latlon_sample(
    latlon: np.ndarray,
    weights: Optional[np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(latlon) == 0:
        raise ValueError("empty lat/lon pool")
    replace = len(latlon) < n
    if weights is not None and weights.sum() > 0:
        prob = weights.astype(float) / float(weights.sum())
        idx = rng.choice(len(latlon), size=n, replace=True, p=prob)
    else:
        idx = rng.choice(len(latlon), size=n, replace=replace)
    return latlon[idx].astype(float)


def load_foursquare_xy_points(args: argparse.Namespace, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """Load Foursquare venue/check-in data and return a km-coordinate point pool."""
    venue_file = _find_dataset_file(args.foursquare_root, args.foursquare_venue_file, ["venues", "venue"])
    checkin_file = ""
    try:
        checkin_file = _find_dataset_file(args.foursquare_root, args.foursquare_checkin_file, ["checkins", "checkin"])
    except FileNotFoundError as exc:
        if not args.disable_checkin_weighting:
            print(f"Warning: {exc}; falling back to uniform venue sampling.")

    print(f"Foursquare venue file:  {venue_file}")
    if checkin_file:
        print(f"Foursquare checkin file: {checkin_file}")

    venues, meta = read_foursquare_venues(
        venue_file,
        venue_id_col=args.foursquare_venue_id_col,
        lat_col=args.foursquare_venue_lat_col,
        lon_col=args.foursquare_venue_lon_col,
    )
    all_ids = set(venues.keys())
    counts: Dict[str, int] = {}
    if checkin_file and not args.disable_checkin_weighting:
        counts, count_meta = read_foursquare_checkin_counts(
            checkin_file,
            valid_ids=all_ids,
            venue_col=args.foursquare_checkin_venue_col,
            scan_limit=args.foursquare_scan_limit,
        )
        meta.update(count_meta)
        if not counts:
            print("Warning: check-in file was found, but no check-in venue IDs matched parsed venue IDs; using uniform venue sampling.")
    else:
        meta["foursquare_checkin_rows_scanned"] = 0.0
        meta["foursquare_checkin_rows_matched"] = 0.0
        meta["foursquare_checkin_weighted_venues"] = 0.0

    user_bbox = parse_bbox(args.foursquare_bbox)
    effective_bbox = user_bbox

    all_latlon = np.asarray(list(venues.values()), dtype=float)
    if effective_bbox is None and not args.disable_auto_bbox:
        # Prefer check-in weighted dense-region selection when counts are available.
        if counts:
            ids_with_counts = [vid for vid in venues.keys() if counts.get(vid, 0) > 0]
            latlon_auto = np.asarray([venues[vid] for vid in ids_with_counts], dtype=float)
            weights_auto = np.asarray([counts[vid] for vid in ids_with_counts], dtype=float)
            auto_n = min(max(1, int(args.foursquare_auto_sample)), int(args.foursquare_max_points))
            auto_sample = _weighted_latlon_sample(latlon_auto, weights_auto, auto_n, rng)
        else:
            auto_n = min(max(1, int(args.foursquare_auto_sample)), len(all_latlon))
            auto_sample = _weighted_latlon_sample(all_latlon, None, auto_n, rng)
        effective_bbox = auto_dense_bbox(
            auto_sample,
            cell_deg=args.foursquare_auto_cell_deg,
            window_deg=args.foursquare_auto_window_deg,
        )
        print(
            "Auto bbox selected: "
            f"min_lat={effective_bbox[0]:.6f}, max_lat={effective_bbox[1]:.6f}, "
            f"min_lon={effective_bbox[2]:.6f}, max_lon={effective_bbox[3]:.6f}"
        )

    filtered_ids: List[str] = []
    filtered_latlon_list: List[Tuple[float, float]] = []
    for vid, (lat, lon) in venues.items():
        if in_bbox(lat, lon, effective_bbox):
            filtered_ids.append(vid)
            filtered_latlon_list.append((lat, lon))

    if not filtered_latlon_list:
        raise ValueError(
            "No Foursquare venues remain after bbox filtering. "
            "Try --disable-auto-bbox or a different --foursquare-bbox."
        )

    min_count = max(1, int(args.foursquare_min_checkin_count))
    if counts:
        selected = [(vid, venues[vid]) for vid in filtered_ids if counts.get(vid, 0) >= min_count]
        if not selected:
            print(
                "Warning: no venue in the selected bbox has enough check-ins; "
                "falling back to uniform sampling over filtered venues."
            )
            selected_ids = filtered_ids
            selected_latlon = np.asarray(filtered_latlon_list, dtype=float)
            weights = None
        else:
            selected_ids = [x[0] for x in selected]
            selected_latlon = np.asarray([x[1] for x in selected], dtype=float)
            weights = np.asarray([counts[vid] for vid in selected_ids], dtype=float)
    else:
        selected_ids = filtered_ids
        selected_latlon = np.asarray(filtered_latlon_list, dtype=float)
        weights = None

    pool_n = int(max(args.foursquare_max_points, args.num_workers + args.num_tasks))
    pool_latlon = _weighted_latlon_sample(selected_latlon, weights, pool_n, rng)
    xy, xy_meta = latlon_to_xy_km(pool_latlon)
    meta.update(xy_meta)
    meta["data_source"] = "foursquare"
    meta["foursquare_venue_file"] = venue_file
    meta["foursquare_checkin_file"] = checkin_file if checkin_file else "none"
    meta["foursquare_filtered_venues"] = float(len(filtered_ids))
    meta["foursquare_pool_points"] = float(len(pool_latlon))
    meta["foursquare_sampling"] = "checkin_weighted" if weights is not None else "uniform_venue"
    meta["distance_unit"] = "km"
    if effective_bbox is not None:
        meta["bbox_min_lat"] = effective_bbox[0]
        meta["bbox_max_lat"] = effective_bbox[1]
        meta["bbox_min_lon"] = effective_bbox[2]
        meta["bbox_max_lon"] = effective_bbox[3]

    print(
        f"Loaded Foursquare points: venues={int(meta['foursquare_unique_venues'])}, "
        f"filtered_venues={len(filtered_ids)}, sampled_pool={len(pool_latlon)}, "
        f"sampling={meta['foursquare_sampling']}"
    )
    print(f"Projected local area: width={meta['width_km']:.3f} km, height={meta['height_km']:.3f} km")
    return xy.astype(float), meta

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
    mode: str = "corridor",
) -> np.ndarray:
    """Generate a synthetic spatial point pool.

    Modes:
        uniform:
            Uniform points over the whole rectangle. This is the least structured
            synthetic baseline.
        mixture:
            Four Gaussian-like urban hotspots. This was used in the previous
            synthetic experiment.
        corridor:
            Hub-and-corridor mobility pattern. Most points are sampled along
            several narrow corridors connecting activity hubs, with additional
            points around the hubs and a small uniform background component.
            This creates a different synthetic dataset from the Gaussian mixture:
            spatial density is concentrated on route-like structures rather than
            isotropic clusters.
    """
    if mode == "uniform":
        return rng.uniform([0.0, 0.0], [width, height], size=(n, 2)).astype(float)

    if mode == "mixture":
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

    if mode != "corridor":
        raise ValueError("mode must be 'corridor', 'mixture', or 'uniform'")

    # A route-like synthetic city: five activity hubs connected by corridors.
    # Units are km in the same rectangular coordinate system used by the rest
    # of the experiment.
    hubs = np.asarray(
        [
            [0.18 * width, 0.22 * height],  # southwest residential hub
            [0.82 * width, 0.25 * height],  # southeast business hub
            [0.75 * width, 0.78 * height],  # northeast hub
            [0.26 * width, 0.72 * height],  # northwest hub
            [0.52 * width, 0.50 * height],  # central transfer hub
        ],
        dtype=float,
    )
    edges = np.asarray(
        [
            [0, 4],
            [1, 4],
            [2, 4],
            [3, 4],
            [0, 3],
            [1, 2],
            [0, 1],
        ],
        dtype=int,
    )
    edge_probs = np.asarray([0.18, 0.18, 0.16, 0.16, 0.10, 0.10, 0.12], dtype=float)
    edge_probs = edge_probs / edge_probs.sum()

    n_corridor = int(round(0.70 * n))
    n_hub = int(round(0.25 * n))
    n_background = max(0, n - n_corridor - n_hub)

    # Corridor component: sample along line segments, then add small
    # perpendicular/isotropic Gaussian jitter to avoid perfectly thin lines.
    edge_choices = rng.choice(len(edges), size=n_corridor, p=edge_probs)
    t = rng.uniform(0.0, 1.0, size=(n_corridor, 1))
    start = hubs[edges[edge_choices, 0]]
    end = hubs[edges[edge_choices, 1]]
    corridor_pts = (1.0 - t) * start + t * end
    corridor_sigma = 0.020 * min(width, height)
    corridor_pts = corridor_pts + rng.normal(0.0, corridor_sigma, size=(n_corridor, 2))

    # Hub component: dense activity around hubs.
    hub_probs = np.asarray([0.22, 0.22, 0.18, 0.18, 0.20], dtype=float)
    hub_probs = hub_probs / hub_probs.sum()
    hub_choices = rng.choice(len(hubs), size=n_hub, p=hub_probs)
    hub_sigma = np.asarray([0.045 * width, 0.045 * height])
    hub_pts = hubs[hub_choices] + rng.normal(0.0, hub_sigma, size=(n_hub, 2))

    # A small background component prevents the task assignment problem from
    # becoming too degenerate and mimics occasional off-route check-ins.
    bg_pts = rng.uniform([0.0, 0.0], [width, height], size=(n_background, 2))

    pts = np.vstack([corridor_pts, hub_pts, bg_pts])
    pts = pts[rng.permutation(len(pts))]
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
            raise ValueError("For HSTGreedy diagnostics, grid_size must be a power of two, e.g., 8, 16, 32, 64")
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

    This matcher uses continuous true coordinates and does not discretize
    locations to grid centers before matching.
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


def adaptive_srr_schedule(epsilon: float, switch_epsilon: float = 8.0) -> str:
    """Probability schedule for SRR-SelfFirst-Adaptive.

    Empirical ablation results showed that the linear schedule is more stable
    around moderate privacy budgets, while the two-level schedule can still be
    useful at very high privacy budgets. Therefore the adaptive variant uses:
        epsilon <= switch_epsilon: LinearSchedule
        epsilon >  switch_epsilon: TwoLevelSchedule
    """
    return "linear" if float(epsilon) <= float(switch_epsilon) else "two_level"


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def build_mechanism(
    name: str,
    domain: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    hst: CompleteQuadtreeHST,
    srr_precompute_max_domain: int,
    grid_size: int,
    local_srr_radii: Sequence[int],
    srr_probability_schedule: str,
    self_first_high_groups: int,
    adaptive_switch_epsilon: float = 8.0,
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
    if name == "SRR-SelfFirst-Adaptive":
        precompute = len(domain) <= srr_precompute_max_domain
        adaptive_schedule = adaptive_srr_schedule(epsilon, adaptive_switch_epsilon)
        adaptive_high_groups = self_first_high_groups if adaptive_schedule == "two_level" else 1
        return LocalSRRMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=rng,
            radii=local_srr_radii,
            probability_schedule=adaptive_schedule,
            precompute_orders=precompute,
            self_first=True,
            two_level_high_groups=adaptive_high_groups,
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
    srr_precompute_max_domain: int,
    grid_size: int,
    local_srr_radii: Sequence[int],
    srr_probability_schedule: str,
    self_first_high_groups: int,
    adaptive_switch_epsilon: float,
    include_hst_greedy: bool,
    include_srr_local: bool,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    # The main comparison among privacy mechanisms is inside the LDP family:
    # GRR, HR, OLH-H, PLDP/OUE, and SRR-SelfFirst. SRR-Local is optional ablation only.
    base_mechanisms = [
        ("GRR", "GRR"),
        ("HR", "HR"),
        ("OLH-H", "OLH-H"),
        ("PLDP", "PLDP"),
        ("SRR-SelfFirst", "SRR-SelfFirst"),
        ("SRR-SelfFirst-Adaptive", "SRR-SelfFirst-Adaptive"),
    ]
    if include_srr_local:
        # SRR-Local is useful as an ablation variant of SRR-SelfFirst.
        # It is excluded from main figures by default because
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
                hst,
                srr_precompute_max_domain,
                grid_size,
                local_srr_radii,
                srr_probability_schedule,
                self_first_high_groups,
                adaptive_switch_epsilon,
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
    rows = filter_excluded_result_methods(rows)
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


EXCLUDED_RESULT_METHODS = {"TrueLocation-Greedy", "Grid-NoPrivacy-Greedy"}


def filter_excluded_result_methods(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Remove no-privacy baseline rows from all exported results."""
    return [r for r in rows if str(r.get("method", "")) not in EXCLUDED_RESULT_METHODS]


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    rows = filter_excluded_result_methods(rows)
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


def _format_txt_value(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def write_series_txt(path: str, series: Sequence[Sequence[float]]) -> None:
    """Write plot data as plain text: one curve per line, comma-separated values."""
    with open(path, "w", encoding="utf-8") as f:
        for values in series:
            f.write(",".join(_format_txt_value(v) for v in values))
            f.write("\n")


def _ablation_plot_method_order(
    rows: List[Dict[str, float]],
    variants: Sequence[str],
    include_reference: bool = True,
) -> List[str]:
    rows = filter_excluded_result_methods(rows)
    methods: List[str] = []
    methods.extend([v for v in variants if any(str(r.get("method")) == v for r in rows)])
    if include_reference:
        for m in ["GRR-Greedy", "OLH-H-Greedy", "PLDP-Greedy"]:
            if any(str(r.get("method")) == m for r in rows):
                methods.append(m)
    return methods


def write_ablation_avg_distance_txt(
    path: str,
    rows: List[Dict[str, float]],
    variants: Sequence[str],
    include_reference: bool = True,
) -> None:
    rows = filter_excluded_result_methods(rows)
    series: List[List[float]] = []
    for method in _ablation_plot_method_order(rows, variants, include_reference=include_reference):
        sub = [r for r in rows if str(r.get("method")) == method]
        sub = sorted(sub, key=lambda r: float(r["epsilon"]))
        series.append([float(r["avg_true_distance_km_mean"]) for r in sub])
    write_series_txt(path, series)


def write_ablation_perturbation_distance_txt(path: str, rows: List[Dict[str, float]]) -> None:
    rows = filter_excluded_result_methods(rows)
    series: List[List[float]] = []
    for method in sorted({str(r.get("method")) for r in rows}):
        sub = [r for r in rows if str(r.get("method")) == method]
        sub = sorted(sub, key=lambda r: float(r["epsilon"]))
        series.append([mean_perturbation_from_summary_row(r) for r in sub])
    write_series_txt(path, series)


def write_ablation_gap_vs_full_txt(
    path: str,
    contribution_rows: List[Dict[str, float]],
    variants: Sequence[str],
) -> None:
    series: List[List[float]] = []
    for variant in variants:
        if variant == "Full-SRR-SelfFirst":
            continue
        sub = [
            r for r in contribution_rows
            if str(r.get("compared_variant", r.get("removed_component_variant"))) == variant
        ]
        sub = sorted(sub, key=lambda r: float(r["epsilon"]))
        if sub:
            series.append([float(r["avg_distance_gap_vs_full_km"]) for r in sub])
    write_series_txt(path, series)


def plot_results(path: str, rows: List[Dict[str, float]]) -> None:
    rows = filter_excluded_result_methods(rows)
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
    rows = filter_excluded_result_methods(rows)
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
    "SRR-SelfFirst-Greedy",
    "SRR-SelfFirst-Adaptive-Greedy",
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
    "SRR-SelfFirst-Greedy",
    "SRR-SelfFirst-Adaptive-Greedy",
]

PERTURBATION_CALIBRATION_METHODS = [
    "GRR-Greedy",
    "HR-Greedy",
    "OLH-H-Greedy",
    "PLDP-Greedy",
    "SRR-SelfFirst-Greedy",
    "SRR-SelfFirst-Adaptive-Greedy",
]

GRID_SENSITIVITY_METHODS = [
    "GRR-Greedy",
    "OLH-H-Greedy",
    "PLDP-Greedy",
    "SRR-SelfFirst-Greedy",
    "SRR-SelfFirst-Adaptive-Greedy",
]

WORKLOAD_SENSITIVITY_METHODS = [
    "GRR-Greedy",
    "HR-Greedy",
    "OLH-H-Greedy",
    "PLDP-Greedy",
    "SRR-SelfFirst-Greedy",
    "SRR-SelfFirst-Adaptive-Greedy",
]


def filter_summary_rows(rows: List[Dict[str, float]], methods: Sequence[str]) -> List[Dict[str, float]]:
    rows = filter_excluded_result_methods(rows)
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
            "--width", str(args.width),
            "--height", str(args.height),
            "--synthetic-pool-size", str(args.synthetic_pool_size),
            "--data-mode", args.data_mode,
            "--foursquare-root", args.foursquare_root,
            "--foursquare-venue-file", args.foursquare_venue_file,
            "--foursquare-checkin-file", args.foursquare_checkin_file,
            "--foursquare-bbox", args.foursquare_bbox,
            "--foursquare-max-points", str(args.foursquare_max_points),
            "--foursquare-auto-sample", str(args.foursquare_auto_sample),
            "--foursquare-scan-limit", str(args.foursquare_scan_limit),
            "--foursquare-auto-cell-deg", str(args.foursquare_auto_cell_deg),
            "--foursquare-auto-window-deg", str(args.foursquare_auto_window_deg),
            "--foursquare-min-checkin-count", str(args.foursquare_min_checkin_count),
            "--foursquare-venue-id-col", str(args.foursquare_venue_id_col),
            "--foursquare-venue-lat-col", str(args.foursquare_venue_lat_col),
            "--foursquare-venue-lon-col", str(args.foursquare_venue_lon_col),
            "--foursquare-checkin-venue-col", str(args.foursquare_checkin_venue_col),
            "--grid-size", str(g),
            "--num-workers", str(args.num_workers),
            "--num-tasks", str(args.num_tasks),
            "--epsilons", epsilons,
            "--repeats", str(repeats),
            "--local-srr-radii", args.local_srr_radii,
            "--srr-probability-schedule", args.srr_probability_schedule,
            "--self-first-high-groups", str(args.self_first_high_groups),
            "--adaptive-switch-epsilon", str(args.adaptive_switch_epsilon),
            "--srr-precompute-max-domain", str(args.srr_precompute_max_domain),
            "--seed", str(args.seed),
            "--out-dir", child_out,
        ]
        if args.disable_auto_bbox:
            cmd += ["--disable-auto-bbox"]
        if args.disable_checkin_weighting:
            cmd += ["--disable-checkin-weighting"]
        if args.include_hst_greedy:
            cmd += ["--include-hst-greedy"]
        if args.include_srr_local:
            cmd += ["--include-srr-local"]
        print("\n[grid sensitivity] running:", " ".join(cmd))
        child_env = os.environ.copy()
        child_env["SRR_SKIP_CODE_WORKLOAD"] = "1"
        subprocess.run(cmd, check=True, env=child_env)

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
            "--width", str(args.width),
            "--height", str(args.height),
            "--synthetic-pool-size", str(args.synthetic_pool_size),
            "--data-mode", args.data_mode,
            "--foursquare-root", args.foursquare_root,
            "--foursquare-venue-file", args.foursquare_venue_file,
            "--foursquare-checkin-file", args.foursquare_checkin_file,
            "--foursquare-bbox", args.foursquare_bbox,
            "--foursquare-max-points", str(args.foursquare_max_points),
            "--foursquare-auto-sample", str(args.foursquare_auto_sample),
            "--foursquare-scan-limit", str(args.foursquare_scan_limit),
            "--foursquare-auto-cell-deg", str(args.foursquare_auto_cell_deg),
            "--foursquare-auto-window-deg", str(args.foursquare_auto_window_deg),
            "--foursquare-min-checkin-count", str(args.foursquare_min_checkin_count),
            "--foursquare-venue-id-col", str(args.foursquare_venue_id_col),
            "--foursquare-venue-lat-col", str(args.foursquare_venue_lat_col),
            "--foursquare-venue-lon-col", str(args.foursquare_venue_lon_col),
            "--foursquare-checkin-venue-col", str(args.foursquare_checkin_venue_col),
            "--grid-size", str(args.grid_size),
            "--num-workers", str(num_workers),
            "--num-tasks", str(num_tasks),
            "--epsilons", f"{eps:g}",
            "--repeats", str(repeats),
            "--local-srr-radii", args.local_srr_radii,
            "--srr-probability-schedule", args.srr_probability_schedule,
            "--self-first-high-groups", str(args.self_first_high_groups),
            "--adaptive-switch-epsilon", str(args.adaptive_switch_epsilon),
            "--srr-precompute-max-domain", str(args.srr_precompute_max_domain),
            "--seed", str(args.seed),
            "--out-dir", child_out,
            "--disable-perturbation-calibration",
        ]
        if args.disable_auto_bbox:
            cmd += ["--disable-auto-bbox"]
        if args.disable_checkin_weighting:
            cmd += ["--disable-checkin-weighting"]
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


# ---------------------------------------------------------------------------
# Ablation experiment
# ---------------------------------------------------------------------------

ABLATION_VARIANT_ORDER = [
    "Full-SRR-SelfFirst",
    "SRR-SelfFirst-Adaptive",
    "NoSelfFirst",
    "SelfCellOnly",
    "NoTwoLevel",
    "LinearSchedule",
]

ABLATION_VARIANT_DESCRIPTIONS = {
    "Full-SRR-SelfFirst": "Complete SRR-SelfFirst: true-cell-first grouping, local rings, two-level schedule, and the configured high groups.",
    "SRR-SelfFirst-Adaptive": "Keep self-first grouping and adapt the probability schedule: LinearSchedule when epsilon <= adaptive_switch_epsilon, TwoLevelSchedule when epsilon > adaptive_switch_epsilon.",
    "NoSelfFirst": "Remove the self-first design: the nearest 1-hop neighborhood is used as the first high-probability group.",
    "SelfCellOnly": "Keep self-first grouping but remove the 1-hop ring from the highest probability set.",
    "NoTwoLevel": "Keep self-first grouping but replace the two-level schedule with an exponential staircase schedule.",
    "LinearSchedule": "Keep self-first grouping but replace the two-level schedule with a linear staircase schedule.",
}

ABLATION_REFERENCE_METHODS = [
    "GRR-Greedy",
    "OLH-H-Greedy",
    "PLDP-Greedy",
]


def parse_ablation_variants(s: str) -> List[str]:
    """Parse and validate the SRR ablation variant list."""
    aliases = {
        "full": "Full-SRR-SelfFirst",
        "full-srr-selffirst": "Full-SRR-SelfFirst",
        "srr-selffirst": "Full-SRR-SelfFirst",
        "adaptive": "SRR-SelfFirst-Adaptive",
        "adaptive-schedule": "SRR-SelfFirst-Adaptive",
        "adaptive_schedule": "SRR-SelfFirst-Adaptive",
        "srr-selffirst-adaptive": "SRR-SelfFirst-Adaptive",
        "srr_self_first_adaptive": "SRR-SelfFirst-Adaptive",
        "srrselffirstadaptive": "SRR-SelfFirst-Adaptive",
        "noselffirst": "NoSelfFirst",
        "no-self-first": "NoSelfFirst",
        "no_self_first": "NoSelfFirst",
        "selfcellonly": "SelfCellOnly",
        "self-cell-only": "SelfCellOnly",
        "self_cell_only": "SelfCellOnly",
        "notwolevel": "NoTwoLevel",
        "no-two-level": "NoTwoLevel",
        "no_two_level": "NoTwoLevel",
        "linearschedule": "LinearSchedule",
        "linear-schedule": "LinearSchedule",
        "linear_schedule": "LinearSchedule",
    }
    out: List[str] = []
    for raw in s.split(","):
        token = raw.strip()
        if not token:
            continue
        key = token.lower().replace(" ", "")
        value = aliases.get(key, token)
        if value not in ABLATION_VARIANT_ORDER:
            raise ValueError(
                f"Unknown ablation variant '{token}'. Available variants: "
                + ",".join(ABLATION_VARIANT_ORDER)
            )
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("ablation variant list is empty")
    # Keep a stable paper-friendly order even if the user supplies a shuffled list.
    order = {name: i for i, name in enumerate(ABLATION_VARIANT_ORDER)}
    return sorted(out, key=lambda name: order[name])


def build_ablation_mechanism(
    variant: str,
    domain: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    srr_precompute_max_domain: int,
    grid_size: int,
    local_srr_radii: Sequence[int],
    full_probability_schedule: str,
    full_self_first_high_groups: int,
    adaptive_switch_epsilon: float = 8.0,
):
    """Build one SRR-family ablation mechanism.

    The variants are designed to isolate one design choice at a time while
    keeping the same finite-domain LDP constraint. All variants use the same
    domain, grid, epsilon, worker/task samples, and online greedy matcher.
    """
    precompute = len(domain) <= srr_precompute_max_domain

    if variant == "Full-SRR-SelfFirst":
        return LocalSRRMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=rng,
            radii=local_srr_radii,
            probability_schedule=full_probability_schedule,
            precompute_orders=precompute,
            self_first=True,
            two_level_high_groups=full_self_first_high_groups,
        )

    if variant == "SRR-SelfFirst-Adaptive":
        adaptive_schedule = adaptive_srr_schedule(epsilon, adaptive_switch_epsilon)
        adaptive_high_groups = full_self_first_high_groups if adaptive_schedule == "two_level" else 1
        return LocalSRRMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=rng,
            radii=local_srr_radii,
            probability_schedule=adaptive_schedule,
            precompute_orders=precompute,
            self_first=True,
            two_level_high_groups=adaptive_high_groups,
        )

    if variant == "NoSelfFirst":
        return LocalSRRMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=rng,
            radii=local_srr_radii,
            probability_schedule=full_probability_schedule,
            precompute_orders=precompute,
            self_first=False,
            two_level_high_groups=1,
        )

    if variant == "SelfCellOnly":
        return LocalSRRMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=rng,
            radii=local_srr_radii,
            probability_schedule="two_level",
            precompute_orders=precompute,
            self_first=True,
            two_level_high_groups=1,
        )

    if variant == "NoTwoLevel":
        return LocalSRRMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=rng,
            radii=local_srr_radii,
            probability_schedule="exponential",
            precompute_orders=precompute,
            self_first=True,
            two_level_high_groups=1,
        )

    if variant == "LinearSchedule":
        return LocalSRRMechanism(
            domain=domain,
            grid_size=grid_size,
            epsilon=epsilon,
            rng=rng,
            radii=local_srr_radii,
            probability_schedule="linear",
            precompute_orders=precompute,
            self_first=True,
            two_level_high_groups=1,
        )


    raise ValueError(f"Unknown ablation variant: {variant}")


def run_one_setting_ablation(
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
    srr_precompute_max_domain: int,
    grid_size: int,
    local_srr_radii: Sequence[int],
    srr_probability_schedule: str,
    self_first_high_groups: int,
    adaptive_switch_epsilon: float,
    ablation_variants: Sequence[str],
    include_reference_baselines: bool,
) -> List[Dict[str, float]]:
    """Run one epsilon setting for the SRR ablation suite."""
    rows: List[Dict[str, float]] = []

    if include_reference_baselines:
        reference_specs = [
            ("GRR-Greedy", "GRR", "euclidean"),
            ("OLH-H-Greedy", "OLH-H", "euclidean"),
            ("PLDP-Greedy", "PLDP", "euclidean"),

        ]
        for method_label, mech_name, matcher in reference_specs:
            t0 = time.time()
            mech_rng = np.random.default_rng(stable_seed(seed, epsilon, "ABLATION_REF_" + mech_name))
            mech = build_mechanism(
                mech_name,
                domain,
                epsilon,
                mech_rng,
                hst,
                srr_precompute_max_domain,
                grid_size,
                local_srr_radii,
                srr_probability_schedule,
                self_first_high_groups,
                adaptive_switch_epsilon,
            )
            w_noisy = mech.perturb_indices(workers_idx)
            t_noisy = mech.perturb_indices(tasks_idx)
            w_pdist = perturbation_distance(workers_idx, w_noisy, domain)
            t_pdist = perturbation_distance(tasks_idx, t_noisy, domain)
            dist_matrix = euclid_dist_matrix if matcher == "euclidean" else hst_dist_matrix
            pairs = online_greedy_match(w_noisy, t_noisy, dist_matrix)
            total_dist = evaluate_true_distance(pairs, workers_true, tasks_true)
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "method": method_label,
                    "total_true_distance_km": total_dist,
                    "avg_true_distance_km": total_dist / len(tasks_true),
                    "worker_perturbation_distance_km": w_pdist,
                    "task_perturbation_distance_km": t_pdist,
                    "runtime_sec": time.time() - t0,
                    "ablation_family": "reference",
                }
            )

    for variant in ablation_variants:
        t0 = time.time()
        mech_rng = np.random.default_rng(stable_seed(seed, epsilon, "ABLATION_" + variant))
        mech = build_ablation_mechanism(
            variant=variant,
            domain=domain,
            epsilon=epsilon,
            rng=mech_rng,
            srr_precompute_max_domain=srr_precompute_max_domain,
            grid_size=grid_size,
            local_srr_radii=local_srr_radii,
            full_probability_schedule=srr_probability_schedule,
            full_self_first_high_groups=self_first_high_groups,
            adaptive_switch_epsilon=adaptive_switch_epsilon,
        )
        w_noisy = mech.perturb_indices(workers_idx)
        t_noisy = mech.perturb_indices(tasks_idx)
        w_pdist = perturbation_distance(workers_idx, w_noisy, domain)
        t_pdist = perturbation_distance(tasks_idx, t_noisy, domain)
        pairs = online_greedy_match(w_noisy, t_noisy, euclid_dist_matrix)
        total_dist = evaluate_true_distance(pairs, workers_true, tasks_true)
        rows.append(
            {
                "epsilon": float(epsilon),
                "method": variant,
                "total_true_distance_km": total_dist,
                "avg_true_distance_km": total_dist / len(tasks_true),
                "worker_perturbation_distance_km": w_pdist,
                "task_perturbation_distance_km": t_pdist,
                "runtime_sec": time.time() - t0,
                "ablation_family": "srr_ablation",
            }
        )

    return rows


def make_ablation_contribution_rows(
    summary_rows: List[Dict[str, float]],
    ablation_variants: Sequence[str],
    full_method: str = "Full-SRR-SelfFirst",
) -> List[Dict[str, float]]:
    """Compute each variant's distance gap compared with Full-SRR-SelfFirst.

    Positive avg_distance_gap_vs_full means the variant is worse than Full.
    Negative means the variant outperforms Full under that epsilon/sample.
    """
    full_by_eps: Dict[float, Dict[str, float]] = {
        float(r["epsilon"]): r for r in summary_rows if str(r.get("method")) == full_method
    }
    out: List[Dict[str, float]] = []
    for variant in ablation_variants:
        if variant == full_method:
            continue
        for r in summary_rows:
            if str(r.get("method")) != variant:
                continue
            eps = float(r["epsilon"])
            full = full_by_eps.get(eps)
            if full is None:
                continue
            full_avg = float(full["avg_true_distance_km_mean"])
            variant_avg = float(r["avg_true_distance_km_mean"])
            gap = variant_avg - full_avg
            pct = 100.0 * gap / max(full_avg, 1e-12)
            out.append(
                {
                    "epsilon": eps,
                    "compared_variant": variant,
                    "removed_component_variant": variant,  # kept for backward-compatible CSV readers
                    "full_method": full_method,
                    "full_avg_true_distance_km": full_avg,
                    "variant_avg_true_distance_km": variant_avg,
                    "avg_distance_gap_vs_full_km": gap,
                    "avg_distance_gap_vs_full_percent": pct,
                    "full_mean_perturbation_km": mean_perturbation_from_summary_row(full),
                    "variant_mean_perturbation_km": mean_perturbation_from_summary_row(r),
                    "interpretation": "positive_gap_means_variant_worse_than_full_negative_gap_means_variant_better",
                }
            )
    return sorted(out, key=lambda x: (float(x["epsilon"]), str(x.get("compared_variant", x.get("removed_component_variant", "")))))


def plot_ablation_gap_vs_full(
    path: str,
    contribution_rows: List[Dict[str, float]],
    variants: Sequence[str],
) -> None:
    """Plot each variant's distance gap compared with Full."""
    if plt is None or not contribution_rows:
        return
    plt.figure(figsize=(10, 6))
    for variant in variants:
        if variant == "Full-SRR-SelfFirst":
            continue
        sub = [
            r for r in contribution_rows
            if str(r.get("compared_variant", r.get("removed_component_variant"))) == variant
        ]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: float(r["epsilon"]))
        x = [float(r["epsilon"]) for r in sub]
        y = [float(r["avg_distance_gap_vs_full_km"]) for r in sub]
        plt.plot(x, y, marker="o", label=variant)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("epsilon")
    plt.ylabel("Average distance gap vs Full (km)")
    plt.title("SRR-SelfFirst ablation: distance gap compared with Full")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_ablation_results(
    path: str,
    rows: List[Dict[str, float]],
    variants: Sequence[str],
    include_reference: bool = True,
) -> None:
    """Paper-oriented ablation plot with variants in a stable order."""
    if plt is None or not rows:
        return
    rows = filter_excluded_result_methods(rows)
    methods = _ablation_plot_method_order(rows, variants, include_reference=include_reference)

    plt.figure(figsize=(10, 6))
    for method in methods:
        sub = [r for r in rows if str(r.get("method")) == method]
        sub = sorted(sub, key=lambda r: float(r["epsilon"]))
        x = [float(r["epsilon"]) for r in sub]
        y = [float(r["avg_true_distance_km_mean"]) for r in sub]
        plt.plot(x, y, marker="o", label=method)
    plt.xlabel("epsilon")
    plt.ylabel("Average true travel distance (km)")
    plt.title("SRR-SelfFirst ablation on online task assignment")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_ablation_experiment(
    args: argparse.Namespace,
    pool_points: np.ndarray,
    data_meta: Dict[str, float],
    domain: np.ndarray,
    hst: CompleteQuadtreeHST,
    euclid_dist_matrix: np.ndarray,
    hst_dist_matrix: np.ndarray,
    local_srr_radii: Sequence[int],
    epsilons: Sequence[float],
    distance_unit: str,
) -> None:
    """Run SRR-SelfFirst ablation under the same data/configuration."""
    ablation_variants = parse_ablation_variants(args.ablation_variants)
    if "Full-SRR-SelfFirst" not in ablation_variants:
        ablation_variants = ["Full-SRR-SelfFirst"] + list(ablation_variants)

    os.makedirs(args.out_dir, exist_ok=True)
    print("\nRunning SRR-SelfFirst ablation experiment ...")
    print("Ablation variants:")
    for v in ablation_variants:
        print(f"  - {v}: {ABLATION_VARIANT_DESCRIPTIONS[v]}")

    all_rows: List[Dict[str, float]] = []
    for rep in range(args.repeats):
        rep_seed = args.seed + rep * 1009
        rng = np.random.default_rng(rep_seed)
        workers_true, tasks_true = sample_workers_tasks_from_pool(pool_points, args.num_workers, args.num_tasks, rng)
        workers_idx = map_points_to_domain(workers_true, domain)
        tasks_idx = map_points_to_domain(tasks_true, domain)

        print(f"Ablation repeat {rep + 1}/{args.repeats}: workers={len(workers_true)}, tasks={len(tasks_true)}")
        for eps in epsilons:
            print(f"  epsilon={eps}")
            rows = run_one_setting_ablation(
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
                srr_precompute_max_domain=args.srr_precompute_max_domain,
                grid_size=args.grid_size,
                local_srr_radii=local_srr_radii,
                srr_probability_schedule=args.srr_probability_schedule,
                self_first_high_groups=args.self_first_high_groups,
                adaptive_switch_epsilon=args.adaptive_switch_epsilon,
                ablation_variants=ablation_variants,
                include_reference_baselines=args.ablation_include_reference_baselines,
            )
            for r in rows:
                r["repeat_id"] = rep
                r["distance_unit"] = distance_unit
                r["data_source"] = data_meta.get("data_source", "unknown")
            all_rows.extend(rows)

    raw_path = os.path.join(args.out_dir, "ablation_raw_results.csv")
    summary_path = os.path.join(args.out_dir, "ablation_summary_results.csv")
    contribution_path = os.path.join(args.out_dir, "ablation_contribution_vs_full.csv")
    fig_path = os.path.join(args.out_dir, "ablation_avg_distance_vs_epsilon.png")
    perturb_fig_path = os.path.join(args.out_dir, "ablation_perturbation_distance_vs_epsilon.png")
    gap_fig_path = os.path.join(args.out_dir, "ablation_gap_vs_full.png")
    avg_txt_path = os.path.join(args.out_dir, "ablation_avg_distance_vs_epsilon.txt")
    perturb_txt_path = os.path.join(args.out_dir, "ablation_perturbation_distance_vs_epsilon.txt")
    gap_txt_path = os.path.join(args.out_dir, "ablation_gap_vs_full.txt")
    variant_desc_path = os.path.join(args.out_dir, "ablation_variant_descriptions.csv")
    meta_path = os.path.join(args.out_dir, "ablation_metadata.txt")

    all_rows = filter_excluded_result_methods(all_rows)
    write_csv(raw_path, all_rows)
    summary_rows = aggregate_rows(all_rows)
    write_csv(summary_path, summary_rows)
    contribution_rows = make_ablation_contribution_rows(summary_rows, ablation_variants)
    write_csv(contribution_path, contribution_rows)
    plot_ablation_results(fig_path, summary_rows, ablation_variants, include_reference=True)
    plot_perturbation_results(perturb_fig_path, summary_rows)
    plot_ablation_gap_vs_full(gap_fig_path, contribution_rows, ablation_variants)
    write_ablation_avg_distance_txt(avg_txt_path, summary_rows, ablation_variants, include_reference=True)
    write_ablation_perturbation_distance_txt(perturb_txt_path, summary_rows)
    write_ablation_gap_vs_full_txt(gap_txt_path, contribution_rows, ablation_variants)

    desc_rows = [
        {"variant": v, "description": ABLATION_VARIANT_DESCRIPTIONS[v]}
        for v in ablation_variants
    ]
    write_csv(variant_desc_path, desc_rows)

    meta = dict(data_meta)
    meta["experiment_role"] = "srr_selffirst_ablation"
    meta["ablation_variants"] = ",".join(ablation_variants)
    meta["ablation_full_method"] = "Full-SRR-SelfFirst"
    meta["adaptive_switch_epsilon"] = float(args.adaptive_switch_epsilon)
    meta["adaptive_rule"] = "SRR-SelfFirst-Adaptive uses LinearSchedule when epsilon <= adaptive_switch_epsilon, otherwise TwoLevelSchedule"
    meta["ablation_goal"] = "decompose utility contribution of self-first grouping, high one-hop group, two-level schedule, and equal-size grouping"
    write_metadata(meta_path, meta, args)

    print("\nAblation done.")
    print(f"Ablation raw results:      {raw_path}")
    print(f"Ablation summary results:  {summary_path}")
    print(f"Ablation contribution CSV: {contribution_path}")
    print(f"Ablation metadata:         {meta_path}")
    if plt is not None:
        print(f"Ablation plot:             {fig_path}")
        print(f"Ablation perturb plot:     {perturb_fig_path}")
        print(f"Ablation gap plot:         {gap_fig_path}")
        print(f"Ablation avg txt:          {avg_txt_path}")
        print(f"Ablation perturb txt:      {perturb_txt_path}")
        print(f"Ablation gap txt:          {gap_txt_path}")

    print("\nAblation summary: average true distance")
    for r in summary_rows:
        print(
            f"epsilon={r['epsilon']:<4} method={str(r['method']):<24} "
            f"avg={r['avg_true_distance_km_mean']:.4f} ± {r['avg_true_distance_km_std']:.4f} {distance_unit}"
        )


def write_recommended_formal_commands(path: str, script_name: str) -> None:
    """Write optional commands for robustness/sensitivity experiments."""
    commands = []
    base = (
        f'python {script_name} ^\n'
        '  --foursquare-root "D:\\data2" ^\n'
        '  --foursquare-max-points 400000 ^\n'
    )
    commands.append("REM Main formal Foursquare experiment")
    commands.append(base +
        '  --grid-size 64 ^\n'
        '  --num-workers 2000 ^\n'
        '  --num-tasks 1000 ^\n'
        '  --epsilons 0.1,0.5,1.0,2.0,4.0,6.0,8.0,10.0 ^\n'
        '  --repeats 10 ^\n'
        '  --out-dir foursquare_formal_main_g64_w2000_t1000_r10')
    commands.append("\nREM Grid-size sensitivity: run grid 8, 16, and 32")
    commands.append(base +
        '  --run-grid-sensitivity ^\n'
        '  --grid-sensitivity-sizes 8,16,32 ^\n'
        '  --grid-sensitivity-epsilons 2.0,4.0,6.0,8.0 ^\n'
        '  --grid-sensitivity-repeats 5 ^\n'
        '  --out-dir foursquare_grid_sensitivity')
    commands.append("\nREM Main experiment plus workload sensitivity")
    commands.append(base +
        '  --grid-size 64 ^\n'
        '  --out-dir foursquare_formal_with_workload')
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

def benchmark_client_perturbation(
    mechanism_func,
    locations,
    eps,
    repeat=5,
    warmup=1000
):
    """
    测量客户端单个位置的平均扰动耗时。

    mechanism_func: 扰动函数，例如 srr_selffirst_perturb
    locations: 输入位置列表或网格编号列表
    eps: 隐私预算
    repeat: 重复实验次数
    warmup: 预热次数，不计入最终时间
    """

    # 预热，避免首次运行带来的额外开销
    for i in range(min(warmup, len(locations))):
        mechanism_func(locations[i], eps)

    avg_times = []

    for _ in range(repeat):
        start = time.perf_counter_ns()

        for loc in locations:
            mechanism_func(loc, eps)

        end = time.perf_counter_ns()

        total_time_ns = end - start
        avg_time_us = total_time_ns / len(locations) / 1000.0
        avg_times.append(avg_time_us)

    return {
        "mean_us_per_location": np.mean(avg_times),
        "std_us_per_location": np.std(avg_times),
        "min_us_per_location": np.min(avg_times),
        "max_us_per_location": np.max(avg_times)
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GRR / HR / PLDP / SRR variants with OLH-H baseline for online task assignment on Foursquare user dataset")

    # Foursquare data source. The five .7z files should already be extracted under this folder.
    parser.add_argument("--foursquare-root", type=str, default=r"D:\data2", help="Folder containing extracted foursquare-user-dataset-master files, default D:\\data2")
    parser.add_argument("--foursquare-venue-file", type=str, default="", help="Optional explicit venues file path. If empty, search recursively under --foursquare-root.")
    parser.add_argument("--foursquare-checkin-file", type=str, default="", help="Optional explicit checkins file path. If empty, search recursively under --foursquare-root.")
    parser.add_argument("--foursquare-bbox", type=str, default="", help="Optional bbox min_lat,max_lat,min_lon,max_lon. If empty, select a dense local region automatically.")
    parser.add_argument("--disable-auto-bbox", action="store_true", help="Use all parsed Foursquare locations instead of selecting a dense local region automatically.")
    parser.add_argument("--foursquare-max-points", type=int, default=400000, help="Number of candidate locations sampled into the worker/task pool.")
    parser.add_argument("--foursquare-auto-sample", type=int, default=100000, help="Number of points used to choose the automatic dense bbox.")
    parser.add_argument("--foursquare-scan-limit", type=int, default=0, help="Maximum check-in lines to scan; 0 means no limit.")
    parser.add_argument("--foursquare-auto-cell-deg", type=float, default=0.25, help="Coarse cell size in degrees for automatic dense-region selection.")
    parser.add_argument("--foursquare-auto-window-deg", type=float, default=0.8, help="Window size in degrees for automatic dense-region selection.")
    parser.add_argument("--disable-checkin-weighting", action="store_true", help="Sample venues uniformly instead of using check-in counts as weights.")
    parser.add_argument("--foursquare-min-checkin-count", type=int, default=1, help="Minimum check-in count for a venue to be used when check-in weighting succeeds.")
    parser.add_argument("--foursquare-venue-id-col", type=int, default=-1, help="Venue ID column in the venues file; -1 means auto-detect.")
    parser.add_argument("--foursquare-venue-lat-col", type=int, default=-1, help="Venue latitude column in the venues file; -1 means auto-detect.")
    parser.add_argument("--foursquare-venue-lon-col", type=int, default=-1, help="Venue longitude column in the venues file; -1 means auto-detect.")
    parser.add_argument("--foursquare-checkin-venue-col", type=int, default=-1, help="Venue ID column in the checkins file; -1 means auto-detect.")
    # Kept only for compatibility with old generated commands; Foursquare runs ignore these synthetic options.
    parser.add_argument("--width", type=float, default=40.0, help=argparse.SUPPRESS)
    parser.add_argument("--height", type=float, default=40.0, help=argparse.SUPPRESS)
    parser.add_argument("--synthetic-pool-size", type=int, default=400000, help=argparse.SUPPRESS)
    parser.add_argument("--data-mode", type=str, default="corridor", choices=["corridor", "mixture", "uniform"], help=argparse.SUPPRESS)

    # Experiment parameters.
    parser.add_argument("--grid-size", type=int, default=32, help="Grid side length; must be power of two for HST-based diagnostics. Default 64 matches the latest formal Geolife/Gowalla configuration.")
    parser.add_argument("--num-workers", type=int, default=2000, help="Formal main experiment default: 2000")
    parser.add_argument("--num-tasks", type=int, default=1000, help="Formal main experiment default: 1000")
    parser.add_argument("--epsilons", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0", help="Formal main epsilon list")
    parser.add_argument("--local-srr-radii", type=str, default="1,2,3,4", help="Neighborhood radii for SRR-Local, e.g. 1,2,3,4 creates groups <=1, <=2, <=3, <=4, and rest")
    parser.add_argument("--srr-probability-schedule", type=str, default="two_level", choices=["linear", "exponential", "two_level"], help="Per-location staircase shape for SRR-Local and SRR-SelfFirst. Default two_level is more aggressive for task assignment.")
    parser.add_argument("--self-first-high-groups", type=int, default=2, help="For SRR-SelfFirst with two_level schedule, number of nearest groups using the highest per-location probability. 2 means true cell + 1-hop ring.")
    parser.add_argument("--adaptive-switch-epsilon", type=float, default=8.0, help="For SRR-SelfFirst-Adaptive: use LinearSchedule when epsilon <= this value, otherwise TwoLevelSchedule. Default 8.0.")
    parser.add_argument("--srr-precompute-max-domain", type=int, default=4096, help="Precompute SRR distance orders only if |D| <= this value")
    parser.add_argument("--include-hst-greedy", action="store_true", help="Also output HSTGreedy curves for GRR/HR/OLH-H/PLDP/SRR. Off by default to keep figures compact.")
    parser.add_argument("--include-srr-local", action="store_true", help="Include SRR-Local as an ablation/appendix method. Off by default because it nearly overlaps SRR-SelfFirst in the main setting.")
    parser.add_argument("--repeats", type=int, default=10, help="Formal main experiment default: 10 repeats")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="foursquare_ablation")
    parser.add_argument("--write-formal-commands", action="store_true", help="Also write a Windows .bat file containing recommended formal/sensitivity runs into out-dir")

    # Grid-size sensitivity mode. This runs this same script as child processes
    # for several grid sizes, then combines their summaries.
    parser.add_argument("--run-grid-sensitivity", action="store_true", help="Run grid-size sensitivity suite and combine results, instead of a single-grid run")
    parser.add_argument("--grid-sensitivity-sizes", type=str, default="8,16,32", help="Grid sizes used by --run-grid-sensitivity")
    parser.add_argument("--grid-sensitivity-epsilons", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0", help="Epsilons used by --run-grid-sensitivity")
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

    # Ablation mode. This runs an SRR-SelfFirst ablation suite under the same
    # Foursquare data, grid, worker/task, epsilon, and repeat configuration.
    parser.add_argument("--run-ablation", action="store_true", default=IN_CODE_RUN_ABLATION_ONLY, help="Run SRR-SelfFirst ablation instead of the normal main experiment.")
    parser.add_argument("--also-run-ablation", action="store_true", default=IN_CODE_RUN_ABLATION_AFTER_MAIN, help="After the normal main experiment, also run SRR-SelfFirst ablation in <out-dir>/ablation.")
    parser.add_argument("--ablation-variants", type=str, default=DEFAULT_ABLATION_VARIANTS, help="Comma-separated ablation variants. Available: Full-SRR-SelfFirst, SRR-SelfFirst-Adaptive, NoSelfFirst, SelfCellOnly, NoTwoLevel, LinearSchedule.")
    parser.add_argument("--ablation-include-reference-baselines", action="store_true", help="Also include GRR, PLDP and OLH-H baseline curves in the ablation result files.")

    args = parser.parse_args()

    if args.num_workers < args.num_tasks:
        raise ValueError("num-workers must be >= num-tasks")
    if args.grid_size <= 1 or (args.grid_size & (args.grid_size - 1)) != 0:
        raise ValueError("--grid-size must be a power of two, e.g., 8, 16, 32, 64")
    if args.grid_size * args.grid_size > 8192:
        raise ValueError("Domain too large for this simple script. Use grid-size <= 64, preferably 16 or 32.")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.run_grid_sensitivity:
        run_grid_sensitivity_via_subprocess(args)
        return

    if args.run_workload_sensitivity:
        run_workload_sensitivity_via_subprocess(args)
        return

    epsilons = parse_epsilons(args.epsilons)
    local_srr_radii = parse_int_list(args.local_srr_radii)
    base_rng = set_seed(args.seed)

    # Load the real Foursquare point pool. The point pool is sampled from venues,
    # preferably weighted by check-in counts, and projected to a local kilometer plane.
    pool_points, data_meta = load_foursquare_xy_points(args, base_rng)
    width = float(data_meta.get("width_km", max(float(pool_points[:, 0].max()), 1e-9)))
    height = float(data_meta.get("height_km", max(float(pool_points[:, 1].max()), 1e-9)))
    distance_unit = "km"
    data_meta["experiment_role"] = "formal_foursquare_main_or_formal_override"
    print(f"Foursquare point pool ready: n={len(pool_points)}, area={width:.3f} km x {height:.3f} km")

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
    data_meta["adaptive_switch_epsilon"] = float(args.adaptive_switch_epsilon)
    data_meta["adaptive_rule"] = "SRR-SelfFirst-Adaptive uses LinearSchedule when epsilon <= adaptive_switch_epsilon, otherwise TwoLevelSchedule"
    data_meta["experiment_role"] = "formal_main_or_formal_override"
    data_meta["paper_positioning"] = "adaptation study: task-oriented SRR variants for online assignment under LDP"
    data_meta["olhh_role"] = "OLH-H is used as the L-SRR baseline comparison method."
    data_meta["in_code_run_workload_after_main"] = bool(IN_CODE_RUN_WORKLOAD_AFTER_MAIN)
    data_meta["in_code_run_workload_only"] = bool(IN_CODE_RUN_WORKLOAD_ONLY)
    data_meta["workload_child_process_guard"] = bool(_IN_WORKLOAD_CHILD_PROCESS)

    print(f"Domain size: {len(domain)} ({args.grid_size} x {args.grid_size})")
    print(f"Distance unit: {distance_unit}")
    print("Precomputing distance matrices ...")
    euclid_dist_matrix = make_euclidean_distance_matrix(domain)
    hst_dist_matrix = hst.tree_distance_matrix()

    if args.run_ablation:
        run_ablation_experiment(
            args=args,
            pool_points=pool_points,
            data_meta=data_meta,
            domain=domain,
            hst=hst,
            euclid_dist_matrix=euclid_dist_matrix,
            hst_dist_matrix=hst_dist_matrix,
            local_srr_radii=local_srr_radii,
            epsilons=epsilons,
            distance_unit=distance_unit,
        )
        return

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
                srr_precompute_max_domain=args.srr_precompute_max_domain,
                grid_size=args.grid_size,
                local_srr_radii=local_srr_radii,
                srr_probability_schedule=args.srr_probability_schedule,
                self_first_high_groups=args.self_first_high_groups,
                adaptive_switch_epsilon=args.adaptive_switch_epsilon,
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

    ablation_after_main_dir = None
    if args.also_run_ablation:
        print("\nRunning SRR-SelfFirst ablation after main experiment ...")
        ablation_args = argparse.Namespace(**vars(args))
        ablation_args.out_dir = os.path.join(args.out_dir, "ablation")
        ablation_args.run_ablation = True
        ablation_args.also_run_ablation = False
        ablation_args.run_grid_sensitivity = False
        ablation_args.run_workload_sensitivity = False
        ablation_args.also_run_workload_sensitivity = False
        run_ablation_experiment(
            args=ablation_args,
            pool_points=pool_points,
            data_meta=data_meta,
            domain=domain,
            hst=hst,
            euclid_dist_matrix=euclid_dist_matrix,
            hst_dist_matrix=hst_dist_matrix,
            local_srr_radii=local_srr_radii,
            epsilons=epsilons,
            distance_unit=distance_unit,
        )
        ablation_after_main_dir = ablation_args.out_dir

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
    if workload_after_main_dir is not None:
        print(f"Workload dir:    {workload_after_main_dir}")
        print(f"Workers plot:    {os.path.join(workload_after_main_dir, 'total_distance_vs_workers.png')}")
        print(f"Tasks plot:      {os.path.join(workload_after_main_dir, 'total_distance_vs_tasks.png')}")
    if ablation_after_main_dir is not None:
        print(f"Ablation dir:    {ablation_after_main_dir}")
        print(f"Ablation plot:   {os.path.join(ablation_after_main_dir, 'ablation_avg_distance_vs_epsilon.png')}")
        print(f"Ablation gap:    {os.path.join(ablation_after_main_dir, 'ablation_gap_vs_full.png')}")

    print("\nSummary: average true distance")
    for r in summary_rows:
        print(
            f"epsilon={r['epsilon']:<4} method={str(r['method']):<18} "
            f"avg={r['avg_true_distance_km_mean']:.4f} ± {r['avg_true_distance_km_std']:.4f} {distance_unit}"
        )


if __name__ == "__main__":
    main()
