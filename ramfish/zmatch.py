"""Finding the matching depth (Z plane) in every round of a thick 3D sample.

The alignment steps work on flat (2D) images. For thin samples (a wing) the maximum projection of the whole
stack is fine. For a thick sample (a zebrafish brain) the projection piles up cells from many depths, so each
round must instead use a thin slab around the SAME depth in the tissue. Round 1 sets that depth; every other
round is searched for it.

Before the rough alignment the rounds can still be shifted, turned or mirrored in X and Y, so planes cannot be
compared pixel by pixel. Instead every plane is described by numbers that do not change when the image is moved,
turned or mirrored: how much of the plane is tissue, how bright the tissue is (mean, spread, 10th / 50th / 90th
percentiles), and how much fine detail it has. Going down the stack these numbers form curves. The curves of a
round are slid along the curves of Round 1 (in micrometers, so different Z steps are fine, and also turned
upside down in case the sample was mounted the other way up), and the shift where they agree best gives the
matching depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

N_FEATURES = 8


def _small(plane: np.ndarray, max_dim: int = 512) -> np.ndarray:
    k = max(1, int(np.ceil(max(plane.shape) / max_dim)))
    return np.asarray(plane[::k, ::k], dtype=np.float32)


def plane_features(vol_zyx: np.ndarray) -> np.ndarray:
    """One row of position-independent numbers per plane (Z x N_FEATURES)."""
    vol = np.stack([_small(p) for p in vol_zyx])
    blurred = np.stack([gaussian_filter(p, 2.0) for p in vol])
    from skimage.filters import threshold_otsu
    sample = blurred[:, ::4, ::4].ravel()
    try:
        thr = 0.5 * float(threshold_otsu(sample))
    except ValueError:
        thr = float(np.percentile(sample, 85))
    rows = []
    for p, b in zip(vol, blurred):
        tissue = b > thr
        frac = float(tissue.mean())
        vals = p[tissue] if tissue.sum() > 50 else p.ravel()
        q10, q50, q90 = np.percentile(vals, [10, 50, 90])
        g = np.hypot(sobel(b, 0), sobel(b, 1))
        detail = float(np.mean(g[tissue])) if tissue.sum() > 50 else float(g.mean())
        lap = p - gaussian_filter(p, 1.5)
        sharp = float(np.std(lap[tissue])) if tissue.sum() > 50 else float(lap.std())
        rows.append([frac, float(vals.mean()), float(vals.std()), q10, q50, q90, detail, sharp])
    return np.asarray(rows, dtype=np.float64)


def _resample(features: np.ndarray, z_step_um: float, grid_um: float) -> np.ndarray:
    """Features on a regular depth grid (grid_um apart), starting at the first plane."""
    n = features.shape[0]
    depth = np.arange(n) * z_step_um
    grid = np.arange(0.0, depth[-1] + 1e-9, grid_um) if n > 1 else np.array([0.0])
    return np.stack([np.interp(grid, depth, features[:, j]) for j in range(features.shape[1])], axis=1)


def _standardize(f: np.ndarray) -> np.ndarray:
    sd = f.std(axis=0)
    sd[sd == 0] = 1.0
    return (f - f.mean(axis=0)) / sd


@dataclass
class DepthMatch:
    z_index: int          # matching plane in the moving stack
    score: float          # agreement of the curves (correlation, -1 to 1)
    upside_down: bool     # the moving stack runs the other way
    offset_um: float      # depth of the moving stack's first plane on Round 1's depth axis


def match_depth(ref_features: np.ndarray, ref_step_um: float, ref_z: int,
                mov_features: np.ndarray, mov_step_um: float, grid_um: float = 1.0,
                min_overlap: float = 0.5) -> DepthMatch:
    """Plane of the moving stack at the depth of plane ``ref_z`` of the reference stack."""
    ref_step_um = float(ref_step_um or 1.0)
    mov_step_um = float(mov_step_um or 1.0)
    if mov_features.shape[0] == 1:
        return DepthMatch(0, 0.0, False, 0.0)
    R = _standardize(_resample(ref_features, ref_step_um, grid_um))
    best = None
    for flip in (False, True):
        mf = mov_features[::-1] if flip else mov_features
        M = _standardize(_resample(mf, mov_step_um, grid_um))
        need = max(5, int(min_overlap * min(len(R), len(M))))
        for shift in range(-(len(M) - need), len(R) - need + 1):
            r0, m0 = max(0, shift), max(0, -shift)
            n = min(len(R) - r0, len(M) - m0)
            if n < need:
                continue
            a, b = R[r0:r0 + n], M[m0:m0 + n]
            a = a - a.mean(axis=0)
            b = b - b.mean(axis=0)
            den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
            ok = den > 0
            if not ok.any():
                continue
            score = float(np.mean((a * b).sum(axis=0)[ok] / den[ok]))
            # prefer larger overlaps a little, so a short accidental match does not win
            key = score * (0.9 + 0.1 * n / max(len(R), len(M)))
            if best is None or key > best[0]:
                best = (key, score, flip, shift)
    if best is None:
        return DepthMatch(int(mov_features.shape[0] // 2), 0.0, False, 0.0)
    _, score, flip, shift = best
    offset_um = shift * grid_um                      # depth (on Round 1's axis) of the moving stack's first plane
    depth_in_mov = ref_z * ref_step_um - offset_um   # micrometers from the moving stack's first plane
    z = int(round(depth_in_mov / mov_step_um))
    n_mov = mov_features.shape[0]
    z = int(np.clip(z, 0, n_mov - 1))
    if flip:
        z = n_mov - 1 - z
    return DepthMatch(z, score, flip, offset_um)


def sharpest_plane(features: np.ndarray) -> int:
    """Default reference depth for Round 1: the plane with the most fine detail inside the tissue."""
    return int(np.argmax(features[:, 7] * np.sqrt(np.maximum(features[:, 0], 1e-6))))


def slab_planes(n_z: int, z: int, z_step_um: float, thickness_um: float) -> tuple[int, int]:
    """First and last plane (inclusive) of a slab ``thickness_um`` thick centered on plane ``z``."""
    half = int(round(0.5 * float(thickness_um) / float(z_step_um or 1.0)))
    return max(0, z - half), min(n_z - 1, z + half)


def slab_projection(vol_zyx: np.ndarray, z0: int, z1: int) -> np.ndarray:
    return np.asarray(vol_zyx[z0:z1 + 1]).max(axis=0)
