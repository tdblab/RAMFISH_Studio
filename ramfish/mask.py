"""Tissue masks: a black-and-white outline of the tissue, made by saturating one channel (or all channels).

For rounds without a nuclear stain, the outline of the tissue is what lines the rounds up. The steps follow
D-CIPHER's tissue mask: blur the image so faint tissue glow becomes one smooth area, threshold it low (Otsu's
threshold times a factor below 1 lights up the whole tissue), keep the largest piece, close small gaps in the
edge, fill holes, and optionally grow the outline a little.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, label

MASK_ALL = "all channels merged"

DEFAULT_MASK_SETTINGS = {"blur_px": 8.0, "threshold_factor": 0.5, "closing_px": 15, "dilation_px": 0}


def _dilate(mask: np.ndarray, r: float) -> np.ndarray:
    if r <= 0 or not mask.any():
        return mask.copy()
    return distance_transform_edt(~mask) <= r


def _erode(mask: np.ndarray, r: float) -> np.ndarray:
    if r <= 0 or mask.all():
        return mask.copy()
    return distance_transform_edt(mask) > r


def _otsu(values: np.ndarray) -> float:
    from skimage.filters import threshold_otsu
    try:
        return float(threshold_otsu(values))
    except ValueError:
        return float(np.percentile(values, 85))


def merge_for_mask(planes: list[np.ndarray]) -> np.ndarray:
    """All channels on one scale (each divided by its own 99.5th percentile), brightest value per pixel."""
    out = None
    for p in planes:
        p = np.asarray(p, dtype=np.float32)
        hi = float(np.percentile(p, 99.5))
        lo = float(np.percentile(p, 0.5))
        scaled = (p - lo) / (hi - lo) if hi > lo else np.zeros_like(p)
        out = scaled if out is None else np.maximum(out, scaled)
    return out


def tissue_mask(plane: np.ndarray, blur_px=8.0, threshold_factor=0.5, closing_px=15, dilation_px=0) -> np.ndarray:
    """Boolean tissue outline of one 2D image (raw intensities, before normalization)."""
    img = np.asarray(plane, dtype=np.float32)
    # Work on a reduced copy of large images: the outline does not need full resolution, and it is much faster.
    k = max(1, int(max(img.shape) // 1024))
    small = img[::k, ::k] if k > 1 else img
    blurred = gaussian_filter(small, blur_px / k) if blur_px > 0 else small
    # a few very bright specks (dust, debris) would pull the threshold far above the tissue: leave the brightest
    # 0.5 % of pixels out when choosing it
    thr = _otsu(np.minimum(blurred, np.percentile(blurred, 99.5))) * float(threshold_factor)
    binary = blurred > thr
    lab, n = label(binary)
    if n > 0:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        binary = lab == counts.argmax()
    if closing_px > 0:
        binary = _erode(_dilate(binary, closing_px / k), closing_px / k)
    binary = binary_fill_holes(binary)
    if dilation_px > 0:
        binary = _dilate(binary, dilation_px / k)
    if k > 1:
        import cv2
        binary = cv2.resize(binary.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return binary


def mask_image(mask: np.ndarray) -> np.ndarray:
    """A boolean mask as a 16-bit image (tissue = 65535), the way RAMFISH stores `mask` images."""
    return mask.astype(np.float32) * 65535.0
