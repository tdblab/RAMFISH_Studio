"""Step 1 processing: the functions of ramfish_processor.py, unchanged."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter as cpu_gaussian_filter

# ==========================================
# HARDWARE ACCELERATION DETECTION
# ==========================================
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
    _ = cp.zeros((1,))
    HAS_GPU = True
except Exception:
    HAS_GPU = False

RES_MODES = ["Original Resolution", "2K (2048 x 2048)", "4K (4096 x 4096)", "Custom Scaling Factor"]
OUTPUT_FORMATS = [".tif", ".jpg", ".png", ".bmp"]


def apply_ramfish_normalization(plane_data, do_norm, blur_sigma, percentile_val, target_intensity):
    """Applies GPU/CPU background subtraction and percentile scaling."""
    if not do_norm:
        return plane_data.astype(np.float32)

    # Hardware routing
    xp = cp if HAS_GPU else np
    blur_filter = gpu_gaussian_filter if HAS_GPU else cpu_gaussian_filter

    img_data = xp.array(plane_data.astype(np.float32))

    # 1. Background Subtraction
    background = blur_filter(img_data, sigma=blur_sigma)
    img_bg_subtracted = img_data - background
    img_bg_subtracted[img_bg_subtracted < 0] = 0

    # 2. Percentile Peak Calculation
    p_target = float(xp.percentile(img_bg_subtracted, percentile_val))

    if p_target == 0:
        return plane_data.astype(np.float32)  # Fallback if blank

    # 3. Scaling
    scaling_factor = target_intensity / p_target
    img_normalized = img_bg_subtracted * scaling_factor

    if HAS_GPU:
        return cp.asnumpy(img_normalized)
    return img_normalized


def format_and_resize(plane_f32, target_format, res_mode, custom_scale):
    """Clips to bit-depth and resizes spatially."""
    # 1. Bit Depth Conversion
    if target_format.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Map 0-60000 to 0-255 safely
        final_img = np.clip((plane_f32 / 65535.0) * 255.0, 0, 255).astype(np.uint8)
    else:
        final_img = np.clip(plane_f32, 0, 65535).astype(np.uint16)

    # 2. Spatial Resizing
    h, w = final_img.shape[:2]
    new_w, new_h = w, h

    if res_mode == "2K (2048 x 2048)":
        new_w, new_h = 2048, 2048
    elif res_mode == "4K (4096 x 4096)":
        new_w, new_h = 4096, 4096
    elif res_mode == "Custom Scaling Factor":
        new_w = int(w * custom_scale)
        new_h = int(h * custom_scale)

    if (new_w, new_h) != (w, h):
        interpolation = cv2.INTER_LANCZOS4 if new_w > w else cv2.INTER_AREA
        final_img = cv2.resize(final_img, (new_w, new_h), interpolation=interpolation)

    return final_img
