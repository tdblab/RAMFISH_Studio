"""Step 2 processing: the preview and transform code of ramfish_rough_aligner.py, unchanged."""

from __future__ import annotations

import base64

import cv2
import numpy as np


def preview(img: np.ndarray) -> dict:
    """Normalize to 8-bit and send a compressed base64 JPEG to the UI (as /api/preview)."""
    h, w = img.shape[:2]

    # Convert to 8-bit for web display
    if img.dtype == np.uint16 or img.dtype == np.float32 or img.dtype == np.float64:
        img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        img_8u = img.copy()

    # Handle channels
    if img_8u.ndim == 3 and img_8u.shape[2] > 3:
        img_8u = img_8u[:, :, :3]  # Strip extra channels for UI preview
    elif img_8u.ndim == 2:
        img_8u = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)

    # Scale down for browser performance (Max 1000px dimension)
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_8u, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_8u

    _, buffer = cv2.imencode('.jpg', img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return {
        "src": f"data:image/jpeg;base64,{b64_str}",
        "orig_w": int(w),
        "orig_h": int(h),
        "prev_w": img_resized.shape[1],
        "prev_h": img_resized.shape[0],
    }


def apply(img: np.ndarray, data: dict) -> np.ndarray:
    """Apply the UI's affine transform to one image (as /api/align_folder)."""
    tx_ui = float(data.get('tx', 0))
    ty_ui = float(data.get('ty', 0))
    rot = float(data.get('rotation', 0))
    flip = bool(data.get('flip', False))
    img_scale = float(data.get('img_scale', 1.0))
    scale_ratio = float(data.get('scale_ratio', 1.0))

    # Convert UI pixels to Native Image pixels based on scale ratio
    real_tx = tx_ui * scale_ratio
    real_ty = ty_ui * scale_ratio

    h, w = img.shape[:2]

    if flip:
        img = cv2.flip(img, 1)

    # Negative rot to mathematically match CSS clockwise rotation
    # Replaced 1.0 with img_scale to apply the scale transformation
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -rot, img_scale)
    M[0, 2] += real_tx
    M[1, 2] += real_ty

    # Handle multi-channel TIFFs properly
    if img.ndim == 3 and img.shape[2] > 4:
        aligned = np.zeros_like(img)
        for c in range(img.shape[2]):
            aligned[:, :, c] = cv2.warpAffine(img[:, :, c], M, (w, h), flags=cv2.INTER_LINEAR)
    else:
        aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return aligned
