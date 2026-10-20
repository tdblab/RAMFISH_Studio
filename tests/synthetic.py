"""Synthetic multi-round RAMFISH dataset with known ground truth.

Used for the in-app demo and the test-suite: every round shows the same
tissue, but rounds 2+ are rotated / shifted / flipped / gently warped, the
way a re-mounted sample would be.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np



DEMO_LAYOUT = {
    1: ["DAPI", "wg", "ci"],
    2: ["DAPI", "optix", "spalt"],
    3: ["DAPI", "dll"],
}

# Rough-aligner settings (native pixels, scale_ratio 1) that bring each round back onto Round 1.
DEMO_POSES = {
    2: {"rotation": -24.0, "img_scale": 1.0, "tx": -22.0, "ty": 15.0, "flip": False, "scale_ratio": 1.0},
    3: {"rotation": 35.0, "img_scale": 1.0, "tx": 18.0, "ty": -12.0, "flip": True, "scale_ratio": 1.0},
}


def _pose_matrix(p, shape):
    """The matrix ramfish_rough_aligner applies after its optional flip."""
    h, w = shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -p["rotation"], p["img_scale"])
    m[0, 2] += p["tx"] * p["scale_ratio"]
    m[1, 2] += p["ty"] * p["scale_ratio"]
    return m


@dataclass
class DemoData:
    images: dict[int, dict[str, np.ndarray]]      # round -> channel -> uint16
    truth_spots: dict[str, np.ndarray]             # gene -> (N, 2) xy in Round 1 frame
    poses: dict[int, dict]                         # rough-aligner settings that undo each round
    tissue_mask: np.ndarray


def _tissue(shape, rng):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.52
    # Asymmetric "wing" so that flips and rotations are identifiable.
    r = ((xx - cx) / (w * 0.36)) ** 2 + ((yy - cy) / (h * 0.27)) ** 2
    lobe = ((xx - w * 0.68) / (w * 0.16)) ** 2 + ((yy - h * 0.28) / (h * 0.12)) ** 2
    return (r < 1.0) | (lobe < 1.0)


def _render_points(shape, pts, sigma, amps):
    img = np.zeros(shape, np.float32)
    h, w = shape
    for (x, y), a in zip(pts, amps):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            img[yi, xi] += a
    k = int(6 * sigma) | 1
    img = cv2.GaussianBlur(img, (k, k), sigma) * (2 * np.pi * sigma * sigma)
    return img


def _region(name, xx, yy, w, h):
    if name == "wg":
        return np.abs(yy - h * 0.52) < h * 0.05
    if name == "ci":
        return xx < w * 0.45
    if name == "optix":
        return xx > w * 0.6
    if name == "spalt":
        return np.abs(xx - w * 0.5) < w * 0.09
    if name == "dll":
        return ((xx - w * 0.68) ** 2 + (yy - h * 0.28) ** 2) < (w * 0.1) ** 2
    raise KeyError(name)


def _smooth_warp(img, rng, amplitude):
    """Gentle non-rigid deformation (what the B-spline stage must undo)."""
    h, w = img.shape
    coarse = rng.normal(0, amplitude, size=(2, 4, 4)).astype(np.float32)
    dx = cv2.resize(coarse[0], (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(coarse[1], (w, h), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderValue=0)


def make_demo(shape=(640, 640), seed: int = 7, warp_amplitude: float = 2.0, n_spots: int = 220,
              with_mask: bool = False) -> DemoData:
    """Synthetic dataset; ``with_mask`` adds a noisy tissue-mask channel ``mask`` to every round."""
    rng = np.random.default_rng(seed)
    h, w = shape
    mask = _tissue(shape, rng)
    yy, xx = np.mgrid[0:h, 0:w]

    # Nuclei: jittered grid inside the tissue.
    step = 13
    gy, gx = np.mgrid[step // 2:h:step, step // 2:w:step]
    nuc = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    nuc += rng.uniform(-3.5, 3.5, nuc.shape)
    inside = mask[np.clip(nuc[:, 1].astype(int), 0, h - 1), np.clip(nuc[:, 0].astype(int), 0, w - 1)]
    nuc = nuc[inside]
    dapi = _render_points(shape, nuc, 3.2, rng.uniform(0.5, 1.0, len(nuc)))
    dapi = dapi / max(dapi.max(), 1e-6) * 0.75
    glow = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 25) * 0.08

    canonical: dict[str, np.ndarray] = {"DAPI": dapi + glow}
    truth: dict[str, np.ndarray] = {}
    genes = [g for chans in DEMO_LAYOUT.values() for g in chans if g != "DAPI"]
    for g in genes:
        region = _region(g, xx, yy, w, h) & mask
        cand = np.argwhere(region)
        pick = cand[rng.choice(len(cand), size=min(n_spots, len(cand)), replace=False)]
        pts = pick[:, ::-1].astype(np.float32) + rng.uniform(-0.4, 0.4, (len(pick), 2))
        # Drop spots that would merge with a neighbour (keeps ground truth unambiguous).
        keep = []
        for i, p in enumerate(pts):
            if all(np.hypot(*(p - pts[j])) > 6 for j in keep):
                keep.append(i)
        pts = pts[keep]
        truth[g] = pts
        spots = _render_points(shape, pts, 1.6, rng.uniform(0.6, 1.0, len(pts)))
        canonical[g] = spots + glow * 1.5

    if with_mask:
        # Filled silhouette with a few holes and debris specks, like a thresholded scan.
        m = mask.astype(np.float32)
        for _ in range(6):
            cy, cx = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 4, 3 * w // 4)
            cv2.circle(m, (int(cx), int(cy)), int(rng.integers(3, 8)), 0.0, -1)
        for _ in range(60):
            cy, cx = rng.integers(0, h), rng.integers(0, w)
            cv2.circle(m, (int(cx), int(cy)), int(rng.integers(1, 4)), 1.0, -1)
        canonical["mask"] = m * 0.9

    layout = {r: (c + ["mask"] if with_mask else c) for r, c in DEMO_LAYOUT.items()}
    images: dict[int, dict[str, np.ndarray]] = {}
    for rnd, chans in layout.items():
        images[rnd] = {}
        warp_seed = rng.integers(1 << 31)
        for c in chans:
            img = canonical[c]
            if rnd in DEMO_POSES:
                img = _smooth_warp(img, np.random.default_rng(warp_seed), warp_amplitude)
                # Render the moving round so that the rough aligner's transform undoes it.
                m = _pose_matrix(DEMO_POSES[rnd], shape)
                img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
                if DEMO_POSES[rnd]["flip"]:
                    img = cv2.flip(img, 1)
            noisy = img + rng.normal(0, 0.012, img.shape).astype(np.float32) + 0.03
            images[rnd][c] = np.clip(noisy * 65535 * 0.35, 0, 65535).astype(np.uint16)
    return DemoData(images=images, truth_spots=truth, poses=DEMO_POSES, tissue_mask=mask)
