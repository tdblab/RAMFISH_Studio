"""Rough alignment of 3D stacks: each round's volume is moved, turned, tilted and scaled onto Round 1's.

Everything is in micrometers, so rounds imaged with other plane spacings, pixel sizes or starting depths fit
together. The volumes are compared as tissue boundaries (from the mask made from the gene channels, or from DAPI),
which is light enough to redraw as the user moves the sliders.

Once every round is placed, each plane of Round 1 gets the matching plane of every other round, cut from that
round's stack (a real imaged plane when there is no tilt; a plane cut across the stack when tilted). Only the
depths that every round covers go on; the other planes are skipped (their files are kept).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .mask import MASK_ALL, mask_image, tissue_mask
from .normalize import apply_ramfish_normalization, format_and_resize, resize_plane
from .project import Project, is_mask
from . import sections as sec

KEYS = ("tx", "ty", "tz", "rot", "tilt_x", "tilt_y", "scale", "flip")
DEFAULT = {"tx": 0.0, "ty": 0.0, "tz": 0.0, "rot": 0.0, "tilt_x": 0.0, "tilt_y": 0.0, "scale": 1.0, "flip": False}
USE_MASK = "mask"
USE_DAPI = "dapi"
USE_LABELS = {USE_MASK: "Mask (tissue outline from the gene channels)", USE_DAPI: "DAPI (tissue outline from the nuclei)"}
SMALL_W = 256          # width of the small stacks used for the view and the first guess
KEEP_FRACTION = 0.9    # a depth is kept when this share of Round 1's tissue falls inside every round's stack


# ------------------------------------------------------------------------------------------ geometry
def geometry(project: Project, r: int) -> dict:
    """Size of a round's stack: planes, height and width (pixels), pixel size and plane spacing (µm)."""
    import tifffile
    chans = project.raw_channels(r)
    info = project.raw_info(r, chans[0])
    with tifffile.TiffFile(project.path("raw", r, chans[0])) as tf:
        shape = tf.series[0].shape
    n, h, w = (shape[-3] if len(shape) >= 3 else 1), shape[-2], shape[-1]
    px = (info.get("pixel_um") or [None, None, None])[-1] or 1.0
    return {"n": int(n), "h": int(h), "w": int(w), "px": float(px), "dz": float(sec.z_step(project, r))}


def _center(g: dict) -> np.ndarray:
    return np.array([g["w"] * g["px"] / 2, g["h"] * g["px"] / 2, (g["n"] - 1) * g["dz"] / 2], dtype=np.float64)


def rotation(p: dict) -> np.ndarray:
    """Turn around the vertical axis, then tilt front/back (around X) and left/right (around Y); degrees."""
    a, b, c = (math.radians(float(p.get(k, 0.0))) for k in ("rot", "tilt_x", "tilt_y"))
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    ry = np.array([[math.cos(c), 0, math.sin(c)], [0, 1, 0], [-math.sin(c), 0, math.cos(c)]])
    return rz @ rx @ ry


def _linear(p: dict) -> np.ndarray:
    """Round r (µm, around its center) -> Round 1 (µm, around its center): rotation · scale · mirror."""
    s = float(p.get("scale", 1.0)) or 1.0
    d = np.diag([(-s if p.get("flip") else s), s, 1.0])
    return rotation(p) @ d


def to_ref(pts: np.ndarray, g_r: dict, g_0: dict, p: dict) -> np.ndarray:
    """Points (N, 3: x, y, z in µm) of round r, placed in Round 1's space."""
    t = np.array([p.get("tx", 0.0), p.get("ty", 0.0), p.get("tz", 0.0)], dtype=np.float64)
    return (pts - _center(g_r)) @ _linear(p).T + _center(g_0) + t


def from_ref_affine(g_r: dict, g_0: dict, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """(A, b) with point_in_round_r = A @ point_in_round_1 + b (µm)."""
    t = np.array([p.get("tx", 0.0), p.get("ty", 0.0), p.get("tz", 0.0)], dtype=np.float64)
    A = np.linalg.inv(_linear(p))
    return A, _center(g_r) - A @ (_center(g_0) + t)


def is_flat(p: dict) -> bool:
    return abs(float(p.get("tilt_x", 0))) < 1e-6 and abs(float(p.get("tilt_y", 0))) < 1e-6


# ------------------------------------------------------------------------------------------ small boundary stacks
def small_volume(project: Project, r: int, use: str = USE_MASK, width: int = SMALL_W) -> tuple[np.ndarray, dict]:
    """The round's tissue boundary as a small stack (0/1 per pixel) and its geometry.

    ``use``: the mask made from the gene channels (like step 1's mask) or the outline of the DAPI signal."""
    vol, _, gs = small_stacks(project, r, use, width)
    return vol, gs


def small_stacks(project: Project, r: int, use: str = USE_MASK, width: int = SMALL_W):
    """(boundary stack, signal stack, geometry): the signal is the gene channels merged (or DAPI), 0 to 1."""
    g = geometry(project, r)
    k = width / float(g["w"])
    h = max(1, int(round(g["h"] * k)))
    chans = [c for c in project.raw_channels(r) if not is_mask(c)]
    dapi = [c for c in chans if c.lower() == "dapi"]
    pick = dapi if (use == USE_DAPI and dapi) else [c for c in chans if c.lower() != "dapi"] or chans
    merged = None
    for c in pick:
        v = project.load_raw(r, c)
        s = np.stack([cv2.resize(pl.astype(np.float32), (width, h), interpolation=cv2.INTER_AREA) for pl in v])
        lo, hi = np.percentile(s, [0.5, 99.5])
        s = np.clip((s - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(s)
        merged = s if merged is None else np.maximum(merged, s)
    ms = sec.main_settings(project)["mask"]
    ms = {"blur_px": ms.get("blur_px", 8.0) * k, "threshold_factor": ms.get("threshold_factor", 0.5),
          "closing_px": ms.get("closing_px", 15) * k, "dilation_px": ms.get("dilation_px", 0) * k}
    if use == USE_DAPI and dapi:      # nuclei are separate dots: blur more so the outline closes around them
        ms["blur_px"] = max(ms["blur_px"], 4.0)
        ms["closing_px"] = max(ms["closing_px"], 6.0)
    vol = np.stack([tissue_mask(pl, **ms) for pl in merged]).astype(np.float32)
    gs = dict(g, h=h, w=width, px=g["px"] / k)
    return vol, merged.astype(np.float32), gs


def outline_points(vol: np.ndarray, gs: dict, every: int = 1, max_pts: int = 400) -> list[np.ndarray]:
    """Boundary line of every plane as (N, 3) µm points (x, y, z in the round's own space), one array per plane."""
    out = []
    for z in range(0, vol.shape[0], every):
        cs, _ = cv2.findContours((vol[z] > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if len(c) < 8:
                continue
            c = c[:, 0, :].astype(np.float64)
            if len(c) > max_pts:
                c = c[:: int(math.ceil(len(c) / max_pts))]
            c = np.vstack([c, c[:1]])
            out.append(np.column_stack([c[:, 0] * gs["px"], c[:, 1] * gs["px"], np.full(len(c), z * gs["dz"])]))
    return out


# ------------------------------------------------------------------------------------------ cutting planes
def cut_plane(vol: np.ndarray, g_r: dict, g_0: dict, p: dict, z0_um: float, out_hw: tuple[int, int],
              out_px: float, nearest_plane: bool = True) -> np.ndarray | None:
    """Round r's picture at Round 1's depth ``z0_um`` on a grid of ``out_hw`` pixels of ``out_px`` µm.

    Without tilt this is one real plane of round r (the nearest), moved and turned; with tilt the plane is cut
    across the stack (filled in between planes). None when the depth is outside round r's stack."""
    A, b = from_ref_affine(g_r, g_0, p)
    H, W = out_hw
    n = vol.shape[0]
    if is_flat(p) and nearest_plane:
        zr = (A[2, 2] * z0_um + b[2]) / g_r["dz"]
        zi = int(round(zr))
        if zi < 0 or zi > n - 1 or abs(zr - zi) > 0.5 + 1e-6:
            return None
        # output pixel (j, i) -> input pixel (x, y)
        M = np.array([[A[0, 0] * out_px / g_r["px"], A[0, 1] * out_px / g_r["px"], (A[0, 2] * z0_um + b[0]) / g_r["px"]],
                      [A[1, 0] * out_px / g_r["px"], A[1, 1] * out_px / g_r["px"], (A[1, 2] * z0_um + b[1]) / g_r["px"]]])
        src = vol[zi].astype(np.float32)
        return cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = np.zeros((H, W), np.float32)
    jj = np.arange(W, dtype=np.float64) * out_px
    step = max(1, 4_000_000 // max(W, 1))
    any_inside = False
    for i0 in range(0, H, step):
        ii = np.arange(i0, min(H, i0 + step), dtype=np.float64) * out_px
        X, Y = np.meshgrid(jj, ii)
        P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z0_um)])
        Q = A @ P + b[:, None]
        zc = Q[2] / g_r["dz"]
        inside = (zc >= -0.5) & (zc <= n - 0.5)
        any_inside |= bool(inside.any())
        coords = np.stack([np.clip(zc, 0, n - 1), Q[1] / g_r["px"], Q[0] / g_r["px"]])
        vals = map_coordinates(vol, coords, order=1, mode="constant", cval=0.0, prefilter=False)
        vals[~inside] = 0
        out[i0:i0 + len(ii)] = vals.reshape(len(ii), W)
    return out if any_inside else None


def inside_fraction(ref_pts: np.ndarray, g_r: dict, g_0: dict, p: dict) -> np.ndarray:
    """For points of Round 1's tissue (N, 3 µm) at many depths: whether each falls within round r's stack."""
    A, b = from_ref_affine(g_r, g_0, p)
    zc = (ref_pts @ A.T + b)[:, 2] / g_r["dz"]
    return (zc >= -0.5) & (zc <= g_r["n"] - 0.5)


def skipped(project: Project) -> dict[int, set[int]]:
    """Planes the user left out, per round (counted from 0): {round: {plane, ...}}."""
    return {int(r): {int(z) for z in zs} for r, zs in project.meta.get("rough3d_skip", {}).items()}


def kept_planes(project: Project, params: dict, ref_small: tuple[np.ndarray, dict] | None = None) -> list[int]:
    """Planes of Round 1 covered by every other round (at least KEEP_FRACTION of the tissue), without the planes
    the user left out: a depth goes when Round 1's plane, or the matching plane of another round, was left out."""
    skip = skipped(project)
    rounds = sec.stack_rounds(project)
    r0 = rounds[0]
    g0 = geometry(project, r0)
    vol0, gs0 = ref_small if ref_small else small_volume(project, r0)
    mip = vol0.max(0) > 0.5
    ys, xs = np.nonzero(mip)
    if len(xs) == 0:
        ys, xs = np.nonzero(np.ones_like(mip))
    pick = np.linspace(0, len(xs) - 1, min(len(xs), 1500)).astype(int)
    xy = np.column_stack([xs[pick] * gs0["px"], ys[pick] * gs0["px"]])
    keep = []
    for z in range(g0["n"]):
        if z in skip.get(r0, set()):
            continue
        pts = np.column_stack([xy, np.full(len(xy), z * g0["dz"])])
        ok = True
        for r in rounds[1:]:
            p = params.get(str(r))
            if p is None:
                ok = False
                break
            if inside_fraction(pts, geometry(project, r), g0, p).mean() < KEEP_FRACTION:
                ok = False
                break
            if skip.get(r) and int(round(matching_plane(project, r, p, z * g0["dz"]))) in skip[r]:
                ok = False
                break
        if ok:
            keep.append(z)
    return keep


def matching_plane(project: Project, r: int, p: dict, z0_um: float) -> float:
    """Plane of round r (counted from 0, may be fractional with tilt) at Round 1's depth, at the frame center."""
    r0 = sec.stack_rounds(project)[0]
    g0, gr = geometry(project, r0), geometry(project, r)
    A, b = from_ref_affine(gr, g0, p)
    c = _center(g0)
    q = A @ np.array([c[0], c[1], z0_um]) + b
    return float(q[2] / gr["dz"])


# ------------------------------------------------------------------------------------------ first guess
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def _band(x: np.ndarray) -> np.ndarray:
    """Region-sized detail of a picture (no pixel noise, no overall glow), on one scale."""
    x = gaussian_filter(x, 1.0) - gaussian_filter(x, 8.0)
    sd = x.std()
    return (x / sd).astype(np.float32) if sd > 0 else x.astype(np.float32)


def _turn_and_shift(fixed: np.ndarray, moving: np.ndarray, px: float) -> dict:
    """Turn, shift and size that put ``moving`` onto ``fixed`` (both centered on canvases of the same size):
    mutual information from starting angles all around the circle, the best end judged on region-sized detail.
    Returned as this module's numbers (degrees, µm, scale)."""
    import SimpleITK as sitk
    f = sitk.GetImageFromArray(fixed.astype(np.float32))
    m = sitk.GetImageFromArray(moving.astype(np.float32))
    bf = _band(fixed)
    best = None
    for ang in np.radians(np.arange(-180, 180, 30)):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(50)
        reg.SetMetricSamplingStrategy(reg.NONE)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300)
        reg.SetOptimizerScalesFromPhysicalShift()
        init = sitk.Similarity2DTransform()
        init.SetCenter([fixed.shape[1] / 2.0, fixed.shape[0] / 2.0])
        init.SetAngle(float(ang))
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SetInitialTransform(init, inPlace=False)
        try:
            t = reg.Execute(f, m)
        except Exception:
            continue
        t = sitk.Similarity2DTransform(t.GetNthTransform(0) if hasattr(t, "GetNthTransform") else t)
        w = sitk.GetArrayFromImage(sitk.Resample(m, f, t, sitk.sitkLinear, 0.0))
        score = _corr(bf, _band(w))
        if 0.7 < t.GetScale() < 1.4 and (best is None or score > best[0]):
            best = (score, t)
    if best is None:
        return {"rot": 0.0, "tx": 0.0, "ty": 0.0, "scale": 1.0}
    t = best[1]
    th, sp = t.GetAngle(), t.GetScale()
    c = np.array(t.GetCenter())
    tt = np.array(t.GetTranslation())
    C = np.array([fixed.shape[1] / 2.0, fixed.shape[0] / 2.0])
    Rm = np.array([[math.cos(-th), -math.sin(-th)], [math.sin(-th), math.cos(-th)]])
    T = -(C - c) - (Rm @ (c + tt - C)) / sp            # canvas pixels (see the derivation in the docs)
    return {"rot": float(math.degrees(-th)), "tx": float(T[0] * px), "ty": float(T[1] * px), "scale": float(1.0 / sp)}


def auto_guess(project: Project, r: int, use: str = USE_MASK, small: dict | None = None) -> dict:
    """A first placement of round r: turn and shift from the tissue seen from above (its texture: gene channels or
    DAPI, which tells head from tail better than the outline), then the depth (up/down) where the plane outlines
    agree best. No tilt (set it by hand if the sample sat at an angle)."""
    rounds = sec.stack_rounds(project)
    r0 = rounds[0]
    small = small or {}
    v0, s0, gs0 = small.get(r0) or small_stacks(project, r0, use)
    vr, sr, gsr = small.get(r) or small_stacks(project, r, use)
    # 1) seen from above: turn + shift (on a common canvas, both centered)
    pxc = gs0["px"]
    size = int(max(gs0["w"], gs0["h"], gsr["w"] * gsr["px"] / pxc, gsr["h"] * gsr["px"] / pxc) * 1.3) + 2

    def canvas(mip, gs):
        k = gs["px"] / pxc
        m = cv2.resize(mip, (max(1, int(round(gs["w"] * k))), max(1, int(round(gs["h"] * k)))), interpolation=cv2.INTER_AREA)
        c = np.zeros((size, size), np.float32)
        y0, x0 = (size - m.shape[0]) // 2, (size - m.shape[1]) // 2
        c[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        return c

    a = canvas(s0.max(0), gs0) * gaussian_filter(canvas(v0.max(0), gs0), 2.0)
    bm = canvas(sr.max(0), gsr) * gaussian_filter(canvas(vr.max(0), gsr), 2.0)
    p = dict(DEFAULT, **_turn_and_shift(a, bm, pxc))
    # 2) depth: slide round r up and down; keep the shift where the plane outlines agree best
    g0, gr = gs0, gsr
    span0, spanr = (g0["n"] - 1) * g0["dz"], (gr["n"] - 1) * gr["dz"]
    need = max(2, int(0.3 * min(g0["n"], gr["n"])))
    skip = skipped(project)
    ref_planes = [gaussian_filter(v0[z], 1.5) for z in range(g0["n"])]
    best_tz, best_s = 0.0, -2.0
    stepz = min(g0["dz"], gr["dz"]) / 2
    for tz in np.arange(-(span0 + spanr) / 2, (span0 + spanr) / 2 + 1e-6, stepz):
        q = dict(p, tz=float(tz))
        scores = []
        A, b = from_ref_affine(gr, g0, q)
        for z in range(g0["n"]):
            if z in skip.get(r0, set()) or int(round((A[2, 2] * z * g0["dz"] + b[2]) / gr["dz"])) in skip.get(r, set()):
                continue
            cut = cut_plane(vr, gr, g0, q, z * g0["dz"], (g0["h"], g0["w"]), g0["px"])
            if cut is not None:
                scores.append(_corr(ref_planes[z], gaussian_filter(cut, 1.5)))
        if len(scores) >= need:
            s = float(np.mean(scores))
            if s > best_s:
                best_s, best_tz = s, float(tz)
    p["tz"] = round(best_tz, 2)
    p["tx"], p["ty"], p["rot"], p["scale"] = round(p["tx"], 1), round(p["ty"], 1), round(p["rot"], 1), round(p["scale"], 3)
    return p


# ------------------------------------------------------------------------------------------ the z-planes
def build_zplanes(project: Project, params: dict, export: bool = True, progress=None) -> list[Project]:
    """One small project per kept plane of Round 1, with the matching plane of every round cut from its stack,
    enlarged and normalized like step 1 (and a tissue mask per plane when the rounds do not all have DAPI).

    ``export``: also save the aligned planes as TIFFs in 2_rough_aligned/planes/Round_<n>/z<plane>_<depth>um/."""
    import shutil
    st_ = sec.main_settings(project)
    res = st_["res"]
    rounds = sec.stack_rounds(project)
    r0 = rounds[0]
    g0 = geometry(project, r0)
    planes = kept_planes(project, params)
    if not planes:
        raise ValueError("No depth is covered by every round: check the up/down position of each round.")
    dapi_everywhere = all(any(c.lower() == "dapi" for c in project.raw_channels(r)) for r in rounds)
    make_mask = not dapi_everywhere and st_["mask_source"] != "none"
    base = project.root / sec.SECTIONS_DIR
    base.mkdir(exist_ok=True)
    for old in sec.section_projects(project):
        shutil.rmtree(old.root)
    exp_dir = project.step_dir("rough") / sec.PLANES_DIR
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    subs = []
    for k, z in enumerate(planes, 1):
        d = round(z * g0["dz"], 2)
        sub = Project(base / sec.section_name(k, d))
        src = {str(r0): float(z)}
        for r in rounds[1:]:
            src[str(r)] = round(matching_plane(project, r, params[str(r)], z * g0["dz"]), 2)
        sub.meta["section"] = {"parent": project.name, "number": k, "depth_um": d, "planes": {r_: int(round(v)) for r_, v in src.items()},
                               "planes_exact": src, "slab_um": 0.0, "aligned_3d": True}
        if project.meta.get("align_with"):
            sub.meta["align_with"] = dict(project.meta["align_with"])
        sub.save()
        subs.append(sub)
    total = sum(len(project.raw_channels(r)) for r in rounds)
    done = 0
    for r in rounds:
        gr = geometry(project, r)
        p = params.get(str(r), DEFAULT) if r != r0 else DEFAULT
        mask_base = [None] * len(planes)
        for c in project.raw_channels(r):
            vol = project.load_raw(r, c).astype(np.float32)
            m_ = is_mask(c)
            norm = dict(st_["norm"], do_norm=False) if m_ else st_["norm"]
            for i, z in enumerate(planes):
                if r == r0:
                    flat = vol[z]
                else:
                    flat = cut_plane(vol, gr, g0, p, z * g0["dz"], (g0["h"], g0["w"]), g0["px"])
                    if flat is None:
                        flat = np.zeros((g0["h"], g0["w"]), np.float32)
                big = resize_plane(flat, res["res_mode"], res.get("custom_scale", 1.0), nearest=m_)
                img = format_and_resize(apply_ramfish_normalization(big, **norm), ".tif", **sec.SAME_SIZE)
                subs[i].add_image(r, c, img)
                if export:
                    _export(project, r, z, g0, c, img)
                if make_mask and not m_ and c.lower() != "dapi":
                    if st_["mask_source"] == MASK_ALL or project.raw_info(r, c).get("dye") == st_["mask_source"]:
                        lo, hi = np.percentile(flat, [0.5, 99.5])
                        s = (flat - lo) / (hi - lo) if hi > lo else np.zeros_like(flat)
                        mask_base[i] = s if mask_base[i] is None else np.maximum(mask_base[i], s)
            del vol
            done += 1
            if progress:
                progress(done / max(total, 1), f"Round {r}: {c}")
        if make_mask and not any(is_mask(c) for c in project.raw_channels(r)):
            for i, z in enumerate(planes):
                if mask_base[i] is None:
                    continue
                m = tissue_mask(mask_base[i], **st_["mask"])
                img = format_and_resize(resize_plane(mask_image(m), res["res_mode"], res.get("custom_scale", 1.0), nearest=True),
                                        ".tif", **sec.SAME_SIZE)
                subs[i].add_image(r, "mask", img)
                if export:
                    _export(project, r, z, g0, "mask", img)
    for sub in subs:
        sub.log("section", {"aligned_3d": True}, **sub.meta["section"])
    project.meta["rough3d_built"] = {"params": {k: dict(v) for k, v in params.items()}, "planes": planes,
                                     "skip": dict(project.meta.get("rough3d_skip", {}))}
    project.save()
    project.log("rough_align_3d", {k: dict(v) for k, v in params.items()}, kept_planes=planes)
    return subs


def _export(project: Project, r: int, z: int, g0: dict, channel: str, img: np.ndarray) -> None:
    import tifffile
    d = project.step_dir("rough") / sec.PLANES_DIR / f"Round_{r}" / f"z{z + 1:02d}_{z * g0['dz']:.1f}um"
    d.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(d / f"{channel}.tif", np.asarray(img).astype(np.uint16))
