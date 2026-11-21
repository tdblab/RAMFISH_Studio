"""Sections of thick 3D samples: the same depth in every round, analyzed one section at a time.

A thick sample (a zebrafish brain) cannot be flattened into one picture: a projection piles up cells from many
depths. Instead, Round 1's stack is cut into sections (for example every 10 micrometers), and for each section the
plane at the same depth is found in every other round. Each section then becomes a small project of its own,
``<project>/sections/S1_3um/``, with one 2D image per round and gene (a thin slab around the matching plane),
and the usual steps (fine alignment, spot counter, merge) run on it unchanged.

Finding the matching plane: the rounds are first lined up in X and Y (with the rough alignment of step 2 when it
has been done, otherwise automatically), then every plane of a round is compared with every plane of Round 1,
picture to picture. In the table of scores the matching planes form a ridge running down both stacks. The ridge is
followed with the depth order kept (a deeper plane in Round 1 never matches a shallower plane in the other round),
which pools the evidence of all planes instead of trusting one comparison. The result is a suggestion: the page
shows the planes side by side so they can be checked and changed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from . import rough_align as ra
from . import zmatch
from .mask import DEFAULT_MASK_SETTINGS, MASK_ALL, mask_image, tissue_mask
from .normalize import RES_4K, apply_ramfish_normalization, format_and_resize, resize_plane, target_size
from .project import Project, is_mask

SMALL = 8                     # suggestions are worked out on images 8 times smaller than the 4K frame
COMPARE_GENES = "genes"       # all gene channels merged (present in every round)
COMPARE_DAPI = "dapi"         # DAPI, for rounds where both have it
COMPARE_MASK = "mask"         # the tissue outline of each plane (made from the gene channels, like step 1's mask)
COMPARE_LABELS = {COMPARE_MASK: "Tissue outline (mask)", COMPARE_DAPI: "DAPI (where both rounds have it)",
                  COMPARE_GENES: "All gene channels merged"}
SECTIONS_DIR = "sections"
SAME_SIZE = {"res_mode": "Original Resolution", "custom_scale": 1.0}


# ------------------------------------------------------------------------------------------ settings
def main_settings(project: Project) -> dict:
    """Normalization, resolution and mask settings of the project's last 'Normalize' run (step 1)."""
    last = next((h for h in reversed(project.meta.get("history", [])) if h.get("step") == "normalize"), {})
    norm = last.get("params") or {"do_norm": True, "blur_sigma": 50.0, "percentile_val": 99.5, "target_intensity": 60000.0}
    return {"norm": norm,
            "res": {"res_mode": last.get("res_mode", RES_4K), "custom_scale": float(last.get("custom_scale", 1.0))},
            "mask_source": last.get("mask", MASK_ALL),
            "mask": last.get("mask_settings") or dict(DEFAULT_MASK_SETTINGS)}


def stack_rounds(project: Project) -> list[int]:
    """Rounds whose raw images are 3D stacks."""
    out = []
    for r in project.raw_rounds():
        chans = project.raw_channels(r)
        if chans and int(project.raw_info(r, chans[0]).get("z_planes") or 1) > 1:
            out.append(r)
    return out


def z_step(project: Project, r: int) -> float:
    chans = project.raw_channels(r)
    return float(project.raw_info(r, chans[0]).get("z_step_um") or 1.0) if chans else 1.0


def n_planes(project: Project, r: int) -> int:
    chans = project.raw_channels(r)
    return int(project.raw_info(r, chans[0]).get("z_planes") or 1) if chans else 1


def section_depths(n_z: int, step_um: float, first_um: float, spacing_um: float) -> list[float]:
    """Depths (micrometers from Round 1's first plane) of the sections that fit in Round 1's stack."""
    last = (n_z - 1) * step_um
    out, d = [], float(first_um)
    while d <= last + 1e-6 and spacing_um > 0:
        out.append(round(d, 2))
        d += spacing_um
    return out


# ------------------------------------------------------------------------------------------ small stacks
def frame_size(project: Project, r: int, res: dict) -> tuple[int, int]:
    """(height, width) of the round's 2D images in the project (the 4K frame by default)."""
    import tifffile
    chans = project.raw_channels(r)
    with tifffile.TiffFile(project.path("raw", r, chans[0])) as tf:   # only the size, not the pixels
        h, w = tf.series[0].shape[-2:]
    return target_size(h, w, res["res_mode"], res.get("custom_scale", 1.0))


def small_stack(project: Project, r: int, use: str, frame_hw: tuple[int, int]) -> np.ndarray:
    """The round's stack, SMALL times smaller than the frame, for comparing planes: DAPI, or the gene channels
    merged (each on its own brightness scale, brightest value per pixel)."""
    h, w = frame_hw[0] // SMALL, frame_hw[1] // SMALL
    if use == COMPARE_MASK:
        return _small_mask_stack(project, r, (h, w))
    chans = [c for c in project.raw_channels(r) if not is_mask(c)]
    dapi = [c for c in chans if c.lower() == "dapi"]
    pick = dapi if use == COMPARE_DAPI and dapi else [c for c in chans if c.lower() != "dapi"] or chans
    out = None
    for c in pick:
        v = project.load_raw(r, c)
        s = np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA) for p in v])
        lo, hi = np.percentile(s, [0.5, 99.5])
        s = np.clip((s - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(s)
        out = s if out is None else np.maximum(out, s)
    return out.astype(np.float32)


def _small_mask_stack(project: Project, r: int, hw: tuple[int, int]) -> np.ndarray:
    """Tissue outline of every plane (step 1's mask settings, scaled to the small size), softened at the edge so
    planes whose outlines are close still score as close."""
    import tifffile
    h, w = hw
    genes = small_stack(project, r, COMPARE_GENES, (h * SMALL, w * SMALL))
    chans = [c for c in project.raw_channels(r) if not is_mask(c)]
    with tifffile.TiffFile(project.path("raw", r, chans[0])) as tf:
        raw_w = tf.series[0].shape[-1]
    k = w / float(raw_w)                        # raw pixels -> small pixels
    ms = dict(main_settings(project)["mask"])
    ms = {"blur_px": ms.get("blur_px", 8.0) * k, "threshold_factor": ms.get("threshold_factor", 0.5),
          "closing_px": ms.get("closing_px", 15) * k, "dilation_px": ms.get("dilation_px", 0) * k}
    return np.stack([gaussian_filter(tissue_mask(p, **ms).astype(np.float32), 2.0) for p in genes]).astype(np.float32)


def band(x: np.ndarray) -> np.ndarray:
    """Brain-region-sized detail only (no pixel noise, no overall glow), on one scale."""
    x = gaussian_filter(x, 1.5) - gaussian_filter(x, 15)
    return (x - x.mean()) / (x.std() + 1e-9)


# ------------------------------------------------------------------------------------------ X/Y line-up
def rough_warp(params: dict, frame_w: int):
    """The rough alignment of step 2, applied to a small image (its numbers are for the full frame)."""
    def warp(img):
        k = img.shape[1] / float(frame_w)
        return ra.apply(np.ascontiguousarray(img, dtype=np.float32), dict(params, scale_ratio=float(params.get("scale_ratio", 1.0)) * k))
    return warp


def auto_warp(fixed: np.ndarray, moving: np.ndarray):
    """Line up two projections automatically (shift, turn and size; mutual information), for when step 2 was skipped."""
    import SimpleITK as sitk
    f = sitk.GetImageFromArray(fixed.astype(np.float32))
    m = sitk.GetImageFromArray(moving.astype(np.float32))
    best = None
    for ang in np.radians(np.arange(-30, 31, 10)):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(50)
        reg.SetMetricSamplingStrategy(reg.NONE)      # every pixel (the images are small): same answer every time
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300)
        reg.SetOptimizerScalesFromPhysicalShift()
        init = sitk.CenteredTransformInitializer(f, m, sitk.Similarity2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
        init.SetAngle(float(ang))
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SetInitialTransform(init, inPlace=False)
        try:
            t = reg.Execute(f, m)
        except Exception:
            continue
        # the starting angle that ends best, judged by the same picture comparison as the planes
        w = sitk.GetArrayFromImage(sitk.Resample(m, f, t, sitk.sitkLinear, 0.0))
        score = float((band(fixed) * band(w)).mean())
        if best is None or score > best[0]:
            best = (score, t)
    t = best[1] if best else sitk.Transform(2, sitk.sitkIdentity)

    def warp(img):
        out = sitk.Resample(sitk.GetImageFromArray(np.asarray(img, dtype=np.float32)), f, t, sitk.sitkLinear, 0.0)
        return sitk.GetArrayFromImage(out)
    return warp


# ------------------------------------------------------------------------------------------ matching planes
def score_matrix(ref: np.ndarray, mov_warped: np.ndarray) -> np.ndarray:
    """How alike every Round 1 plane (rows) and every plane of the other round (columns) are (-1 to 1)."""
    a = [band(p) for p in ref]
    b = [band(p) for p in mov_warped]
    return np.array([[float((x * y).mean()) for y in b] for x in a])


def follow_ridge(M: np.ndarray, max_step: int = 2) -> tuple[list[int], float]:
    """Best matching plane for every Round 1 plane, keeping the depth order (each step down in Round 1 moves
    0 to ``max_step`` planes down in the other round). Returns (planes, mean score along the path)."""
    n, m = M.shape
    S = np.full((n, m), -1e9)
    B = np.zeros((n, m), dtype=int)
    S[0] = M[0]
    for i in range(1, n):
        for j in range(m):
            lo = max(0, j - max_step)
            k = lo + int(np.argmax(S[i - 1, lo:j + 1]))
            S[i, j] = S[i - 1, k] + M[i, j]
            B[i, j] = k
    j = int(np.argmax(S[-1]))
    path = [j]
    for i in range(n - 1, 0, -1):
        j = int(B[i, j])
        path.append(j)
    return path[::-1], float(S[-1].max()) / n


def suggest_planes(project: Project, use: str = COMPARE_MASK, progress=None) -> dict:
    """Matching plane in every round for every plane of Round 1. Saved in the project and returned:
    {"use": ..., "rounds": {r: {"planes": [...], "score": s, "upside_down": bool, "xy": "rough alignment"|"automatic"}}}"""
    rounds = stack_rounds(project)
    if not rounds:
        return {}
    r0 = rounds[0]
    res = main_settings(project)["res"]
    hw = frame_size(project, r0, res)
    ref = small_stack(project, r0, use, hw)
    ref_dapi = any(c.lower() == "dapi" for c in project.raw_channels(r0))
    out = {"use": use, "reference": r0, "rounds": {}}
    for i, r in enumerate(rounds[1:]):
        if progress:
            progress(i / max(len(rounds) - 1, 1), f"Comparing Round {r} with Round {r0}...")
        both_dapi = ref_dapi and any(c.lower() == "dapi" for c in project.raw_channels(r))
        pair = use if use == COMPARE_MASK else (COMPARE_DAPI if (use == COMPARE_DAPI and both_dapi) else COMPARE_GENES)
        a = ref if pair == use else small_stack(project, r0, pair, hw)
        hw_r = frame_size(project, r, res)
        mov = small_stack(project, r, pair, hw_r)
        params = project.meta.get("rough_params", {}).get(str(r))
        fit = lambda img: img if img.shape == a.shape[1:] else \
            cv2.resize(img, (a.shape[2], a.shape[1]), interpolation=cv2.INTER_AREA)
        if params:   # step 2's numbers are for this round's own frame; the result is then fitted to Round 1's
            rw = rough_warp(params, hw_r[1])
            warp, how = (lambda img: fit(rw(img))), "rough alignment"
        else:
            warp, how = auto_warp(a.max(0), np.stack([fit(p) for p in mov]).max(0)), "automatic"
            aw = warp
            warp = lambda img: aw(fit(img))
        mw = np.stack([warp(p) for p in mov])
        M = score_matrix(a, mw)
        fwd, s_fwd = follow_ridge(M)
        rev, s_rev = follow_ridge(M[:, ::-1])
        upside = s_rev > s_fwd + 0.01
        planes = [mov.shape[0] - 1 - j for j in rev] if upside else fwd
        out["rounds"][str(r)] = {"planes": [int(z) for z in planes], "score": round(max(s_fwd, s_rev), 3),
                                 "upside_down": bool(upside), "xy": how, "compared": pair}
    project.meta["section_suggestions"] = out
    project.save()
    if progress:
        progress(1.0, "Done")
    return out


def warp_for_display(project: Project, r: int, use: str = COMPARE_GENES):
    """(small stack of round r lined up with Round 1, small stack of Round 1) for showing planes side by side."""
    rounds = stack_rounds(project)
    r0 = rounds[0]
    res = main_settings(project)["res"]
    hw = frame_size(project, r0, res)
    a = small_stack(project, r0, use, hw)
    if r == r0:
        return a, a
    hw_r = frame_size(project, r, res)
    mov = small_stack(project, r, use, hw_r)
    fit = lambda img: img if img.shape == a.shape[1:] else cv2.resize(img, (a.shape[2], a.shape[1]), interpolation=cv2.INTER_AREA)
    params = project.meta.get("rough_params", {}).get(str(r))
    if params:
        rw = rough_warp(params, hw_r[1])
        return np.stack([fit(rw(p)) for p in mov]), a
    aw = auto_warp(a.max(0), np.stack([fit(p) for p in mov]).max(0))
    return np.stack([aw(fit(p)) for p in mov]), a


def planes_for_depth(project: Project, suggestion: dict, depth_um: float) -> dict[int, int]:
    """{round: plane} for one section depth (Round 1's plane is the one nearest to the depth)."""
    rounds = stack_rounds(project)
    r0 = rounds[0]
    z0 = int(np.clip(round(depth_um / z_step(project, r0)), 0, n_planes(project, r0) - 1))
    out = {r0: z0}
    for r in rounds[1:]:
        s = (suggestion or {}).get("rounds", {}).get(str(r))
        if s:
            out[r] = int(s["planes"][z0])
        else:   # no suggestion yet: the same depth in micrometers
            out[r] = int(np.clip(round(depth_um / z_step(project, r)), 0, n_planes(project, r) - 1))
    return out


# ------------------------------------------------------------------------------------------ section projects
def section_name(k: int, depth_um: float) -> str:
    return f"S{k}_{depth_um:g}um".replace(".", "p")


def section_projects(project: Project) -> list[Project]:
    d = project.root / SECTIONS_DIR
    if not d.is_dir():
        return []
    subs = [Project(p) for p in sorted(d.iterdir()) if (p / ".ramfish_project.json").exists()]
    return sorted(subs, key=lambda s: s.meta.get("section", {}).get("depth_um", 0))


def build_sections(project: Project, sections: list[dict], slab_um: float, progress=None) -> list[Project]:
    """Make one project per section. ``sections``: [{"depth_um": d, "planes": {round: plane}}, ...].

    For every round and channel: a slab of ``slab_um`` around the section's plane is projected, enlarged to the
    project's size (4K by default), normalized with step 1's settings and saved; a tissue mask is made the same way
    as in step 1 when the rounds do not all have DAPI; and step 2's rough alignment is copied, so the section can go
    straight to fine alignment.
    """
    import shutil
    st_ = main_settings(project)
    rounds = [r for r in project.raw_rounds()]
    dapi_everywhere = all(any(c.lower() == "dapi" for c in project.raw_channels(r)) for r in rounds)
    make_mask = not dapi_everywhere and st_["mask_source"] != "none"
    base = project.root / SECTIONS_DIR
    base.mkdir(exist_ok=True)
    for old in section_projects(project):   # new sections replace the ones made before (other depths would mix in)
        shutil.rmtree(old.root)
    subs = []
    for k, sec in enumerate(sections, 1):
        d = base / section_name(k, sec["depth_um"])
        if d.exists():
            shutil.rmtree(d)
        sub = Project(d)
        sub.meta["section"] = {"parent": project.name, "number": k, "depth_um": sec["depth_um"],
                               "planes": {str(r): int(z) for r, z in sec["planes"].items()}, "slab_um": slab_um}
        if project.meta.get("align_with"):
            sub.meta["align_with"] = dict(project.meta["align_with"])
        sub.save()
        subs.append(sub)
    total = sum(len(project.raw_channels(r)) for r in rounds)
    done = 0
    for r in rounds:
        n_z, step = n_planes(project, r), z_step(project, r)
        mask_base = [None] * len(sections)
        for c in project.raw_channels(r):
            vol = project.load_raw(r, c)
            norm = dict(st_["norm"], do_norm=False) if is_mask(c) else st_["norm"]
            for i, sec in enumerate(sections):
                z = int(sec["planes"].get(r, sec["planes"].get(str(r), 0)))
                z0, z1 = zmatch.slab_planes(n_z, z, step, slab_um) if n_z > 1 else (0, 0)
                flat = vol[z0:z1 + 1].max(axis=0)
                big = resize_plane(flat, st_["res"]["res_mode"], st_["res"].get("custom_scale", 1.0), nearest=is_mask(c))
                subs[i].add_image(r, c, format_and_resize(apply_ramfish_normalization(big, **norm), ".tif", **SAME_SIZE))
                if make_mask and not is_mask(c) and c.lower() != "dapi":
                    if st_["mask_source"] == MASK_ALL or project.raw_info(r, c).get("dye") == st_["mask_source"]:
                        lo, hi = np.percentile(flat, [0.5, 99.5])
                        s = (flat - lo) / (hi - lo) if hi > lo else np.zeros_like(flat)
                        mask_base[i] = s if mask_base[i] is None else np.maximum(mask_base[i], s)
            del vol
            done += 1
            if progress:
                progress(done / max(total, 1), f"Round {r}: {c}")
        if make_mask and not any(is_mask(c) for c in project.raw_channels(r)):
            for i in range(len(sections)):
                if mask_base[i] is None:
                    continue
                m = tissue_mask(mask_base[i], **st_["mask"])
                img = resize_plane(mask_image(m), st_["res"]["res_mode"], st_["res"].get("custom_scale", 1.0), nearest=True)
                subs[i].add_image(r, "mask", format_and_resize(img, ".tif", **SAME_SIZE))
    # step 2's rough alignment, copied into every section
    for sub in subs:
        for r in sub.rounds():
            params = project.meta.get("rough_params", {}).get(str(r))
            if not params or r == sub.reference_round:
                continue
            for c in sub.channels(r):
                import tifffile
                img = tifffile.imread(sub.path("normalized", r, c))
                out = sub.path("rough", r, c)
                out.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(out, ra.apply(img, params))
            sub.meta.setdefault("rough_params", {})[str(r)] = dict(params)
        if sub.meta.get("rough_params"):
            sub.write_reference_copies("rough")
        sub.log("section", {"slab_um": slab_um}, **sub.meta["section"])
    project.log("sections", {"slab_um": slab_um}, sections=[{"depth_um": s["depth_um"], "planes": {str(r): int(z) for r, z in s["planes"].items()}}
                                                           for s in sections])
    return subs


def combined_counts(project: Project):
    """Spot counts of every section in one table (rows: section and gene)."""
    import pandas as pd
    frames = []
    for sub in section_projects(project):
        f = sub.step_dir("spots") / "data_sheets" / "BATCH_SUMMARY_COUNTS.csv"
        if f.exists():
            df = pd.read_csv(f)
            sec = sub.meta.get("section", {})
            df.insert(0, "Depth in Round 1 (um)", sec.get("depth_um"))
            df.insert(0, "Section", sub.root.name)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


PLANES_DIR = "planes"     # 2_rough_aligned/planes/Round_<n>/z<plane>_<depth>um/: the aligned planes as TIFFs
