"""Step 1 in two parts: (A) read microscope files into raw TIFFs, (B) make the normalized 2D images from them.

A. Every channel of every file is saved as it was read (all Z planes, original intensities, pixel size kept)
   in ``1_import_and_normalize/raw/Round_<n>/<gene>.tif``. The round and the gene of every channel come from the
   file name and the file's own channel information (see :mod:`ramfish.channels`).
B. From the raw TIFFs: one 2D image per channel (the maximum projection of the stack, or for thick samples a thin
   slab at the matching depth, see :mod:`ramfish.zmatch`), a tissue mask where needed (see :mod:`ramfish.mask`),
   the RAMFISH normalization, and the exported copies. Part B can be run again with other settings without
   reading the microscope files again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import dataset
from .channels import HOW_SINGLE, assign_channels, dye_label, oir_planned_z, parse_filename
from .io import open_with_info
from .mask import DEFAULT_MASK_SETTINGS, MASK_ALL, mask_image, tissue_mask
from .normalize import RES_4K, apply_ramfish_normalization, format_and_resize, resize_plane
from .project import Project, clean_name, is_mask
from . import zmatch

SAME_SIZE = {"res_mode": "Original Resolution", "custom_scale": 1.0}   # already resized: keep the size
DEPTH_MIP = "mip"        # maximum projection of the whole stack (thin samples)
DEPTH_MATCH = "match"    # slab at the matching depth (thick samples)


@dataclass
class ReadResult:
    file: str
    round: int | None
    channels: list[dict] = field(default_factory=list)   # one row per channel for the check table
    warnings: list[str] = field(default_factory=list)


def read_file(project: Project, path: str | Path, display_name: str, fallback_round: int | None,
              relpath: str | None = None, omit_tokens=None, amplifier_dyes=None,
              single_name: str | None = None) -> ReadResult:
    """Part A for one file: work out round and genes, save every channel as a raw TIFF.

    ``single_name`` names a single-channel file (e.g. the DAPI / mask name that :func:`dataset.scan` settled on).
    """
    path = Path(path)
    single_name_override = single_name
    shape, info, get_channel = open_with_info(path)    # channels are read one at a time
    n_c, n_z = shape[0], shape[1]
    pixel_um = info.get("pixel_um")
    base = Path(display_name).name
    folder_round, single_name = dataset.parse_name(relpath or base)
    parsed = parse_filename(base, omit_tokens)
    res = ReadResult(display_name, None)

    if n_c == 1:
        rnd = folder_round if folder_round is not None else (parsed.round if parsed.round is not None else fallback_round)
        meta = info["channels"][0] if info["channels"] else None
        assignments = [(0, single_name_override or single_name, meta.dye if meta else None, HOW_SINGLE)]
    else:
        rnd = folder_round if folder_round is not None else (parsed.round if parsed.round is not None else fallback_round)
        found, warns = assign_channels(parsed, info["channels"], amplifier_dyes)
        res.warnings += warns
        assignments = [(a.index, a.name, a.dye, a.how) for a in found]
    res.round = rnd
    planned = oir_planned_z(path) if path.suffix.lower() == ".oir" else None
    planned_z = planned[0] if planned else None
    if planned and planned[0] > n_z:
        step = planned[1] or (pixel_um[0] if pixel_um and pixel_um[0] else 0)
        res.warnings.append(
            f"{display_name}: only {n_z} of the {planned[0]} planes set up on the microscope are in this file "
            f"({(n_z - 1) * step:.0f} of {(planned[0] - 1) * step:.0f} µm). The microscope software saves the rest of a "
            f"large stack in extra files with the same name plus _00001, _00002, … (no extension). Add them next to "
            f"the .oir (or in the same folder of the zip) and import again to get the whole stack.")
    if rnd is None:
        res.warnings.append(f"{display_name}: no round in the file name and no round given; the file was skipped.")
        return res
    if parsed.round is not None and folder_round is None and fallback_round is not None and parsed.round != fallback_round:
        res.warnings.append(f"{display_name}: the file name says Round {parsed.round}; that was used "
                            f"(not the Round box, {fallback_round}).")

    for idx, name, dye, how in assignments:
        name = clean_name(name)
        # the same name from another file of this round would overwrite it: keep both
        existing = project.raw_info(rnd, name)
        if existing and existing.get("file") != display_name:
            k = 2
            while project.raw_info(rnd, f"{name}_{k}"):
                k += 1
            res.warnings.append(f"Round {rnd} already has '{name}' from {existing.get('file')}; this one is '{name}_{k}'.")
            name = f"{name}_{k}"
        z_step = pixel_um[0] if pixel_um and pixel_um[0] else None
        row = {"file": display_name, "channel_in_file": idx + 1, "dye": dye or "", "how": how,
               "z_planes": int(n_z), "z_planes_planned": planned_z, "z_step_um": z_step,
               "pixel_um": [float(v) if v else None for v in pixel_um] if pixel_um else None,
               "source_name": info["channels"][idx].name if idx < len(info["channels"]) else ""}
        saved = project.add_raw(rnd, name, get_channel(idx), row)
        res.channels.append({"Round": rnd, "Gene / channel": saved, "Dye": dye_label(dye), "Channel in file": idx + 1,
                             "Matched by": how, "Z planes": int(n_z), "File": display_name})
    project.log("read raw", {}, round=rnd, file=display_name, channels=[c["Gene / channel"] for c in res.channels])
    return res


# ------------------------------------------------------------------------------------------------ part B
@dataclass
class BuildSettings:
    norm: dict
    res: dict = field(default_factory=lambda: {"res_mode": RES_4K, "custom_scale": 1.0})
    target_format: str = ".tif"
    apply_mip: bool = True
    depth_mode: str = DEPTH_MIP
    slab_um: float = 10.0
    ref_plane: int | None = None                       # Round 1's depth; None = automatic (sharpest plane)
    manual_planes: dict = field(default_factory=dict)  # {round: plane} typed in by the user
    mask_mode: str = "auto"                            # "auto" (when a round has no DAPI), "always", "never"
    mask_source: str = MASK_ALL                        # MASK_ALL or a dye key ("alexa488", ...)
    mask: dict = field(default_factory=lambda: dict(DEFAULT_MASK_SETTINGS))


def _has_dapi(project: Project, rnd: int) -> str | None:
    return next((c for c in project.raw_channels(rnd) if c.lower() == "dapi"), None)


def _gene_channels(project: Project, rnd: int) -> list[str]:
    return [c for c in project.raw_channels(rnd) if not is_mask(c)]


def _scaled(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol[:, ::4, ::4], [0.5, 99.5])
    return (vol - lo) / (hi - lo) if hi > lo else np.zeros_like(vol)


def alignment_volume(project: Project, rnd: int, use_dapi: bool, channels: list[str] | None = None) -> np.ndarray:
    """The stack the depth search and the mask look at: DAPI, or all channels merged (each on its own scale)."""
    if use_dapi:
        return project.load_raw(rnd, _has_dapi(project, rnd))
    out = None
    for c in channels or _gene_channels(project, rnd):
        v = _scaled(project.load_raw(rnd, c))
        out = v if out is None else np.maximum(out, v)
    return out


def find_depths(project: Project, settings: BuildSettings, use_dapi: bool) -> dict:
    """Plane (and slab) to use in every round. Round 1 sets the depth; the others are matched to it."""
    rounds = project.raw_rounds()
    if not rounds:
        return {}
    out = {}
    r0 = rounds[0]
    feats = {}
    steps = {}
    for r in rounds:
        chans = project.raw_channels(r)
        info = project.raw_info(r, chans[0]) if chans else {}
        steps[r] = float(info.get("z_step_um") or 1.0)
        n_z = int(info.get("z_planes") or 1)
        if n_z > 1:
            feats[r] = zmatch.plane_features(alignment_volume(project, r, use_dapi))
    for r in rounds:
        n_z = feats[r].shape[0] if r in feats else 1
        entry = {"n_z": n_z, "z_step_um": steps[r], "auto": True, "score": None, "upside_down": False}
        if n_z == 1:
            entry.update(z=0, z0=0, z1=0, auto=False)
        elif r in settings.manual_planes and settings.manual_planes[r] is not None:
            entry.update(z=int(np.clip(settings.manual_planes[r], 0, n_z - 1)), auto=False)
        elif r == r0:
            ref = settings.ref_plane
            entry.update(z=int(np.clip(ref, 0, n_z - 1)) if ref is not None else zmatch.sharpest_plane(feats[r]),
                         auto=ref is None)
        elif r0 in feats:
            m = zmatch.match_depth(feats[r0], steps[r0], out[r0]["z"], feats[r], steps[r])
            entry.update(z=m.z_index, score=round(m.score, 3), upside_down=m.upside_down)
        else:
            entry.update(z=n_z // 2)       # Round 1 is a single plane: nothing to match, use the middle
        if n_z > 1:
            entry["z0"], entry["z1"] = zmatch.slab_planes(n_z, entry["z"], steps[r], settings.slab_um)
        out[r] = entry
    return out


def build(project: Project, settings: BuildSettings, progress=None) -> list[str]:
    """Part B for every round with raw TIFFs. Returns warnings to show."""
    warnings: list[str] = []
    rounds = project.raw_rounds()
    if not rounds:
        return ["No raw images yet: read the microscope files first."]
    dapi_everywhere = all(_has_dapi(project, r) for r in rounds)
    make_mask = settings.mask_mode == "always" or (settings.mask_mode == "auto" and not dapi_everywhere)
    use_dapi_for_depth = dapi_everywhere
    any_3d = any(int(project.raw_info(r, c).get("z_planes") or 1) > 1 for r in rounds for c in project.raw_channels(r)[:1])
    depths = find_depths(project, settings, use_dapi_for_depth) if (settings.depth_mode == DEPTH_MATCH and any_3d) else {}
    project.meta["depth"] = {str(r): d for r, d in depths.items()}
    for r, d in depths.items():
        if d.get("upside_down"):
            warnings.append(f"Round {r} seems to be mounted the other way up: its depth was searched from the other end.")
        if d.get("score") is not None and d["score"] < 0.5:
            warnings.append(f"Round {r}: the matching depth is uncertain (agreement {d['score']:.2f}). Check the "
                            "images and type the plane yourself if needed.")

    out_root = project.step_dir("exports")
    prefix = "NORM_" if settings.norm.get("do_norm") else "RAW_"
    total = sum(len(project.raw_channels(r)) for r in rounds)
    done = 0
    for r in rounds:
        d = depths.get(r)
        out = out_root / f"Round_{r}"
        out.mkdir(parents=True, exist_ok=True)
        keep = []
        raw_planes = {}
        own_mask = False
        for c in project.raw_channels(r):
            norm = dict(settings.norm, do_norm=False) if is_mask(c) else settings.norm   # a mask image is kept as it is
            own_mask = own_mask or is_mask(c)
            vol = project.load_raw(r, c)
            n_z = vol.shape[0]
            if n_z == 1:
                flat = vol[0]
            elif d is not None:
                flat = zmatch.slab_projection(vol, d["z0"], d["z1"])
            else:
                flat = vol.max(axis=0)
            if not is_mask(c):
                raw_planes[c] = flat
            # Every image is first resized (by default to 4K, longest side 4096 px), then normalized, so the
            # normalization and the spot counter work on the full-size 16-bit TIFF.
            big = lambda img: resize_plane(img, settings.res["res_mode"], settings.res.get("custom_scale", 1.0),
                                           nearest=is_mask(c))
            working = apply_ramfish_normalization(big(flat), **norm)
            # exported copies: every Z plane, the projection of the whole stack and the slab used
            for z in range(n_z):
                z_label = f"_Z{z}" if n_z > 1 else ""
                plane = working if n_z == 1 else apply_ramfish_normalization(big(vol[z]), **norm)
                _write(out / f"{prefix}R{r}_{c}{z_label}{settings.target_format}",
                       format_and_resize(plane, settings.target_format, **SAME_SIZE))
            if n_z > 1 and settings.apply_mip:
                _write(out / f"{prefix}R{r}_{c}_MIP{settings.target_format}",
                       format_and_resize(apply_ramfish_normalization(big(vol.max(axis=0)), **norm),
                                         settings.target_format, **SAME_SIZE))
            if n_z > 1 and d is not None:
                _write(out / f"{prefix}R{r}_{c}_SLAB_Z{d['z0']}-Z{d['z1']}{settings.target_format}",
                       format_and_resize(working, settings.target_format, **SAME_SIZE))
            keep.append(project.add_image(r, c, format_and_resize(working, ".tif", **SAME_SIZE)))
            del vol
            done += 1
            if progress:
                progress(done / max(total, 1), f"Round {r}: {c}")

        if make_mask and not own_mask and raw_planes:
            src = [c for c in raw_planes if c.lower() != "dapi"] or list(raw_planes)
            if settings.mask_source != MASK_ALL:
                pick = [c for c in src if project.raw_info(r, c).get("dye") == settings.mask_source]
                if pick:
                    src = pick[:1]
                else:
                    warnings.append(f"Round {r} has no {dye_label(settings.mask_source)} channel: its mask was made "
                                    "from all channels merged.")
            from .mask import merge_for_mask
            base = raw_planes[src[0]] if len(src) == 1 else merge_for_mask([raw_planes[c] for c in src])
            m = tissue_mask(base, **settings.mask)       # made at the microscope's resolution (faster), then resized
            img = resize_plane(mask_image(m), settings.res["res_mode"], settings.res.get("custom_scale", 1.0), nearest=True)
            keep.append(project.add_image(r, "mask", format_and_resize(img, ".tif", **SAME_SIZE)))
            _write(out / f"MASK_R{r}{settings.target_format}", format_and_resize(img, settings.target_format, **SAME_SIZE))
            if m.mean() < 0.02 or m.mean() > 0.98:
                warnings.append(f"Round {r}: the tissue mask covers {100 * m.mean():.0f}% of the image. Check it.")
        # images of this round that are no longer made (e.g. a mask from an earlier run)
        for c in project.channels(r):
            if c not in keep and c not in project.raw_channels(r):
                project.remove_channel(r, c)
    project.log("normalize", settings.norm, **settings.res, depth_mode=settings.depth_mode, slab_um=settings.slab_um,
                mask=(settings.mask_source if make_mask else "none"), mask_settings=settings.mask,
                depths={str(k): v for k, v in depths.items()})
    return warnings


def _write(path: Path, img: np.ndarray) -> None:
    import imageio.v3 as iio
    iio.imwrite(str(path), img)
