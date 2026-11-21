"""Steps 3 and 4 on every section of a 3D project, and the merge across depth.

``run_sections`` runs fine alignment and the spot counter on every section in turn, with exactly the steps of the
Fine alignment and Spot counter pages. ``composite`` draws one merged picture the way the Multiplex merge page does
(each layer scaled to its own brightest pixel, faint signal below the threshold hidden, gamma applied, colors added
at 70 %, and a key with the gene names), so the picture of every depth can be shown behind one slider.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .fine_align import apply_transforms, compute_transforms, match_fixed_shape_centered, read_image_bytes
from .normalize import format_and_resize
from .io import read_plane
from .project import Project, is_alignment_channel
from .spots import DEFAULT_CONFIG, process_single_image

# The merge page's default colors, in the same order (DAPI is light grey)
PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 165, 0),
           (128, 0, 128), (0, 128, 128), (128, 128, 0), (255, 100, 100), (100, 255, 100), (100, 100, 255),
           (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 200, 0), (255, 0, 200), (0, 255, 200),
           (200, 255, 0), (200, 0, 255), (0, 200, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 64, 0),
           (128, 0, 64), (0, 128, 64), (0, 64, 128), (64, 0, 128)]
DAPI_GREY = (200, 200, 200)
SOURCE_SPOTS = "Synthetic spots (step 4)"
SOURCE_ALIGNED = "Aligned images (step 3)"


# ------------------------------------------------------------------------------------------ steps 3 and 4
def fine_align(project: Project, bend_masks: bool = False, progress=None) -> None:
    """The Fine alignment page's 'Run Batch Alignment', for one project."""
    r0 = project.reference_round
    rounds = [r for r in project.rounds() if r != r0 and project.alignment_options(r)]
    out_dir = project.step_dir("fine")
    out_dir.mkdir(exist_ok=True)
    for r in rounds:
        project.clear_stage("fine", r)
    for i, r in enumerate(rounds):
        ch = project.alignment_channel(r)
        fixed = read_image_bytes(project.path("normalized", r0, project.reference_partner(r, ch)).read_bytes())
        moving = match_fixed_shape_centered(read_image_bytes(project.path(project.stage_of(r, ch, "rough"), r, ch).read_bytes()),
                                            fixed.shape)
        bend = ch.lower() != "mask" or bend_masks
        affine_tx, bspline_tx, fixed_img = compute_transforms(fixed, moving, bspline=bend)
        for c in project.channels(r):
            target = match_fixed_shape_centered(read_image_bytes(project.path(project.stage_of(r, c, "rough"), r, c).read_bytes()),
                                                fixed.shape)
            cv2.imwrite(str(project.path("fine", r, c)), apply_transforms(affine_tx, bspline_tx, fixed_img, target))
        project.log("fine_align", {"affine": "GD lr=1.0, 100 it", "bspline": "mesh 8x8, LBFGSB 50 it" if bend else "off"},
                    round=r, aligned_on=ch, channels=project.channels(r))
        if progress:
            progress((i + 1) / max(len(rounds), 1))
    project.write_reference_copies("fine")


def count_spots(project: Project, settings: dict | None = None, progress=None) -> pd.DataFrame:
    """The Spot counter page's 'Run spot counter', for one project. Returns the counts per gene."""
    settings = dict(DEFAULT_CONFIG["settings"], **(settings or {}))
    base = project.step_dir("spots")
    if base.exists():
        shutil.rmtree(base)
    dirs = {"csv": base / "data_sheets" / "individual_csvs", "img": base / "spot_images", "synth": base / "synthetic_spots"}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    genes = project.gene_targets()
    stats, errors = [], []
    for i, (r, c, label) in enumerate(genes):
        res = process_single_image(project.path(project.stage_of(r, c), r, c), dirs, {"settings": settings}, label)
        if res["status"] == "success":
            stats.append({"Gene": res["gene"], "Total_Spots": res["count"]})
        elif res["status"] != "skipped":
            errors.append({"File": res["gene"], "Error": res["reason"]})
        if progress:
            progress((i + 1) / max(len(genes), 1))
    summary = pd.DataFrame(stats)
    summary.to_csv(base / "data_sheets" / "BATCH_SUMMARY_COUNTS.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(base / "data_sheets" / "ERROR_LOG.csv", index=False)
    csvs = list(dirs["csv"].glob("*_spots.csv"))
    if csvs:
        pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True).to_csv(base / "data_sheets" / "ramfish_master_spots.csv", index=False)
    project.log("spot_detection", settings, counts={s["Gene"]: s["Total_Spots"] for s in stats})
    return summary


def run_sections(subs: list[Project], spot_settings: dict | None = None, bend_masks: bool = False, progress=None) -> None:
    """Fine alignment, then the spot counter, on every section in turn."""
    n = max(len(subs), 1)
    for i, sub in enumerate(subs):
        name = sub.root.name
        fine_align(sub, bend_masks, progress=(lambda f, i=i, name=name: progress((i + 0.7 * f) / n, f"{name}: fine alignment")) if progress else None)
        count_spots(sub, spot_settings, progress=(lambda f, i=i, name=name: progress((i + 0.7 + 0.3 * f) / n, f"{name}: spot counter")) if progress else None)


# ------------------------------------------------------------------------------------------ merge across depth
def layer_names(sub: Project) -> list[str]:
    """Layers a section offers: its gene labels, then DAPI (or the mask) as the tissue background."""
    names = [label for _, _, label in sub.gene_targets()]
    bg = sub.background_channel()
    if bg:
        names.append("DAPI" if bg.lower() == "dapi" else bg)
    return names


def layer_image(sub: Project, name: str, source: str) -> np.ndarray | None:
    """8-bit gray image of one layer of a section, as the merge page loads it (None if missing)."""
    bg = sub.background_channel()
    if bg and name in ("DAPI", bg):
        p = sub.path("normalized", sub.reference_round, bg)
        return format_and_resize(read_plane(p), ".jpg", "Original Resolution", 1.0) if p.exists() else None
    where = {label: (r, c) for r, c, label in sub.gene_targets()}
    if name not in where:
        return None
    r, c = where[name]
    if source == SOURCE_SPOTS:
        p = sub.step_dir("spots") / "synthetic_spots" / f"{name}.jpg"
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if (sub.step_dir("spots") / "data_sheets" / "BATCH_SUMMARY_COUNTS.csv").exists():
            # counted, but no dots in this section (the spot counter then saves no picture): an empty layer
            stage = sub.stage_of(r, c)
            if stage:
                import tifffile
                with tifffile.TiffFile(sub.path(stage, r, c)) as tf:
                    h, w = tf.series[0].shape[-2:]
                return np.zeros((h, w), dtype=np.uint8)
        return None
    stage = sub.stage_of(r, c)
    return format_and_resize(read_plane(sub.path(stage, r, c)), ".jpg", "Original Resolution", 1.0) if stage else None


def composite(layers: list[tuple[str, np.ndarray, tuple]], threshold: float = 0.15, gamma: float = 1.5,
              legend_pos: str = "top_right", text_scale: float = 0.65, label: str | None = None) -> np.ndarray:
    """RGB picture of the layers [(name, 8-bit gray image, (r, g, b))], drawn like the Multiplex merge page."""
    from PIL import Image, ImageDraw, ImageFont
    h, w = layers[0][1].shape[:2]
    comp = np.zeros((h, w, 3), dtype=np.float32)
    for _, img, color in layers:
        g = img.astype(np.float32)
        if g.shape != (h, w):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        norm = g / max(float(g.max()), 1.0)
        norm = np.where(norm < threshold, 0.0, norm ** gamma)
        comp += norm[..., None] * (np.asarray(color, dtype=np.float32) * 0.7)
    out = Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    base = w / 1000.0
    scale = base * text_scale
    radius, spacing, font_size = 15 * scale, 60 * scale, 40 * scale
    mx = my = 40 * base
    font = _font(max(8, int(font_size)))
    if getattr(font, "_ramfish_builtin", False) and label:
        label = label.replace("µ", "u")    # the built-in font has no µ
    names = [n for n, _, _ in layers]
    text_w = max(draw.textlength(n, font=font) for n in names)
    lw, lh = radius * 2 + 20 + text_w, len(names) * spacing
    x0 = w - lw - mx if "right" in legend_pos else mx
    y0 = (h - lh - my + spacing / 2) if "bottom" in legend_pos else my + spacing / 2
    for i, (n, _, color) in enumerate(layers):
        y = y0 + i * spacing
        draw.ellipse([x0, y - radius, x0 + 2 * radius, y + radius], fill=tuple(int(v) for v in color))
        draw.text((x0 + 2 * radius + 15 * scale, y), n, fill="white", font=font, anchor="lm")
    if label:   # the depth, at the opposite corner of the key
        lx = mx if "right" in legend_pos else w - mx
        ly = h - my if "bottom" not in legend_pos else my
        draw.text((lx, ly), label, fill="white", font=font, anchor=("ld" if "right" in legend_pos else "rd") if "bottom" not in legend_pos
                  else ("la" if "right" in legend_pos else "ra"))
    return np.asarray(out)


def _font(size: int):
    """Arial (Windows, macOS) or DejaVu Sans / Liberation Sans (Linux); Pillow's own font if none is found."""
    from PIL import ImageFont
    for name in ("arial.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
                 "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    font._ramfish_builtin = True
    return font


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in c)
