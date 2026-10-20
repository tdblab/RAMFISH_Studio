"""The five processing steps, using the functions taken from the original scripts."""

import numpy as np
import pytest
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from ramfish import fine_align as fa
from ramfish import rough_align as ra
from ramfish.io import load_stack, project_stack, to_16bit_scale
from ramfish.normalize import apply_ramfish_normalization, format_and_resize
from ramfish.project import Project
from ramfish.spots import DEFAULT_CONFIG, process_single_image


def ncc(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# ------------------------------------------------------------------ 1. import
def test_bit_depths_share_one_intensity_scale():
    assert to_16bit_scale(np.array([255], np.uint8))[0] == pytest.approx(65535)
    assert to_16bit_scale(np.array([4095], np.uint16))[0] == 4095


def test_tiff_axes_and_mip(tmp_path):
    vol = np.random.default_rng(0).integers(0, 1000, (5, 2, 32, 40), dtype=np.uint16)  # Z C Y X
    tifffile.imwrite(tmp_path / "s.tif", vol, imagej=True, metadata={"axes": "ZCYX"})
    stack = load_stack(tmp_path / "s.tif")
    assert stack.shape == (2, 5, 32, 40)
    np.testing.assert_array_equal(project_stack(stack, 1, "mip"), vol[:, 1].max(0))


@pytest.mark.parametrize("chans", [(2,), (0, 1), (0, 2)])   # blue mask, yellow, magenta exports
def test_pseudocolour_rgb_tiff_is_one_channel(tmp_path, chans):
    rgb = np.zeros((20, 30, 3), np.uint8)
    for c in chans:
        rgb[3:9, 4:20, c] = 180
    tifffile.imwrite(tmp_path / "m.tif", rgb, photometric="rgb")
    stack = load_stack(tmp_path / "m.tif")
    assert stack.shape == (1, 1, 20, 30) and stack[0, 0, 5, 5] == pytest.approx(180 * 257)


def test_rgb_composite_keeps_three_channels(tmp_path):
    rng = np.random.default_rng(3)   # two independent, strong signals (e.g. a gene in red, DAPI in blue)
    rgb = np.zeros((20, 30, 3), np.uint8)
    rgb[..., 0] = rng.integers(0, 256, (20, 30))
    rgb[..., 2] = rng.integers(0, 256, (20, 30))
    tifffile.imwrite(tmp_path / "c.tif", rgb, photometric="rgb")
    assert load_stack(tmp_path / "c.tif").shape == (3, 1, 20, 30)


def test_proprietary_formats_go_through_aicsimageio(tmp_path, monkeypatch):
    from ramfish import io as rio

    calls = []

    class FakeImg:
        def __init__(self, path, reader=None):
            calls.append((path.suffix, reader))

        def get_image_data(self, order):
            assert order == "CZYX"
            return np.ones((2, 3, 4, 5), np.uint16)

    monkeypatch.setattr(rio, "_AICS", (FakeImg, "BioformatsReader"))
    for ext in (".oir", ".czi"):
        f = tmp_path / f"x{ext}"
        f.write_bytes(b"0")
        assert load_stack(f).shape == (2, 3, 4, 5)
    assert calls == [(".oir", "BioformatsReader"), (".czi", None)]   # as in ramfish_processor.py


def test_normalization_is_the_original_formula(demo):
    img = to_16bit_scale(demo.images[1]["wg"])
    out = apply_ramfish_normalization(img, True, 30.0, 99.5, 60000.0)
    sub = np.maximum(img - gaussian_filter(img, 30.0), 0)
    np.testing.assert_allclose(out, sub * (60000.0 / np.percentile(sub, 99.5)), rtol=1e-5)
    assert format_and_resize(out, ".jpg", "Custom Scaling Factor", 0.5).shape == (320, 320)
    assert format_and_resize(out, ".tif", "Original Resolution", 1).dtype == np.uint16


# --------------------------------------------------------------- 2. rough
@pytest.mark.parametrize("rnd", [2, 3])
def test_rough_alignment_undoes_the_pose(demo, rnd):
    fixed = demo.images[1]["DAPI"]
    moving = demo.images[rnd]["DAPI"]
    aligned = ra.apply(moving, demo.poses[rnd])
    assert ncc(fixed, aligned) > 0.9 > ncc(fixed, moving)


def test_preview_scale_ratio():
    p = ra.preview(np.zeros((2000, 1500), np.uint16))
    assert (p["prev_w"], p["prev_h"]) == (750, 1000) and p["orig_w"] / p["prev_w"] == 2.0
    assert p["src"].startswith("data:image/jpeg;base64,")


# ----------------------------------------------------------------- 3. fine
@pytest.mark.parametrize("ref", ["DAPI", "mask"])
def test_fine_alignment_on_dapi_or_mask(ref):
    from synthetic import make_demo

    d = make_demo(with_mask=True)
    fixed = d.images[1][ref]
    rough = ra.apply(d.images[2][ref], d.poses[2])
    affine, bspline, ref_img = fa.compute_transforms(fixed, fa.match_fixed_shape_centered(rough, fixed.shape))
    gene_rough = ra.apply(d.images[2]["optix"], d.poses[2])
    gene = fa.apply_transforms(affine, bspline, ref_img, gene_rough)
    assert gene.dtype == np.uint16
    dapi_after = fa.apply_transforms(affine, bspline, ref_img, ra.apply(d.images[2]["DAPI"], d.poses[2]))
    assert ncc(d.images[1]["DAPI"], dapi_after) >= ncc(d.images[1]["DAPI"], ra.apply(d.images[2]["DAPI"], d.poses[2])) - 0.01


# ----------------------------------------------------------------- 4. spots
def test_spot_counter_outputs(tmp_path, demo):
    img = format_and_resize(apply_ramfish_normalization(to_16bit_scale(demo.images[1]["ci"]), True, 50, 99.5, 60000),
                            ".tif", "Original Resolution", 1)
    tifffile.imwrite(tmp_path / "ci.tif", img)
    dirs = {"csv": tmp_path / "csv", "img": tmp_path / "img", "synth": tmp_path / "synth"}
    for p in dirs.values():
        p.mkdir()
    res = process_single_image(tmp_path / "ci.tif", dirs, DEFAULT_CONFIG, "ci")
    assert res["status"] == "success"
    assert (dirs["img"] / "ci_val.jpg").exists() and (dirs["synth"] / "ci.jpg").exists()
    import pandas as pd

    df = pd.read_csv(dirs["csv"] / "ci_spots.csv")
    dist, _ = cKDTree(demo.truth_spots["ci"]).query(df[["x", "y"]].to_numpy())
    assert res["count"] == len(df) and (dist <= 2).mean() > 0.9
    assert process_single_image(tmp_path / "ci.tif", dirs, DEFAULT_CONFIG, "DAPI")["status"] == "skipped"


# --------------------------------------------------------------- project
def test_alignment_channel_dapi_then_mask_and_rename(tmp_path):
    p = Project(tmp_path)
    z = np.zeros((8, 8))
    for ch in ("DAPI", "mask", "wg"):
        p.add_image(1, ch, z)
    for ch in ("DAPI", "mask", "ci"):
        p.add_image(2, ch, z)
    for ch in ("Mask", "dll"):
        p.add_image(3, ch, z)
    p.add_image(4, "omb", z)
    assert p.alignment_options(2) == ["DAPI", "mask"] and p.alignment_channel(2) == "DAPI"
    assert p.alignment_channel(3) == "Mask" and p.reference_partner(3, "Mask") == "mask"
    assert p.alignment_channel(4) is None
    assert sorted(lbl for _, _, lbl in p.gene_targets()) == ["ci", "dll", "omb", "wg"]
    assert p.rename_channel(4, "omb", "optix") == "optix" and p.has("normalized", 4, "optix")
    assert p.background_channel() == "DAPI"


def test_orange_pseudocolour_is_one_channel(tmp_path):
    """Orange = full red + 65% green: one signal at two strengths, not two channels."""
    sig = np.random.default_rng(1).integers(0, 255, (40, 50)).astype(np.float32)
    rgb = np.zeros((40, 50, 3), np.uint8)
    rgb[..., 0] = sig.astype(np.uint8)
    rgb[..., 1] = np.round(sig * 0.65).astype(np.uint8)
    tifffile.imwrite(tmp_path / "o.tif", rgb, photometric="rgb")
    assert load_stack(tmp_path / "o.tif").shape == (1, 1, 40, 50)


def test_mask_rounds_can_skip_bending():
    from synthetic import make_demo

    d = make_demo(with_mask=True)
    fixed, rough = d.images[1]["mask"], ra.apply(d.images[2]["mask"], d.poses[2])
    affine, bspline, ref = fa.compute_transforms(fixed, rough, bspline=False)
    assert bspline.GetTransformEnum() == fa.sitk.sitkIdentity
    out = fa.apply_transforms(affine, bspline, ref, ra.apply(d.images[2]["DAPI"], d.poses[2]))
    assert ncc(d.images[1]["DAPI"], out) > 0.8


def test_faint_colour_trace_is_not_a_channel(tmp_path):
    """Blue signal + a green trace of at most 3/255 + empty red (a real Round 6 export) = one image."""
    rng = np.random.default_rng(2)
    rgb = np.zeros((40, 50, 3), np.uint8)
    rgb[..., 2] = rng.integers(0, 256, (40, 50))
    rgb[..., 1] = (rgb[..., 2] // 90).astype(np.uint8)   # 0..2, loosely follows blue
    tifffile.imwrite(tmp_path / "ci.tif", rgb, photometric="rgb")
    stack = load_stack(tmp_path / "ci.tif")
    assert stack.shape == (1, 1, 40, 50)
    np.testing.assert_array_equal(stack[0, 0], rgb[..., 2].astype(np.float32) * 257)
