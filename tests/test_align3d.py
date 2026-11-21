"""3D rough alignment: moving a round's volume onto Round 1's, the first guess and the z-planes."""

import cv2
import numpy as np
import pytest

from ramfish import align3d as a3, importer, sections as sec
from ramfish.project import Project
from test_sections import NORM, RES, brain, ome


def turned(stack, ang, dx, dy):
    h, w = stack.shape[-2:]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[:, 2] += (dx, dy)
    return np.stack([cv2.warpAffine(p.astype(np.float32), M, (w, h)) for p in stack])


@pytest.fixture()
def moved(tmp_path):
    """Round 2 starts 6 µm deeper with other plane spacing; Round 3 is also turned by 25° and shifted."""
    rng = np.random.default_rng(3)
    def make(z_um, step, name, ang=0.0, dx=0.0, dy=0.0):
        g1, g2 = brain(z_um, shape=(180, 220))
        if ang or dx or dy:
            g1, g2 = turned(g1, ang, dx, dy), turned(g2, ang, dx, dy)
        data = np.stack([g1, g2]) * 3000 + rng.normal(100, 5, (2,) + g1.shape)
        return ome(tmp_path / name, np.clip(data, 0, 65535), step)
    files = [make(np.arange(0, 40.1, 3.0), 3.0, "round1_gad2_alexa546__oxt_alexa488.ome.tif"),
             make(np.arange(6, 46.1, 2.5), 2.5, "round2_tph2_alexa546__omp_alexa488.ome.tif"),
             make(np.arange(-6, 30.1, 3.0), 3.0, "round3_chatb_alexa546__elavl3_alexa488.ome.tif", 25.0, 12.0, -8.0)]
    p = Project(tmp_path / "proj")
    for f in files:
        importer.read_file(p, f, f.name, None)
    importer.build(p, importer.BuildSettings(norm=NORM, res=RES))
    return p


def test_geometry_round_trip():
    g0 = {"n": 10, "h": 100, "w": 120, "px": 0.5, "dz": 3.0}
    gr = {"n": 14, "h": 90, "w": 110, "px": 0.6, "dz": 2.5}
    p = dict(a3.DEFAULT, tx=5, ty=-3, tz=4, rot=30, tilt_x=5, tilt_y=-4, scale=1.1, flip=True)
    pts = np.random.default_rng(0).uniform(0, 50, (20, 3))
    back = a3.to_ref(pts, gr, g0, p)
    A, b = a3.from_ref_affine(gr, g0, p)
    assert np.allclose(back @ A.T + b, pts)


def test_first_guess_finds_depth_turn_and_shift(moved):
    p = moved
    g0 = a3.geometry(p, 1)
    v0, gs0 = a3.small_volume(p, 1)
    for r, tz_true in ((2, 6.0 + 20.0 - 19.5),     # plane 0 of Round 2 sits at 6 µm in Round 1
                       (3, -6.0 + 18.0 - 19.5)):   # Round 3 starts 6 µm higher
        q = a3.auto_guess(p, r)
        assert abs(q["tz"] - tz_true) <= 3.0, (r, q)
        vr, gsr = a3.small_volume(p, r)
        scores = []
        for z in range(3, 10):
            cut = a3.cut_plane(vr, gsr, gs0, q, z * g0["dz"], v0.shape[1:], gs0["px"])
            if cut is not None:
                scores.append(a3._corr(v0[z], cut))
        assert np.mean(scores) > 0.8, (r, q, scores)


def test_kept_planes_and_zplanes(moved):
    p = moved
    params = {str(r): a3.auto_guess(p, r) for r in (2, 3)}
    keep = a3.kept_planes(p, params)
    # Round 1 covers 0-39 µm, Round 2 6-46, Round 3 -6-30: all three overlap at about 6-30 µm
    depths = [z * 3.0 for z in keep]
    assert min(depths) >= 3.0 and max(depths) <= 33.0 and len(keep) >= 6, depths
    subs = a3.build_zplanes(p, params)
    assert len(subs) == len(keep) and all(s.meta["section"]["aligned_3d"] for s in subs)
    assert sec.section_projects(p)[0].rounds() == [1, 2, 3]
    assert (p.step_dir("rough") / "planes" / "Round_3").is_dir()
    # tilted cut: still a picture, inside the stack
    q = dict(params["3"], tilt_x=3.0)
    vr, gsr = a3.small_volume(p, 3)
    v0, gs0 = a3.small_volume(p, 1)
    assert a3.cut_plane(vr, gsr, gs0, q, 18.0, v0.shape[1:], gs0["px"]) is not None


def test_left_out_planes_drop_their_depths(moved):
    p = moved
    params = {str(r): a3.auto_guess(p, r) for r in (2, 3)}
    keep = a3.kept_planes(p, params)
    first = keep[0]
    p.meta["rough3d_skip"] = {"1": [first]}          # a bad plane of Round 1
    p.save()
    keep2 = a3.kept_planes(p, params)
    assert first not in keep2 and len(keep2) == len(keep) - 1
    z = keep2[2]                                     # a bad plane of Round 2: the depth that uses it goes
    zr = int(round(a3.matching_plane(p, 2, params["2"], z * 3.0)))
    p.meta["rough3d_skip"] = {"1": [first], "2": [zr]}
    p.save()
    keep3 = a3.kept_planes(p, params)
    assert z not in keep3 and set(keep3) < set(keep2)
    subs = a3.build_zplanes(p, params, export=False)
    assert len(subs) == len(keep3)
