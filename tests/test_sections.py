"""Sections of 3D stacks: matching planes across rounds and one small project per section."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy.ndimage import gaussian_filter

from ramfish import importer, sections as sec
from ramfish.project import Project

NORM = dict(do_norm=True, blur_sigma=50.0, percentile_val=99.5, target_intensity=60000.0)
RES = dict(res_mode="Original Resolution", custom_scale=1.0)


def brain(z_um, seed=0, shape=(160, 200)):
    """Tissue whose outline and patterns change with depth (positions in micrometers); two gene channels."""
    rng = np.random.default_rng(seed)
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    blobs = [(rng.uniform(0, 60), rng.uniform(20, H - 20), rng.uniform(20, W - 20), rng.uniform(6, 14)) for _ in range(60)]
    g1, g2 = [], []
    for z in z_um:
        ry, rx = 45 + 25 * np.exp(-((z - 20) / 25) ** 2), 60 + 30 * np.exp(-((z - 15) / 20) ** 2)
        tissue = ((yy - H / 2) / ry) ** 2 + ((xx - W / 2) / rx) ** 2 < 1
        a = 0.2 * tissue.astype(float)
        for bz, by, bx, br in blobs:
            w = np.exp(-((bz - z) / 4.0) ** 2)
            if w > 0.02:
                a += w * np.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2 * br ** 2))) * tissue
        g1.append(gaussian_filter(a, 1.0))
        g2.append(gaussian_filter(np.roll(a, 7, axis=1) * 0.7 + 0.1 * tissue, 1.0))
    return np.stack(g1), np.stack(g2)


def ome(path, czyx, step):
    tifffile.imwrite(str(path), czyx.astype(np.uint16), ome=True, metadata={
        "axes": "CZYX", "PhysicalSizeZ": step, "PhysicalSizeY": 1.0, "PhysicalSizeX": 1.0,
        "Channel": {"Name": ["CH1", "CH2"], "Fluor": ["Alexa Fluor 546", "Alexa Fluor 488"], "ExcitationWavelength": [570.0, 500.0]}})
    return path


@pytest.fixture()
def stacks(tmp_path):
    rng = np.random.default_rng(1)
    def make(z_um, step, name, seed):
        g1, g2 = brain(z_um)
        data = np.stack([g1, g2]) * 3000 + rng.normal(100, 5, (2,) + g1.shape)
        return ome(tmp_path / name, np.clip(data, 0, 65535), step)
    f1 = make(np.arange(0, 40.1, 3.0), 3.0, "round1_gad2_alexa546__oxt_alexa488.ome.tif", 1)        # 14 planes
    f2 = make(np.arange(6, 46.1, 2.5), 2.5, "round2_tph2_alexa546__omp_alexa488.ome.tif", 2)       # starts 6 um deeper
    p = Project(tmp_path / "proj")
    for f in (f1, f2):
        importer.read_file(p, f, f.name, None)
    importer.build(p, importer.BuildSettings(norm=NORM, res=RES))
    return p


def test_depths_and_ridge():
    assert sec.section_depths(11, 3.19, 3.0, 10.0) == [3.0, 13.0, 23.0]
    M = np.full((5, 6), 0.0)
    for i, j in enumerate([1, 1, 2, 4, 5]):
        M[i, j] = 1.0
    M[2, 0] = 0.9                      # a better-looking plane that would break the depth order
    path, score = sec.follow_ridge(M)
    assert path == [1, 1, 2, 4, 5] and score > 0.9


def test_matching_planes_follow_the_depth(stacks):
    p = stacks
    s = sec.suggest_planes(p, sec.COMPARE_GENES)
    m = s["rounds"]["2"]
    assert not m["upside_down"] and m["xy"] == "automatic"
    # Round 1 plane at 21 um is 15 um into Round 2's stack, which starts at 6 um: plane 6
    assert abs(m["planes"][7] - 6) <= 1, m["planes"]
    assert m["planes"] == sorted(m["planes"])          # depth order kept
    assert sec.planes_for_depth(p, s, 21.0)[1] == 7


def test_sections_become_projects(stacks):
    p = stacks
    s = sec.suggest_planes(p)
    plan = [{"depth_um": d, "planes": sec.planes_for_depth(p, s, d)} for d in sec.section_depths(14, 3.0, 6.0, 10.0)]
    p.meta.setdefault("rough_params", {})["2"] = {"tx": 3.0, "ty": -2.0, "rotation": 0.0, "img_scale": 1.0, "flip": False,
                                                  "scale_ratio": 1.0}
    subs = sec.build_sections(p, plan, slab_um=6.0)
    assert [x.root.name for x in subs] == ["S1_6um", "S2_16um", "S3_26um", "S4_36um"]
    sub = subs[1]
    assert sub.rounds() == [1, 2] and sub.channels(2) == ["tph2", "omp", "mask"]     # no DAPI: a mask per round
    assert sub.meta["section"]["planes"] == {"1": plan[1]["planes"][1], "2": plan[1]["planes"][2]}
    img = tifffile.imread(sub.path("normalized", 2, "tph2"))
    assert img.dtype == np.uint16 and img.shape == (160, 200)
    assert sub.stage_of(2, "tph2") == "rough"                # step 2's alignment was copied
    assert sec.section_projects(p)[0].root.name == "S1_6um"
    # the main project is untouched
    assert p.channels(1) == ["gad2", "oxt", "mask"] and p.stage_of(2, "tph2") == "normalized"


def test_step2_in_3d_mode_makes_zplanes(stacks, monkeypatch):
    """Step 2 shows the 3D aligner for stacks: suggest, save, make the z-planes; steps 3 and 4 then run on one."""
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    p = stacks
    monkeypatch.setenv("RAMFISH_PROJECT_DIR", str(p.root))
    at = AppTest.from_file(str(root / "app_pages" / "rough_align.py"), default_timeout=600).run()
    assert not at.exception
    assert at.radio(key="data_kind_pick").value.startswith("3D")
    next(b for b in at.button if b.label.startswith("🔮 Suggest")).click()
    at = at.run()
    assert not at.exception
    assert abs(at.slider(key="a3_2_tz").value - 6.5) <= 3.0     # Round 2 starts 6 µm deeper
    next(b for b in at.button if b.label.startswith("✅ Save Round 2")).click()
    at = at.run()
    assert not at.exception and "2" in Project(p.root).meta["rough3d"]
    next(b for b in at.button if b.label.startswith("🧊 Make the z-planes")).click()
    at = at.run()
    assert not at.exception
    subs = sec.section_projects(Project(p.root))
    assert len(subs) >= 8
    monkeypatch.setenv("RAMFISH_PROJECT_DIR", str(subs[0].root))
    for page, button in (("fine_align", "Run Batch Alignment"), ("spot_counter", "Run spot counter")):
        at = AppTest.from_file(str(root / "app_pages" / f"{page}.py"), default_timeout=300).run()
        next(b for b in at.button if b.label.startswith(button)).click()
        at = at.run()
        assert not at.exception, page
    assert (subs[0].step_dir("spots") / "data_sheets" / "BATCH_SUMMARY_COUNTS.csv").exists()
    assert not sec.combined_counts(Project(p.root)).empty


def test_all_sections_and_merge_across_depth(stacks, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from ramfish import depth as dp
    p = stacks
    s = sec.suggest_planes(p)
    plan = [{"depth_um": d, "planes": sec.planes_for_depth(p, s, d)} for d in sec.section_depths(14, 3.0, 6.0, 10.0)]
    subs = sec.build_sections(p, plan, slab_um=6.0)
    dp.run_sections(subs)
    for sub in subs:   # steps 3 and 4 ran on every section, as the pages would have
        assert (sub.step_dir("spots") / "data_sheets" / "BATCH_SUMMARY_COUNTS.csv").exists()
        assert sub.stage_of(2, "tph2") == "fine"
    table = sec.combined_counts(p)
    assert sorted(table["Section"].unique()) == ["S1_6um", "S2_16um", "S3_26um", "S4_36um"]
    # one merged picture per depth, drawn like the merge page
    names = dp.layer_names(subs[1])
    assert names == ["gad2", "oxt", "tph2", "omp", "mask"]  # as on the merge page: no DAPI, so the mask is the background
    lay = [(n, dp.layer_image(subs[1], n, dp.SOURCE_ALIGNED), dp.PALETTE[i]) for i, n in enumerate(names[:4])]
    pic = dp.composite(lay, label="16 µm")
    assert pic.shape == (160, 200, 3) and pic.dtype == np.uint8 and pic.max() > 0
    spots = [(n, dp.layer_image(subs[1], n, dp.SOURCE_SPOTS), (255, 0, 0)) for n in names[:4]]
    assert all(img is not None for _, img, _ in spots)
    # the merge page: Z slider and one picture per depth
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("RAMFISH_PROJECT_DIR", str(p.root))
    at = AppTest.from_file(str(root / "app_pages" / "merge.py"), default_timeout=600).run()
    assert not at.exception
    at.select_slider(key="z_idx_merge").set_value(2).run()
    assert not at.exception
    next(b for b in at.button if b.label.startswith("Prepare all depths")).click()
    at = at.run()
    assert not at.exception
    assert len(list(Project(p.root).step_dir("merge").glob("depth_*.jpg"))) == 4


def test_every_z_plane_with_z_slider_on_steps_3_to_5(stacks, monkeypatch):
    """Every plane of Round 1 becomes a z-plane; steps 3, 4 and 5 show a Z slider and run on all planes."""
    from streamlit.testing.v1 import AppTest
    p = stacks
    s = sec.suggest_planes(p)
    n0, step0 = sec.n_planes(p, 1), sec.z_step(p, 1)
    depths = sec.section_depths(n0, step0, 0.0, step0)
    assert len(depths) == n0
    subs = sec.build_sections(p, [{"depth_um": d, "planes": sec.planes_for_depth(p, s, d)} for d in depths[:3]], slab_um=step0)
    assert [x.root.name for x in sec.section_projects(p)] == [x.root.name for x in subs]   # earlier sections replaced
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("RAMFISH_PROJECT_DIR", str(p.root))
    at = AppTest.from_file(str(root / "app_pages" / "fine_align.py"), default_timeout=900).run()
    assert not at.exception
    assert at.select_slider(key="z_idx_fine").value == 0
    next(b for b in at.button if b.label.startswith("Run Batch Alignment on all")).click()
    at = at.run()
    assert not at.exception
    assert all(x.stage_of(2, "tph2") == "fine" for x in subs)
    at = AppTest.from_file(str(root / "app_pages" / "spot_counter.py"), default_timeout=900).run()
    assert not at.exception
    next(b for b in at.button if b.label.startswith("Run spot counter on all")).click()
    at = at.run()
    assert not at.exception
    assert all((x.step_dir("spots") / "data_sheets" / "BATCH_SUMMARY_COUNTS.csv").exists() for x in subs)
    assert any("every z-plane" in str(h.value) for h in at.subheader)
    at = AppTest.from_file(str(root / "app_pages" / "merge.py"), default_timeout=900).run()
    assert not at.exception
    at.select_slider(key="z_idx_merge").set_value(2).run()
    assert not at.exception
    assert len(at.image) >= 1, "the merged picture of the chosen z-plane is shown"


def test_matching_planes_on_the_tissue_outline(stacks):
    """The default: planes matched on each plane's tissue outline."""
    p = stacks
    s = sec.suggest_planes(p)
    assert s["use"] == sec.COMPARE_MASK
    m = s["rounds"]["2"]
    assert m["compared"] == sec.COMPARE_MASK and not m["upside_down"]
    assert abs(m["planes"][7] - 6) <= 2, m["planes"]     # Round 1 at 21 um = Round 2 plane 6 (it starts at 6 um)
    assert m["planes"] == sorted(m["planes"])
