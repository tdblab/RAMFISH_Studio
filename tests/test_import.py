"""Step 1 from multi-channel microscope files: raw TIFFs, gene names, tissue masks and matched depths."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy.ndimage import gaussian_filter, rotate

from ramfish import importer
from ramfish.channels import HOW_DYE_NAME, HOW_WAVELENGTH
from ramfish.project import Project

NORM = dict(do_norm=True, blur_sigma=50.0, percentile_val=99.5, target_intensity=60000.0)
RES = dict(res_mode="Original Resolution", custom_scale=1.0)


def ome_tif(path, czyx, names, fluors, excitations, pixel_um=(1.0, 0.5, 0.5)):
    tifffile.imwrite(str(path), czyx.astype(np.uint16), ome=True, metadata={
        "axes": "CZYX", "PhysicalSizeZ": pixel_um[0], "PhysicalSizeY": pixel_um[1], "PhysicalSizeX": pixel_um[2],
        "Channel": {"Name": names, "Fluor": fluors, "ExcitationWavelength": excitations}})
    return path


def wing(shape=(256, 256), shift=(0, 0), seed=0):
    """A bright blob of tissue on a dark background, with a few gene patterns inside."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    tissue = (((yy - 128 - shift[0]) / 90.0) ** 2 + ((xx - 120 - shift[1]) / 70.0) ** 2) < 1
    chans = []
    for k in range(4):
        img = 300.0 * tissue + rng.normal(100, 10, shape)
        img += 2500.0 * tissue * (np.sin((xx + 20 * k) / 12.0) > 0.7)
        chans.append(gaussian_filter(img, 1.0))
    return np.stack(chans)[:, None], tissue


@pytest.fixture()
def wing_files(tmp_path):
    # channels stored 488, 514, 546, 647; the names list the genes in another order
    a, t1 = wing(seed=1)
    b, t2 = wing(shift=(10, -8), seed=2)
    f1 = ome_tif(tmp_path / "round1_dpp_alexa546__optix_alexa647__omb_alexa488__hh_alexa514__1.ome.tif", a,
                 ["CH1", "CH2", "CH3", "CH4"], ["Alexa Fluor 488", "Alexa Fluor 514", "Alexa Fluor 546", "Alexa Fluor 647"],
                 [500.0, 530.0, 570.0, 650.0])
    f2 = ome_tif(tmp_path / "round2_wg_alexa546__fz4_alex647__fz9_alexa488__ptc_alex514_.ome.tif", b,
                 ["CH1", "CH2", "CH3", "CH4"], ["", "", "", ""], [500.0, 530.0, 570.0, 650.0])
    return [f1, f2], (t1, t2)


def test_wing_rounds_without_dapi(tmp_path, wing_files):
    files, (t1, t2) = wing_files
    p = Project(tmp_path / "proj")
    reads = [importer.read_file(p, f, f.name, fallback_round=7) for f in files]
    assert [r.round for r in reads] == [1, 2]                 # from the file names, not the Round box
    assert [c["Gene / channel"] for c in reads[0].channels] == ["omb", "hh", "dpp", "optix"]
    assert {c["Matched by"] for c in reads[0].channels} == {HOW_DYE_NAME}
    assert [c["Gene / channel"] for c in reads[1].channels] == ["fz9", "ptc", "wg", "fz4"]
    assert {c["Matched by"] for c in reads[1].channels} == {HOW_WAVELENGTH}

    raw = p.path("raw", 1, "dpp")
    assert raw.exists() and "1_import_and_normalize/raw/Round_1" in raw.as_posix()
    assert np.allclose(p.load_raw(1, "dpp")[0], tifffile.imread(files[0])[2])   # untouched intensities

    warns = importer.build(p, importer.BuildSettings(norm=NORM, res=RES))
    assert not warns
    for r, truth in ((1, t1), (2, t2)):
        assert p.channels(r)[-1] == "mask"
        m = p.load(r, "mask", stage="normalized") > 0
        dice = 2 * (m & truth).sum() / (m.sum() + truth.sum())
        assert dice > 0.9, dice
    assert p.alignment_options(2) == ["mask"]
    exports = p.step_dir("exports") / "Round_1"
    assert (exports / "NORM_R1_dpp.tif").exists() and (exports / "MASK_R1.tif").exists()

    # settings changed afterwards: no mask any more, nothing re-read from the microscope files
    importer.build(p, importer.BuildSettings(norm=NORM, res=RES, mask_mode="never"))
    assert "mask" not in p.channels(1) and p.raw_channels(1) == ["omb", "hh", "dpp", "optix"]

    # renaming a gene renames its raw TIFF too
    p.rename_channel(2, "wg", "Wnt1")
    assert p.path("raw", 2, "Wnt1").exists() and p.raw_info(2, "Wnt1")["dye"] == "alexa546"


def test_mask_from_one_dye(tmp_path, wing_files):
    files, _ = wing_files
    p = Project(tmp_path / "proj")
    for f in files:
        importer.read_file(p, f, f.name, None)
    warns = importer.build(p, importer.BuildSettings(norm=NORM, res=RES, mask_source="alexa488"))
    assert not warns and p.load(2, "mask", stage="normalized").max() == 65535


def brain(z_um, seed=0):
    """Thick tissue whose outline and cell density change with depth (positions in micrometers)."""
    rng = np.random.default_rng(seed)
    H = W = 128
    yy, xx = np.mgrid[0:H, 0:W]
    n = 4000
    nz, ny, nx = rng.uniform(0, 160, n), rng.uniform(0, H, n), rng.uniform(0, W, n)
    keep = rng.uniform(0, 1, n) < (0.35 + 0.65 * np.exp(-((nz - 70) / 18) ** 2))
    nz, ny, nx = nz[keep], ny[keep], nx[keep]
    planes = []
    for z in z_um:
        ry, rx = 20 + 35 * np.exp(-((z - 50) / 60) ** 2), 15 + 45 * np.exp(-((z - 40) / 50) ** 2)
        tissue = ((yy - 58) / ry) ** 2 + ((xx - 70) / rx) ** 2 < 1
        img = 0.15 * tissue
        sel = np.abs(nz - z) < 4
        iy, ix = ny[sel].astype(int), nx[sel].astype(int)
        ok = tissue[iy, ix]
        np.add.at(img, (iy[ok], ix[ok]), 40 * np.exp(-((nz[sel][ok] - z) / 2.5) ** 2))
        planes.append(gaussian_filter(img, 1.5))
    return np.stack(planes)


def test_thick_stacks_use_the_same_depth(tmp_path):
    rng = np.random.default_rng(3)
    ref = brain(np.arange(10, 130.1, 2.0))                    # Round 1: 61 planes, 2 µm apart
    mov = brain(np.arange(30, 150.1, 3.0))                    # Round 2: 41 planes, 3 µm apart, deeper start
    mov = np.stack([np.fliplr(rotate(m, 90, reshape=False, order=1)) for m in mov])   # turned and mirrored
    to_u16 = lambda v, gain: np.clip(v * 1000 * gain + rng.normal(100, 8, v.shape), 0, 65535)
    gene = lambda v: v * 0.5
    f1 = ome_tif(tmp_path / "fish1_round1_gad2_alexa546.ome.tif", np.stack([to_u16(ref, 1), to_u16(gene(ref), 1)]),
                 ["CH1", "CH2"], ["DAPI", "Alexa Fluor 546"], [405.0, 561.0], pixel_um=(2.0, 0.5, 0.5))
    f2 = ome_tif(tmp_path / "fish1_round2_oxt_alexa546.ome.tif", np.stack([to_u16(mov, 1.7), to_u16(gene(mov), 1.7)]),
                 ["CH1", "CH2"], ["DAPI", "Alexa Fluor 546"], [405.0, 561.0], pixel_um=(3.0, 0.5, 0.5))
    p = Project(tmp_path / "proj")
    for f in (f1, f2):
        r = importer.read_file(p, f, f.name, None)
        assert [c["Gene / channel"] for c in r.channels][0] == "DAPI"
    s = importer.BuildSettings(norm=NORM, res=RES, depth_mode=importer.DEPTH_MATCH, ref_plane=30, slab_um=6.0)
    importer.build(p, s)
    d = p.meta["depth"]
    assert d["1"]["z"] == 30                                  # 70 µm
    assert abs(d["2"]["z"] - 13) <= 1, d["2"]                 # 70 µm in Round 2 = (70 - 30) / 3
    assert (d["2"]["z0"], d["2"]["z1"]) == (d["2"]["z"] - 1, d["2"]["z"] + 1)
    assert "mask" not in p.channels(1)                        # DAPI in every round: no mask needed
    assert any(f.name.startswith("NORM_R2_DAPI_SLAB_") for f in (p.step_dir("exports") / "Round_2").iterdir())

    s.manual_planes = {2: 20}                                 # a plane typed in by the user wins
    importer.build(p, s)
    assert p.meta["depth"]["2"]["z"] == 20 and not p.meta["depth"]["2"]["auto"]


def test_old_projects_are_moved_to_the_new_folder(tmp_path):
    root = tmp_path / "old"
    (root / "1_normalized" / "Round_1").mkdir(parents=True)
    (root / "1_normalized" / "exports" / "Round_1").mkdir(parents=True)
    tifffile.imwrite(str(root / "1_normalized" / "Round_1" / "DAPI.tif"), np.ones((8, 8), np.uint16))
    (root / "1_normalized" / "exports" / "Round_1" / "NORM_x.tif").write_bytes(b"x")
    (root / ".ramfish_project.json").write_text('{"name": "old", "rounds": {"1": ["DAPI"]}, "history": []}')
    p = Project(root)
    assert not (root / "1_normalized").exists()
    assert p.load(1, "DAPI").shape == (8, 8)
    assert (p.step_dir("exports") / "Round_1" / "NORM_x.tif").exists()


def test_import_page_with_raw_images(tmp_path, monkeypatch, wing_files):
    from streamlit.testing.v1 import AppTest
    files, _ = wing_files
    monkeypatch.setenv("RAMFISH_PROJECT_DIR", str(tmp_path / "proj"))
    p = Project(tmp_path / "proj")
    for f in files:
        importer.read_file(p, f, f.name, None)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_pages" / "import_normalize.py"),
                           default_timeout=300).run()
    assert not at.exception
    next(b for b in at.button if b.label.startswith("🔁 Normalize again")).click()
    at = at.run()
    assert not at.exception
    p = Project(tmp_path / "proj")
    assert p.channels(1) == ["omb", "hh", "dpp", "optix", "mask"]
    assert any("Channels" in s.value for s in at.subheader)


def test_images_are_enlarged_to_4k_before_normalization(tmp_path):
    from ramfish.normalize import RES_4K, resize_plane
    a, _ = wing(shape=(256, 192), seed=4)
    f = ome_tif(tmp_path / "round1_dpp_alexa546__optix_alexa647__omb_alexa488__hh_alexa514.ome.tif", a,
                ["CH1", "CH2", "CH3", "CH4"], ["Alexa Fluor 488", "Alexa Fluor 514", "Alexa Fluor 546", "Alexa Fluor 647"],
                [500.0, 530.0, 570.0, 650.0])
    p = Project(tmp_path / "proj")
    importer.read_file(p, f, f.name, None)
    importer.build(p, importer.BuildSettings(norm=NORM))          # default: 4K
    assert importer.BuildSettings(norm=NORM).res["res_mode"] == RES_4K
    img = tifffile.imread(p.path("normalized", 1, "dpp"))
    assert img.shape == (4096, 3072) and img.dtype == np.uint16    # longest side 4096, shape kept, 16-bit TIFF
    m = tifffile.imread(p.path("normalized", 1, "mask"))
    assert m.shape == (4096, 3072) and set(np.unique(m)) <= {0, 65535}
    assert p.load_raw(1, "dpp").shape == (1, 256, 192)              # the raw TIFF keeps the microscope's pixels
    assert tifffile.imread(p.step_dir("exports") / "Round_1" / "NORM_R1_dpp.tif").shape == (4096, 3072)
    # enlarging never makes negative values, and keeps the brightness of a flat area
    flat = resize_plane(np.full((10, 20), 500.0), RES_4K)
    assert flat.shape == (2048, 4096) and np.allclose(flat, 500.0, atol=1)


def test_several_zips_read_as_one(tmp_path):
    """One zip per round (the built-in example) is read as one dataset."""
    import zipfile
    from ramfish import dataset
    for r in (1, 10):
        with zipfile.ZipFile(tmp_path / f"Round_{r}.zip", "w") as z:
            z.writestr(f"Round_{r}/", "")
            z.writestr(f"Round_{r}/{r}_mask.tif", b"x")
            z.writestr(f"Round_{r}/{r}_omb.tif", b"y")
    src = dataset.MultiZipSource(sorted(tmp_path.glob("*.zip")))
    assert sorted(src.names()) == ["Round_1/1_mask.tif", "Round_1/1_omb.tif", "Round_10/10_mask.tif", "Round_10/10_omb.tif"]
    assert src.read("Round_10/10_omb.tif") == b"y"
    src.copy_to("Round_1/1_mask.tif", tmp_path / "m.tif")
    assert (tmp_path / "m.tif").read_bytes() == b"x"
    assert {(e.round, e.channel) for e in dataset.scan(src.names())} == {(1, "mask"), (1, "omb"), (10, "mask"), (10, "omb")}
