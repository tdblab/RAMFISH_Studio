import io
import zipfile

import numpy as np
import pytest
import tifffile

from ramfish import dataset


@pytest.mark.parametrize("path, expected", [
    ("Round_1/DAPI.tif", (1, "DAPI")),
    ("RAMFISH_Project/Round_12/wg.tiff", (12, "wg")),
    ("data/round3/optix.ome.tif", (3, "optix")),
    ("R07/ALIGNED_spalt.tif", (7, "spalt")),
    ("Cycle 2/ci.png", (2, "ci")),
    ("R1_DAPI.tif", (1, "DAPI")),
    ("round-14_dll.tif", (14, "dll")),
    ("Round_2/sub/gene A.tif", (2, "gene_A")),
    ("loose_image.tif", (None, "loose_image")),
    ("rho1.tif", (None, "rho1")),
    ("rawdata/Round_3/3_Dll.tif", (3, "Dll")),
    ("rawdata/Round_7/7_mask.tif", (7, "mask")),
    ("Round_2/5_gene.tif", (2, "5_gene")),   # number differs from the round: keep it
])
def test_parse_name(path, expected):
    assert dataset.parse_name(path) == expected


def test_scan_skips_junk_and_finds_reference():
    names = ["Round_1/DAPI.tif", "Round_1/wg.tif", "__MACOSX/Round_1/._wg.tif", "Round_1/.DS_Store",
             "Round_2/dapi_round2.tif", "Round_2/optix.tif", "Round_2/notes.txt"]
    entries = dataset.scan(names + ["Round_2/tissue_Mask.tif"])
    got = {(e.round, e.channel) for e in entries}
    assert got == {(1, "DAPI"), (1, "wg"), (2, "DAPI"), (2, "optix"), (2, "mask")}


def test_scan_does_not_guess_ambiguous_reference():
    entries = dataset.scan(["Round_1/DAPI_a.tif", "Round_1/DAPI_b.tif"])
    assert {e.channel for e in entries} == {"DAPI_a", "DAPI_b"}


def _zip(files: dict[str, np.ndarray]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, arr in files.items():
            b = io.BytesIO()
            tifffile.imwrite(b, arr)
            zf.writestr(name, b.getvalue())
    buf.seek(0)
    return buf


def test_zip_source_reads_14_rounds():
    rng = np.random.default_rng(0)
    files = {f"exp/Round_{r}/{c}.tif": rng.integers(0, 4000, (32, 32), dtype=np.uint16)
             for r in range(1, 15) for c in ("DAPI", f"gene{r}")}
    src = dataset.ZipSource(_zip(files))
    entries = dataset.scan(src.names())
    assert sorted({e.round for e in entries}) == list(range(1, 15))
    e = next(e for e in entries if e.round == 14 and e.channel == "gene14")
    stack = dataset.load_stack_bytes(e.name, src.read(e.name))
    np.testing.assert_array_equal(stack[0, 0], files[e.name])


def test_folder_source(tmp_path):
    (tmp_path / "Round_1").mkdir()
    tifffile.imwrite(tmp_path / "Round_1" / "DAPI.tif", np.ones((8, 8), np.uint16))
    src = dataset.FolderSource(tmp_path)
    assert [(e.round, e.channel) for e in dataset.scan(src.names())] == [(1, "DAPI")]
