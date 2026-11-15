"""Save a whole project as one .ramfish file and open it again."""

import zipfile

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ramfish.project import META, PROJECT_EXT, Project, load_project_file, save_project_file
from test_app import ROOT


def _small_project(root):
    p = Project(root / "wing_2026")
    p.add_image(1, "DAPI", np.full((40, 50), 1000, np.uint16))
    p.add_image(1, "omb", np.arange(2000, dtype=np.uint16).reshape(40, 50))
    p.add_image(2, "DAPI", np.full((40, 50), 900, np.uint16))
    p.write_stage("rough", 2, "DAPI", np.full((40, 50), 800, np.uint16))
    p.log("test", {"a": 1})
    return p


def test_save_and_load_round_trip(tmp_path):
    p = _small_project(tmp_path / "A")
    out = save_project_file(p, tmp_path / f"wing_2026{PROJECT_EXT}")
    assert zipfile.is_zipfile(out)
    dest = load_project_file(out, tmp_path / "B")
    q = Project(dest)
    assert q.name == "wing_2026"
    assert q.rounds() == [1, 2] and q.channels(1) == ["DAPI", "omb"]
    assert np.array_equal(q.load(1, "omb"), p.load(1, "omb"))
    assert q.has("rough", 2, "DAPI")
    assert q.meta["history"][-1]["step"] == "test"


def test_load_keeps_existing_project(tmp_path):
    p = _small_project(tmp_path)
    data = save_project_file(p)                      # bytes
    dest = load_project_file(data, tmp_path)         # same folder already exists
    assert dest.name == "wing_2026_2"
    assert Project(dest).name == "wing_2026_2"
    assert (tmp_path / "wing_2026" / META).exists()
    dest = load_project_file(data, tmp_path, overwrite=True)
    assert dest.name == "wing_2026"


def test_rejects_other_files(tmp_path):
    with pytest.raises(ValueError):
        load_project_file(b"not a zip", tmp_path)
    bad = tmp_path / "bad.ramfish"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("x/readme.txt", "hello")
    with pytest.raises(ValueError):
        load_project_file(bad, tmp_path)


def test_ignores_paths_outside_project(tmp_path):
    bad = tmp_path / "evil.ramfish"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr(f"p/{META}", '{"name": "p", "rounds": {}, "history": []}')
        zf.writestr("p/../../escaped.txt", "x")
    dest = load_project_file(bad, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()
    assert (dest / META).exists()


def test_sidebar_save_button(tmp_path, monkeypatch):
    monkeypatch.delenv("RAMFISH_PROJECT_DIR", raising=False)
    monkeypatch.setenv("RAMFISH_OUTPUT_DIR", str(tmp_path / "Output"))
    _small_project(tmp_path / "Output")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.session_state["project_name"] = "wing_2026"
    at.run()
    assert not at.exception
    at = next(b for b in at.sidebar.button if b.label.startswith("💾 Save project")).click().run()
    assert not at.exception
    name, data = at.session_state["project_file"]
    assert name == "wing_2026" and zipfile.is_zipfile(__import__("io").BytesIO(data))


def test_reference_copies(tmp_path):
    import numpy as np
    from ramfish.project import Project

    p = Project(tmp_path / "demo")
    p.add_image(1, "mask", np.ones((20, 20), np.uint16))
    p.add_image(1, "wg", np.arange(400, dtype=np.uint16).reshape(20, 20))
    p.add_image(2, "mask", np.ones((20, 20), np.uint16))
    rough = p.write_reference_copies("rough")
    fine = p.write_reference_copies("fine")
    assert [f.name for f in rough] == ["ALIGNED_mask.tif", "ALIGNED_wg.tif"]
    assert [f.name for f in fine] == ["R1_aligned_mask.tif", "R1_aligned_wg.tif"]
    assert all(f.read_bytes() == p.path("normalized", 1, f.name.split("_")[-1][:-4]).read_bytes() for f in rough + fine)
    assert p.stage_of(1, "wg") == "normalized"        # the copies are never read back
    assert p.write_reference_copies("normalized") == []
    p.add_image(1, "wg", np.zeros((20, 20), np.uint16))   # re-importing Round 1 removes its stale copies
    assert not p.path("fine", 1, "wg").exists() and not p.path("rough", 1, "wg").exists()
