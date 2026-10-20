"""Each Streamlit page runs on a small prepared project (rough-alignment UI is tested in a browser)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ramfish import rough_align as ra
from ramfish.io import to_16bit_scale
from ramfish.normalize import apply_ramfish_normalization, format_and_resize
from ramfish.project import Project

ROOT = Path(__file__).resolve().parents[1]


def page(name):
    return AppTest.from_file(str(ROOT / "app_pages" / f"{name}.py"), default_timeout=300)


def click(at, label):
    next(b for b in at.button if b.label.startswith(label)).click()
    return at.run()


@pytest.fixture()
def project(tmp_path, monkeypatch, demo):
    monkeypatch.setenv("RAMFISH_PROJECT_DIR", str(tmp_path))
    p = Project(tmp_path)
    for rnd, chans in demo.images.items():
        for ch, img in chans.items():
            norm = apply_ramfish_normalization(to_16bit_scale(img), True, 50.0, 99.5, 60000.0)   # as step 1
            p.add_image(rnd, ch, format_and_resize(norm, ".tif", "Original Resolution", 1.0))
    return p


def test_entrypoint_starts_on_import(tmp_path, monkeypatch):
    monkeypatch.setenv("RAMFISH_PROJECT_DIR", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert "Import" in at.title[0].value


def test_existing_project_can_be_reopened(tmp_path, monkeypatch):
    """Typing a new name is tested in the browser (AppTest cannot enter new selectbox options)."""
    monkeypatch.delenv("RAMFISH_PROJECT_DIR", raising=False)
    monkeypatch.setenv("RAMFISH_OUTPUT_DIR", str(tmp_path / "Output"))
    at = page("import_normalize").run()
    assert not at.exception
    assert any("Enter a project name" in i.value for i in at.info)
    Project(tmp_path / "Output" / "wing_disc_2026")
    at = page("import_normalize").run()
    assert at.selectbox[0].options == ["wing_disc_2026"]
    at = at.selectbox[0].set_value("wing_disc_2026").run()
    assert not at.exception
    assert at.session_state["project_name"] == "wing_disc_2026"
    assert any("wing_disc_2026" in c.value for c in at.caption)


def test_pages_render(project):
    for name in ("import_normalize", "rough_align", "fine_align", "spot_counter", "merge"):
        at = page(name).run()
        assert not at.exception, name


def test_fine_spots_merge(project, demo):
    import tifffile

    for r in (2, 3):   # what the rough-alignment UI writes on Apply
        for c in project.channels(r):
            out = project.path("rough", r, c)
            out.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(out, ra.apply(tifffile.imread(project.path("normalized", r, c)), demo.poses[r]))
    at = click(page("fine_align").run(), "Run Batch Alignment")
    assert not at.exception
    assert (project.root / "3_fine_aligned" / "R3_aligned_dll.tif").exists()
    assert project.stage_of(3, "dll") == "fine"

    at = click(page("spot_counter").run(), "Run spot counter")
    assert not at.exception
    sheets = project.root / "4_spots" / "data_sheets"
    assert (sheets / "BATCH_SUMMARY_COUNTS.csv").exists() and (sheets / "ramfish_master_spots.csv").exists()
    assert (project.root / "4_spots" / "synthetic_spots" / "dll.jpg").exists()

    at = page("merge").run()
    assert not at.exception
