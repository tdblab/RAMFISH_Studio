"""Streamlit helpers shared by all pages (session workspace, previews)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st

from ramfish.io import downsample_to, read_plane, to_display_u8
from ramfish.project import META, Project

PREVIEW_MAX = 1000


APP_DIR = Path(__file__).resolve().parent


def output_root() -> Path:
    """Folder that holds one sub-folder per project (default: ``Output/`` next to the app)."""
    return Path(os.environ.get("RAMFISH_OUTPUT_DIR", APP_DIR / "Output"))


def existing_projects() -> list[str]:
    root = output_root()
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if (d / META).exists())


def get_project() -> Project | None:
    """The project chosen on step 1 (``RAMFISH_PROJECT_DIR`` pins one folder, e.g. for tests)."""
    fixed = os.environ.get("RAMFISH_PROJECT_DIR")
    if fixed:
        return Project(fixed)
    name = st.session_state.get("project_name")
    return Project(output_root() / name) if name else None


def need_project() -> Project:
    project = get_project()
    if project is None:
        st.info("Start on **1 · Import & normalize** by entering a project name.")
        st.stop()
    return project


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(max_entries=64, show_spinner=False)
def _load_preview(path: str, mtime: float, max_dim: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    img = read_plane(path)
    small, k = downsample_to(img, max_dim)
    return small, k, img.shape


def preview(project: Project, stage: str, rnd: int, channel: str, max_dim: int = PREVIEW_MAX):
    """(downsampled float image, scale factor, full-res shape), cached by mtime."""
    p = project.path(stage, rnd, channel)
    return _load_preview(str(p), _mtime(p), max_dim)


def preview_path(path: str | Path, max_dim: int = PREVIEW_MAX):
    """Like :func:`preview` for an arbitrary image file."""
    return _load_preview(str(path), _mtime(Path(path)), max_dim)


def thumb(img: np.ndarray, size: int = 256) -> np.ndarray:
    small, _ = downsample_to(img, size)
    return to_display_u8(small)


def need_project_data(project: Project) -> bool:
    if project.is_empty():
        st.info("No images yet. Start with **1 · Import & normalize**.")
        return False
    return True


def alignable_rounds(project: Project) -> list[int]:
    """Rounds after Round 1 that share a DAPI or mask image with Round 1."""
    r0 = project.reference_round
    rounds = [r for r in project.rounds() if r != r0 and project.alignment_options(r)]
    missing = [r for r in project.rounds() if r != r0 and not project.alignment_options(r)]
    if missing:
        st.warning(f"Round(s) {', '.join(map(str, missing))} cannot be aligned: they share no `DAPI` or `mask` "
                   "image with Round 1. Name the right file `DAPI` or `mask` on the Import page.")
    return rounds


def how_to(text: str, expanded: bool = True) -> None:
    """Plain-language instructions at the top of a page (open by default)."""
    with st.expander("📖 How to use this page", expanded=expanded):
        st.markdown(text)


# ------------------------------------------------------------------------------------------ 3D stacks: z-planes
def z_main(project: Project) -> Project:
    """The 3D project a z-plane belongs to (the project itself when it is not a z-plane)."""
    return Project(project.root.parent.parent) if project.meta.get("section") else project


def z_planes(project: Project) -> list[Project]:
    """The z-planes made in step 2b (one small project per plane), shallow to deep; [] for a flat project."""
    from ramfish import sections as sec
    return sec.section_projects(z_main(project))


def z_label(sub: Project) -> str:
    s = sub.meta.get("section", {})
    p = s.get("planes", {})
    r0 = min(p, key=int) if p else None
    return f"z = {s.get('depth_um', 0):g} µm" + (f" (Round {r0} plane {int(p[r0]) + 1})" if r0 else "")


def z_select(project: Project, page: str) -> Project:
    """On steps 3 to 5 of a 3D project: the Z slider. Returns the chosen z-plane (or the project itself if flat).

    The chosen depth is kept when moving between pages."""
    subs = z_planes(project)
    if not subs:
        if any(int((project.raw_info(r, c) or {}).get("z_planes") or 1) > 1
               for r in project.raw_rounds() for c in project.raw_channels(r)[:1]):
            st.info("This project has 3D stacks. To run this step on every z-plane on its own (with a Z slider), place "
                    "the stacks and make the z-planes first in **2 · Rough alignment** (3D mode).")
        return project
    if project.meta.get("section") and "z_keep" not in st.session_state:   # opened from 2b: start on that plane
        st.session_state.z_keep = next((i for i, s in enumerate(subs) if s.root == project.root), 0)
    keep = min(int(st.session_state.get("z_keep", 0)), len(subs) - 1)
    st.info(f"🧊 **3D stack of {z_main(project).name}:** this step runs on each of the {len(subs)} z-planes on its own. "
            "Drag the slider to move through the depth.")
    idx = st.select_slider("Z depth (below Round 1's first plane)", options=list(range(len(subs))), value=keep,
                           format_func=lambda i: z_label(subs[i]), key=f"z_idx_{page}")
    st.session_state.z_keep = idx
    return subs[idx]
