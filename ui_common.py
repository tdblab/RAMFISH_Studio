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
