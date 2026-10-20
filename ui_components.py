"""The rough-aligner (ramfish_rough_aligner.py) and merger (ramfish_merger.html) interfaces as Streamlit components."""

from pathlib import Path

import streamlit.components.v1 as components

rough_aligner = components.declare_component(
    "rough_aligner", path=str(Path(__file__).resolve().parent / "components" / "rough_aligner")
)

merger = components.declare_component(
    "merger", path=str(Path(__file__).resolve().parent / "components" / "merger")
)
