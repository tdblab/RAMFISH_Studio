"""RAMFISH Studio - Streamlit entry point.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ramfish import __version__
from ui_common import get_project

ASSETS = Path(__file__).parent / "assets"
st.set_page_config(page_title="RAMFISH Studio", page_icon=str(ASSETS / "ramfish-icon.png"), layout="wide")
st.logo(str(ASSETS / "ramfish-icon.png"), size="large")

pages = [
    st.Page("app_pages/import_normalize.py", title="1 · Import & normalize", icon="📥", default=True),
    st.Page("app_pages/rough_align.py", title="2 · Rough alignment", icon="🧭"),
    st.Page("app_pages/fine_align.py", title="3 · Fine alignment", icon="🎯"),
    st.Page("app_pages/spot_counter.py", title="4 · Spot counter", icon="🔴"),
    st.Page("app_pages/merge.py", title="5 · Multiplex merge", icon="🌈"),
]
nav = st.navigation(pages)

project = get_project()
with st.sidebar:
    st.image(str(ASSETS / "ramfish-logo.png"), width=170)
    st.caption(f"RAMFISH Studio v{__version__}")
    if project is not None:
        rounds = project.rounds()
        st.markdown(f"**Project:** {project.name}  \n{len(rounds)} round(s), "
                    f"{sum(len(project.channels(r)) for r in rounds)} image(s)")
        st.caption(f"Saved in `{project.root}`")

nav.run()
