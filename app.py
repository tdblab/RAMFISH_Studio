"""RAMFISH Studio - Streamlit entry point.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import streamlit as st

from ramfish import __version__
from ramfish.project import PROJECT_EXT, load_project_file, save_project_file
from ui_common import get_project, output_root

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
        if st.button("Prepare project download"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(project.root.rglob("*")):
                    if f.is_file():
                        zf.write(f, f"{project.name}/{f.relative_to(project.root).as_posix()}")
            st.session_state.project_zip = (project.name, buf.getvalue())
        if st.session_state.get("project_zip", (None,))[0] == project.name:
            st.download_button("Download whole project (.zip)", st.session_state.project_zip[1],
                               file_name=f"{project.name}.zip", mime="application/zip")

    if not os.environ.get("RAMFISH_PROJECT_DIR"):
        st.divider()
        st.markdown("**Project file**")
        if project is not None:
            if st.button("💾 Save project", help=f"Pack the whole project into one {PROJECT_EXT} file "
                                                  "that you can keep, share or open later."):
                with st.spinner("Packing project..."):
                    st.session_state.project_file = (project.name, save_project_file(project))
            if st.session_state.get("project_file", (None,))[0] == project.name:
                st.download_button(f"Download {project.name}{PROJECT_EXT}", st.session_state.project_file[1],
                                   file_name=f"{project.name}{PROJECT_EXT}", mime="application/octet-stream")
        gen = st.session_state.get("open_gen", 0)
        upload = st.file_uploader(f"📂 Load project ({PROJECT_EXT})", type=[PROJECT_EXT.lstrip(".")],
                                  key=f"open_project_{gen}",
                                  help=f"Open a project saved with 💾 Save project. It is copied into "
                                       f"{output_root()} and becomes the current project.")
        if upload is not None:
            try:
                with st.spinner("Opening project..."):
                    dest = load_project_file(upload, output_root())
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.project_name = dest.name
                st.session_state.pop("project_file", None)
                st.session_state.open_gen = gen + 1
                st.session_state.opened_msg = f"Opened project **{dest.name}**."
                st.rerun()
        if st.session_state.get("opened_msg"):
            st.success(st.session_state.pop("opened_msg"))

nav.run()
