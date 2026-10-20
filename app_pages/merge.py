import base64
from pathlib import Path

import cv2
import streamlit as st

from ramfish.io import read_plane
from ramfish.normalize import format_and_resize
from ui_common import how_to, need_project, need_project_data
from ui_components import merger

project = need_project()
st.title("🌈 Multiplex merge")
how_to("""
**What this step does:** overlays the genes of all rounds in one coloured picture of the tissue.

1. **Source:** *Synthetic spots* uses the dots found in step 4 (clean dots on black). *Aligned images* uses the
   aligned microscope images from step 3 (the real signal, useful for expression domains).
2. **Images to load:** every gene is listed. Click ✕ to leave a gene out, or pick it again from the list to put it
   back. A layer called **DAPI** is drawn as a grey background that shows the tissue.
3. In the merger below, choose a **colour** for each gene, then click **Generate Composite**. You can then fine-tune:
   - **Background Threshold:** hides faint signal below this level. Raise it to clean up haze.
   - **Gamma Correction:** above 1 makes strong signal stand out more; below 1 brightens faint signal.
   - **Legend Position / Legend Size Scale:** where the key with the gene names goes, and how big it is.
   Click **Generate Composite** again after each change.
4. Click **Download High-Res Image**. The picture is downloaded *and* saved in `5_merge/multiplex_composite.jpg`.
""")

if not need_project_data(project):
    st.stop()

synth_dir = project.step_dir("spots") / "synthetic_spots"
sources = (["Synthetic spots (step 4)"] if synth_dir.exists() and any(synth_dir.glob("*.jpg")) else []) + \
          ["Aligned images (step 3)"]
c1, c2 = st.columns([1, 2])
source = c1.radio("Source", sources)

r0 = project.reference_round
layers: dict[str, Path] = {}   # layer name -> image file
if source.startswith("Synthetic"):
    layers.update({p.stem: p for p in sorted(synth_dir.glob("*.jpg"))})
else:
    layers.update({label: project.path(project.stage_of(r, c), r, c) for r, c, label in project.gene_targets()})
bg = project.background_channel()
if bg:
    layers["DAPI" if bg.lower() == "dapi" else bg] = project.path("normalized", r0, bg)

# Images of rounds that were never aligned would sit in the wrong place: leave them out by default.
where = {label: (r, c) for r, c, label in project.gene_targets()}
unaligned = [n for n in layers if n in where and where[n][0] != r0
             and project.stage_of(*where[n]) not in ("rough", "fine")]
if unaligned:
    st.warning("Not aligned to Round 1 yet, so left out of the picture: " + ", ".join(f"`{n}`" for n in unaligned)
               + ". Align their rounds in steps 2 and 3 first (you can still add them below).")
chosen = c2.multiselect("Images to load (Max 30)", list(layers),
                        default=[n for n in layers if n.lower() != "mask" and n not in unaligned][:30])


@st.cache_data(max_entries=64, show_spinner="Loading images...")
def _data_url(path: str, mtime: float) -> str:
    if path.endswith(".jpg"):
        data = Path(path).read_bytes()
    else:   # 16-bit project image -> 8-bit JPEG, mapped as in ramfish_processor.py
        img = format_and_resize(read_plane(path), ".jpg", "Original Resolution", 1.0)
        data = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


if chosen:
    files = [{"name": f"{n}.jpg", "url": _data_url(str(layers[n]), layers[n].stat().st_mtime)} for n in chosen]
    token = source + "|" + "|".join(f"{n}:{layers[n].stat().st_mtime}" for n in chosen)
    result = merger(files=files, token=token, key="merger", default=None)
    if result and result.get("nonce") != st.session_state.get("merge_saved"):
        st.session_state.merge_saved = result["nonce"]
        out = project.step_dir("merge")
        out.mkdir(exist_ok=True)
        (out / result["name"]).write_bytes(base64.b64decode(result["data"].split(",", 1)[1]))
        st.session_state.merge_msg = f"Saved to `{out / result['name']}`"
    if st.session_state.get("merge_msg"):
        st.success(st.session_state.merge_msg)
