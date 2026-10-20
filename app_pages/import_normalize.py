import io
import os
import tempfile
import zipfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import streamlit as st

from ramfish import dataset
from ramfish.io import SUPPORTED_EXTENSIONS, load_stack
from ramfish.normalize import HAS_GPU, OUTPUT_FORMATS, RES_MODES, apply_ramfish_normalization, format_and_resize
from ramfish.project import clean_name
from ui_common import existing_projects, get_project, how_to, output_root, preview, thumb

CLEAN_EXTS = list(set([ext.split('.')[-1] for ext in SUPPORTED_EXTENSIONS]))

st.title("📥 Import & normalize")
how_to("""
**What this step does:** brings your microscope images into the project and makes their brightness comparable
between rounds (it removes the uneven background glow and scales every image the same way).

1. **Project name:** type a name for this experiment (for example `wing_disc_Sept2026`) and press **Enter**.
   To continue an earlier experiment, pick its name from the list instead. All results are saved in
   `Output/<project name>/`.
2. **Your images:** choose one of the two options under *1. Locate Data*:
   - **Microscopy files (one round):** set the **Round** number, then drag in all the images of that round
     (`.oir`, `.czi`, `.lif`, `.nd2`, `.tif`, …). Repeat for every round, starting with Round 1.
   - **Dataset .zip:** one zip file with a folder per round, called `Round_1`, `Round_2`, … Each folder holds the
     images of that round.
3. **Name your files before uploading**, so the app knows what is what:
   - the nuclear stain must be called **DAPI** (for example `DAPI.tif` or `1_DAPI.tif`);
   - if you have no DAPI, the tissue outline image must be called **mask**;
   - every other file is named after its gene, for example `wg.tif` or `1_wg.tif`.
4. Leave the settings as they are (they suit most experiments) and click **🚀 Execute Unified Pipeline**.
5. Check the **Rounds** table at the bottom of the page: every round should list `DAPI` or `mask` plus its genes.
   You can correct names there, then click **Save changes**.

**Round 1 is the reference:** all other rounds will be lined up to it. **Next:** go to **2 · Rough alignment**.
""")

if not os.environ.get("RAMFISH_PROJECT_DIR"):
    names = existing_projects()
    current = st.session_state.get("project_name")
    choice = st.selectbox(
        "Project name", names, index=names.index(current) if current in names else None,
        accept_new_options=True, placeholder="Type a new project name, or pick an existing project",
        help=f"Everything is saved in {output_root()}/<project name>/",
    )
    if choice and clean_name(choice) != current:
        st.session_state.project_name = clean_name(choice)
        st.rerun()
project = get_project()
if project is None:
    st.info(f"Enter a project name to start. All results are saved in `{output_root()}/<project name>/`.")
    st.stop()
st.caption(f"📁 Saving to `{project.root}`")

col1, col2, col3 = st.columns([1.5, 1, 1.2])

with col1:
    st.subheader("1. Locate Data")
    mode = st.radio("Source", ["Microscopy files (one round)", "Dataset .zip (Round_1/, Round_2/, …)"],
                    label_visibility="collapsed")
    gen = st.session_state.get("upload_gen", 0)
    uploaded_files, uploaded_zip, rnd = None, None, None
    if mode.startswith("Microscopy"):
        existing = project.rounds()
        rnd = int(st.number_input("Round", min_value=1, max_value=99, value=(max(existing) + 1) if existing else 1))
        uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True, type=CLEAN_EXTS,
                                          help="Supports .oir, .czi, .lif, .tif, and more.", key=f"files_{gen}")
        if uploaded_files:
            st.success(f"Loaded {len(uploaded_files)} file(s) ready for processing.")
    else:
        uploaded_zip = st.file_uploader("Dataset archive (.zip)", type=["zip"], key=f"zip_{gen}",
                                        help="One folder per round (Round_1/, Round_2/, …); file names become channel "
                                             "names, e.g. Round_1/1_DAPI.tif → DAPI.")
    st.info(f"**Hardware Engine:** {'🚀 GPU (CUDA)' if HAS_GPU else '💻 CPU (SciPy)'}")

with col2:
    st.subheader("2. Export Settings")
    target_format = st.selectbox("Output Format:", OUTPUT_FORMATS)
    st.caption("File type of the exported copies. **.tif** keeps the full image quality (recommended); "
               ".jpg/.png/.bmp are smaller pictures for viewing.")
    apply_mip = st.checkbox("Export Max Intensity Projection (MIP) along with Z-slices", value=True)
    st.caption("For 3D stacks: also save one flat image made of the brightest point through the stack. "
               "The next steps always use this flat image.")
    res_mode = st.selectbox("Target Resolution:", RES_MODES)
    st.caption("Keep **Original Resolution** unless the images are very large and the computer is slow.")
    custom_scale = 1.0
    if res_mode == "Custom Scaling Factor":
        custom_scale = st.slider("Scale Factor:", min_value=0.25, max_value=4.0, value=1.0, step=0.25)

with col3:
    st.subheader("3. RAMFISH Normalization")
    do_norm = st.toggle("Apply RAMFISH Normalization", value=True)
    st.caption("Removes the uneven background glow and gives every round the same brightness. Keep it on.")
    blur_sigma = st.number_input("Gaussian Blur Sigma (radius):", min_value=1.0, max_value=200.0, value=50.0, disabled=not do_norm)
    st.caption("Size (in pixels) of the background glow that is removed. It must be larger than your biggest real "
               "structure. 50 suits most images; increase it if large expression domains look dimmed.")
    percentile_val = st.number_input("Robust Percentile Bound:", min_value=90.0, max_value=100.0, value=99.5, step=0.1, disabled=not do_norm)
    st.caption("Which brightness counts as the 'brightest real signal': 99.5 = the brightest 0.5 % of pixels. "
               "Lower it if a few very bright specks make the rest of the image look dark.")
    target_int = st.number_input("Target Intensity Scaling:", min_value=1000.0, max_value=65535.0, value=60000.0, disabled=not do_norm)
    st.caption("How bright that signal becomes (maximum possible: 65535). Leave at 60000.")
    st.caption("Images named `mask` are kept exactly as they are.")

norm_settings = {"do_norm": do_norm, "blur_sigma": blur_sigma, "percentile_val": percentile_val, "target_intensity": target_int}
res_settings = {"res_mode": res_mode, "custom_scale": custom_scale}
output_dir = project.step_dir("normalized") / "exports"


def process_and_export(file_path, rnd, base_name):
    """ramfish_processor.py's export loop, plus registering each channel's image for the next steps."""
    file_path = Path(file_path)
    raw_volume = load_stack(file_path)   # CZYX, any supported microscopy format
    num_channels, num_z = raw_volume.shape[0], raw_volume.shape[1]
    prefix = "NORM_" if norm_settings["do_norm"] else "RAW_"
    out = output_dir / f"Round_{rnd}"
    out.mkdir(parents=True, exist_ok=True)
    clean_name = "".join(ch for ch in file_path.stem if ch.isprintable()).strip()
    names = []

    for c in range(num_channels):
        name = base_name if num_channels == 1 else f"{base_name}_C{c}"
        settings = dict(norm_settings, do_norm=False) if name.lower() == "mask" else norm_settings
        # 1. ALWAYS export single slices (or single image if no z-stack)
        for z in range(num_z):
            z_label = f"_Z{z}" if num_z > 1 else ""
            norm_plane = apply_ramfish_normalization(raw_volume[c][z], **settings)
            final_img = format_and_resize(norm_plane, target_format, **res_settings)
            iio.imwrite(str(out / f"{prefix}{clean_name}_C{c}{z_label}{target_format}"), final_img)
            working = norm_plane
        # 2. IN ADDITION, export MIP if requested and it is a z-stack
        if num_z > 1:
            working = apply_ramfish_normalization(np.max(raw_volume[c], axis=0), **settings)
            if apply_mip:
                final_img = format_and_resize(working, target_format, **res_settings)
                iio.imwrite(str(out / f"{prefix}{clean_name}_C{c}_MIP{target_format}"), final_img)
        # The next steps use this image (the MIP for z-stacks).
        names.append(project.add_image(rnd, name, format_and_resize(working, ".tif", **res_settings)))
    project.log("import", norm_settings, round=rnd, file=file_path.name, channels=names, **res_settings)


st.divider()

if st.button("🚀 Execute Unified Pipeline", type="primary", width="stretch"):
    jobs = []   # (round, channel name, file name, data getter)
    if uploaded_files:
        for f in uploaded_files:
            jobs.append((rnd, dataset.parse_name(f"Round_{rnd}/{f.name}")[1], f.name, f.getbuffer))
    elif uploaded_zip is not None:
        src = dataset.ZipSource(uploaded_zip)
        for e in dataset.scan(src.names()):
            if e.round is not None:
                jobs.append((e.round, e.channel, Path(e.name).name, lambda n=e.name: src.read(n)))
    if not jobs:
        st.warning("No files uploaded to process.")
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="ramfish_upload_"))
        progress_bar = st.progress(0)
        status_text = st.empty()
        success_count = 0
        for i, (r, name, fname, data) in enumerate(jobs):
            status_text.text(f"Processing ({i + 1}/{len(jobs)}): {fname}...")
            temp_file_path = temp_dir / fname
            temp_file_path.write_bytes(bytes(data()))
            try:
                process_and_export(temp_file_path, r, name)
                success_count += 1
            except Exception as e:
                st.error(f"Failed to process {fname}: {e}")
            temp_file_path.unlink(missing_ok=True)
            progress_bar.progress((i + 1) / len(jobs))
        temp_dir.rmdir()
        status_text.text(f"Complete! Successfully exported {success_count} out of {len(jobs)} files to "
                         f"'{project.root.name}/{project.step_dir('normalized').name}'.")
        st.session_state.upload_gen = gen + 1
        st.session_state.pop("norm_zip", None)
        st.balloons()

if output_dir.exists() and any(p.is_file() for p in output_dir.rglob("*")):
    if st.button("Prepare 1_normalized download"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in project.step_dir("normalized").rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(project.root).as_posix())
        st.session_state.norm_zip = buf.getvalue()
    if st.session_state.get("norm_zip"):
        st.download_button("Download 1_normalized (.zip)", st.session_state.norm_zip,
                           file_name=f"{project.name}_1_normalized.zip", mime="application/zip")

# -------------------------------------------------------------------- rounds
if project.rounds():
    st.divider()
    st.subheader("Rounds")
    st.caption("Round 1 is the reference. Each round needs an image named `DAPI` or `mask` for alignment; the "
               "other names are used as gene names in the later steps. Edit names here if needed.")
    rows = [{"Round": r, "Channel": c, "Delete": False} for r in project.rounds() for c in project.channels(r)]
    edited = st.data_editor(pd.DataFrame(rows), hide_index=True, disabled=["Round"], key="channels_table")
    if st.button("Save changes"):
        for old, new in zip(rows, edited.to_dict("records")):
            if new["Delete"]:
                project.remove_channel(old["Round"], old["Channel"])
            elif new["Channel"] != old["Channel"]:
                project.rename_channel(old["Round"], old["Channel"], new["Channel"])
        st.rerun()
    for r in project.rounds():
        st.markdown(f"**Round {r}**" + ("  · reference" if r == project.reference_round else ""))
        if r != project.reference_round and not project.alignment_options(r):
            st.warning("No `DAPI` or `mask` image shared with Round 1 — this round cannot be aligned.")
        cols = st.columns(8)
        for i, c in enumerate(project.channels(r)):
            img, _, _ = preview(project, "normalized", r, c, 256)
            cols[i % 8].image(thumb(img, 160), caption=c, width="stretch")
