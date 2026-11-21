import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ramfish import dataset, importer
from ramfish.channels import DEFAULT_AMPLIFIER_DYES, DEFAULT_OMIT_TOKENS, DYE_LASER_NM, dye_key, dye_label, oir_part_stem, oir_parts_for
from ramfish.io import SUPPORTED_EXTENSIONS, SUPPORTED_EXTS, file_ext
from ramfish.mask import DEFAULT_MASK_SETTINGS, MASK_ALL
from ramfish.normalize import HAS_GPU, OUTPUT_FORMATS, RES_MODES
from ramfish.project import clean_name
from ui_common import existing_projects, get_project, how_to, output_root, preview, thumb

EXAMPLE_ZIP = Path(__file__).resolve().parents[1] / "Rawfile_Example" / "rawdata.zip"
EXAMPLE_LABEL = "Example data (two rounds of a butterfly wing, built in)"

CLEAN_EXTS = list(set([ext.split('.')[-1] for ext in SUPPORTED_EXTENSIONS]))

st.title("📥 Import & normalize")
how_to("""
**What this step does:** reads your microscope files, works out which round and which gene every channel is,
saves every channel as it came from the microscope (the **raw** TIFFs), and then makes the brightness comparable
between rounds (it removes the uneven background glow and scales every image the same way).

1. **Project name:** type a name for this experiment (for example `wing_disc_Sept2026`) and press **Enter**.
   To continue an earlier experiment, pick its name from the list instead. All results are saved in
   `Output/<project name>/`.
2. **Your images:** choose one of the two options under *1. Locate Data*:
   - **Microscopy files:** drag in the files of one or more rounds (`.oir`, `.czi`, `.lif`, `.nd2`, `.tif`, …).
     A large Olympus stack is saved as the `.oir` plus extra files with the same name and `_00001`, `_00002`, …
     (no extension) that hold the rest of the planes: add them too, here or in the same folder of the zip.
   - **Folder on this computer:** type the path of a folder that holds the files (sub-folders `Round_1`, `Round_2`,
     … or file names starting with their round). Nothing is uploaded or copied, so this is the way for large 3D stacks.
   - **Example data:** two rounds of a butterfly wing that come with the app, to try every step in a few
     minutes without your own files.
   - **Dataset .zip:** one zip file with a folder per round, called `Round_1`, `Round_2`, … Each folder holds the
     images of that round.
3. **Name your files so the app knows what is what.** A file with several channels is named after its round and
   its genes, each gene followed by its dye, for example
   `round1_dpp_alexa546__optix_alexa647__omb_alexa488__hh_alexa514.oir`.
   The app matches every gene to the right channel by its dye (using the dye names or laser wavelengths the
   microscope saved in the file), so the order of the genes in the name does not matter. Dyes can also be written
   as `AF546`, `546` or in brackets (`dpp(546)`), and HCR amplifiers as `dpp(b1)` (see *File name reading*).
   A file with one channel is named after its gene (`wg.tif`), the nuclear stain **DAPI**, or a tissue outline **mask**.
4. **No DAPI?** The app makes a tissue **mask** for every round by saturating the channels (all channels merged,
   or one dye you choose). The rounds are then lined up by the outline of the tissue.
5. **3D stacks of thick tissue** (for example a brain): keep *Maximum projection of the whole stack* under
   *5. 3D stacks*. Step 2 then lines the stacks up in 3D and makes one flat image set per depth. (*Same depth in
   every round* instead keeps a single thin slab at one depth.)
6. Click **🚀 Execute Unified Pipeline**, then check the **Channels** table at the bottom: every channel should
   have the right gene and dye. Correct a name there and click **Save changes**. To change the settings
   afterwards, click **🔁 Normalize again** (the microscope files are not read again).

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
    mode = st.radio("Source", ["Microscopy files", "Folder on this computer (best for large stacks)",
                               "Dataset .zip (Round_1/, Round_2/, …)", EXAMPLE_LABEL], label_visibility="collapsed")
    gen = st.session_state.get("upload_gen", 0)
    uploaded_files, uploaded_zip, rnd, folder_src = None, None, None, None
    if mode.startswith("Folder"):
        folder_text = st.text_input("Folder", key="src_folder", placeholder=r"e.g. D:\data\fish_1",
                                    help="The files are read where they are, nothing is uploaded or copied, so even "
                                         "very large stacks fit in memory. Sub-folders are included: name them Round_1, "
                                         "Round_2, … or start each file name with its round (round1_…).")
        if folder_text.strip():
            try:
                folder_src = dataset.FolderSource(folder_text.strip().strip('"'))
                f_names = folder_src.names()
                f_entries = dataset.scan(f_names)
                f_parts = sum(len(oir_parts_for(e.name, f_names)) for e in f_entries)
                f_rounds = sorted({e.round for e in f_entries if e.round is not None})
                if f_entries:
                    st.success(f"Found {len(f_entries)} image file(s) in Round " + ", ".join(map(str, f_rounds))
                               + (f", plus {f_parts} extra part(s) of large .oir stacks." if f_parts else ".")
                               + (f" {sum(e.round is None for e in f_entries)} file(s) have no round and will be "
                                  "skipped." if any(e.round is None for e in f_entries) else ""))
                else:
                    st.warning("No image files found in this folder.")
            except (NotADirectoryError, OSError) as e:
                folder_src = None
                st.error(f"Cannot open this folder: {e}")
    elif mode.startswith("Microscopy"):
        existing = project.rounds()
        rnd = int(st.number_input("Round (used only for files whose name has no round)", min_value=1, max_value=99,
                                  value=(max(existing) + 1) if existing else 1,
                                  help="A file called round2_... or R2_... goes to Round 2 whatever this box says."))
        # no type filter: the extra _00001 files of large .oir stacks have no extension (checked below instead)
        uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True,
                                          help="Supports .oir, .czi, .lif, .tif, and more. Files of several rounds "
                                               "can be added at once. For a large .oir stack, also add the extra files "
                                               "with the same name plus _00001, _00002, … (they hold the rest of the "
                                               "planes).", key=f"files_{gen}")
        if uploaded_files:
            n_parts = sum(1 for f in uploaded_files if oir_part_stem(f.name))
            st.success(f"Loaded {len(uploaded_files) - n_parts} file(s) ready for processing"
                       + (f", plus {n_parts} extra part(s) of large .oir stacks." if n_parts else "."))
    elif mode == EXAMPLE_LABEL:
        if EXAMPLE_ZIP.exists():
            uploaded_zip = str(EXAMPLE_ZIP)       # read where it is: nothing to upload
            ex = {}
            for e in dataset.scan(dataset.ZipSource(uploaded_zip).names()):
                ex.setdefault(e.round, []).append(e.channel or Path(e.name).stem)
            st.success("Built-in example, ready to run: " + "; ".join(
                f"Round {r}: {', '.join(sorted(ex[r]))}" for r in sorted(k for k in ex if k is not None))
                + ". Name the project above, then click **Execute**.")
        else:
            st.warning("The example data is not in this copy of the app (`Rawfile_Example/rawdata.zip`).")
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
    st.caption("For 3D stacks: also save one flat image made of the brightest point through the stack.")
    res_mode = st.selectbox("Target Resolution:", RES_MODES)
    st.caption("Keep **4K**: every image is enlarged (longest side 4096 pixels, shape kept) before normalization, "
               "and saved as a 16-bit TIFF. The spot counter's dot sizes are set for images of this size; at the "
               "microscope's own size (often 2048 px) the dots are half as wide and many are missed.")
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

col4, col5 = st.columns(2)
MASK_MODES = {"Only when a round has no DAPI (recommended)": "auto", "Always": "always", "Never": "never"}
with col4:
    st.subheader("4. Tissue mask (for alignment without DAPI)")
    mask_mode = MASK_MODES[st.radio("Make a tissue mask", list(MASK_MODES), key="mask_mode")]
    st.caption("A mask is the outline of the tissue, made by saturating the image: a low threshold lights up the "
               "whole tissue. It is made from the raw images, before normalization, and every round gets one.")
    raw_dyes = sorted({project.raw_info(r, c).get("dye") for r in project.raw_rounds() for c in project.raw_channels(r)}
                      - {None, "", "dapi", "hoechst"})
    source_options = [MASK_ALL] + (raw_dyes or sorted(k for k in DYE_LASER_NM if k.startswith("alexa")))
    mask_source = st.selectbox("Saturate", source_options, key="mask_source",
                               format_func=lambda k: "All channels merged (recommended)" if k == MASK_ALL else f"The {dye_label(k)} channel",
                               disabled=mask_mode == "never")
    st.caption("All channels merged gives an outline that does not depend on which genes happen to be bright in a "
               "round. Pick one dye to use the same channel in every round instead.")
    with st.expander("Mask settings"):
        m_blur = st.number_input("Blur (pixels)", 0.0, 50.0, float(DEFAULT_MASK_SETTINGS["blur_px"]), 1.0,
                                 help="Smooths the image first, so the faint glow of the tissue becomes one area.")
        m_thr = st.slider("Threshold (fraction of Otsu's threshold)", 0.1, 1.5, float(DEFAULT_MASK_SETTINGS["threshold_factor"]), 0.05,
                          help="Lower takes in fainter tissue; higher keeps only the brighter parts.")
        m_close = st.number_input("Close gaps in the edge (pixels)", 0, 100, int(DEFAULT_MASK_SETTINGS["closing_px"]), 1)
        m_grow = st.number_input("Grow the outline (pixels)", 0, 100, int(DEFAULT_MASK_SETTINGS["dilation_px"]), 1)

DEPTH_MODES = {"Maximum projection of the whole stack (recommended; 3D stacks are lined up in step 2)": importer.DEPTH_MIP,
               "Same depth in every round (one thin slab only)": importer.DEPTH_MATCH}
with col5:
    st.subheader("5. 3D stacks")
    depth_mode = DEPTH_MODES[st.radio("2D image used for alignment and spot counting", list(DEPTH_MODES), key="depth_mode")]
    st.caption("To analyze every depth of a thick sample, keep the maximum projection: step 2 lines the stacks up in "
               "3D and makes one flat image set per depth. *Same depth in every round* keeps one thin slab instead.")
    slab_um = st.number_input("Slab thickness (µm)", 1.0, 200.0, 10.0, 1.0, disabled=depth_mode != importer.DEPTH_MATCH,
                              help="The planes within this thickness around the matched depth are projected into one image.")
    auto_ref = st.checkbox("Round 1 depth: the sharpest plane (automatic)", value=True, disabled=depth_mode != importer.DEPTH_MATCH)
    ref_plane = None
    if not auto_ref and depth_mode == importer.DEPTH_MATCH:
        ref_plane = int(st.number_input("Round 1 plane (first plane = 0)", 0, 9999, 0, 1))
    st.caption("Every other round is searched for the same depth, also if it was imaged with another Z step or "
               "mounted the other way up. You can type a plane yourself in the Depth table below.")

with st.expander("File name reading"):
    st.caption("HCR amplifier → dye, for names such as `dpp(b1)`. Change it if your amplifiers carry other dyes.")
    amp_rows = [{"Amplifier": k, "Dye": dye_label(v)} for k, v in DEFAULT_AMPLIFIER_DYES.items()]
    amp_table = st.data_editor(pd.DataFrame(amp_rows), hide_index=True, num_rows="dynamic", key="amp_table")
    amplifier_dyes = {str(r["Amplifier"]).strip().lower(): dye_key(r["Dye"]) for r in amp_table.to_dict("records")
                      if str(r.get("Amplifier") or "").strip() and dye_key(r.get("Dye"))}
    omit_text = st.text_input("Words in file names that are not genes (comma-separated)",
                              ", ".join(sorted(DEFAULT_OMIT_TOKENS)))
    omit_tokens = {w.strip().lower() for w in omit_text.split(",") if w.strip()}

norm_settings = {"do_norm": do_norm, "blur_sigma": blur_sigma, "percentile_val": percentile_val, "target_intensity": target_int}
res_settings = {"res_mode": res_mode, "custom_scale": custom_scale}


def build_settings():
    manual = {int(k): v for k, v in st.session_state.get("manual_planes", {}).items()}
    return importer.BuildSettings(
        norm=norm_settings, res=res_settings, target_format=target_format, apply_mip=apply_mip,
        depth_mode=depth_mode, slab_um=slab_um, ref_plane=ref_plane, manual_planes=manual,
        mask_mode=mask_mode, mask_source=mask_source,
        mask={"blur_px": m_blur, "threshold_factor": m_thr, "closing_px": m_close, "dilation_px": m_grow})


def run_build():
    bar = st.progress(0.0, text="Normalizing...")
    warns = importer.build(project, build_settings(), progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
    bar.empty()
    for w in warns:
        st.warning(w)
    st.session_state.pop("norm_zip", None)


st.divider()

b1, b2 = st.columns([2, 1])
run_all = b1.button("🚀 Execute Unified Pipeline", type="primary", width="stretch")
redo = b2.button("🔁 Normalize again (from the raw TIFFs)", width="stretch", disabled=not project.raw_rounds(),
                 help="Uses the current settings on the raw TIFFs already saved; the microscope files are not read again.")

def _upload_saver(f):
    return lambda dest: Path(dest).write_bytes(f.getbuffer())


if run_all:
    jobs = []   # (display name, relative path, single-channel name, save(dest) or a path on disk, [(extra part name, save(dest))])
    skipped = []
    if uploaded_files:
        names = [f.name for f in uploaded_files]
        by_name = {f.name: f for f in uploaded_files}
        all_parts = {n for f in uploaded_files for n in oir_parts_for(f.name, names)}
        for f in uploaded_files:
            if f.name in all_parts:
                continue
            if file_ext(f.name) not in SUPPORTED_EXTS:
                skipped.append(f.name)
                continue
            jobs.append((f.name, None, None, _upload_saver(f),
                         [(n, _upload_saver(by_name[n])) for n in oir_parts_for(f.name, names)]))
    elif folder_src is not None:
        # read in place: the .oir and its _00001 parts are already side by side on disk
        for e in dataset.scan(folder_src.names()):
            if e.round is None:
                skipped.append(e.name)
                continue
            jobs.append((Path(e.name).name, e.name, e.channel, folder_src.root / e.name, []))
    elif uploaded_zip is not None:
        src = dataset.ZipSource(uploaded_zip)
        names = src.names()
        all_parts = set()
        for e in dataset.scan(names):
            parts = oir_parts_for(e.name, names)
            all_parts.update(parts)
            jobs.append((Path(e.name).name, e.name, e.channel, lambda d, n=e.name: src.copy_to(n, d),
                         [(Path(n).name, lambda d, n=n: src.copy_to(n, d)) for n in parts]))
        skipped = [n for n in names if oir_part_stem(n) and n not in all_parts]
    for n in skipped:
        st.warning(f"Skipped {n}: not an image file, no round in its name or folder, or an extra part whose .oir "
                   "file was not added.")
    if not jobs:
        st.warning("No files uploaded to process.")
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="ramfish_upload_"))
        progress_bar = st.progress(0)
        status_text = st.empty()
        ok = 0
        for i, (fname, rel, single, save, parts) in enumerate(jobs):
            extra = f" (+ {len(parts)} extra part{'s' if len(parts) > 1 else ''})" if parts else ""
            status_text.text(f"Reading ({i + 1}/{len(jobs)}): {fname}{extra}...")
            if isinstance(save, Path):          # a file on this computer: read where it is
                temp_file_path, written = save, []
            else:
                temp_file_path = temp_dir / fname
                written = [temp_file_path] + [temp_dir / Path(pn).name for pn, _ in parts]
            try:
                for (_, write), dest in zip([(fname, save)] + parts, written):
                    write(dest)     # same folder and name as on the microscope PC, so the reader finds the parts
                res = importer.read_file(project, temp_file_path, fname, rnd, relpath=rel, omit_tokens=omit_tokens,
                                         amplifier_dyes=amplifier_dyes, single_name=single)
                for w in res.warnings:
                    st.warning(w)
                ok += bool(res.channels)
            except Exception as e:
                st.error(f"Failed to read {fname}: {e}")
            for w in written:
                w.unlink(missing_ok=True)
            progress_bar.progress((i + 1) / len(jobs))
        temp_dir.rmdir()
        status_text.text(f"Read {ok} of {len(jobs)} files into '{project.root.name}/1_import_and_normalize/raw'. Normalizing...")
        run_build()
        status_text.text(f"Complete! {ok} of {len(jobs)} files read and normalized in '{project.root.name}/1_import_and_normalize'.")
        st.session_state.upload_gen = gen + 1
        st.balloons()
elif redo:
    run_build()
    st.success("Normalized again with the current settings.")

step1 = project.step_dir("import")
if step1.exists() and any(p.is_file() for p in step1.rglob("*")):
    d1, d2 = st.columns([1, 2])
    with_raw = d2.checkbox("Include the raw TIFFs in the download (much larger)", value=False)
    if d1.button("Prepare step 1 download"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in step1.rglob("*"):
                if p.is_file() and (with_raw or project.step_dir("raw") not in p.parents):
                    zf.write(p, p.relative_to(project.root).as_posix())
        st.session_state.norm_zip = buf.getvalue()
    if st.session_state.get("norm_zip"):
        st.download_button("Download 1_import_and_normalize (.zip)", st.session_state.norm_zip,
                           file_name=f"{project.name}_1_import_and_normalize.zip", mime="application/zip")

# -------------------------------------------------------------------- channels
if project.rounds():
    st.divider()
    st.subheader("Channels")
    st.caption("Round 1 is the reference. Each round needs an image named `DAPI` or `mask` for alignment; the "
               "other names are used as gene names in the later steps. Check that every gene has the right dye, "
               "and edit names here if needed.")
    rows = []
    for r in project.rounds():
        for c in project.channels(r):
            info = project.raw_info(r, c)
            rows.append({"Round": r, "Channel": c, "Dye": dye_label(info.get("dye")) if info else "",
                         "Channel in file": info.get("channel_in_file", "") if info else "",
                         "Matched by": info.get("how", "") if info else ("made by the app" if c == "mask" else ""),
                         "File": info.get("file", "") if info else "", "Delete": False})
    edited = st.data_editor(pd.DataFrame(rows), hide_index=True, key="channels_table",
                            disabled=["Round", "Dye", "Channel in file", "Matched by", "File"])
    if st.button("Save changes"):
        for old, new in zip(rows, edited.to_dict("records")):
            if new["Delete"]:
                project.remove_channel(old["Round"], old["Channel"])
            elif new["Channel"] != old["Channel"]:
                project.rename_channel(old["Round"], old["Channel"], new["Channel"])
        st.rerun()

    depth = project.meta.get("depth") or {}
    if any(int(d.get("n_z", 1)) > 1 for d in depth.values()):
        st.subheader("Depth")
        st.caption("Plane of every round used for the 2D images (first plane = 0), and the slab around it. "
                   "Agreement says how well the depth curves of a round match Round 1 (1 = perfect). Type another "
                   "plane to override it, then click **🔁 Normalize again**.")
        drows = [{"Round": int(r), "Planes": d["n_z"], "Z step (µm)": d.get("z_step_um"), "Plane used": d.get("z"),
                  "Slab": f"{d.get('z0')} to {d.get('z1')}", "Agreement": d.get("score"),
                  "Upside down": bool(d.get("upside_down")), "Set by": "you" if not d.get("auto") else "app"}
                 for r, d in sorted(depth.items(), key=lambda kv: int(kv[0]))]
        dedit = st.data_editor(pd.DataFrame(drows), hide_index=True, key="depth_table",
                               disabled=["Round", "Planes", "Z step (µm)", "Slab", "Agreement", "Upside down", "Set by"])
        manual = {}
        for old, new in zip(drows, dedit.to_dict("records")):
            if new["Plane used"] != old["Plane used"] and new["Plane used"] is not None and not pd.isna(new["Plane used"]):
                manual[str(old["Round"])] = int(new["Plane used"])
            elif old["Set by"] == "you":
                manual[str(old["Round"])] = int(old["Plane used"])
        st.session_state.manual_planes = manual

    for r in project.rounds():
        st.markdown(f"**Round {r}**" + ("  · reference" if r == project.reference_round else ""))
        if r != project.reference_round and not project.alignment_options(r):
            st.warning("No `DAPI` or `mask` image shared with Round 1 — this round cannot be aligned.")
        cols = st.columns(8)
        for i, c in enumerate(project.channels(r)):
            img, _, _ = preview(project, "normalized", r, c, 256)
            cols[i % 8].image(thumb(img, 160), caption=c, width="stretch")
