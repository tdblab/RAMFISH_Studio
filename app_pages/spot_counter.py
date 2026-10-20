import io
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from ramfish.spots import DEFAULT_CONFIG, process_single_image
from ui_common import how_to, need_project, need_project_data

project = need_project()
st.title("🔴 Spot counter")
how_to("""
**What this step does:** finds the bright mRNA dots in every gene image, counts them, and records where each dot is.
DAPI and mask images are skipped.

1. For a first try, keep the settings below as they are and click **Run spot counter**.
2. When it finishes, choose a gene under **Validation image**: every dot that was counted has a **red circle**.
3. Judge the circles and adjust one setting at a time, then click **Run spot counter** again:
   - real dots **without** a circle → lower **blob_threshold** (for example 0.30 → 0.20) or lower **intensity_cutoff**;
   - circles on **background haze** (autofluorescence) → raise **intensity_cutoff** (for example 0.25 → 0.35);
   - **large** blobs or clusters missed → raise **blob_max_sigma**; **very small** dots missed → lower **blob_min_sigma**.

Saved in `4_spots/`: **data_sheets/** (spreadsheets you can open in Excel: `BATCH_SUMMARY_COUNTS.csv` = number of
dots per gene, `ramfish_master_spots.csv` = position of every dot), **spot_images/** (the red-circle check
pictures) and **synthetic_spots/** (white dots on black, used in step 5). **Next:** **5 · Multiplex merge**.
""")
if not need_project_data(project):
    st.stop()
genes = project.gene_targets()
if not genes:
    st.info("No gene images yet (only DAPI / mask).")
    st.stop()

st.subheader("Settings")
d = DEFAULT_CONFIG["settings"]
EXPLAIN = {
    "blob_min_sigma": "**Smallest dot size** to look for, in pixels. Default 2. Lower it if tiny dots are missed.",
    "blob_max_sigma": "**Largest dot size** to look for, in pixels. Default 5. Raise it if big dots or clusters are missed.",
    "blob_num_sigma": "**How many sizes** between the smallest and largest are tried. Default 4. More = slower; "
                      "rarely needs changing.",
    "blob_threshold": "**How clearly a dot must stand out** from its surroundings (0 to 1). Default 0.30. "
                      "Lower = finds fainter dots, but also more false ones.",
    "intensity_cutoff": "**Minimum brightness** of a dot (0 to 1, where 1 is the brightest possible). Default 0.25 "
                        "= 25 % of full brightness. Raise it to ignore tissue autofluorescence.",
    "synthetic_spot_radius": "**Drawing size** of each dot in the step 5 picture, in pixels. Default 4. "
                             "Does not change the counting.",
}
c1, c2, c3 = st.columns(3)
settings = {}
for col, key, kind, lo, hi, step in (
        (c1, "blob_min_sigma", float, 0.5, 50.0, 0.5), (c1, "blob_max_sigma", float, 0.5, 50.0, 0.5),
        (c2, "blob_threshold", float, 0.0, 1.0, 0.01), (c2, "intensity_cutoff", float, 0.0, 1.0, 0.01),
        (c3, "blob_num_sigma", int, 1, 50, 1), (c3, "synthetic_spot_radius", int, 1, 50, 1)):
    with col:
        settings[key] = st.number_input(key, kind(lo), kind(hi), kind(d[key]), kind(step))
        st.caption(EXPLAIN[key])
st.caption("Gene images that will be counted: " + ", ".join(f"`{label}` (Round {r})" for r, _, label in genes))

base = project.step_dir("spots")
dirs = {"csv": base / "data_sheets" / "individual_csvs", "img": base / "spot_images", "synth": base / "synthetic_spots"}

if st.button("Run spot counter", type="primary"):
    if base.exists():
        shutil.rmtree(base)
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    config = {"settings": settings}
    summary_stats, error_log = [], []
    bar = st.progress(0.0, text="Processing Batch")
    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_single_image, project.path(project.stage_of(r, c), r, c), dirs, config, label): label
                   for r, c, label in genes}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result["status"] == "success":
                summary_stats.append({'Gene': result["gene"], 'Total_Spots': result["count"]})
            elif result["status"] != "skipped":
                error_log.append({'File': result["gene"], 'Error': result["reason"]})
            bar.progress((i + 1) / len(futures), text=f"Processing Batch ({i + 1}/{len(futures)})")

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(base / "data_sheets" / "BATCH_SUMMARY_COUNTS.csv", index=False)
    if error_log:
        pd.DataFrame(error_log).to_csv(base / "data_sheets" / "ERROR_LOG.csv", index=False)
    all_csvs = list(dirs["csv"].glob("*_spots.csv"))
    if all_csvs:
        pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True).to_csv(
            base / "data_sheets" / "ramfish_master_spots.csv", index=False)
    project.log("spot_detection", settings, counts={s["Gene"]: s["Total_Spots"] for s in summary_stats})
    st.session_state.pop("spots_zip", None)

summary_path = base / "data_sheets" / "BATCH_SUMMARY_COUNTS.csv"
if summary_path.exists():
    st.subheader("Batch Processing Complete")
    summary_df = pd.read_csv(summary_path)
    st.metric("Total Transcripts Detected", int(summary_df["Total_Spots"].sum()) if not summary_df.empty else 0)
    st.dataframe(summary_df, hide_index=True)
    err = base / "data_sheets" / "ERROR_LOG.csv"
    if err.exists():
        st.error("Errors during processing:")
        st.dataframe(pd.read_csv(err), hide_index=True)
    shown = [g for g in summary_df["Gene"] if (dirs["img"] / f"{g}_val.jpg").exists()]
    if shown:
        g = st.selectbox("Validation image", shown)
        a, b = st.columns(2)
        a.image(str(dirs["img"] / f"{g}_val.jpg"), caption="spot_images/", width="stretch")
        b.image(str(dirs["synth"] / f"{g}.jpg"), caption="synthetic_spots/", width="stretch")
    if st.button("Prepare results download"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in base.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(base).as_posix())
        st.session_state.spots_zip = buf.getvalue()
    if st.session_state.get("spots_zip"):
        st.download_button("Download data_sheets / spot_images / synthetic_spots (.zip)", st.session_state.spots_zip,
                           file_name=f"{project.name}_4_spots.zip", mime="application/zip")
