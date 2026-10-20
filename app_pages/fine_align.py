import io
import zipfile

import cv2
import streamlit as st

from ramfish.fine_align import apply_transforms, compute_transforms, match_fixed_shape_centered, read_image_bytes
from ui_common import alignable_rounds, how_to, need_project, need_project_data, preview_path, thumb

project = need_project()
st.title("🎯 Fine alignment")
how_to("""
**What this step does:** automatically corrects the small differences that remain between each round and Round 1,
and applies exactly the same correction to every gene image of that round. Afterwards all images line up with
Round 1. Rounds with **DAPI** also get local stretching/squeezing corrected; rounds lined up on a **mask** are only
rotated, shifted and scaled (a mask has no detail inside the tissue, so bending it would smear the genes).

1. Check the list below: for every round it shows which image is used to line it up (**DAPI**, or **mask** if the
   round has no DAPI) and whether you rough-aligned it in step 2.
2. Click **Run Batch Alignment**. Large images take a few minutes per round: keep this page open until the bar
   reaches 100 %.
3. The aligned images then appear at the bottom. Scroll through them: the tissue should sit in the same place in
   every picture. If a round looks wrong, redo its rough alignment in step 2 and run this step again.

Saved in `3_fine_aligned/` (for example `R2_aligned_ptc.tif` = gene *ptc* from Round 2).
**Next:** **4 · Spot counter** to count mRNA dots, or **5 · Multiplex merge** for a combined picture.
""")
if not need_project_data(project):
    st.stop()
r0 = project.reference_round
rounds = alignable_rounds(project)
if not rounds:
    st.info("Import at least one more round with a `DAPI` or `mask` image.")
    st.stop()

st.markdown(f"**Round {r0} (Global Fixed Reference)**")
for r in rounds:
    ch = project.alignment_channel(r)
    src = project.stage_of(r, ch, "rough")
    targets = [c for c in project.channels(r) if c != ch]
    st.markdown(f"- **Round {r} Alignment** — structural reference: `{ch}` "
                f"({'roughly aligned' if src == 'rough' else 'not roughly aligned'}); target genes: "
                + ", ".join(f"`{t}`" for t in targets))

out_dir = project.step_dir("fine")

if any(project.alignment_channel(r).lower() == "mask" for r in rounds):
    bend_masks = st.checkbox(
        "Rounds aligned on a mask: also correct local stretching (B-Spline)", value=False,
        help="Off (recommended): mask rounds are lined up by rotating, shifting and scaling only. A mask is plain "
             "white inside, so the B-Spline can only follow the outline and tends to smear the inside of the tissue.")
    st.caption("**Recommended: off.** Rounds lined up on DAPI always get the full correction (affine + B-Spline). "
               "Rounds lined up on a mask get rotation / shift / scale only, because a mask has no detail inside the "
               "tissue and bending it would smear the gene images.")
else:
    bend_masks = False

if st.button("Run Batch Alignment", type="primary"):
    progress = st.progress(0, text="Initializing images in memory...")
    total_steps = sum(1 + len(project.channels(r)) for r in rounds)
    step = 0
    out_dir.mkdir(exist_ok=True)
    for r in rounds:
        project.clear_stage("fine", r)
    try:
        for r in rounds:
            ch = project.alignment_channel(r)
            fixed_ref_np = read_image_bytes(project.path("normalized", r0, project.reference_partner(r, ch)).read_bytes())
            progress.progress(int(step / total_steps * 90), text=f"Computing B-Spline Transform for Round {r}...")
            moving_ref_raw = read_image_bytes(project.path(project.stage_of(r, ch, "rough"), r, ch).read_bytes())
            moving_ref_np = match_fixed_shape_centered(moving_ref_raw, fixed_ref_np.shape)

            # Heavy computation
            bend = ch.lower() != "mask" or bend_masks
            affine_tx, bspline_tx, fixed_img_sitk = compute_transforms(fixed_ref_np, moving_ref_np, bspline=bend)
            step += 1

            for c in project.channels(r):
                progress.progress(int(step / total_steps * 90), text=f"Applying warp to {c} (Round {r})...")
                target_raw = read_image_bytes(project.path(project.stage_of(r, c, "rough"), r, c).read_bytes())
                target_np = match_fixed_shape_centered(target_raw, fixed_ref_np.shape)
                aligned_np = apply_transforms(affine_tx, bspline_tx, fixed_img_sitk, target_np)
                cv2.imwrite(str(project.path("fine", r, c)), aligned_np)
                step += 1
            project.log("fine_align", {"affine": "GD lr=1.0, 100 it", "bspline": "mesh 8x8, LBFGSB 50 it" if bend else "off"},
                        round=r, aligned_on=ch, channels=project.channels(r))
        progress.progress(100, text="Alignment completely successfully!")
        st.session_state.pop("aligned_zip", None)
    except Exception as e:
        st.error(f"Error: {e}")

files = sorted(out_dir.glob("R*_aligned_*.tif")) if out_dir.exists() else []
if files:
    st.subheader("Alignment Complete")
    cols = st.columns(4)
    for i, p in enumerate(files):
        cols[i % 4].image(thumb(preview_path(p, 512)[0], 300), caption=p.name, width="stretch")
    if st.button("Prepare 3_fine_aligned download"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in files:
                zf.write(p, f"{out_dir.name}/{p.name}")
        st.session_state.aligned_zip = buf.getvalue()
    if st.session_state.get("aligned_zip"):
        st.download_button("Download 3_fine_aligned (.zip)", st.session_state.aligned_zip,
                           file_name=f"{project.name}_3_fine_aligned.zip", mime="application/zip")
