import tifffile
import streamlit as st

from ramfish import rough_align as ra
from ui_components import rough_aligner
from ui_common import alignable_rounds, how_to, need_project, need_project_data

project = need_project()
# Compact page so the aligner fits on one screen.
st.markdown("<style>.block-container{padding-top:2.5rem;padding-bottom:0}</style>", unsafe_allow_html=True)

if not need_project_data(project):
    st.stop()

# ---- 2D images or 3D stacks: the 3D stacks get their own aligner
from ramfish import sections as _sec   # noqa: E402
_stacks = _sec.stack_rounds(project)
_planes = [_sec.n_planes(project, r) for r in _stacks]
KIND_2D, KIND_3D = "2D: flat images", "3D: stacks of planes"
_sure = len(_stacks) >= 2 and min(_planes) >= 4          # clearly stacks
_guess = KIND_3D if len(_stacks) >= 2 else KIND_2D
_kind = project.meta.get("data_kind") or _guess
kind = st.radio("Your images are", [KIND_2D, KIND_3D], index=[KIND_2D, KIND_3D].index(_kind), horizontal=True,
                key="data_kind_pick",
                help="3D stacks are aligned as whole volumes (also up and down), then cut into one flat project per depth.")
if not project.meta.get("data_kind") and not _sure and _stacks:
    st.caption("The app could not tell for sure whether these are flat images or stacks: please choose.")
elif _stacks:
    st.caption("Detected: " + ", ".join(f"Round {r} has {n} planes" for r, n in zip(_stacks, _planes)) + ".")
if kind != project.meta.get("data_kind"):
    project.meta["data_kind"] = kind
    project.save()
if kind == KIND_3D:
    if len(_stacks) < 2:
        st.warning("3D alignment needs the stacks (all planes) of at least two rounds: import the microscope files in "
                   "**1 · Import & normalize**.")
        st.stop()
    from ui_align3d import render as _render3d   # noqa: E402
    _render3d(project)
    st.stop()

r0 = project.reference_round
rounds = alignable_rounds(project)
if not rounds:
    st.info("Import at least one more round with a `DAPI` or `mask` image.")
    st.stop()

t, c1, c2 = st.columns([1.3, 1, 1], vertical_alignment="bottom")
t.markdown("### 🧭 Rough alignment")
how_to("""
**What this step does:** lines up each round with Round 1 by hand, so that the automatic fine alignment (step 3)
starts close to the right answer. This matters most when a sample was rotated, flipped or shifted between rounds.

1. Pick a round under **Round to align**. Round 1 is always the reference and never moves.
2. In the dark viewer, **Round 1** is shown in **magenta** and the **selected round** (its DAPI image, or its mask
   if there is no DAPI) in **cyan**. Where the two match, the tissue turns **white**. Untick **Magenta + cyan
   colors** to see both in gray, with the selected round see-through on top.
3. Use the sliders until the two pictures overlap: **Rotation** turns the top picture, **X / Y Translate** slides it
   left–right / up–down, **Image Scale** makes it bigger or smaller, **Flip Horizontal** mirrors it, and **Target
   Opacity** makes it fainter or stronger. You can also type exact numbers in the boxes.
4. Zoom in with the **mouse wheel** and **drag** the picture to check the edges closely.
5. Click **Apply Rough Alignment to Round**: every image of that round is moved in exactly the same way.
   Then pick the next round.

A round that already overlaps well can be skipped. Saved in `2_rough_aligned/`, together with an unchanged copy
of Round 1. **Next:** **3 · Fine alignment**.
""", expanded=False)
rnd = c1.selectbox("Round to align (Round 1 is the fixed reference)", rounds,
                   format_func=lambda r: f"Round {r}" + (" ✓ applied" if str(r) in project.meta.get("rough_params", {}) else ""))
options = project.alignment_options(rnd)
if len(options) > 1:
    choice = c2.radio("Align using", options, horizontal=True, index=options.index(project.alignment_channel(rnd)))
    if choice != project.alignment_channel(rnd):
        project.set_alignment_channel(rnd, choice)
ch = project.alignment_channel(rnd)
ch0 = project.reference_partner(rnd, ch)


@st.cache_data(max_entries=16, show_spinner="Generating previews...")
def _preview(path: str, mtime: float) -> dict:
    return ra.preview(tifffile.imread(path))


ref_path, tgt_path = project.path("normalized", r0, ch0), project.path("normalized", rnd, ch)
ref = _preview(str(ref_path), ref_path.stat().st_mtime)
tgt = _preview(str(tgt_path), tgt_path.stat().st_mtime)
saved = project.meta.get("rough_params", {}).get(str(rnd), {})
initial = {k: float(saved.get(k, d)) for k, d in (("img_scale", 1.0), ("rotation", 0.0), ("tx", 0.0), ("ty", 0.0))}
initial["flip"] = bool(saved.get("flip", False))
status_key = f"rough_status_{rnd}"

value = rough_aligner(
    ref=ref, target=tgt, initial=initial, height=715, key=f"rough_{rnd}", default=None,
    token=f"{rnd}:{ch}:{tgt_path.stat().st_mtime}:{ref_path.stat().st_mtime}",
    info=f"Reference (magenta, underneath): <b>Round {r0} · {ch0}</b><br>Moving (cyan, on top): <b>Round {rnd} · {ch}</b>"
         "<ol style='color:#ccc; padding-left:18px; margin:8px 0 0 0'>"
         "<li>Move the sliders until the two pictures overlap (white where they match).</li>"
         "<li>Mouse wheel = zoom, drag = move the view.</li>"
         "<li>Click <b>Apply Rough Alignment to Round</b>.</li></ol>",
    status=st.session_state.get(status_key),
)

if value and value.get("nonce") != st.session_state.get(f"rough_done_{rnd}"):
    st.session_state[f"rough_done_{rnd}"] = value["nonce"]
    try:
        chans = project.channels(rnd)
        for c in chans:
            img = tifffile.imread(project.path("normalized", rnd, c))
            out = project.path("rough", rnd, c)
            out.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(out, ra.apply(img, value))
        project.write_reference_copies("rough")   # Round 1 unchanged, so the folder holds every round
        project.clear_stage("fine", rnd)
        params = {k: value[k] for k in ("tx", "ty", "rotation", "img_scale", "flip", "scale_ratio")}
        project.meta.setdefault("rough_params", {})[str(rnd)] = params
        project.save()
        project.log("rough_align", params, round=rnd, aligned_on=ch, channels=chans)
        msg = {"ok": True, "message": f"Successfully aligned {len(chans)} images in Round {rnd}!"}
    except Exception as exc:
        msg = {"ok": False, "message": f"Error: {exc}"}
    st.session_state[status_key] = {**msg, "nonce": value["nonce"]}
    st.rerun()
