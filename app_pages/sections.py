import os

import cv2

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ramfish import sections as sec
from ramfish.io import to_display_u8
from ui_common import how_to, need_project, need_project_data, output_root

project = need_project()
st.title("🧊 Sections (3D stacks)")
how_to("""
**What this step is for:** thick samples imaged as 3D stacks (for example a zebrafish brain). Each round is a pile of
planes. A flat projection squeezes all planes into one picture, so cells from many depths land on top of each other.
Here the app works out which plane of every round shows the same tissue as each plane of Round 1, then bundles those
planes into one flat "z-plane" per depth. Steps 3 to 5 then treat each z-plane like a wing image. Thin samples such
as wings do not need this step.

Why the planes need matching: the rounds do not start at the same depth and have different plane spacings (for
example 3.2 µm in Round 1, 3.7 µm in Round 2), so plane 5 of Round 2 is usually not the same tissue as plane 5 of
Round 1.

1. Before this page: import all rounds in **1 · Import & normalize** (keep *Maximum projection of the whole stack*:
   the flat projections are only used for step 2) and line every round up with Round 1 in **2 · Rough alignment**.
2. **A · Export:** click **Export the rough-aligned z-planes**. Every plane of every round is moved with step 2's
   alignment and saved as a file, so all planes sit on top of Round 1.
3. **B · Match:** keep **Every z-plane of Round 1** and click **Suggest matching planes** for a first guess.
4. Check the guess by eye: pick a Round 1 plane with the top slider. Each other round shows its matching plane:
   the tissue outline of Round 1 in magenta and of the other round in green, white where they overlap. Move a
   round's slider until the colored rims are thinnest. (Choose **DAPI** under *Match planes using* to compare the
   nuclei instead.) The table lists all picks.
5. Click **Make the sections**. For every Round 1 plane, the matched planes of all rounds are bundled into one small
   flat project (in `sections/` inside this project), one per depth.
6. Go on to **3 · Fine alignment**, **4 · Spot counter** and **5 · Multiplex merge**. Each has a **Z depth** slider at
   the top to choose the depth you look at, and a button to run that step on every depth.
""")

if project.meta.get("section"):
    s = project.meta["section"]
    st.info(f"You are inside section **{project.name}** ({s.get('depth_um')} µm) of **{s.get('parent')}**. "
            "Use **⬅ Back to the main project** in the left menu to change or add sections.")
    st.stop()
if not need_project_data(project):
    st.stop()
rounds = sec.stack_rounds(project)
if len(rounds) < 2:
    st.info("This step is for 3D stacks: import at least two rounds whose files have more than one Z plane.")
    st.stop()

r0 = rounds[0]
n0, step0 = sec.n_planes(project, r0), sec.z_step(project, r0)
st.caption("Stacks: " + " · ".join(f"Round {r}: {sec.n_planes(project, r)} planes, {sec.z_step(project, r):g} µm apart "
                                     f"({(sec.n_planes(project, r) - 1) * sec.z_step(project, r):.0f} µm deep)" for r in rounds))
partial = []
for r in rounds:
    info = next((project.raw_info(r, c) for c in project.raw_channels(r)), {}) or {}
    planned, got = info.get("z_planes_planned"), info.get("z_planes")
    if planned and got and planned > got:
        partial.append(f"Round {r} ({got} of {planned} planes)")
if partial:
    st.warning("Part of the stack is missing: " + ", ".join(partial) + ". The microscope software saves the rest of a "
               "large .oir stack in extra files with the same name plus _00001, _00002, … (no extension). Import the "
               ".oir again together with those files (step 1) to get the whole depth.")
missing_rough = [r for r in rounds[1:] if str(r) not in project.meta.get("rough_params", {})]
if missing_rough:
    st.warning("Not lined up in step 2 yet: Round " + ", ".join(map(str, missing_rough)) + ". The suggestions will line "
               "these rounds up automatically, but the sections start better aligned after step 2.")

st.subheader("A · Export every z-plane after rough alignment")
st.caption("Every plane of every round is enlarged and normalized like step 1, moved with step 2's rough alignment and "
           "saved in `2_rough_aligned/planes/Round_<n>/z<plane>_<depth>um/` (one TIFF per gene, plus a tissue mask "
           "per plane for rounds without DAPI). Open them in Fiji if you like: they already line up with Round 1. "
           "The pictures below use them to match the planes.")
if sec.planes_exported(project):
    info = project.meta.get("planes_export", {})
    st.success(f"{info.get('files', 0)} plane images saved in `{project.step_dir('rough') / sec.PLANES_DIR}`.")
    if sec.planes_outdated(project):
        st.warning("The rough alignment changed after the export: export again so the planes line up.")
if st.button("📤 Export the rough-aligned z-planes", disabled=bool(missing_rough),
             help="Align every round in step 2 first." if missing_rough else None):
    bar = st.progress(0.0, text="Exporting...")
    n = sec.export_planes(project, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
    bar.empty()
    st.session_state.pop("pick_cache", None)
    st.rerun()

st.subheader("B · Match the planes")
plan = project.meta.get("section_plan", {})
EVERY = "Every z-plane of Round 1 (recommended)"
SPACED = "Thicker sections at a set spacing"
pmode = st.radio("Planes to analyze", [EVERY, SPACED], index=0 if plan.get("mode", "every") == "every" else 1,
                 horizontal=True, key="sec_mode",
                 help="Every z-plane: each plane of Round 1 (and the matching plane of every other round) is aligned, "
                      "counted and merged on its own. Sections: a few planes projected together, every few µm.")
c1, c2, c3, c4 = st.columns(4)
if pmode == EVERY:
    first, spacing, slab = 0.0, float(step0), float(step0)    # one plane each (a slab of one plane step)
    c1.metric("z-planes", n0)
    c2.metric("Step", f"{step0:g} µm")
    c3.metric("Depth", f"{(n0 - 1) * step0:.0f} µm")
else:
    first = c1.number_input("First section (µm below Round 1's first plane)", 0.0, float((n0 - 1) * step0),
                            float(plan.get("first_um", min(3.0, (n0 - 1) * step0))), 0.5)
    spacing = c2.number_input("Distance between sections (µm)", 1.0, 500.0, float(plan.get("spacing_um", 10.0)), 0.5)
    slab = c3.number_input("Section thickness (µm)", 0.5, 100.0, float(plan.get("slab_um", round(2 * step0 + 0.01, 1))), 0.5,
                           help="The planes within this thickness around the matching plane are projected into one image. "
                                "About 2 to 3 planes keeps cells from neighboring sections out.")
USES = [sec.COMPARE_MASK, sec.COMPARE_DAPI, sec.COMPARE_GENES]
use = c4.radio("Match planes using", USES, index=USES.index(plan.get("use", sec.COMPARE_MASK)) if plan.get("use", sec.COMPARE_MASK) in USES else 0,
               format_func=lambda u: sec.COMPARE_LABELS[u],
               help="Tissue outline (default): the edge of the tissue in each plane, made like step 1's mask; the same "
                    "kind of picture in every round. DAPI: the nuclei, used only for rounds that have DAPI. Used for "
                    "the suggestion and for the pictures you check by eye.")
depths = sec.section_depths(n0, step0, first, spacing)
st.markdown(f"**{len(depths)} section(s)** at " + ", ".join(f"{d:g} µm" for d in depths) + " in Round 1.")
project.meta["section_plan"] = {"mode": "every" if pmode == EVERY else "spaced", "first_um": first,
                                "spacing_um": spacing, "slab_um": slab, "use": use}

if st.button("🔎 Suggest matching planes", type="primary"):
    bar = st.progress(0.0, text="Reading the stacks...")
    sec.suggest_planes(project, use, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
    bar.empty()
    for k in [k for k in st.session_state if str(k).startswith(("secz_", "pick_"))]:
        del st.session_state[k]
    project.meta.pop("plane_picks", None)
    project.save()
    st.rerun()
project.save()
sug = project.meta.get("section_suggestions", {})
if sug:
    rows = [{"Round": int(r), "Lined up in X/Y by": s["xy"], "Compared using": sec.COMPARE_LABELS.get(s.get("compared"), s.get("compared")),
             "Agreement (−1 to 1)": s["score"], "Upside down": s["upside_down"]} for r, s in sug.get("rounds", {}).items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if any(s["upside_down"] for s in sug.get("rounds", {}).values()):
        st.warning("A round seems to be mounted the other way up: its planes were matched from the other end. Check it.")


@st.cache_data(max_entries=8, show_spinner="Preparing the pictures...")
def _display(root: str, r: int, use: str, stamp: str):
    moved, ref = sec.warp_for_display(sec.Project(root), r, use)
    return moved, ref


def _stamp(r):
    p = project.path("raw", r, project.raw_channels(r)[0])
    return f"{p.stat().st_mtime}:{project.meta.get('rough_params', {}).get(str(r))}"


def _overlay(a, b):
    a8, b8 = to_display_u8(a), to_display_u8(b)
    return np.dstack([a8, b8, a8])


@st.cache_data(max_entries=256, show_spinner=False)
def _plane(root: str, r: int, z: int, use: str, stamp: str):
    return sec.plane_view(sec.Project(root), r, z, use, max_dim=900)


def _pick_view():
    """Round 1's plane next to the chosen plane of every other round (tissue outline by default, or DAPI / genes)."""
    stamp = str(project.meta.get("planes_export", {}))
    picks = project.meta.setdefault("plane_picks", {})
    rows = sec.picked_planes(project, depths)
    labels = [f"plane {row['planes'][r0] + 1} · {row['depth_um']:g} µm" for row in rows]
    st.caption(f"Pick, for each plane of Round {r0}, the plane of every other round that shows the same depth: Round "
               f"{r0} in magenta, the other round in green, white where they agree. With the tissue outline, the "
               "colored rims show where the edges differ: move the slider until the rims are thinnest. Start from the "
               "suggestion. Your picks are saved with the project.")
    i = labels.index(st.select_slider(f"Round {r0} plane", options=labels, key="pick_nav"))
    row = rows[i]
    z0 = row["planes"][r0]
    cols = st.columns(len(rounds[1:]))
    for col, r in zip(cols, rounds[1:]):
        n_r, step_r = sec.n_planes(project, r), sec.z_step(project, r)
        key = f"pick_{z0}_{r}"
        if key not in st.session_state:
            st.session_state[key] = int(row["planes"][r]) + 1
        z = col.slider(f"Round {r} plane", 1, n_r, key=key) - 1
        if z != int(row["planes"][r]):
            picks.setdefault(str(z0), {})[str(r)] = int(z)
            project.save()
        a, what_a = _plane(str(project.root), r0, z0, use, stamp)
        b, what_b = _plane(str(project.root), r, z, use, stamp)
        if a is None or b is None:
            col.warning("No picture for this plane: export the z-planes again (step A).")
            continue
        if what_a != what_b and what_b.startswith("outline"):   # this round has no DAPI: compare outlines
            a, what_a = _plane(str(project.root), r0, z0, sec.COMPARE_MASK, stamp)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        col.image(_overlay(a, b), width="stretch",
                  caption=f"Round {r0} plane {z0 + 1} {what_a} (magenta) + Round {r} plane {z + 1} {what_b}, "
                          f"{z * step_r:.1f} µm (green)")
    rows = sec.picked_planes(project, depths)
    table = pd.DataFrame([{f"Round {r0} plane": row["planes"][r0] + 1, "Depth (µm)": row["depth_um"],
                           **{f"Round {r}": row["planes"][r] + 1 for r in rounds[1:]}} for row in rows])
    st.dataframe(table, hide_index=True)
    back = [r for r in rounds[1:] if any(b["planes"][r] < a["planes"][r] for a, b in zip(rows, rows[1:]))]
    if back:
        st.warning("The planes go back up in Round " + ", ".join(map(str, back)) + ": check those picks.")
    return rows


chosen = []
if sec.planes_exported(project):
    chosen = _pick_view()
for k, depth in enumerate([] if chosen else depths, 1):
    planes = sec.planes_for_depth(project, sug, depth)
    with st.expander(f"Section {k}: {depth:g} µm (Round {r0} plane {planes[r0]})", expanded=(k == 1)):
        ref_stack = None
        cols = st.columns(len(rounds[1:]))
        picked = {r0: planes[r0]}
        for col, r in zip(cols, rounds[1:]):
            key = f"secz_{k}_{r}"
            if key not in st.session_state:
                st.session_state[key] = int(planes[r])
            z = col.slider(f"Round {r} plane", 0, sec.n_planes(project, r) - 1, key=key,
                           help=f"Suggested: plane {planes[r]} ({planes[r] * sec.z_step(project, r):.0f} µm)")
            picked[r] = int(z)
            moved, ref = _display(str(project.root), r, use, _stamp(r) + _stamp(r0))
            col.image(_overlay(ref[planes[r0]], moved[z]), width="stretch",
                      caption=f"Round {r0} plane {planes[r0]} (magenta) + Round {r} plane {z}, {z * sec.z_step(project, r):.0f} µm (green)")
        chosen.append({"depth_um": depth, "planes": picked})

st.divider()
if sec.section_projects(project):
    st.caption("Making the sections again replaces the ones already made, with their results of steps 3 to 5.")
if st.button("🧊 Make the sections", type="primary", disabled=not depths):
    bar = st.progress(0.0, text="Making the sections...")
    subs = sec.build_sections(project, chosen, slab, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
    bar.empty()
    st.success(f"Made {len(subs)} section(s) in `{project.root / sec.SECTIONS_DIR}`.")


def _open(sub):
    st.session_state.project_name = sub.root.relative_to(output_root()).as_posix()
    st.session_state.pop("project_zip", None)


subs = sec.section_projects(project)
if subs:
    st.subheader("Sections")
    pinned = bool(os.environ.get("RAMFISH_PROJECT_DIR"))
    for sub in subs:
        s = sub.meta.get("section", {})
        fine = any(sub.step_dir("fine").glob("R*_aligned_*")) if sub.step_dir("fine").exists() else False
        spots = (sub.step_dir("spots") / "data_sheets" / "BATCH_SUMMARY_COUNTS.csv").exists()
        a, b = st.columns([4, 1])
        a.markdown(f"**{sub.name}** · {s.get('depth_um')} µm · planes " +
                   ", ".join(f"R{r}: {z}" for r, z in s.get("planes", {}).items()) +
                   f" · fine alignment {'✓' if fine else '–'} · spots {'✓' if spots else '–'}")
        if not pinned and b.button("Open", key=f"open_{sub.name}"):
            _open(sub)
            st.switch_page("app_pages/fine_align.py")
    table = sec.combined_counts(project)
    if not table.empty:
        st.subheader("Counts of all sections")
        st.dataframe(table, hide_index=True)
        st.download_button("📋 Download counts of all sections", table.to_csv(index=False).encode("utf-8"),
                           file_name=f"{project.name}_sections_counts.csv", mime="text/csv")

# ==========================================================================================================
# Steps 3 and 4 on all sections, and the merge across depth
# ==========================================================================================================
from ramfish import depth as dp
from ui_depth import depth_merge
from ramfish.spots import DEFAULT_CONFIG

subs = sec.section_projects(project)
if subs:
    st.divider()
    st.subheader("▶ Fine alignment and spot counter on all sections")
    st.caption("Runs **3 · Fine alignment** and then **4 · Spot counter** on every section in turn, with the settings "
               "below (the same as on those pages). Large stacks take a few minutes per section.")
    with st.expander("Spot counter settings (the defaults suit most images)"):
        d = DEFAULT_CONFIG["settings"]
        sc = st.columns(3)
        spot_settings = {
            "blob_min_sigma": sc[0].number_input("blob_min_sigma", 0.5, 50.0, float(d["blob_min_sigma"]), 0.5, key="dp_min"),
            "blob_max_sigma": sc[0].number_input("blob_max_sigma", 0.5, 50.0, float(d["blob_max_sigma"]), 0.5, key="dp_max"),
            "blob_threshold": sc[1].number_input("blob_threshold", 0.0, 1.0, float(d["blob_threshold"]), 0.01, key="dp_thr"),
            "intensity_cutoff": sc[1].number_input("intensity_cutoff", 0.0, 1.0, float(d["intensity_cutoff"]), 0.01, key="dp_cut"),
            "blob_num_sigma": sc[2].number_input("blob_num_sigma", 1, 50, int(d["blob_num_sigma"]), 1, key="dp_num"),
            "synthetic_spot_radius": sc[2].number_input("synthetic_spot_radius", 1, 50, int(d["synthetic_spot_radius"]), 1, key="dp_rad"),
        }
        bend = st.checkbox("Rounds aligned on a mask: also correct local stretching (B-Spline)", value=False, key="dp_bend",
                           help="Off (recommended), as on the Fine alignment page.")
    if st.button("▶ Run fine alignment and spot counter on all sections", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        dp.run_sections(subs, spot_settings, bend, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
        bar.empty()
        st.success(f"Done: {len(subs)} section(s) aligned and counted.")
        st.rerun()

    st.divider()
    depth_merge(project, subs)
