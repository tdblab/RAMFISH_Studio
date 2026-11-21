"""Step 2 for 3D stacks: an interactive 3D view where each round's volume is placed onto Round 1's."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ramfish import align3d as a3, sections as sec
from ramfish.project import Project
from ui_common import how_to

MAGENTA, GREEN = "#d61f9c", "#2fbf4a"


@st.cache_data(max_entries=24, show_spinner="Reading the stack...")
def _small(root: str, r: int, use: str, stamp: str):
    return a3.small_stacks(Project(root), r, use)


def _lines(segments: list[np.ndarray], z_stretch: float):
    xs, ys, zs = [], [], []
    for s in segments:
        xs += list(s[:, 0]) + [None]
        ys += list(s[:, 1]) + [None]
        zs += list(-s[:, 2] * z_stretch) + [None]     # depth goes down
    return xs, ys, zs


def _overlay(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rgb = np.zeros(a.shape + (3,), np.float32)
    rgb[..., 0] = a
    rgb[..., 2] = a
    rgb[..., 1] = b
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def render(project: Project) -> None:
    rounds = sec.stack_rounds(project)
    r0 = rounds[0]
    st.markdown("### 🧭 Rough alignment (3D stacks)")
    how_to("""
**What this step does:** the rounds of a thick sample were not imaged from the same starting plane, and the
sample may sit shifted, turned or tilted. Here you place each round's whole volume onto Round 1's, roughly: it
does not need to be perfect, the fine alignment (step 3) does the rest.

1. Pick a round under **Round to place**. Round 1 is the fixed reference.
2. Choose what the volumes are drawn from: the **mask** (tissue outline made from the gene channels) or **DAPI**
   (tissue outline from the nuclei). Each plane's outline is drawn as a line: Round 1 in magenta, the round you
   move in green. Drag inside the 3D view to look around it; the mouse wheel zooms.
3. Click **Suggest a position** for a first guess, then move the round with the sliders: **Up / down** (depth, in
   µm) fixes a stack that started at another plane; **Left / right** and **Front / back** slide it; **Turn**,
   **Tilt** and **Scale** turn, tilt and resize it. **Stretch depth** only changes the view (stacks are thin).
4. Check one depth closely under **Check one depth**: Round 1 in magenta, the moved round in green, white where
   they overlap.
5. Click **Save this round's position**, then do the next round.
   Under **Leave out planes**, pick any bad plane (out of focus, empty, damaged) of any round: every depth that
   would use it is dropped.
6. When every round is saved, click **Make the z-planes**. Only the depths of Round 1 that every round covers
   are kept; for each, the matching plane of every round is cut from its stack and becomes one small flat
   project. Planes outside the shared volume are left out (their files are kept). **Next:** steps 3 to 5, which
   have a **Z depth** slider.
""", expanded=False)
    saved_all = project.meta.setdefault("rough3d", {})
    c1, c2 = st.columns([1, 2])
    r = c1.selectbox("Round to place (Round 1 is the fixed reference)", rounds[1:],
                     format_func=lambda x: f"Round {x}" + (" ✓ saved" if str(x) in saved_all else ""), key="a3_round")
    use = c2.radio("Draw the volumes from", [a3.USE_MASK, a3.USE_DAPI], horizontal=True,
                   format_func=lambda u: a3.USE_LABELS[u], key="a3_use",
                   index=[a3.USE_MASK, a3.USE_DAPI].index(project.meta.get("rough3d_use", a3.USE_MASK)))
    if use != project.meta.get("rough3d_use", a3.USE_MASK):
        project.meta["rough3d_use"] = use
        project.save()
    if use == a3.USE_DAPI:
        no_dapi = [x for x in rounds if not any(c.lower() == "dapi" for c in project.raw_channels(x))]
        if no_dapi:
            st.caption("No DAPI in Round " + ", ".join(map(str, no_dapi)) + ": its outline comes from the gene channels.")
    stamp = str(project.meta.get("raw", {}).get(str(r0), {}).keys()) + str(project.meta.get("history", [])[-1:])
    v0, s0, gs0 = _small(str(project.root), r0, use, stamp)
    vr, sr, gsr = _small(str(project.root), r, use, stamp)
    g0, gr = a3.geometry(project, r0), a3.geometry(project, r)

    # ---- the round's position (kept per round in the session; saved values come first)
    keys = {k: f"a3_{r}_{k}" for k in a3.KEYS}
    start = dict(a3.DEFAULT, **saved_all.get(str(r), {}))
    for k, key in keys.items():
        if key not in st.session_state:
            st.session_state[key] = start[k] if k == "flip" else float(start[k])
    w_um, h_um = g0["w"] * g0["px"], g0["h"] * g0["px"]
    span = (g0["n"] - 1) * g0["dz"] + (gr["n"] - 1) * gr["dz"]

    b1, b2, b3 = st.columns(3)
    if b1.button("🔮 Suggest a position", width="stretch",
                 help="Turn and shift from the tissue seen from above, then the depth where the plane outlines agree best."):
        with st.spinner("Looking for the best position..."):
            q = a3.auto_guess(project, r, use, small={r0: (v0, s0, gs0), r: (vr, sr, gsr)})
        for k, key in keys.items():
            st.session_state[key] = q[k] if k == "flip" else float(q[k])
        st.rerun()
    if b2.button("↺ Reset", width="stretch"):
        for k, key in keys.items():
            st.session_state[key] = a3.DEFAULT[k] if k == "flip" else float(a3.DEFAULT[k])
        st.rerun()

    left, right = st.columns([1, 2.2])
    with left:
        st.slider("Up / down (µm)", -float(span), float(span), step=0.5, key=keys["tz"],
                  help="Moves the whole stack deeper (+) or higher (−) against Round 1.")
        st.slider("Left / right (µm)", -w_um / 2, w_um / 2, step=1.0, key=keys["tx"])
        st.slider("Front / back (µm)", -h_um / 2, h_um / 2, step=1.0, key=keys["ty"])
        st.slider("Turn (degrees)", -180.0, 180.0, step=0.5, key=keys["rot"])
        st.slider("Tilt front / back (degrees)", -30.0, 30.0, step=0.5, key=keys["tilt_x"],
                  help="Tilted planes are cut across the stack, so they look softer than real planes.")
        st.slider("Tilt left / right (degrees)", -30.0, 30.0, step=0.5, key=keys["tilt_y"])
        st.slider("Scale", 0.5, 2.0, step=0.01, key=keys["scale"])
        st.checkbox("Mirror left–right", key=keys["flip"])
        z_stretch = st.slider("Stretch depth (view only)", 1.0, 20.0, 6.0, 0.5, key="a3_zstretch",
                              help="The stacks are much thinner than they are wide; stretching the depth makes the planes easier to tell apart.")
    p = {k: st.session_state[key] for k, key in keys.items()}

    # ---- 3D view
    seg0 = a3.outline_points(v0, gs0)
    segr = [a3.to_ref(s, gsr, gs0, p) for s in a3.outline_points(vr, gsr)]
    fig = go.Figure()
    for segs, col, name in ((seg0, MAGENTA, f"Round {r0}"), (segr, GREEN, f"Round {r}")):
        x, y, z = _lines(segs, z_stretch)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=col, width=3), name=name,
                                   hoverinfo="skip"))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0), uirevision="a3", legend=dict(x=0.02, y=0.98),
                      scene=dict(aspectmode="data", xaxis_title="X (µm)", yaxis_title="Y (µm)",
                                 zaxis_title=f"Depth (µm × {z_stretch:g})",
                                 yaxis=dict(autorange="reversed"), bgcolor="#0e1117",
                                 xaxis=dict(color="#bbb"), zaxis=dict(color="#bbb")))
    right.plotly_chart(fig, width="stretch", key="a3_fig")

    # ---- check one depth
    st.markdown("**Check one depth**")
    labels = [f"plane {z + 1} · {z * g0['dz']:.1f} µm" for z in range(g0["n"])]
    zl = st.select_slider(f"Round {r0} plane", options=labels, value=labels[len(labels) // 2], key="a3_check")
    z = labels.index(zl)
    cut_m = a3.cut_plane(vr, gsr, gs0, p, z * g0["dz"], v0.shape[1:], gs0["px"])
    cut_s = a3.cut_plane(sr, gsr, gs0, p, z * g0["dz"], v0.shape[1:], gs0["px"])
    k1, k2 = st.columns([1, 1.4])
    if cut_m is None:
        k1.warning(f"Round {r} has no plane at this depth.")
    else:
        zr = a3.matching_plane(project, r, p, z * g0["dz"])
        what = "DAPI" if use == a3.USE_DAPI else "genes"
        k1.image(_overlay(s0[z] / max(float(s0[z].max()), 1e-6), cut_s / max(float(cut_s.max()), 1e-6)), width="stretch",
                 caption=f"Round {r0} plane {z + 1} {what} (magenta) + Round {r} at the same depth, about its plane "
                         f"{zr + 1:.1f} (green): white where they agree")
        k1.caption(f"Outlines overlap: {a3._corr(v0[z], cut_m):.2f} (1 = perfect)")
    with k2:
        if st.button(f"✅ Save Round {r}'s position", type="primary"):
            saved_all[str(r)] = {k: (bool(v) if k == "flip" else round(float(v), 3)) for k, v in p.items()}
            project.meta["rough3d_use"] = use
            project.save()
            project.log("rough_align_3d_round", saved_all[str(r)], round=r, drawn_from=use)
            st.rerun()
        if str(r) in saved_all and saved_all[str(r)] != {k: (bool(v) if k == "flip" else round(float(v), 3)) for k, v in p.items()}:
            st.caption("The sliders differ from the saved position: save again to keep them.")

    # ---- planes to leave out (bad or empty planes, in any round)
    st.markdown("**Leave out planes**")
    st.caption("Planes that are bad (out of focus, empty, damaged) can be left out. Every depth that would use a "
               "left-out plane, of Round 1 or the matching plane of another round, is dropped from steps 3 to 5. "
               "The files are kept.")
    skip_all = {str(k): sorted(v) for k, v in a3.skipped(project).items()}
    cols = st.columns(len(rounds))
    for col, x in zip(cols, rounds):
        gx = g0 if x == r0 else a3.geometry(project, x)
        opts = list(range(gx["n"]))
        chosen = col.multiselect(f"Round {x}", opts, default=[z for z in skip_all.get(str(x), []) if z in opts],
                                 format_func=lambda z, gx=gx: f"plane {z + 1} · {z * gx['dz']:.1f} µm", key=f"a3_skip_{x}")
        if sorted(chosen) != skip_all.get(str(x), []):
            project.meta.setdefault("rough3d_skip", {})[str(x)] = sorted(int(z) for z in chosen)
            if not chosen:
                project.meta["rough3d_skip"].pop(str(x), None)
            project.save()

    # ---- the z-planes
    st.divider()
    missing = [x for x in rounds[1:] if str(x) not in saved_all]
    if missing:
        st.info("Save the position of Round " + ", ".join(map(str, missing)) + " to make the z-planes.")
        return
    keep = a3.kept_planes(project, saved_all, ref_small=(v0, gs0))
    if keep:
        st.success(f"Depths covered by every round: Round {r0} planes {keep[0] + 1} to {keep[-1] + 1} "
                   f"({keep[0] * g0['dz']:.1f} to {keep[-1] * g0['dz']:.1f} µm), {len(keep)} z-planes. "
                   + (f"The other {g0['n'] - len(keep)} plane{'s' if g0['n'] - len(keep) != 1 else ''} of Round {r0} "
                      "(outside the shared volume or left out above), and every plane of the other rounds outside this "
                      "volume, are left out of steps 3 to 5 (their files are kept)." if len(keep) < g0["n"] else
                      "Planes of the other rounds outside this volume are left out of steps 3 to 5 (their files are kept)."))
        rows = []
        for zz in keep:
            row = {f"Round {r0} plane": zz + 1, "Depth (µm)": round(zz * g0["dz"], 2)}
            for x in rounds[1:]:
                row[f"Round {x} plane"] = round(a3.matching_plane(project, x, saved_all[str(x)], zz * g0["dz"]) + 1, 1)
            rows.append(row)
        with st.expander("Which plane of every round goes with each depth"):
            st.dataframe(rows, hide_index=True)
    else:
        st.warning("No depth is covered by every round: check the up/down position of each round.")
    built = project.meta.get("rough3d_built", {})
    if built and (built.get("params") != saved_all or built.get("skip", {}) != project.meta.get("rough3d_skip", {})):
        st.warning("A position or a left-out plane changed after the z-planes were made: make them again.")
    export = st.checkbox("Also save the aligned planes as TIFFs for Fiji (2_rough_aligned/planes/Round_<n>/z<plane>_<depth>um/)",
                         value=True, key="a3_export")
    if sec.section_projects(project):
        st.caption("Making the z-planes again replaces the ones already made, with their results of steps 3 to 5.")
    if st.button("🧊 Make the z-planes", type="primary", disabled=not keep):
        bar = st.progress(0.0, text="Starting...")
        subs = a3.build_zplanes(project, saved_all, export=export, progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
        bar.progress(1.0, text=f"{len(subs)} z-planes made.")
        st.success(f"{len(subs)} z-planes made. Go on to **3 · Fine alignment**: it has a Z depth slider.")
