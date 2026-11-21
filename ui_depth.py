"""The colored merge of a 3D project, one z-plane at a time (used by 2b · Sections and 5 · Multiplex merge)."""

from __future__ import annotations

import io

import cv2
import streamlit as st

from ramfish import depth as dp
from ramfish.project import Project


def depth_merge(project: Project, subs: list[Project], pick: Project | None = None) -> None:
    """Merge controls and picture. ``pick``: the z-plane to show (chosen elsewhere, e.g. the page's Z slider);
    None shows a Depth slider here."""
    counted = all((s.step_dir("spots") / "data_sheets" / "BATCH_SUMMARY_COUNTS.csv").exists() for s in subs)
    if pick is None:
        st.subheader("🌈 Merge across depth")
        st.caption("One colored picture per z-plane, drawn like **5 · Multiplex merge**. Choose the genes and colors "
                   "once, then drag the **Depth** slider to move through the tissue.")
    else:
        st.caption("Choose the genes and colors once, then drag the **Z depth** slider at the top: the picture follows "
                   "with the same settings. The depth is written in the corner of the picture.")
    sources = ([dp.SOURCE_SPOTS] if counted else []) + [dp.SOURCE_ALIGNED]
    m1, m2 = st.columns([1, 2])
    source = m1.radio("Source", sources, key="dp_source",
                      help="Synthetic spots: the dots of step 4 (run the spot counter on all sections first). "
                           "Aligned images: the aligned microscope images of step 3.")
    names = dp.layer_names(subs[0])
    chosen = m2.multiselect("Layers", names, default=[n for n in names if n.lower() != "mask"][:30], key="dp_layers")
    if chosen:
        colors = {}
        cc = st.columns(min(6, len(chosen)))
        k = 0
        for i, n in enumerate(chosen):
            default = dp.DAPI_GREY if n.upper() == "DAPI" else dp.PALETTE[k % len(dp.PALETTE)]
            if n.upper() != "DAPI":
                k += 1
            colors[n] = dp.hex_to_rgb(cc[i % len(cc)].color_picker(n, dp.rgb_to_hex(default), key=f"dp_col_{n}"))
        s1, s2, s3, s4 = st.columns(4)
        thr = s1.slider("Background Threshold", 0.0, 0.5, 0.15, 0.01, key="dp_bthr")
        gam = s2.slider("Gamma Correction", 0.5, 3.0, 1.5, 0.1, key="dp_gam")
        pos = s3.selectbox("Legend Position", ["top_right", "bottom_right", "top_left", "bottom_left"], key="dp_pos",
                           format_func=lambda p: p.replace("_", " ").title())
        tsc = s4.slider("Legend Size Scale", 0.2, 2.0, 0.65, 0.05, key="dp_tsc")
        labels = [f"z = {s.meta.get('section', {}).get('depth_um', 0):g} µm" for s in subs]
        if pick is None:
            depth_label = st.select_slider("Depth (in Round 1)", options=labels, key="dp_depth")
            sub = subs[labels.index(depth_label)]
        else:
            sub = pick
            depth_label = labels[[x.root for x in subs].index(pick.root)]

        @st.cache_data(max_entries=256, show_spinner=False)
        def _small_layer(root: str, name: str, source: str, stamp: float, max_dim: int = 1600):
            img = dp.layer_image(Project(root), name, source)
            if img is None:
                return None
            k = min(1.0, max_dim / max(img.shape))
            return cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA) if k < 1 else img

        def _stamp(s):
            f = s.root / ".ramfish_project.json"
            return f.stat().st_mtime if f.exists() else 0.0

        def _layers(s, small=True):
            out = []
            for n in chosen:
                img = _small_layer(str(s.root), n, source, _stamp(s)) if small else dp.layer_image(s, n, source)
                if img is not None:
                    out.append((n, img, colors[n]))
            return out

        lay = _layers(sub)
        missing = [n for n in chosen if n not in [x[0] for x in lay]]
        if missing:
            st.warning(f"Not available in this section: {', '.join(missing)}"
                       + (" (run the spot counter on all sections first)." if source == dp.SOURCE_SPOTS else "."))
        if lay:
            pic = dp.composite(lay, thr, gam, pos, tsc, label=depth_label)
            st.image(pic, caption=f"{sub.root.name}: {depth_label} below Round 1's first plane", width="stretch")

            def _full_jpg(s, lab):
                full = dp.composite(_layers(s, small=False), thr, gam, pos, tsc, label=lab)
                return cv2.imencode(".jpg", cv2.cvtColor(full, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()

            d1, d2 = st.columns(2)
            if d1.button("Prepare this depth at full size"):
                st.session_state.dp_one = (sub.root.name, _full_jpg(sub, depth_label))
            if st.session_state.get("dp_one", (None,))[0] == sub.root.name:
                d1.download_button(f"⬇️ Download {sub.root.name}.jpg", st.session_state.dp_one[1],
                                   file_name=f"{project.name}_{sub.root.name}_merge.jpg", mime="image/jpeg")
            if d2.button("Prepare all depths (.zip, also saved in 5_merge/)"):
                import zipfile
                buf = io.BytesIO()
                out_dir = project.step_dir("merge")
                out_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(buf, "w") as zf:
                    for s, lab in zip(subs, labels):
                        data = _full_jpg(s, lab)
                        (out_dir / f"depth_{s.root.name}.jpg").write_bytes(data)
                        zf.writestr(f"{project.name}_{s.root.name}_merge.jpg", data)
                st.session_state.dp_all = buf.getvalue()
            if st.session_state.get("dp_all"):
                d2.download_button("⬇️ Download all depths (.zip)", st.session_state.dp_all,
                                   file_name=f"{project.name}_merge_all_depths.zip", mime="application/zip")
