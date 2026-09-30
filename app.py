"""IOP Benthic Survey prep tool. Run locally:  streamlit run app.py"""
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import bruv_core as core

CAL_FILE = Path.home() / ".iop_bruv_calibration.json"
LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]

st.set_page_config(page_title="IOP Benthic Survey Prep", layout="wide")
st.title("Benthic survey: select & crop")

# ------------------------------------------------------------------ sidebar
loc = st.sidebar.selectbox("1. Project location", list(core.SITES))
site = st.sidebar.selectbox(
    "2. Dive site", core.SITES[loc], format_func=lambda s: f"{s[0]} ({s[1]}) - {s[2]}"
)
survey_date = st.sidebar.date_input("Survey date", date.today())
name = core.survey_name(survey_date, site[1])
st.sidebar.markdown("**Export name**")
st.sidebar.code(name, language=None)

src = st.sidebar.text_input("Source folder (contains subfolders 25, 20, 15 ...)")
out_root = st.sidebar.text_input("Output folder", str(Path.home() / "BenthicSurveys"))
depths = sorted(
    st.sidebar.multiselect("Depths (m)", [30, 25, 20, 15, 10, 5], default=[25, 20, 15, 10, 5]),
    reverse=True,
)
k = st.sidebar.number_input("Pictures per depth", 1, 10, 5)

L = core.layout(out_root, name)
ss = st.session_state
ss.setdefault("pts", [])
ss.setdefault("last_click", None)

tab1, tab2, tab3 = st.tabs(["1. Import", "2. Calibrate quadrat", "3. Select & export"])

# ------------------------------------------------------------------- import
with tab1:
    st.write(f"Creates `{L['base']}` with **reserved / chosen / cropped** folders.")
    if st.button("Create folders & import photos", disabled=not src):
        if not Path(src).is_dir():
            st.error("Source folder not found.")
        else:
            counts = core.ingest(src, out_root, name, depths)
            for d, c in counts.items():
                (st.success if c else st.warning)(f"{d} m: {c} photos" if c else f"{d} m: folder not found")

# --------------------------------------------------------------- calibration
with tab2:
    st.write(
        "The camera is fixed to the frame, so the quadrat sits in the same place in every photo. "
        "Click the 4 corners of the **string square** once, in order: "
        + ", ".join(LABELS) + "."
    )
    first = None
    for d in depths:
        dd = L["reserved"] / f"{d}m"
        if dd.is_dir() and core.list_images(dd):
            first = core.list_images(dd)[0]
            break
    if first is None:
        st.info("Import photos first.")
    else:
        rgb, _ = core.load_rgb(first)
        H, W = rgb.shape[:2]
        disp_w = 1100
        scale = W / disp_w
        disp = cv2.resize(rgb, (disp_w, int(H / scale)))
        for i, (x, y) in enumerate(ss.pts):
            cv2.circle(disp, (int(x / scale), int(y / scale)), 8, (255, 0, 0), -1)
            cv2.putText(disp, str(i + 1), (int(x / scale) + 10, int(y / scale) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        click = streamlit_image_coordinates(Image.fromarray(disp), key="cal")
        if click and (click["x"], click["y"]) != ss.last_click and len(ss.pts) < 4:
            ss.last_click = (click["x"], click["y"])
            ss.pts.append((click["x"] * scale, click["y"] * scale))
            st.rerun()

        c1, c2, c3 = st.columns(3)
        if len(ss.pts) < 4:
            c1.write(f"Next: **{LABELS[len(ss.pts)]}** corner")
        if c2.button("Reset points"):
            ss.pts, ss.last_click = [], None
            st.rerun()
        if c3.button("Load last calibration") and CAL_FILE.exists():
            ss.pts = json.loads(CAL_FILE.read_text())
            st.rerun()
        if len(ss.pts) == 4:
            CAL_FILE.write_text(json.dumps(ss.pts))
            (L["base"]).mkdir(parents=True, exist_ok=True)
            (L["base"] / "calibration.json").write_text(json.dumps(ss.pts))
            st.success("Calibration saved. Preview of the crop:")
            st.image(core.warp_quadrat(rgb, ss.pts), width=400)


# ------------------------------------------------------ selection & export
@st.cache_data(show_spinner=False)
def scored(folder, corners):
    files = core.list_images(folder)
    raw = [core.score_image(f, corners) for f in files]
    return [f.name for f in files], core.combine_scores(raw)


with tab3:
    if len(ss.pts) != 4:
        st.info("Calibrate the quadrat first (tab 2).")
    else:
        corners = tuple(map(tuple, ss.pts))
        selection = {}
        for d in depths:
            folder = L["reserved"] / f"{d}m"
            if not folder.is_dir():
                continue
            with st.spinner(f"Scoring {d} m ..."):
                names, sc = scored(str(folder), corners)
            if len(names) < k:
                st.warning(f"{d} m: only {len(names)} photos")
            auto = core.pick_evenly(names, sc, k)
            st.subheader(f"{d} m  ({len(names)} photos)")
            chosen = st.multiselect(f"Selected ({d} m)", names, default=auto, key=f"sel{d}")
            cols = st.columns(max(len(chosen), 1))
            for col, f in zip(cols, chosen):
                rgb, _ = core.load_rgb(folder / f)
                sq = cv2.resize(core.warp_quadrat(rgb, corners), (300, 300))
                col.image(sq, caption=f"{f}\nscore {sc[names.index(f)]:.2f}")
            if len(chosen) != k:
                st.warning(f"{d} m: {len(chosen)} selected, expected {k}")
            selection[d] = chosen

        st.caption("Check each thumbnail for obstructions (hand, pointer, torch) and that "
                   "at least 20% of the frame is coral/algae/rubble. Swap any bad ones above.")
        if st.button("Export to chosen + cropped", type="primary"):
            n = core.export(out_root, name, selection, corners)
            st.success(f"{n} photos copied to chosen and cropped in `{L['base']}`")
