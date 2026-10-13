"""IOP Benthic Survey prep tool — v2.  Run locally:  streamlit run app.py"""
import json
from datetime import date
from pathlib import Path
 
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
 
import benthic_core as core
 
CAL_FILE = Path.home() / ".iop_bruv_calibration.json"
LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]
CAL_THUMB_N = 12          # how many thumbnails shown in picker
THUMB_W = 160             # display width of each calibration thumbnail (px)
HIGH_OBS_CONF = 0.70      # auto-reject obstruction threshold
 
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
 
# BM collision warning
if site[1] == "BM":
    st.sidebar.warning(
        "⚠️ Code **BM** is shared by Batumulapan (Penida) and Blue Magic (Raja). "
        "The exported folder name alone does not disambiguate. "
        "Recommend the org rename one of the two codes."
    )
 
def select_folder(title="Select Folder", initialdir=""):
    import tkinter as tk
    from tkinter import filedialog
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        folder = filedialog.askdirectory(master=root, title=title, initialdir=initialdir if initialdir else None)
        root.destroy()
        return folder
    except Exception:
        return ""
 
if "src_path" not in st.session_state:
    st.session_state["src_path"] = ""
if "out_root_path" not in st.session_state:
    st.session_state["out_root_path"] = str(Path.home() / "BenthicSurveys")
 
st.sidebar.markdown("**Source folder** *(contains 25, 20, 15...)*")
col_src1, col_src2 = st.sidebar.columns([0.8, 0.2])
with col_src1:
    src_val = st.text_input("Source folder", value=st.session_state["src_path"], label_visibility="collapsed", key="src_input")
    if src_val != st.session_state["src_path"]:
        st.session_state["src_path"] = src_val
with col_src2:
    if st.button("📁", key="browse_src", help="Browse source folder"):
        chosen = select_folder("Select Source Folder", st.session_state["src_path"])
        if chosen:
            st.session_state["src_path"] = chosen
            st.rerun()
 
src = st.session_state["src_path"]
 
st.sidebar.markdown("**Output folder**")
col_out1, col_out2 = st.sidebar.columns([0.8, 0.2])
with col_out1:
    out_val = st.text_input("Output folder", value=st.session_state["out_root_path"], label_visibility="collapsed", key="out_input")
    if out_val != st.session_state["out_root_path"]:
        st.session_state["out_root_path"] = out_val
with col_out2:
    if st.button("📁", key="browse_out", help="Browse output folder"):
        chosen = select_folder("Select Output Folder", st.session_state["out_root_path"])
        if chosen:
            st.session_state["out_root_path"] = chosen
            st.rerun()
 
out_root = st.session_state["out_root_path"]
depths = sorted(
    st.sidebar.multiselect("Depths (m)", [30, 25, 20, 15, 10, 5], default=[25, 20, 15, 10, 5]),
    reverse=True,
)
k = st.sidebar.number_input("Pictures per depth", 1, 10, 5)
auto_detect_on = st.sidebar.toggle("Auto-detect corners per photo", value=False)
cal_thumb_n = st.sidebar.number_input(
    "Calibration thumbnail count", min_value=4, max_value=36, value=CAL_THUMB_N
)
 
L = core.layout(out_root, name)
ss = st.session_state
ss.setdefault("pts", [])
ss.setdefault("last_click", None)
ss.setdefault("cal_source_idx", 0)   # which thumbnail was chosen for calibration
ss.setdefault("cal_source_depth", None)
 
# ---- review state
ss.setdefault("review_active", False)
ss.setdefault("seg_candidates", {})   # {depth: {seg_i: [ranked photo names]}}
ss.setdefault("cand_idx", {})          # {(depth, seg_i): index currently shown/selected}
ss.setdefault("auto_rejected", [])     # [(depth, seg_i, photo, obs_conf)]
ss.setdefault("lightbox", None)        # (depth, seg_i) of the open viewer, or None
 
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
                (st.success if c else st.warning)(
                    f"{d} m: {c} photos" if c else f"{d} m: folder not found"
                )
 
# --------------------------------------------------------------- calibration
with tab2:
    st.write(
        "Camera is fixed to frame — calibrate once per survey. "
        "**Step 1:** pick the cleanest thumbnail (all 4 string corners visible, no hands). "
        "**Step 2:** click the 4 corners in order: " + ", ".join(LABELS) + "."
    )
 
    # Collect candidate images for the picker
    cal_candidates = []
    for d in depths:
        dd = L["reserved"] / f"{d}m"
        if dd.is_dir():
            imgs = core.list_images(dd)
            if imgs:
                cal_candidates = imgs[:cal_thumb_n]
                if ss.cal_source_depth is None:
                    ss.cal_source_depth = d
                break
 
    if not cal_candidates:
        st.info("Import photos first.")
    else:
        # --- thumbnail grid picker
        st.markdown("#### Step 1 — pick calibration image")
        chosen_source = cal_candidates[ss.cal_source_idx]
        cols_per_row = min(6, len(cal_candidates))
        rows = [cal_candidates[i:i + cols_per_row] for i in range(0, len(cal_candidates), cols_per_row)]
        for row in rows:
            thumb_cols = st.columns(len(row))
            for ci, (col, img_path) in enumerate(zip(thumb_cols, row)):
                global_idx = cal_candidates.index(img_path)
                rgb_t, _ = core.load_rgb(img_path)
                thumb = cv2.resize(rgb_t, (THUMB_W, THUMB_W))
                border_color = "#00cc66" if global_idx == ss.cal_source_idx else "#444"
                col.markdown(
                    f'<div style="border:3px solid {border_color};border-radius:4px;'
                    f'display:inline-block;margin:2px;">'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                col.image(thumb, caption=f"#{global_idx + 1}", width=THUMB_W)
                if col.button(f"Use #{global_idx + 1}", key=f"cal_pick_{global_idx}",
                              type="primary" if global_idx == ss.cal_source_idx else "secondary"):
                    ss.cal_source_idx = global_idx
                    ss.pts = []
                    ss.last_click = None
                    st.rerun()
 
        st.divider()
 
        # --- 4-click corner calibration on chosen source
        st.markdown(f"#### Step 2 — click corners on image #{ss.cal_source_idx + 1}")
        rgb, _ = core.load_rgb(chosen_source)
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
            cal_data = {
                "corners": ss.pts,
                "source_image": str(chosen_source),
                "source_depth": ss.cal_source_depth,
                "source_thumb_idx": ss.cal_source_idx,
            }
            CAL_FILE.write_text(json.dumps(cal_data))
            (L["base"]).mkdir(parents=True, exist_ok=True)
            (L["base"] / "calibration.json").write_text(json.dumps(cal_data))
            st.success("Calibration saved. Preview of the crop:")
            st.image(core.warp_quadrat(rgb, ss.pts), width=400)
 
 
# ------------------------------------------------------ selection & export
@st.cache_data(show_spinner=False)
def scored(folder, corners):
    files = core.list_images(folder)
    raw = [core.score_image(f, corners) for f in files]
    return [f.name for f in files], core.combine_scores(raw)
 
 
def _get_corners_for_photo(rgb, fallback_corners):
    """Return corners to use for this photo, respecting auto_detect_on toggle."""
    if not auto_detect_on:
        return fallback_corners, 1.0, "manual"
    detected, conf, method = core.detect_corners(rgb)
    if detected is not None and method == "classical":
        return detected, conf, method
    return fallback_corners, conf, "fallback"
 
 
def _build_seg_candidates(depths, L, corners, k):
    """Pre-compute ranked candidate lists per depth/segment. Returns dict."""
    result = {}
    for d in depths:
        folder = L["reserved"] / f"{d}m"
        if not folder.is_dir():
            continue
        names, sc = scored(str(folder), tuple(map(tuple, corners)))
        segs = np.array_split(np.arange(len(names)), k)
        result[d] = {}
        for seg_i, seg in enumerate(segs):
            ranked = sorted(seg, key=lambda i: sc[i], reverse=True)
            result[d][seg_i] = [names[i] for i in ranked]
    return result
 
 
GRID_THUMB = 180   # px, square thumbnail in the grid
LB_SIZE = 640       # px, lightbox square image size
ICON_R = 22          # px, reject-icon radius
ICON_MARGIN = 16     # px, distance of icon centre from top-right corner
 
 
def _draw_reject_icon(sq_img):
    """Return a copy of sq_img with an X-in-circle drawn top-right, plus (cx, cy, r)."""
    h, w = sq_img.shape[:2]
    cx, cy, r = w - ICON_MARGIN - ICON_R, ICON_MARGIN + ICON_R, ICON_R
    out = sq_img.copy()
    overlay = out.copy()
    cv2.circle(overlay, (cx, cy), r, (30, 30, 30), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    d = int(r * 0.5)
    cv2.line(out, (cx - d, cy - d), (cx + d, cy + d), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(out, (cx - d, cy + d), (cx + d, cy - d), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), r, (255, 255, 255), 2, cv2.LINE_AA)
    return out, (cx, cy, r)
 
 
def _candidate_image(folder, photo, corners, size):
    rgb, _ = core.load_rgb(folder / photo)
    per_photo_corners, conf, method = _get_corners_for_photo(rgb, corners) if auto_detect_on \
        else (corners, 1.0, "manual")
    sq = cv2.resize(core.warp_quadrat(rgb, per_photo_corners), (size, size))
    return sq, conf, method
 
 
def _reject_current(d, seg_i, photo, score_val):
    """Drop `photo` from this segment's candidate pool; pull from the next segment if exhausted."""
    ss.seg_candidates[d][seg_i] = [p for p in ss.seg_candidates[d][seg_i] if p != photo]
    core.log_decision(L["base"], {
        "photo": photo, "depth": d, "segment": seg_i, "action": "reject", "score": score_val,
    })
    ss.cand_idx[(d, seg_i)] = 0
    if not ss.seg_candidates[d][seg_i]:
        adj_seg = (seg_i + 1) % k
        used = {ss.seg_candidates[d][s][ss.cand_idx.get((d, s), 0)]
                for s in ss.seg_candidates[d] if ss.seg_candidates[d][s]}
        pulled = [p for p in ss.seg_candidates.get(d, {}).get(adj_seg, []) if p not in used]
        if pulled:
            ss.seg_candidates[d][seg_i] = pulled
            st.toast(f"Segment {seg_i + 1} ran out — pulled a spare from segment {adj_seg + 1}.")
 
 
with tab3:
    if len(ss.pts) != 4:
        st.info("Calibrate the quadrat first (tab 2).")
    else:
        corners = ss.pts
 
        if not ss.review_active:
            if st.button("Build selection grid", type="primary"):
                with st.spinner("Scoring all depths..."):
                    ss.seg_candidates = _build_seg_candidates(depths, L, corners, k)
                ss.cand_idx = {}
                ss.auto_rejected = []
                ss.lightbox = None
                ss.review_active = True
                st.rerun()
        else:
            active_depths = [d for d in depths if d in ss.seg_candidates]
 
            # ---- auto-reject high-confidence obstructions, once, before showing anything
            if not ss.get("_obs_scanned"):
                for d in active_depths:
                    folder = L["reserved"] / f"{d}m"
                    for seg_i, cands in ss.seg_candidates[d].items():
                        kept = []
                        for photo in cands:
                            rgb_c, _ = core.load_rgb(folder / photo)
                            sq_c = core.warp_quadrat(rgb_c, corners)
                            has_obs, _, obs_conf = core.detect_obstruction(sq_c)
                            if has_obs and obs_conf >= HIGH_OBS_CONF:
                                ss.auto_rejected.append((d, seg_i, photo, obs_conf))
                                core.log_decision(L["base"], {
                                    "photo": photo, "depth": d, "segment": seg_i,
                                    "action": "auto_reject", "score": 0.0,
                                    "obstruction_conf": obs_conf,
                                })
                            else:
                                kept.append(photo)
                        ss.seg_candidates[d][seg_i] = kept or cands  # never leave a segment empty here
                ss._obs_scanned = True
 
            # ================================================= LIGHTBOX VIEW
            if ss.lightbox is not None:
                d, seg_i = ss.lightbox
                cands = ss.seg_candidates[d][seg_i]
                if not cands:
                    st.error(f"No candidates left for {d} m, segment {seg_i + 1}.")
                    if st.button("‹ Back to grid"):
                        ss.lightbox = None
                        st.rerun()
                else:
                    idx = min(ss.cand_idx.get((d, seg_i), 0), len(cands) - 1)
                    photo = cands[idx]
                    folder = L["reserved"] / f"{d}m"
                    sq, corner_conf, corner_method = _candidate_image(folder, photo, corners, LB_SIZE)
                    disp, (icx, icy, ir) = _draw_reject_icon(sq)
 
                    names_d, sc_d = scored(str(folder), tuple(map(tuple, corners)))
                    score_val = sc_d[names_d.index(photo)] if photo in names_d else 0.0
 
                    st.markdown(
                        f"**{d} m — segment {seg_i + 1} of {k}**  &nbsp;·&nbsp; "
                        f"candidate {idx + 1} of {len(cands)}  &nbsp;·&nbsp; score {score_val:.2f}"
                        + (f"  &nbsp;·&nbsp; corners: {corner_method} ({corner_conf:.0%})"
                           if auto_detect_on else "")
                    )
 
                    click = streamlit_image_coordinates(Image.fromarray(disp), key=f"lb_{d}_{seg_i}_{idx}")
                    if click:
                        cx, cy = click["x"], click["y"]
                        if (cx - icx) ** 2 + (cy - icy) ** 2 <= (ir + 6) ** 2:
                            _reject_current(d, seg_i, photo, score_val)
                            st.rerun()
                        elif cx < LB_SIZE * 0.33 and idx > 0:
                            ss.cand_idx[(d, seg_i)] = idx - 1
                            st.rerun()
                        elif cx > LB_SIZE * 0.66 and idx < len(cands) - 1:
                            ss.cand_idx[(d, seg_i)] = idx + 1
                            st.rerun()
 
                    st.caption(f"`{photo}` — click the left/right edge of the photo to slide between "
                               "candidates, or the ✕ to reject it and jump to the next best.")
 
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    if bc1.button("‹ Slide left", disabled=idx == 0, use_container_width=True):
                        ss.cand_idx[(d, seg_i)] = idx - 1
                        st.rerun()
                    if bc2.button("✕ Reject", use_container_width=True):
                        _reject_current(d, seg_i, photo, score_val)
                        st.rerun()
                    if bc3.button("Slide right ›", disabled=idx == len(cands) - 1, use_container_width=True):
                        ss.cand_idx[(d, seg_i)] = idx + 1
                        st.rerun()
                    if bc4.button("Back to grid", type="primary", use_container_width=True):
                        ss.lightbox = None
                        st.rerun()
 
            # =================================================== GRID VIEW
            else:
                st.caption("Click any photo to open it — slide left/right there to browse ranked "
                           "candidates, click the ✕ to reject and auto-advance.")
                for d in active_depths:
                    st.subheader(f"{d} m")
                    folder = L["reserved"] / f"{d}m"
                    cols = st.columns(k)
                    for seg_i in range(k):
                        cands = ss.seg_candidates[d].get(seg_i, [])
                        with cols[seg_i]:
                            if not cands:
                                st.error(f"seg {seg_i + 1}: none left")
                                continue
                            idx = min(ss.cand_idx.get((d, seg_i), 0), len(cands) - 1)
                            photo = cands[idx]
                            sq, _, _ = _candidate_image(folder, photo, corners, GRID_THUMB)
                            st.image(sq, caption=f"seg {seg_i + 1}  ({len(cands)} left)")
                            if st.button("🔍 View", key=f"view_{d}_{seg_i}", use_container_width=True):
                                ss.lightbox = (d, seg_i)
                                st.rerun()
 
            if ss.auto_rejected:
                with st.expander(f"Auto-rejected photos ({len(ss.auto_rejected)})"):
                    for depth, seg_i, photo, obs_conf in ss.auto_rejected:
                        st.write(f"• {photo} — depth {depth} m, segment {seg_i + 1}, "
                                 f"obstruction conf {obs_conf:.0%}")
 
        # --- Export section (always visible once the grid has been built)
        st.divider()
        selection = {}
        for d in ss.seg_candidates if ss.review_active else []:
            picks = []
            for seg_i in range(k):
                cands = ss.seg_candidates[d].get(seg_i, [])
                if cands:
                    idx = min(ss.cand_idx.get((d, seg_i), 0), len(cands) - 1)
                    picks.append(cands[idx])
            selection[d] = picks
 
        if any(selection.values()):
            st.subheader("Current picks (top of each segment unless you slid/rejected)")
            for d, photos in selection.items():
                if photos:
                    st.markdown(f"**{d} m:** " + ", ".join(photos))
 
        st.caption(
            "Check each photo for obstructions (hand, pointer, torch) and that "
            "at least 20% of the frame is coral/algae/rubble."
        )
 
        export_ready = ss.review_active and all(
            len(selection.get(d, [])) == k for d in depths if d in ss.seg_candidates
        )
        if st.button("Export to chosen + cropped", type="primary", disabled=not export_ready):
            n = core.export(out_root, name, selection, corners, auto_detect=auto_detect_on)
            st.success(f"{n} photos copied to chosen and cropped in `{L['base']}`")
            warn_path = L["base"] / "crop_warnings.txt"
            if warn_path.exists():
                st.warning(f"Some photos used fallback corners — see `{warn_path.name}`")