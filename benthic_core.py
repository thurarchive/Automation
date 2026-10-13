"""Core logic for the IOP benthic survey prep tool (no UI, no internet)."""
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ------------------------------------------------------------------ v2 helpers

# ---- corner detection (Step A classical; Step B / YOLO deferred to v3) -------

# HSV range for the blue string rope.  Generous to tolerate colour-cast at depth.
_STRING_HUE_LO = np.array([90, 40, 40], np.uint8)
_STRING_HUE_HI = np.array([140, 255, 255], np.uint8)

_CONF_THRESHOLD = 0.70  # min confidence to auto-accept classical result


def _cluster_lines(lines, n_clusters=2):
    """Group HoughLinesP segments by angle into n_clusters dominant orientations.
    Uses a simple median split — no sklearn needed."""
    if lines is None or len(lines) == 0:
        return []
    segs = lines.reshape(-1, 4)  # (N, 4) from HoughLinesP (N, 1, 4)
    angles = np.array([np.arctan2(float(y2 - y1), float(x2 - x1))
                       for x1, y1, x2, y2 in segs])
    # fold to [0, pi)
    angles = np.where(angles < 0, angles + np.pi, angles)
    # Split at median angle into two orientation groups
    median = float(np.median(angles))
    groups = [
        segs[np.abs(angles - median) < np.pi / 4],
        segs[np.abs(angles - median) >= np.pi / 4],
    ]
    reps = []
    for grp in groups:
        if len(grp) == 0:
            continue
        lengths = np.hypot(grp[:, 2] - grp[:, 0], grp[:, 3] - grp[:, 1])
        best = grp[np.argmax(lengths)]
        reps.append(tuple(int(v) for v in best))
    return reps


def _line_intersection(l1, l2):
    """Intersection of two lines each given as (x1,y1,x2,y2). Returns (x,y) or None."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _sanity_check(corners, img_h, img_w):
    """Returns confidence in [0,1] for a candidate 4-corner set."""
    pts = np.array(corners, dtype=np.float32)
    area = cv2.contourArea(pts)
    img_area = img_h * img_w
    if area < img_area * 0.02 or area > img_area * 0.95:
        return 0.0
    # convexity
    hull = cv2.convexHull(pts)
    if len(hull) != 4:
        return 0.15
    # aspect ratio: longest side vs shortest
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    ratio = max(sides) / (min(sides) + 1e-6)
    if ratio > 2.0:
        return 0.2
    # score falls off as ratio deviates from 1
    conf = max(0.0, 1.0 - (ratio - 1.0))
    return float(conf)


def detect_corners(image: np.ndarray) -> tuple:
    """Detect the 4 string-quadrat corners in an RGB image.

    Returns:
        corners  : list of 4 [x, y] points (float32), or None on total failure
        confidence: float 0–1
        method   : "classical" | "fallback"
    """
    H, W = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, _STRING_HUE_LO, _STRING_HUE_HI)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=40,
                            minLineLength=min(H, W) // 6, maxLineGap=20)
    if lines is None or len(lines) < 2:
        return None, 0.0, "fallback"

    reps = _cluster_lines(lines, n_clusters=2)
    if len(reps) < 2:
        return None, 0.0, "fallback"

    def extend_line(x1, y1, x2, y2, length=10000):
        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy) + 1e-6
        dx, dy = dx / norm * length, dy / norm * length
        return (x1 - dx, y1 - dy, x1 + dx, y1 + dy)

    def split_parallel(lines_seg, split_by):
        if split_by == "y":
            mids = [(y1 + y2) / 2 for x1, y1, x2, y2 in lines_seg]
        else:
            mids = [(x1 + x2) / 2 for x1, y1, x2, y2 in lines_seg]
        median = float(np.median(mids))
        lo = [s for s, m in zip(lines_seg, mids) if m <= median]
        hi = [s for s, m in zip(lines_seg, mids) if m > median]
        return (lo or lines_seg[:1]), (hi or lines_seg[-1:])

    segs_flat = lines.reshape(-1, 4)
    angles = np.array([np.arctan2(float(y2 - y1), float(x2 - x1))
                       for x1, y1, x2, y2 in segs_flat])
    angles_pos = np.where(angles < 0, angles + np.pi, angles)
    
    is_horiz = (angles_pos < np.pi / 4) | (angles_pos > 3 * np.pi / 4)
    grp_h = [segs_flat[i].tolist() for i in range(len(segs_flat)) if is_horiz[i]]
    grp_v = [segs_flat[i].tolist() for i in range(len(segs_flat)) if not is_horiz[i]]

    if not grp_h or not grp_v:
        return None, 0.0, "fallback"

    h_lo, h_hi = split_parallel(grp_h, "y")   
    v_lo, v_hi = split_parallel(grp_v, "x")   

    def longest(segs):
        return max(segs, key=lambda s: np.hypot(s[2] - s[0], s[3] - s[1]))

    lines4 = [extend_line(*longest(g)) for g in (h_lo, h_hi, v_lo, v_hi)]
    pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]
    corners = []
    for a, b in pairs:
        pt = _line_intersection(lines4[a], lines4[b])
        if pt is not None:
            corners.append(list(pt))

    if len(corners) != 4:
        return None, 0.0, "fallback"

    conf = _sanity_check(corners, H, W)
    if conf >= _CONF_THRESHOLD:
        return corners, conf, "classical"
    return corners, conf, "fallback"


# ---- obstruction detection ---------------------------------------------------

_YOLO_MODEL = None          # loaded once, cached in module scope
_OBSTRUCTION_CLASSES = {    # COCO class ids that count as obstruction
    0,   # person
}
_OBSTRUCTION_AREA_THRESH = 0.03   # flag if box covers >3% of quadrat area


def _get_yolo():
    """Lazy-load YOLOv8n once; return None if ultralytics unavailable."""
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    try:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO("yolov8n.pt")   # downloads ~6 MB on first call
    except Exception:
        _YOLO_MODEL = None
    return _YOLO_MODEL


def detect_obstruction(cropped: np.ndarray) -> tuple:
    """Detect obstructions (hand, person, pointer) in an already-warped quadrat crop."""
    flagged_boxes = []
    max_conf = 0.0

    model = _get_yolo()
    if model is not None:
        try:
            H, W = cropped.shape[:2]
            quadrat_area = H * W
            bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            results = model(bgr, verbose=False)[0]

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id not in _OBSTRUCTION_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)
                if box_area / quadrat_area > _OBSTRUCTION_AREA_THRESH:
                    flagged_boxes.append([x1, y1, x2, y2])
                    max_conf = max(max_conf, conf)
        except Exception:
            pass

    H, W = cropped.shape[:2]
    quadrat_area = H * W

    # Heuristic: also flag long-thin elongated contours (pointer stick)
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        rect = cv2.minAreaRect(cnt)
        (_, _), (w_r, h_r), _ = rect
        aspect = max(w_r, h_r) / (min(w_r, h_r) + 1e-6)
        if aspect > 5 and cv2.contourArea(cnt) / quadrat_area > _OBSTRUCTION_AREA_THRESH:
            x, y, rw, rh = cv2.boundingRect(cnt)
            flagged_boxes.append([x, y, x + rw, y + rh])
            max_conf = max(max_conf, 0.6)

    return bool(flagged_boxes), flagged_boxes, max_conf


# ---- decision log ------------------------------------------------------------

def log_decision(survey_dir: Path, record: dict) -> None:
    """Append one JSON line to <survey_dir>/decisions.jsonl."""
    survey_dir = Path(survey_dir)
    survey_dir.mkdir(parents=True, exist_ok=True)
    log_path = survey_dir / "decisions.jsonl"
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

# ---------------------------------------------------------------- site data
SITES = {
    "Penida": [
        ("SD Point", "SD", "Penida North"), ("Ped", "PD", "Penida North"),
        ("Sental", "SN", "Penida North"), ("Batumulapan", "BM", "Penida East"),
        ("Karangsari", "KS", "Penida East"), ("Toyapakeh", "TP", "Penida West"),
        ("Mangrove", "MN", "Penida West"), ("Crystal Bay", "CB", "Penida West"),
        ("Manta Bay", "MB", "Penida South"), ("Manta Point", "MP", "Penida South"),
        ("Labyrinth", "LB", "Penida South"), ("Rabbit Hole", "RH", "Penida South"),
    ],
    "Bira": [
        ("Blue Planet", "BP", "Bira West"), ("Bira Dive Camp", "BD", "Bira West"),
        ("Stingray Bay", "SB", "Bira West"), ("Phinisi Point", "PN", "Bira East"),
        ("Marumasa", "MS", "Bira East"), ("Harbour Reef", "HR", "Bira East"),
        ("Kasuso", "KA", "Bira East"), ("Mr. Ramli", "MR", "Liukang"),
        ("Shark Point", "SP", "Liukang"), ("Mola Mola", "MM", "Kambing"),
        ("Eagle Rock", "ER", "Kambing"), ("Great Wall", "GW", "Kambing"),
        ("Coral Garden", "CG", "Kambing"), ("The Cave", "CV", "Kambing"),
    ],
    "Raja": [
        ("Scuba Republic", "SR", "Waigeo"), ("Friwen Batu", "FB", "Gam"),
        ("Friwen Garden", "FG", "Gam"), ("Friwen Wall", "FW", "Gam"),
        ("Batu Lima", "BL", "Gam"), ("Blue Magic", "BM", "Kri"),
        ("Sardine Reef", "SA", "Kri"), ("Chicken Reef", "CR", "Kri"),
        ("Cape Kri", "CK", "Kri"), ("Mioskon", "MI", "Mioskon"),
        ("Lao Lao", "LL", "Mioskon"), ("Yenbuba", "YB", "Mansuar"),
        ("Sauwandarek", "SW", "Mansuar"),
    ],
}

IMG_EXT = {".jpg", ".jpeg", ".png"}


def survey_name(date, code):
    return f"BS_{date:%Y%m%d}_{code}"


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).name)]


def list_images(folder):
    return sorted(
        (p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT),
        key=natural_key,
    )


def find_depth_dir(src, depth):
    for child in Path(src).iterdir():
        if child.is_dir() and re.fullmatch(rf"{depth}\s*m?", child.name, re.I):
            return child
    return None


# ------------------------------------------------------------ folder layout
def layout(root, name):
    base = Path(root) / name
    return {
        "base": base,
        "reserved": base / f"{name} reserved",
        "chosen": base / f"{name} chosen",
        "cropped": base / f"{name} cropped",
    }


def ingest(src, root, name, depths):
    L = layout(root, name)
    for k in ("reserved", "chosen", "cropped"):
        L[k].mkdir(parents=True, exist_ok=True)
    counts = {}
    for d in depths:
        dd = find_depth_dir(src, d)
        if dd is None:
            counts[d] = 0
            continue
        dest = L["reserved"] / f"{d}m"
        dest.mkdir(exist_ok=True)
        imgs = list_images(dd)
        for i, p in enumerate(imgs, 1):
            shutil.copy2(p, dest / f"{name}_{d} ({i}){p.suffix.upper()}")
        counts[d] = len(imgs)
    return counts


# ------------------------------------------------------------ crop / warp
def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def warp_quadrat(rgb, corners, side=None):
    c = order_corners(corners)
    if side is None:
        e = [np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1]),
             np.linalg.norm(c[2] - c[3]), np.linalg.norm(c[3] - c[0])]
        side = int(round(np.mean(e)))
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], np.float32)
    M = cv2.getPerspectiveTransform(c, dst)
    return cv2.warpPerspective(rgb, M, (side, side), flags=cv2.INTER_CUBIC)


def load_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB")), im.info.get("exif")


def crop_in_place(path, corners):
    rgb, exif = load_rgb(path)
    sq = warp_quadrat(rgb, corners)
    kw = {"quality": 95, "subsampling": 0}
    if exif:
        kw["exif"] = exif
    Image.fromarray(sq).save(path, "JPEG" if Path(path).suffix.lower() != ".png" else "PNG",
                             **(kw if Path(path).suffix.lower() != ".png" else {}))


# ------------------------------------------------------------------ scoring
def score_image(path, corners, work_side=600):
    rgb, _ = load_rgb(path)
    sq = warp_quadrat(rgb, corners)
    sq = cv2.resize(sq, (work_side, work_side), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(sq, cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    cells = gray.reshape(4, work_side // 4, 4, work_side // 4).mean(axis=(1, 3))
    unevenness = float(cells.std() / (cells.mean() + 1e-6))
    clipped = float(((gray < 15) | (gray > 240)).mean())
    return {"sharp": sharp, "unevenness": unevenness, "clipped": clipped}


def combine_scores(raw):
    sharp = np.array([r["sharp"] for r in raw])
    sharp_n = sharp / (np.percentile(sharp, 95) + 1e-9)
    sharp_n = np.clip(sharp_n, 0, 1)
    even = np.array([1 - min(r["unevenness"] / 0.5, 1) for r in raw])
    clip = np.array([1 - min(r["clipped"] * 5, 1) for r in raw])
    return (0.6 * sharp_n + 0.3 * even + 0.1 * clip).tolist()


def pick_evenly(names, scores, k=5):
    if len(names) <= k:
        return list(names)
    segs = np.array_split(np.arange(len(names)), k)
    return [names[max(seg, key=lambda i: scores[i])] for seg in segs]


# ------------------------------------------------------------------- export
def export(root, name, selection, corners, auto_detect=False):
    """selection: {depth: [filenames in reserved/<depth>m]}.
    reserved -> chosen -> cropped (copied, then overwritten in place with the crop)."""
    L = layout(root, name)
    n = 0
    warn_path = L["base"] / "crop_warnings.txt"
    
    # clear previous warning logs if starting a new export
    if warn_path.exists():
        warn_path.unlink()
        
    for d, files in selection.items():
        for f in files:
            src = L["reserved"] / f"{d}m" / f
            chosen = L["chosen"] / f
            cropped = L["cropped"] / f
            shutil.copy2(src, chosen)
            shutil.copy2(chosen, cropped)
            
            # Use Auto-Detection if toggled on
            photo_corners = corners
            if auto_detect:
                rgb, _ = load_rgb(src)
                detected, conf, method = detect_corners(rgb)
                
                # Use detected corners if it's confident enough
                if detected is not None and method == "classical":
                    photo_corners = detected
                else:
                    # Log to warnings file if fallback corners are used
                    with open(warn_path, "a") as wf:
                        wf.write(f"{d}m / {f}: fallback corners used (conf={conf:.2f})\n")

            crop_in_place(cropped, photo_corners)
            n += 1
            
    return n