"""Core logic for the IOP benthic survey prep tool (no UI, no internet)."""
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- site data
# (site name, code, locality)
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
    """Find a subfolder named '25', '25m', '25 m' inside src."""
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
    """Copy source photos into reserved/<depth>m/, renamed NAME_DEPTH (n).ext"""
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
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def warp_quadrat(rgb, corners, side=None):
    """Perspective-warp the quadrat (4 clicked string corners) to a true square."""
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
    """Overwrite the file with its square crop ('Save', not 'Save as a copy')."""
    rgb, exif = load_rgb(path)
    sq = warp_quadrat(rgb, corners)
    kw = {"quality": 95, "subsampling": 0}
    if exif:
        kw["exif"] = exif
    Image.fromarray(sq).save(path, "JPEG" if Path(path).suffix.lower() != ".png" else "PNG",
                             **(kw if Path(path).suffix.lower() != ".png" else {}))


# ------------------------------------------------------------------ scoring
def score_image(path, corners, work_side=600):
    """Heuristic quality of the quadrat area. Higher = better."""
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
    """Normalise within a depth batch -> single score per image."""
    sharp = np.array([r["sharp"] for r in raw])
    sharp_n = sharp / (np.percentile(sharp, 95) + 1e-9)
    sharp_n = np.clip(sharp_n, 0, 1)
    even = np.array([1 - min(r["unevenness"] / 0.5, 1) for r in raw])
    clip = np.array([1 - min(r["clipped"] * 5, 1) for r in raw])
    return (0.6 * sharp_n + 0.3 * even + 0.1 * clip).tolist()


def pick_evenly(names, scores, k=5):
    """Split the sequence into k equal segments, take the best-scoring photo of each.
    Gives even spacing (per IOP method) while avoiding bad frames."""
    if len(names) <= k:
        return list(names)
    segs = np.array_split(np.arange(len(names)), k)
    return [names[max(seg, key=lambda i: scores[i])] for seg in segs]


# ------------------------------------------------------------------- export
def export(root, name, selection, corners):
    """selection: {depth: [filenames in reserved/<depth>m]}.
    reserved -> chosen -> cropped (copied, then overwritten in place with the crop)."""
    L = layout(root, name)
    n = 0
    for d, files in selection.items():
        for f in files:
            src = L["reserved"] / f"{d}m" / f
            chosen = L["chosen"] / f
            cropped = L["cropped"] / f
            shutil.copy2(src, chosen)
            shutil.copy2(chosen, cropped)
            crop_in_place(cropped, corners)
            n += 1
    return n
