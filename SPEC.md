# Benthic Survey Prep Tool — Technical Specification

Local-first Streamlit application that automates photo selection and cropping
for the Indo Ocean Project Roving Benthic Survey method, prior to CoralNet
annotation. Runs fully offline; no internet or external API required at
runtime.

Status: v1 (folder import, manual calibration, heuristic scoring) is
implemented in `app.py` / `bruv_core.py`. This document specifies v2:
automatic string detection, calibration image picker, swipe-style review,
and obstruction detection. Sections marked **[v1 — done]** describe existing
behaviour; sections marked **[v2 — build]** are the new work.

---

## 1. Background (source: `guideline.md`)

### 1.1 Locations, sites, codes

| Location | Site | Code | Locality |
|---|---|---|---|
| Penida | SD Point | SD | Penida North |
| Penida | Ped | PD | Penida North |
| Penida | Sental | SN | Penida North |
| Penida | Batumulapan | BM | Penida East |
| Penida | Karangsari | KS | Penida East |
| Penida | Toyapakeh | TP | Penida West |
| Penida | Mangrove | MN | Penida West |
| Penida | Crystal Bay | CB | Penida West |
| Penida | Manta Bay | MB | Penida South |
| Penida | Manta Point | MP | Penida South |
| Penida | Labyrinth | LB | Penida South |
| Penida | Rabbit Hole | RH | Penida South |
| Bira | Blue Planet | BP | Bira West |
| Bira | Bira Dive Camp | BD | Bira West |
| Bira | Stingray Bay | SB | Bira West |
| Bira | Phinisi Point | PN | Bira East |
| Bira | Marumasa | MS | Bira East |
| Bira | Harbour Reef | HR | Bira East |
| Bira | Kasuso | KA | Bira East |
| Bira | Mr. Ramli | MR | Liukang |
| Bira | Shark Point | SP | Liukang |
| Bira | Mola Mola | MM | Kambing |
| Bira | Eagle Rock | ER | Kambing |
| Bira | Great Wall | GW | Kambing |
| Bira | Coral Garden | CG | Kambing |
| Bira | The Cave | CV | Kambing |
| Raja | Scuba Republic | SR | Waigeo |
| Raja | Friwen Batu | FB | Gam |
| Raja | Friwen Garden | FG | Gam |
| Raja | Friwen Wall | FW | Gam |
| Raja | Batu Lima | BL | Gam |
| Raja | Blue Magic | BM | Kri |
| Raja | Sardine Reef | SA | Kri |
| Raja | Chicken Reef | CR | Kri |
| Raja | Cape Kri | CK | Kri |
| Raja | Mioskon | MI | Mioskon |
| Raja | Lao Lao | LL | Mioskon |
| Raja | Yenbuba | YB | Mansuar |
| Raja | Sauwandarek | SW | Mansuar |

> ⚠️ Known code collision: `BM` = Batumulapan (Penida) **and** Blue Magic
> (Raja). Because location is always selected first, `name_survey()` scopes
> the code by location internally, but the exported folder name
> `BS_YYYYMMDD_BM` alone does not disambiguate. Flag this to the user in the
> UI when `BM` is chosen; recommend the org rename one of the two codes.

**Export format:** `BS_YYYYMMDD_CODE` (e.g. `BS_20260406_CG`). `BS` = Benthic
Survey.

### 1.2 Depths surveyed

Per dive: 30, 25, 20, 15, 10, 5 m (guideline.md) or 30–10 m per the printed
manual - not all sites reach every depth (e.g. Manta Bay excludes 30 m). Once
a depth is excluded at a site, exclude it for all subsequent surveys of that
site. 36 photos are taken per depth (5 s interval ⇒ ≥3 min per depth); 5 are
selected per depth for annotation (30 total across 6 depths).

### 1.3 What makes a good benthic picture

1. Not blurry
2. No obstruction (hand, pointer stick, flashlight)
3. As close to the reef as possible, if it's too far don't pick it
4. Light evenly distributed, not too many shadows
5. Shows the substrate condition
6. Minimum 20% of the picture consist of coral / algae / rubble

### 1.4 Directory structure

```
BS_YYYYMMDD_CODE/
├── BS_YYYYMMDD_CODE reserved/     # all raw photos, split into depth subfolders
│   ├── 30m/  25m/  20m/  15m/  10m/
├── BS_YYYYMMDD_CODE chosen/       # 5 best photos per depth, flat
└── BS_YYYYMMDD_CODE cropped/      # copy of chosen, then cropped in place
                                    # (Save, not Save As — overwrite, keep filename)
```

Cropping: square 1:1, exactly matching the string quadrat line, no skew.

---

## 2. v1 — done

- Streamlit sidebar: location → site → date → auto-built export name.
- `ingest()`: copies raw photos from a source folder's depth subfolders into
  `reserved/<depth>m/`, renamed `NAME_DEPTH (n).ext`.
- Manual 4-point calibration (click corners once), reused for the whole
  survey via `getPerspectiveTransform` + `warpPerspective`.
- `score_image()`: Laplacian-variance sharpness, 4×4-cell brightness
  unevenness, clipped-pixel ratio → single 0–1 score.
- `pick_evenly()`: splits each depth's sequence into 5 segments, best score
  per segment (keeps the "evenly spaced" rule from the manual while dodging
  clearly bad frames).
- `export()`: copy to `chosen/`, copy to `cropped/`, overwrite the `cropped/`
  copy in place with the warped square crop (EXIF preserved).

## 3. v2 — build

### 3.1 Calibration image picker

**Problem:** the first photo in a depth sequence is often the diver's hand
holding the frame near a dive computer, not a clean top-down shot — but a
clean shot (no fingers/pointer over the string) usually exists somewhere in
the sequence.

**Requirement:**
- Replace the fixed "use image 1" calibration source with a thumbnail
  strip/grid of the first N photos (configurable, default 12) of the chosen
  depth folder.
- User clicks whichever thumbnail shows all 4 string corners clearly, then
  calibrates on that one, exactly as v1 does today (4-click corners).
- Persist the calibration (`calibration.json` inside the survey's base
  folder, as v1 already does) plus which source image + depth it came from,
  for audit/debugging.
- Calibration remains **one-time per survey** (frame is rigid, camera is
  fixed to it) — v2 does not need per-photo recalibration unless automatic
  detection (3.2) disagrees strongly with the saved calibration, in which
  case flag that photo for manual review rather than silently reusing stale
  corners.

### 3.2 Automatic string-square detection (two-step)

**Step A — classical CV fast path (no model, no training):**
1. Convert to HSV, threshold the known string colour (blue, per sample
   photos) with a generous range to tolerate depth/colour cast.
2. Morphological close → `cv2.HoughLinesP` → cluster into ~2 dominant
   orientations (roughly orthogonal in the warped frame).
3. Intersect the 4 dominant lines → 4 candidate corners → sanity-check
   (convex quadrilateral, area within expected range given known distance
   from the rig, aspect ratio near 1:1 before warp-correction).
4. If sanity checks pass with high confidence → auto-accept corners for that
   photo, skip Step B.

**Step B — object-detection fallback:**
1. When Step A's confidence is low (string occluded, poor contrast, murky
   water), run a small pretrained detector (e.g. YOLOv8n fine-tuned on
   frame-corner or full-quadrat bounding boxes) to localize the quadrat
   region, then run Step A's line/corner search *within* that cropped
   region only, narrowing the search space.
2. If Step B still fails confidence checks, fall back to the last known-good
   calibration for that survey and flag the photo for manual corner review
   (do **not** silently use a wrong crop).

**Training data note:** no labelled data exists yet. Bootstrap the detector
with synthetic training images: composite the known rigid frame geometry
(rectangle of known real-world side length, unaffected by warp) onto a
library of plain reef/ocean background photos at randomized angles,
distances, and lighting, to generate bounding-box labels automatically. Real
photos + manual corner clicks collected during v1/v2 usage (see 3.3) become
real fine-tuning data once enough surveys have been run.

**Interface:** `detect_corners(image) -> (corners, confidence, method)` where
`method` is `"classical"` or `"detector"` or `"fallback"`. `bruv_core.py`'s
existing `warp_quadrat()` / `score_image()` / `crop_in_place()` are reused
unchanged — this only replaces how `corners` is obtained per-photo instead of
once per survey, when auto-detection is enabled.

### 3.3 Swipe-style manual review ("Tinder for reef photos")

Replaces/extends the current `st.multiselect` review step.

**Requirement:**
- After scoring, present the auto-picked candidate for each depth's segment
  one at a time, large preview, with two buttons: **✅ Keep** / **❌ Reject**
  (or left/right swipe gesture if using a touch-friendly component).
- Swipe right to keep, left to reject
- On **Reject**: discard that candidate, immediately show the
  next-best-scoring unpicked photo *from the same segment* (not simply
  "next file in sequence" — respect the scoring order within the segment so
  the replacement is still a good frame, not just the next timestamp). If a
  segment is exhausted (all candidates in it rejected), pull from the
  adjacent segment and note the imbalance to the user.
- On **Keep**: lock that photo for that depth/segment slot, advance to the
  next segment.
- Every accept/reject decision is logged (`decisions.jsonl` per survey:
  `{photo, depth, segment, action, score, timestamp}`). This log is the
  labelled dataset for future obstruction-detector fine-tuning (3.4) and for
  auditing why a given final 25 were chosen.
- Progress indicator: "Depth 15 m — segment 3 of 5".
- Keyboard shortcuts (←/→ or Y/N) for fast review during a real annotation
  session.

### 3.4 Obstruction detection

**Requirement:**
- `detect_obstruction(cropped_image) -> (has_obstruction: bool, mask/boxes, confidence)`.
- v2-minimum: run a pretrained general object/hand detector (e.g. a
  hand-landmark or hand-detection model, plus a simple long-thin-object
  detector for the pointer stick) restricted to the warped quadrat region;
  flag if a detected object's bounding box overlaps more than a small
  threshold (e.g. 3%) of the quadrat area.
- Surface the flag directly in the swipe review (3.3): auto-reject
  candidates with a high-confidence obstruction hit before the human ever
  sees them, but show a small "auto-rejected: possible hand" badge on
  request so the reviewer can override false positives.
- Do **not** attempt full substrate-cover classification (the "≥20%
  coral/algae/rubble" rule) in v2 — that needs a domain-specific segmentation
  model and labelled reef data the project doesn't have yet. Leave it as a
  manual check in the review UI (short reminder text), and revisit once
  `decisions.jsonl` + CoralNet's own point annotations provide enough
  labelled substrate data to bootstrap a classifier.

### 3.5 Module additions

```
bruv_core.py
  + detect_corners(image) -> (corners, confidence, method)
  + detect_obstruction(cropped_image) -> (bool, boxes, confidence)
  + log_decision(survey_dir, record: dict) -> None   # appends decisions.jsonl

app.py
  + Tab "Calibrate": thumbnail grid picker before the 4-click step
  + Tab "Review": swipe UI replacing multiselect, wired to detect_obstruction
                  auto-reject + log_decision
  + Sidebar toggle: "Auto-detect corners per photo" (on/off; off = v1 behaviour,
                     one calibration reused for the whole survey)
```

No new runtime dependency is required for Step A of 3.2 (pure OpenCV). Step B
of 3.2 and 3.4 add a lightweight detector — plan for `ultralytics` (YOLOv8n)
or an equivalently small pretrained model, loaded once and cached, running on
CPU (no GPU assumed).

---

## 4. Acceptance checklist

- [ ] Selecting location + site auto-builds `BS_YYYYMMDD_CODE`, warns on the
      `BM` collision.
- [ ] Calibration step shows a thumbnail grid of the first N photos in the
      chosen depth folder; user picks the cleanest one before clicking corners.
- [ ] With auto-detect on, most photos get corners without a manual click;
      low-confidence photos are clearly flagged rather than silently
      mis-cropped.
- [ ] Review screen shows one candidate at a time with Keep/Reject; Reject
      instantly serves the next-best candidate from the same segment.
- [ ] Every keep/reject is appended to `decisions.jsonl`.
- [ ] Obstruction auto-reject fires on an obvious hand-in-frame test photo.
- [ ] Export still produces `reserved / chosen / cropped`, with `cropped`
      overwritten in place (no `_copy` suffix), EXIF intact.
