# Benthic Survey Prep Tool

Offline tool for the Indo Ocean Project's Roving Benthic Survey workflow:
imports raw GoPro time-lapse photos, auto-picks the best 5 per depth, and
crops them to a clean 1:1 square from the string quadrat — ready to upload to
CoralNet.

Runs entirely on your own machine. No internet connection needed at runtime,
and no photos leave your computer.

See [`SPEC.md`](./SPEC.md) for the full technical specification (current
behaviour + planned automatic corner detection, swipe review, and obstruction
detection).

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/benthic-survey-tool.git
cd benthic-survey-tool

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`, served
locally by your own machine.

## Before you start a survey

Arrange your raw photos in one folder per dive, split into depth
subfolders, e.g.:

```
raw_photos/
├── 25/
├── 20/
├── 15/
├── 10/
└── 5/
```

(Folder names `25`, `25m`, `25 m` are all accepted.)

## Using the app

1. **Location → Dive site → Date** — the sidebar builds the export name
   `BS_YYYYMMDD_CODE` for you.
2. **Import** — point it at your `raw_photos/` folder; it copies and renames
   everything into `<output>/BS_.../BS_..._reserved/<depth>m/`.
3. **Calibrate** — click the 4 string corners once on a clean, unobstructed
   photo (pick from the thumbnail grid — the first frame is often the dive
   computer shot, so choose whichever one shows all 4 corners clearly).
4. **Select & export** — review the auto-picked 5-per-depth selection,
   swap anything you don't like, then export. This produces `chosen/` and a
   `cropped/` folder with each photo overwritten in place as a square crop
   matching the quadrat.

## Project status

- ✅ Location/site picker, auto-naming, folder import
- ✅ Manual quadrat calibration + square crop
- ✅ Heuristic photo scoring (sharpness, lighting evenness, clipping) and
  even-spacing selection
- 🚧 Automatic string-corner detection (classical CV + fallback detector)
- 🚧 Calibration image picker (thumbnail grid)
- 🚧 Swipe-style keep/reject review with auto-next-candidate
- 🚧 Obstruction (hand/pointer) auto-flagging

Details for the items in progress are in [`SPEC.md`](./SPEC.md).

## Reporting a bug / requesting a feature

Please open a GitHub Issue on this repo and include:
- What you were doing (which tab/step)
- What you expected vs. what happened
- A sample photo or screenshot if it's a detection/crop issue (strip GPS
  EXIF if the dive site is sensitive)
- Your OS and Python version

If you'd rather not use GitHub, or for anything involving real survey data,
email **fajrifathur.work@gmail.com**.

## Contributing

Pull requests welcome — see [`SPEC.md`](./SPEC.md) for the architecture and
what's planned next before starting on something large, and open an issue
first to avoid duplicate work.

## License

MIT — see [`LICENSE`](./LICENSE).

## Contact

Fajri Fathur Rahman — fajrifathur.work@gmail.com
