# MoshFlow

A Python desktop application by Team 3 (Digital Image Processing, Team Project 2).
It takes several video clips, lets you choose a datamosh transition for every
cut on a timeline, and exports one video file.

## Set up

Requires Python 3.7 or newer, so the Python that the Visual Studio installer adds
(3.7 with VS 2019, 3.9 with VS 2022) works. The first setup needs internet access
to install NumPy, OpenCV and PySide6.

### Windows, quickest

Double-click `run_windows.bat`. The first run creates `.venv` and installs
`requirements.txt` into it (a few minutes); later runs start the GUI at once.

### Visual Studio 2019 / 2022

1. In the Visual Studio Installer, add the **Python development** workload.
2. Open `Team3_MoshFlow.sln` and press **F5**. This runs `run.py`. If the active
   Python lacks the packages, it uses `.venv` when there is one, or offers to create
   it and install `requirements.txt` (answer Enter in the console window).
3. Or set it up by hand: Solution Explorer > **Python Environments** > right-click the
   active environment > **Install from requirements.txt**, then press F5.

### Command line

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

On Windows, use `py -3 -m venv .venv` and `.venv\Scripts\` instead of `.venv/bin/`.

## Use

1. **Import** videos into the Media Bin.
2. Drag them onto the timeline. Drag to reorder; press Delete to remove. Drag a clip's
   left or right edge to trim it, or play the clip in the preview and press **I** / **O**
   (or **Set start** / **Set end**) to start or end it at the frame shown.
3. Click a transition chip between two clips and choose its effect, duration
   and parameters in the Inspector. A new transition lasts 3 s and starts at the
   cut, so the first clip plays to its end. Clips with another aspect ratio are
   fitted on black bars.
4. Press **Export**. The video is written in the background.

**Edit > Undo** (Ctrl+Z) and **Redo** (Ctrl+Shift+Z) work for every change to the
timeline, the Inspector and the export settings. One slider drag is one undo step.

## Transitions

Ten effects are in the Inspector: Hard Cut, Crossfade and eight datamosh effects
(Keyframe Drop, Motion Bloom, Liquid Melt, Vector Swap, P-frame Pulse, DCT Crush,
Pulse Mosh, Bake Smear). Each one is a file in `moshflow/effects/`; the program
finds them automatically and the Inspector builds their controls.

## Library rule

Apart from OpenCV video input and output, the program uses no library image
processing. All OpenCV calls are in `moshflow/videoio/`. NumPy is used to store
frames and for basic arithmetic.

## Project layout

| Path | Contents |
| --- | --- |
| `Team3_MoshFlow.sln`, `MoshFlow.pyproj` | Visual Studio solution and project |
| `run.py` | Start-up file (F5 in Visual Studio) |
| `run_windows.bat` | Double-click start on Windows, without Visual Studio |
| `moshflow/api/` | The contract between the core and the effects |
| `moshflow/core/` | Project model, timeline resolution, renderer |
| `moshflow/app/` | Project files, effect catalog, render jobs |
| `moshflow/videoio/` | OpenCV video reader and writer |
| `moshflow/toolkit/` | Shared image helpers (block matching, motion compensation, codec-style P-frames, block switching) |
| `moshflow/effects/` | One file per transition effect |
| `moshflow/ui/` | PySide6 interface |
| `moshflow/cli.py` | Command-line rendering |
