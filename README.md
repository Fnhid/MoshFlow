# MoshFlow

A Python desktop application by Team 3 (Digital Image Processing, Team Project 2).
It takes several video clips, lets you choose a datamosh transition for every
cut on a timeline, and exports one video file.

## Set up

Requires Python 3.12.

### Visual Studio 2019 / 2022

1. In the Visual Studio Installer, add the **Python development** workload.
2. Open `Team3_MoshFlow.sln`.
3. In Solution Explorer, right-click **Python Environments > Add Environment**,
   create a virtual environment with Python 3.12, and let it install from `requirements.txt`.
4. Press **F5**. This runs `run.py`, which starts the GUI.

### Command line

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

On Windows, use `.venv\Scripts\` instead of `.venv/bin/`.

## Use

1. **Import** videos into the Media Bin.
2. Drag them onto the timeline. Drag to reorder; press Delete to remove.
3. Click a transition chip between two clips and choose its effect, duration
   and parameters in the Inspector.
4. Press **Export**. The video is written in the background.

**Edit > Undo** (Ctrl+Z) and **Redo** (Ctrl+Shift+Z) work for every change to the
timeline, the Inspector and the export settings. One slider drag is one undo step.

## Work together

Suggested branches (optional):

| Branch | Who | Main file |
| --- | --- | --- |
| `effect/<your-effect>` | Each effect author | `moshflow/effects/<your_effect>.py` |
| `core/<topic>`, `ui/<topic>` | Yuhyeon Cha | Everything else |

Effects depend only on `moshflow.api` and `moshflow.toolkit`, so work on the core and
the UI does not affect them. Run `python -m pytest` before merging into `main`.
See [`CLAUDE.md`](CLAUDE.md), "Git", for more.

## Write an effect

The full contract is in [`docs/API.md`](docs/API.md). Team members using Claude Code
should also read [`CLAUDE.md`](CLAUDE.md).

Each member writes the effect they want; it only has to follow the API. Keyframe Drop,
Motion Bloom and Liquid Melt are finished and appear in the Inspector. The shared helpers, block matching in
`toolkit/motion.py` and motion compensation in `toolkit/warp.py`, work and are
checked by `tests/test_toolkit.py`.

An effect is one file in `moshflow/effects/`. The program finds it automatically, and
the Inspector builds its controls from `param_specs()`.

1. Copy `moshflow/effects/_template.py` to `moshflow/effects/<your_effect>.py`.
2. Rename the class and fill in `info`, `param_specs` and `render`.
3. Test it:

```bash
python -m pytest tests/test_effect_contract.py
python -m moshflow.cli try --effect <your_effect_id> a.mp4 b.mp4 -o test.mp4
```

Inside `render(self, ctx, progress)`:

| Name | Meaning |
| --- | --- |
| `ctx.outgoing`, `ctx.incoming` | Clip A and clip B. `.frame(i)` returns a read-only H x W x 3 uint8 BGR array |
| `ctx.a_window`, `ctx.b_window` | The frames of A and of B that the transition replaces |
| `ctx.length` | Number of frames to yield: `len(a_window) + len(b_window)` |
| `ctx.size`, `ctx.fps` | Output size `(width, height)` and frame rate |
| `ctx.params` | Validated values of your parameters |
| `ctx.seed` | Use for all randomness: `np.random.default_rng(ctx.seed)` |
| `progress.cancelled` | Check it in loops and return when it is true |
| `progress.report(done, total)` | Optional progress for long computations |

Helpers more than one effect needs go into `moshflow/toolkit/`.

The contract test checks every effect automatically: frame count, shape and type,
read-only inputs, reproducible output, cancellation, and windows that use a whole clip.

## Library rule

Apart from OpenCV video input and output, the program uses no library image
processing. All OpenCV calls are in `moshflow/videoio/`. NumPy is used to store frames
and for basic arithmetic. `tests/test_compliance.py` checks the source for this.

## Project layout

| Path | Contents |
| --- | --- |
| `moshflow/api/` | The contract between the core and the effects (API v1) |
| `moshflow/core/` | Project model, timeline resolution, renderer |
| `moshflow/app/` | Project files, effect catalog, render jobs |
| `moshflow/videoio/` | OpenCV video reader and writer |
| `moshflow/toolkit/` | Shared image helpers |
| `moshflow/effects/` | One file per effect |
| `moshflow/ui/` | PySide6 interface |
| `moshflow/cli.py` | Command-line tools |
| `tests/` | Unit, contract, compliance, render and UI tests |
| `tools/update_vs_project.py` | Regenerates the Visual Studio files after adding source files |
| `docs/API.md` | The effect API and toolkit conventions |
| `CLAUDE.md` | Rules and commands for team members working with Claude Code |

## Tests

```bash
python -m pytest
```
