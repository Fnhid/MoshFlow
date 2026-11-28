# MoshFlow: notes for Claude

MoshFlow is Team 3's project for Digital Image Processing, Team Project 2
(Chung-Ang University). It is a Python desktop app: the user lines up video clips
on a timeline, picks a datamosh transition for every cut, and exports one video.

Read `docs/API.md` before changing any effect or toolkit code. It is the contract
between the core and the effects.

## Library rule

Apart from OpenCV video input and output, no library function may do image or
signal processing. Breaking this rule can fail the whole project.

- OpenCV (`cv2`) is used only in `moshflow/videoio/`, and only `VideoCapture`,
  `VideoWriter` and `CAP_PROP_*`. Never add a `cv2` call anywhere else.
- NumPy is for arrays and basic arithmetic only: indexing, slicing, `astype`,
  `+ - * /`, `abs`, `min`, `max`, `sum`, `clip`, `argmin`. Never use `np.fft`,
  `np.convolve`, `np.correlate`, `np.gradient`, `np.interp` or `np.histogram`.
- Never add SciPy, scikit-image, Pillow, imageio, MoviePy, PyAV, FFmpeg, Numba or
  any other processing library, and never add a package to `requirements.txt`.
- Write every image algorithm by hand: block matching, warping, interpolation,
  blending, colour conversion.
- If you are unsure whether a function is allowed, do not use it; ask the user.

`tests/test_compliance.py` checks this, but it cannot catch everything.

## Who owns what

| Area | Owner |
| --- | --- |
| `moshflow/api/`, `core/`, `app/`, `ui/`, `videoio/`, `toolkit/`, `cli.py`, `tests/` | Yuhyeon Cha (team leader) |
| One effect file each in `moshflow/effects/` | Casper Bach, Nicolai Hansen, Viktor Tost |
| Test clips and demo reel | Viktor Tost |

Each member chooses what their effect does; the only requirement is the API in
`docs/API.md`. Work in the files of the person you are helping. Do not change
`moshflow/api/` (API v1 is frozen) or another member's files; if that seems
necessary, say so and let the user ask Yuhyeon. A new shared helper goes in a new
file in `moshflow/toolkit/`.

## Git

The repository is `github.com/Fnhid/MoshFlow`. The habits below are optional;
use whatever works for the person you are helping.

- A branch per person keeps work apart, for example `effect/<your-effect>` for an
  effect and `core/<topic>` or `ui/<topic>` for Yuhyeon's work.
- Keeping an effect branch to that effect's file makes merging easy.
- Before merging into `main`, bring in the latest `main`
  (`git fetch origin` then `git merge origin/main`) and run `python -m pytest`,
  so `main` keeps passing the tests.
- A pull request lets the others see a change before it lands in `main`.
- Do not commit `.venv/`, rendered videos, Word or PowerPoint files.

Work stays apart even without these habits: effects import only `moshflow.api`
and `moshflow.toolkit`, and the core and UI never import an effect directly. So
changes to the core or UI do not break an effect unless the API changes.

## How an effect works

One file in `moshflow/effects/`, one `TransitionEffect` subclass with `@register`.
`render(ctx, progress)` yields exactly `ctx.length` frames of shape `(H, W, 3)`,
`uint8`, BGR. Input frames are read-only. Do the maths in float32
(`frame.astype(np.float32)`) and convert back with `color.to_uint8`. Use
`ctx.seed` for any randomness, check `progress.cancelled` in every loop, and
declare every tunable value in `param_specs()`. No file I/O, UI calls or global state.
Never blend clip A and clip B with a weight (`P + h * (B - P)` looks like a crossfade, not a
datamosh): a block shows one picture or the other. Use `toolkit/refresh.py` to switch blocks to B.

Ten effects are finished: Hard Cut, Crossfade, Keyframe Drop, Motion Bloom, Liquid
Melt, Vector Swap, P-frame Pulse, DCT Crush, Pulse Mosh and Bake Smear. A member
may still change one or write another. Once an effect id has been used in a saved project, keep it stable.
The shared helpers work: `toolkit/motion.py` (`estimate`, `to_field`,
`smooth_field`), `toolkit/warp.py` (`warp`), `toolkit/codec.py` (P-frames) and
`toolkit/refresh.py` (`Takeover`: B wears A away by its own motion, nothing on a
timer). Do not change their signatures or
the vector convention, because other effects may depend on them. When an effect
works, remove `(draft)` from its name.

Vector convention (from `docs/API.md`): `v = (dx, dy)` means
`current(x, y) ≈ reference(x + dx, y + dy)`, and warping is backward:
`out(x, y) = image(x + dx, y + dy)`.

## Commands

Python 3.7 or newer (keep the code 3.7-compatible: Visual Studio 2019 installs 3.7; every module starts with `from __future__ import annotations`, and runtime type expressions use `typing.Optional`/`Union`). On Windows use `.venv\Scripts\python` instead of `.venv/bin/python`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest                                  # must pass before you finish
.venv/bin/python -m pytest tests/test_effect_contract.py    # every effect
.venv/bin/python -m pytest tests/test_toolkit.py            # shared helpers
.venv/bin/python -m moshflow.cli try --effect keyframe_drop a.mp4 b.mp4 -o test.mp4 --size 640x360
.venv/bin/python run.py                                     # the GUI
.venv/bin/python tools/update_vs_project.py                 # after adding or removing a .py file
```

## Definition of done

1. `python -m pytest` passes, including `test_compliance.py` and `test_effect_contract.py`.
2. `moshflow.cli try` renders a short video with the effect, and it looks right.
3. If you added or removed a `.py` file, run `tools/update_vs_project.py` so Visual
   Studio lists it. If two branches both changed `MoshFlow.pyproj`, run it again
   after merging instead of fixing the conflict by hand.
4. No new dependencies, no edits outside your area, no debug files left behind.

## Visual Studio and submission

The source is submitted as a Visual Studio solution, `Team3_MoshFlow.sln` with
`MoshFlow.pyproj`, and graded by opening it in Visual Studio. Keep the solution in
Visual Studio 2019 format (the same environment as the v142 toolset from Team
Project 1); `tools/update_vs_project.py` writes it that way. The v142 toolset
setting itself belongs to C++ projects and does not exist in a Python project.
Never edit the `.sln` or `.pyproj` by hand.

Source code is due on Nov 28, 2026, 10 PM, as `Team3_MoshFlow_Src.zip`.

## Style

- English for all code, comments and messages.
- Match the existing code: short docstrings, few comments, type hints on public
  functions, small functions, NumPy vectorisation instead of per-pixel Python loops
  where it stays readable.
- Explain image-processing choices at the pixel level (what happens to one pixel
  value), because the professor asks questions that way.

## Skills

Project skills live in `.claude/skills/`. Claude Code loads them when you start it
inside this repository; type the name as a command (for example `/new-effect`) or
just describe the task.

| Skill | Use it to |
| --- | --- |
| `/new-effect` | Write your own effect, finish an effect, or fix a contract-test failure |
| `/check-library-rule` | Check a change against the course library rule before a commit or pull request |
| `/run-moshflow` | Run the app without a screen, take screenshots, export a transition and look at its frames |
| `/humanizer` | Make README, docs, reports or pull request text read like a person wrote it |

Use `/check-library-rule` before every pull request that touches `moshflow/`, and
`/run-moshflow` to see an effect's output instead of guessing from the code.
