# MoshFlow API v1

This is the contract between the core (Yuhyeon Cha) and the effects. It is frozen
as API v1: effects may rely on everything below, and the core will not change it
without agreeing with the team first.

If this page and the code disagree, the code in `moshflow/api/` and
`moshflow/toolkit/` is right.

## Where an effect sits

```text
UI (PySide6)  ->  app (project files, render jobs)  ->  core (timeline -> renderer)
                                                          |
                                          moshflow.api  <-+->  videoio (OpenCV I/O only)
                                                |
                                     effects/<one file each>  ->  toolkit (shared helpers)
```

An effect sees only `moshflow.api` and `moshflow.toolkit`. It never sees the UI,
the video files, the encoder or the project file.

## Files

| Path | Contents | Owner |
| --- | --- | --- |
| `moshflow/api/effect.py` | `TransitionEffect`, `TransitionContext`, `EffectInfo` | Yuhyeon Cha |
| `moshflow/api/frames.py` | `Frame`, `FrameSource`, `FrameSink`, `ProgressReporter` | Yuhyeon Cha |
| `moshflow/api/params.py` | `IntParam`, `FloatParam`, `ChoiceParam`, `BoolParam`, validation | Yuhyeon Cha |
| `moshflow/api/registry.py` | `@register`, `get_effect`, `all_effects` | Yuhyeon Cha |
| `moshflow/toolkit/color.py` | `bgr_to_y`, `to_uint8` | Yuhyeon Cha |
| `moshflow/toolkit/resize.py` | `nearest`, `bilinear` | Yuhyeon Cha |
| `moshflow/toolkit/frames.py` | `window_frames`, `last_clean_a` | Yuhyeon Cha |
| `moshflow/toolkit/motion.py` | `estimate`, `to_field`, `smooth_field` | Yuhyeon Cha |
| `moshflow/toolkit/warp.py` | `warp` | Yuhyeon Cha |
| `moshflow/effects/hard_cut.py`, `crossfade.py` | Working reference effects | Yuhyeon Cha |
| `moshflow/effects/keyframe_drop.py`, `motion_bloom.py`, `liquid_melt.py` | Finished effects, one per member | Casper Bach, Nicolai Hansen, Viktor Tost |
| `moshflow/effects/<your_effect>.py` | Your own effect | Its author |

## Frames

A `Frame` is a NumPy array of shape `(H, W, 3)`, dtype `uint8`, in BGR order
(OpenCV's order). Every frame an effect reads and yields has the project size.

Frames handed to an effect are read-only. Convert before doing maths:

```python
picture = clip.frame(i).astype(np.float32)   # a new, writable copy
...
yield color.to_uint8(picture)                # clip to 0..255, round, back to uint8
```

Never do pixel maths in `uint8`: `50 - 80` wraps around to `226`.

## FrameSource

One clip, already conformed to the project size and frame rate.

| Member | Meaning |
| --- | --- |
| `len(source)` | Number of frames |
| `source.frame(i)` | Frame `i` (read-only). Raises `IndexError` outside `0 .. len - 1` |
| `source.size` | `(width, height)` |
| `source.fps` | Frames per second |

Frames are cached, so reading the same frame twice is cheap.

## TransitionContext

Everything an effect gets to render one transition.

| Field | Meaning |
| --- | --- |
| `outgoing` | Clip A, a `FrameSource` |
| `incoming` | Clip B, a `FrameSource` |
| `a_window` | `range` of A's frames that the transition replaces (the end of A) |
| `b_window` | `range` of B's frames that the transition replaces (the start of B) |
| `params` | Your parameters, already validated, as `{id: value}` |
| `seed` | Use for all randomness: `rng = np.random.default_rng(ctx.seed)` |
| `length` | Number of frames to yield: `len(a_window) + len(b_window)` |
| `size` | `(width, height)` of every frame |
| `fps` | Frame rate |

The output replaces A's frames in `a_window` followed by B's frames in `b_window`.
The windows come from the transition's duration and alignment:

| Alignment | `a_window` | `b_window` |
| --- | --- | --- |
| `before` | last n frames of A | empty |
| `center` | last n // 2 frames of A | first n - n // 2 frames of B |
| `after` | empty | first n frames of B |

An effect may read any frame of either clip, including frames outside the windows.
A datamosh, for example, starts from the last clean frame of A before the window. Either window
can be empty, and a window can cover a whole clip.

## TransitionEffect

```python
from moshflow.api import EffectInfo, FloatParam, TransitionEffect, register

@register
class MyEffect(TransitionEffect):
    info = EffectInfo("my_effect", "My Effect", "Your Name", "One sentence on the look.")

    @classmethod
    def param_specs(cls):
        return [FloatParam("strength", "Strength", 0.5, 0.0, 1.0)]

    def render(self, ctx, progress):
        for i in range(ctx.length):
            if progress.cancelled:
                return
            ...
            progress.report(i + 1, ctx.length)
            yield frame
```

- `info.id` is stored in project files: keep it stable once used. It must be unique.
- `info.name` is shown in the UI.
- `render` is a generator. A new effect object is made for every transition, so
  `self` holds nothing from earlier renders.

### The effect contract

`tests/test_effect_contract.py` checks every registered effect automatically:

1. It yields exactly `ctx.length` frames.
2. Every frame is an `(H, W, 3)` `uint8` array of the project size.
3. It never modifies input frames.
4. The same context and seed give the same frames.
5. It stops, yielding nothing more, when `progress.cancelled` is true.
6. It works for every window shape: A only, B only, both, one frame, whole clips.

Checked in review, not by the test: no file I/O, no UI calls, no global state, and
every tunable value is declared in `param_specs()`.

The renderer also checks rules 1 and 2 at run time and stops the export with a
clear error message if an effect breaks them.

## Parameters

| Spec | Fields | Inspector widget |
| --- | --- | --- |
| `IntParam` | `id, label, default, min, max` | Integer spin box |
| `FloatParam` | `id, label, default, min, max` | Decimal spin box |
| `ChoiceParam` | `id, label, default, options` | Drop-down list |
| `BoolParam` | `id, label, default` | Check box |

`ctx.params` always holds every declared id. Missing values get their default.
Unknown ids, wrong types and values outside `min..max` are rejected before
`render` is called, so the effect does not need to check them again.

## Registry and discovery

At start-up the core imports every module in `moshflow/effects/` whose name does not
start with `_`. The `@register` decorator adds the class to the registry. A module
that fails to import does not stop the program: the error is shown in the UI and
by `python -m moshflow.cli effects`, and `test_all_effect_modules_load` fails.

## Toolkit

Helpers that more than one effect needs. They use plain NumPy arithmetic only.

### color, resize, frames (working)

| Function | Returns |
| --- | --- |
| `color.bgr_to_y(frame)` | `(H, W)` float32 luma: `0.114 B + 0.587 G + 0.299 R` |
| `color.to_uint8(values)` | Clips to 0..255, rounds, converts to uint8 |
| `resize.nearest(frame, (w, h))`, `resize.bilinear(frame, (w, h))` | Resized frame |
| `frames.window_frames(ctx)` | `[(clip, index), ...]` for every output frame: A's window, then B's |
| `frames.last_clean_a(ctx)` | A's frame just before the window (A's first frame if there is none) |

### Motion vectors (motion.py and warp.py)

Effects that use these helpers depend on these conventions, so they stay fixed.

- A vector `v = (dx, dy)` says where a pixel came from:
  `current(x, y) ≈ reference(x + dx, y + dy)`. An object that moved 2 pixels to the
  right has `v = (-2, 0)`.
- Block vectors are int32 with shape `(rows, cols, 2)`, where
  `rows = ceil(H / block)` and `cols = ceil(W / block)`. `[..., 0]` is dx, `[..., 1]` is dy.
  The last row and column of blocks may be smaller than `block × block`.
- A field is float32 with shape `(H, W, 2)`: one vector per pixel, in the same order.

| Function | Does |
| --- | --- |
| `motion.estimate(reference, current, block, search)` | Full-search block matching on two `(H, W)` float32 luma images. For each block of `current`, the `d` with `abs(dx), abs(dy) <= search` and the smallest `SAD(d) = Σ abs(current(x) - reference(x + d))`. Candidates reaching outside the image are skipped; ties go to the smaller `abs(dx) + abs(dy)`. Returns block vectors |
| `motion.to_field(vectors, (w, h), block)` | Every pixel takes its block's vector. Returns a field |
| `motion.smooth_field(vectors, (w, h), block, smoothness)` | Bilinear interpolation between block centres, mixed with `to_field` by `smoothness` (0 gives exactly `to_field`) |
| `warp.warp(image, field, sampling="nearest")` | Backward warping: `out(x, y) = image(x + dx, y + dy)`. Positions outside the image are clamped to the edge. `"nearest"` keeps the blocky codec look, `"bilinear"` is smooth. Returns float32, same shape as `image` |

The worked example from the proposal (slides 6 and 7) follows these conventions:
the block `50 60 / 70 80` moves from (1, 1) to (2, 2), so `estimate` gives
`v = (-1, -1)`, and `warp` makes pixel (2, 2) read pixel (1, 1).

`tests/test_toolkit.py` checks these results.

Speed: on a 1280 × 720 frame, `estimate` with block 16 and search 8 takes about
0.35 s on our test laptop, and `warp` about 0.03 s. For long transitions, estimate
on every second pixel (`luma[::2, ::2]`, with half the block and search) and double
the vectors.

## Finished effects

Keyframe Drop, Motion Bloom and Liquid Melt are finished. Each member's file only has
to follow this API; the parameters below are what the Inspector shows today.

| Effect | id | Parameters |
| --- | --- | --- |
| Keyframe Drop | `keyframe_drop` | `block` int 16 (4–64), `search` int 8 (1–32), `heal` float 0.02 (0–1) |
| Motion Bloom | `motion_bloom` | `gain` float 1.0 (0–4), `residual` float 1.0 (0–2), `emerge` float 1.0 (0–3) |
| Liquid Melt | `liquid_melt` | `look` pixel / liquid, `direction` right / left, `pixel` int 8 (1–32), `melt` float 2.0 (0–6), `strength` float 1.5 (0–3), `inertia` float 0.6 (0–0.95), `imprint` float 0.6 (0–1), `settle` float 0.35 (0–1), `block` int 16 (8–64) |

Keyframe Drop works best with the `after` alignment and a clip B with clear motion.
When a new effect works, remove `(draft)` from its `info.name`.

## Library rule

The course allows no library functions except OpenCV video input and output.

| Allowed | Not allowed |
| --- | --- |
| NumPy arrays: creating, indexing, slicing, `astype`, shapes | Any `cv2` call outside `moshflow/videoio/` |
| Element-wise `+ - * /`, `abs`, `min`, `max`, `sum`, `clip`, `argmin`, `round` | `np.fft`, `np.convolve`, `np.correlate`, `np.gradient`, `np.interp`, `np.histogram` |
| Python standard library for program logic | SciPy, scikit-image, Pillow, imageio, MoviePy, PyAV, FFmpeg filters, Numba |
| | Qt image functions (`QImage`, `QTransform`, `scaled`) for processing |

`tests/test_compliance.py` checks the source for these. If you are unsure about a
function, ask before using it.

## Testing an effect

```bash
python -m pytest                                    # everything
python -m pytest tests/test_effect_contract.py      # the contract, for every effect
python -m pytest tests/test_toolkit.py              # the shared helpers
python -m moshflow.cli effects                      # list effects and their parameters
python -m moshflow.cli try --effect keyframe_drop a.mp4 b.mp4 -o test.mp4 --param heal=0.1
```

`try` renders one transition between the end of `a.mp4` and the start of `b.mp4`
(`--duration` seconds, `--align before|center|after`, `--size 640x360` for speed).

## Changing the API

API v1 is frozen. If an effect needs something the API does not offer, ask Yuhyeon
first. An effect can usually do what it needs inside its own file or with a new
toolkit helper, and neither changes the API.
