"""P-frame Pulse: the transition moshes in rhythmic bursts of repeated P-frames.

Emulates moshy's ppulse: every few frames the current P-frame is duplicated a
few times, so the picture blooms in a burst, then the clip's own P-frames take
over again and the damage is carried forward. Over a cut with B's I-frame
dropped, the bursts pull B in step by step.

A P-frame here is a motion field F (one vector per pixel, from block matching:
F(x) says where the content at x came from) plus a residual E, the colour the
motion could not predict. Decoding it onto the picture P gives

    P_new(x) = P(x + F(x)) + E(x)

The transition, frame by frame (both clips keep their real speed):
- The picture starts as A's last clean frame and is decoded from P-frames all
  the way: A's own before the cut, B's after it (B's I-frame is dropped, so B's
  P-frames land on A's picture).
- Pulse: every `interval` frames, the P-frame of that moment is held for `hold`
  frames: its motion is decoded again on every one of them (motion only, as a
  repeated P-frame adds no new picture), so moving areas smear and swell. It is
  mixed with the clip's own current P-frame by the envelope e,
  F = (1 - e) F_clip + e F_held: early in the transition the clip keeps playing
  smoothly with a little extra smear, and only near the cut does a pulse freeze
  it into the repeated motion, so the start does not stutter like dropped frames. The
  held motion has the camera's own movement (its average vector) removed, and
  keeps its pattern but is scaled to an average of PULSE_PX pixels
  per frame (times gain and the envelope), so every pulse is visible. If the
  clip is still when a pulse starts, the pulse uses the latest real motion seen
  instead (starting with B's first motion), so a still clip still pulses. The
  clip keeps running underneath; when the pulse ends, decoding continues with
  the clip's current P-frame, so the picture snaps into its motion again with
  the smear baked in.
- Envelope: the pulses grow stronger towards the cut and weaker after it,
  e = smoothstep rising to 1 at the cut, then falling to 0 at the end.
- Heal: between pulses each pixel moves a step towards the clip's real frame,
  P += h (R - P) with h = heal (1 - e): A is clean at the start, the bloom
  dominates around the cut, and after the cut B heals in from pulse to pulse.
  Right after each pulse the picture snaps back by the full `heal`, even near
  the cut, so the pulses read as beats.
  Over the last LAND part h rises to 1, and the last frame is B's real frame.

Block matching is noisy in flat or blurry areas, and a wrong vector repeated
many times becomes a random streak, so vectors that do not explain the motion
clearly better than (0, 0) are dropped, and block vectors are interpolated
between block centres so no hard block edges appear. Pixels are read with
bilinear sampling, so fractional vectors do not leave a zig-zag.
"""

import numpy as np

from moshflow.api import EffectInfo, FloatParam, IntParam, TransitionEffect, register
from moshflow.toolkit import color, frames, motion, warp

BLOCK = 16           # block size and search range of the motion, at full size
SEARCH = 8
CONFIDENCE = 0.7     # keep a block's vector only if it cuts the block's error to below 70 %
LAND = 0.2           # share of the end over which the picture heals fully into B
PULSE_PX = 6.0       # average motion (pixels per frame) of a pulse at full strength
MOVING_PX = 0.3      # average vector length below which a P-frame counts as still
LOOK_AHEAD = 30      # B frames searched for B's first motion


@register
class PFramePulse(TransitionEffect):
    info = EffectInfo("pframe_pulse", "P-frame Pulse (draft)", "Viktor Tost",
                      "The cut moshes in rhythmic bursts: P-frames repeat in pulses that pull B in.")

    @classmethod
    def param_specs(cls):
        return [
            IntParam("interval", "Interval (frames)", 10, 3, 60),
            IntParam("hold", "Hold (frames)", 5, 1, 30),
            FloatParam("gain", "Gain", 1.5, 0.0, 4.0),
            FloatParam("heal", "Heal", 0.25, 0.0, 1.0),
        ]

    def render(self, ctx, progress):
        p = ctx.params
        width, height = ctx.size
        hold = min(p["hold"], p["interval"] - 1)
        cut = len(ctx.a_window) / ctx.length            # where the cut is, 0..1
        picture = frames.last_clean_a(ctx).astype(np.float32)
        last_field = np.zeros((height, width, 2), np.float32)
        held, held_left, just_ended = last_field, 0, False
        # Motion for pulses when the clip itself is still: the latest real motion
        # seen, starting with B's first motion.
        moving = _first_motion(ctx.incoming, min(ctx.b_window.start, len(ctx.incoming) - 1), ctx.size)
        for i, (clip, index) in enumerate(frames.window_frames(ctx)):
            if progress.cancelled:
                return
            u = (i + 1) / ctx.length
            e = _envelope(u, cut)
            real = clip.frame(index).astype(np.float32)

            if held_left == 0 and i % p["interval"] == p["interval"] - hold and e > 0.05:
                # A pulse starts: hold this moment's P-frame for `hold` frames.
                field = _field_at(clip, index, ctx.size)
                if field is None or not _is_moving(field):
                    field = moving
                held = _normalised(field)
                held_left = hold

            # The clip's own current P-frame. Between pulses it is decoded as it is;
            # during a pulse it is mixed with the held P-frame by the envelope, so a
            # weak pulse (early) leaves the clip playing smoothly and only a strong
            # one (near the cut) freezes it into the repeated motion.
            p_frame = _p_frame_at(clip, index, ctx.size)
            if p_frame is not None:
                last_field, error = p_frame
                if _is_moving(last_field):
                    moving = last_field
            else:
                error = None
            own = 1.0 - e if held_left > 0 else 1.0
            field = own * last_field
            if held_left > 0:
                field = field + e * PULSE_PX * p["gain"] * held
                held_left -= 1
                ending = held_left == 0
            else:
                ending = False
            picture = warp.warp(picture, field, "bilinear")
            if error is not None:
                picture = picture + own * error
            # A step of healing towards the clip's real frame, weaker during a pulse.
            # Right after a pulse the picture snaps back by the full `heal`, even
            # near the cut, so every pulse reads as a beat: smear, snap, smear.
            h = own * p["heal"] * (1.0 - e)
            if just_ended:
                h = max(h, p["heal"])
            picture = picture + h * (real - picture)
            just_ended = ending

            # Landing: heal fully into B over the last part.
            if u > 1.0 - LAND:
                landing = _smoothstep((u - (1.0 - LAND)) / LAND)
                picture = picture + landing * (real - picture)
            if i == ctx.length - 1:
                picture = real.copy()
            picture = np.clip(picture, 0.0, 255.0)

            progress.report(i + 1, ctx.length)
            yield color.to_uint8(picture)


def _is_moving(field: np.ndarray) -> bool:
    """True if the field's average vector is at least MOVING_PX pixels long."""
    return float(((field[..., 0] ** 2 + field[..., 1] ** 2) ** 0.5).mean()) >= MOVING_PX


def _first_motion(clip, start: int, size: tuple[int, int]) -> np.ndarray:
    """The first real motion in a clip from frame `start` on (zeros if it never moves)."""
    for index in range(start + 1, min(start + LOOK_AHEAD, len(clip))):
        field = _field_at(clip, index, size)
        if field is not None and _is_moving(field):
            return field
    width, height = size
    return np.zeros((height, width, 2), np.float32)


def _normalised(field: np.ndarray) -> np.ndarray:
    """The local part of the field, scaled so its average vector length is 1.

    The average vector is the camera moving everything the same way; repeating it
    would only shove the whole picture and snap it back. What is left after
    subtracting it is how things move relative to each other, which makes moving
    areas smear and swell. It keeps its pattern (where things move, and which
    way) but always has the same strength, so even a clip that barely moves
    blooms visibly. A field with no local motion is returned unchanged.
    """
    local = field - field.mean(axis=(0, 1))
    length = float(((local[..., 0] ** 2 + local[..., 1] ** 2) ** 0.5).mean())
    return local / length if length > 1e-3 else field


def _envelope(u: float, cut: float) -> float:
    """Pulse strength at progress u: rises to 1 at the cut, falls to 0 at the end."""
    if u <= cut:
        return _smoothstep(u / cut) if cut > 0 else 1.0
    return 1.0 - _smoothstep((u - cut) / (1.0 - cut))


def _smoothstep(t: float) -> float:
    """0 at t <= 0, 1 at t >= 1, with a gentle start and end: t^2 (3 - 2 t)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _p_frame_at(clip, index: int, size: tuple[int, int]):
    """The P-frame from frame index - 1 to index as (F, E), or None if there is none
    (the clip's first frame, or a repeated duplicate frame)."""
    if index <= 0 or _same(clip.frame(index - 1), clip.frame(index)):
        return None
    field = _field(clip, index - 1, index, size)
    predicted = warp.warp(clip.frame(index - 1), field, "bilinear")
    return field, clip.frame(index).astype(np.float32) - predicted


def _field_at(clip, index: int, size: tuple[int, int]):
    """Only the motion of the P-frame at index (see _p_frame_at), or None."""
    if index <= 0 or _same(clip.frame(index - 1), clip.frame(index)):
        return None
    return _field(clip, index - 1, index, size)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two frames are (almost) identical, i.e. a repeated frame."""
    return float(np.abs(a.astype(np.float32) - b).mean()) < 0.5


def _field(clip, first: int, second: int, size: tuple[int, int]) -> np.ndarray:
    """Motion from frame `first` to frame `second` of a clip, as a smooth field (H, W, 2).

    Block matching runs at half size (every second pixel, half the block and search)
    to keep it fast, and the vectors are doubled back to full size.
    """
    reference = color.bgr_to_y(clip.frame(first))[::2, ::2]
    current = color.bgr_to_y(clip.frame(second))[::2, ::2]
    vectors = motion.estimate(reference, current, BLOCK // 2, SEARCH // 2)
    vectors = _confident(vectors, reference, current, BLOCK // 2)
    return motion.smooth_field(vectors * 2, size, BLOCK, 1.0)


def _confident(vectors: np.ndarray, reference: np.ndarray, current: np.ndarray, block: int) -> np.ndarray:
    """Block vectors with the unreliable ones set to (0, 0).

    For each block, error(v) = sum of |current(x) - reference(x + v)| over its pixels.
    A vector is kept only if error(v) < CONFIDENCE * error(0): the match must be
    clearly better than "nothing moved", otherwise it is just noise.
    """
    height, width = current.shape
    field = motion.to_field(vectors, (width, height), block)
    moved = _block_sums(np.abs(current - warp.warp(reference, field)), block)
    still = _block_sums(np.abs(current - reference), block)
    out = vectors.copy()
    out[~(moved < CONFIDENCE * still)] = 0
    return out


def _block_sums(values: np.ndarray, block: int) -> np.ndarray:
    """Sum of an H x W array over each block; partial blocks at the edges sum what they have."""
    height, width = values.shape
    rows, cols = -(-height // block), -(-width // block)
    padded = np.zeros((rows * block, cols * block), np.float32)
    padded[:height, :width] = values
    return padded.reshape(rows, block, cols, block).sum(axis=(1, 3))
