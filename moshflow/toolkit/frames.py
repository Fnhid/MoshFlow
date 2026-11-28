"""Which source frames a transition covers, so every effect walks them the same way."""

from __future__ import annotations

from moshflow.api import Frame, FrameSource, TransitionContext


def window_frames(ctx: TransitionContext) -> list[tuple[FrameSource, int]]:
    """(clip, frame index) for every output frame: A's window, then B's window."""
    return [(ctx.outgoing, i) for i in ctx.a_window] + [(ctx.incoming, i) for i in ctx.b_window]


def last_clean_a(ctx: TransitionContext) -> Frame:
    """The last frame of A before the window: the picture a datamosh starts from.

    When the window starts at A's first frame, there is none before it, so this
    returns A's first frame.
    """
    return ctx.outgoing.frame(max(ctx.a_window.start - 1, 0))
