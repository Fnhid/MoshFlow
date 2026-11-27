"""Command-line interface, mainly for developing and testing effects.

  python -m moshflow.cli effects
  python -m moshflow.cli try --effect crossfade a.mp4 b.mp4 -o test.mp4 [--param ease=2]
  python -m moshflow.cli render project.moshproj [-o out.mp4]
"""

import argparse
import json
import sys

from moshflow.app.catalog import EffectCatalog
from moshflow.app.render_job import JobProgress, render_project
from moshflow.app.store import load_project
from moshflow.core.project import ALIGNMENTS, Clip, OutputSettings, Project, Transition
from moshflow.videoio import probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="moshflow", description="MoshFlow command-line tools")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("effects", help="list the available effects and their parameters")

    p_try = sub.add_parser("try", help="render one transition between two clips")
    p_try.add_argument("clip_a")
    p_try.add_argument("clip_b")
    p_try.add_argument("--effect", required=True, help="effect id (see 'effects')")
    p_try.add_argument("-o", "--output", default="test.mp4")
    p_try.add_argument("--duration", type=float, default=1.0, help="transition length in seconds")
    p_try.add_argument("--align", choices=ALIGNMENTS, default="after")
    p_try.add_argument("--param", action="append", default=[], metavar="ID=VALUE")
    p_try.add_argument("--seed", type=int, default=0)
    p_try.add_argument("--size", help="output size, e.g. 640x360")
    p_try.add_argument("--fps", type=float)
    p_try.add_argument("--context", type=float, default=1.0,
                       help="seconds of each clip to keep around the transition")

    p_render = sub.add_parser("render", help="render a saved project")
    p_render.add_argument("project")
    p_render.add_argument("-o", "--output")

    args = parser.parse_args(argv)
    catalog = EffectCatalog()
    for failure in catalog.failures:
        print(f"warning: could not load {failure.module}:\n{failure.error}", file=sys.stderr)

    try:
        if args.command == "effects":
            return _list_effects(catalog)
        if args.command == "try":
            project = _try_project(args)
            catalog.get(args.effect)  # fail early on an unknown id
        else:
            project = load_project(args.project)
        return _render(project, args.output)
    except (KeyError, ValueError, IOError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _list_effects(catalog: EffectCatalog) -> int:
    for entry in catalog.entries():
        print(f"{entry.id:<20} {entry.name}  ({entry.author})")
        print(f"{'':<20} {entry.description}")
        for spec in entry.specs:
            print(f"{'':<22}- {spec.id}: {spec}")
    return 0


def _try_project(args) -> Project:
    width = height = None
    if args.size:
        width, height = (int(v) for v in args.size.lower().split("x"))
    params = {}
    for item in args.param:
        key, _, raw = item.partition("=")
        params[key] = _parse_value(raw)
    # Keep only a short piece of each clip so the test render is fast.
    a_info = probe(args.clip_a)
    keep = args.context + args.duration
    return (Project(output=OutputSettings(path=args.output, width=width, height=height, fps=args.fps))
            .with_clip(Clip(args.clip_a, start=max(0.0, a_info.duration - keep)))
            .with_clip(Clip(args.clip_b, end=keep))
            .with_transition(0, Transition(args.effect, args.duration, args.align, params, args.seed)))


def _parse_value(raw: str) -> object:
    try:
        return json.loads(raw)       # numbers, true / false
    except json.JSONDecodeError:
        return raw                   # plain strings for ChoiceParam


def _render(project: Project, output: str | None) -> int:
    def show(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            print(f"\rrendering {done}/{total} frames ({100 * done // max(total, 1)}%)", end="", flush=True)

    result = render_project(project, JobProgress(show), output)
    print(f"\nwrote {result.path}: {result.frames} frames, {result.size[0]}x{result.size[1]} "
          f"@ {result.fps:g} fps in {result.seconds:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
