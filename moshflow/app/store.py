"""ProjectStore: saves and loads projects as JSON (.moshproj)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, fields, replace

from moshflow.core.project import Clip, OutputSettings, Project, Transition

EXTENSION = ".moshproj"
FILE_VERSION = 1


def save_project(project: Project, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project_to_dict(project), f, indent=2)


def load_project(path: str) -> Project:
    """Loads a project; relative clip and output paths are relative to the file."""
    with open(path, encoding="utf-8") as f:
        project = project_from_dict(json.load(f))
    base = os.path.dirname(os.path.abspath(path))
    clips = tuple(replace(clip, path=_resolve(base, clip.path)) for clip in project.clips)
    output = replace(project.output, path=_resolve(base, project.output.path))
    return replace(project, clips=clips, output=output)


def project_to_dict(project: Project) -> dict:
    return {
        "version": FILE_VERSION,
        "clips": [asdict(clip) for clip in project.clips],
        "transitions": [_transition_to_dict(t) for t in project.transitions],
        "output": asdict(project.output),
    }


def project_from_dict(data: dict) -> Project:
    version = data.get("version", FILE_VERSION)
    if version > FILE_VERSION:
        raise ValueError(f"the project file is version {version}; this MoshFlow reads up to {FILE_VERSION}")
    project = Project(
        clips=tuple(Clip(**c) for c in data.get("clips", [])),
        transitions=tuple(Transition(**t) for t in data.get("transitions", [])),
        output=OutputSettings(**data.get("output", {})),
    )
    project.validate()
    return project


def _transition_to_dict(transition: Transition) -> dict:
    # params is a read-only mapping, which asdict() cannot copy.
    data = {f.name: getattr(transition, f.name) for f in fields(transition)}
    data["params"] = dict(transition.params)
    return data


def _resolve(base: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))
