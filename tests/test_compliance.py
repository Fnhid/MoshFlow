"""Course rule: apart from OpenCV video I/O, no library image processing.

This walks the syntax tree of every source file under moshflow/, so comments
and docstrings do not count. It fails on any forbidden import or call.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent / "moshflow"
SOURCES = sorted(ROOT.rglob("*.py"))

CV2_ALLOWED_DIR = ROOT / "videoio"
CV2_ALLOWED = {"VideoCapture", "VideoWriter"}          # plus CAP_PROP_* constants
BANNED_MODULES = {"scipy", "skimage", "PIL", "numba", "imageio", "moviepy", "ffmpeg", "av", "sklearn"}
BANNED_NUMPY = {"fft", "convolve", "correlate", "gradient", "interp", "histogram", "histogram2d", "digitize"}
BANNED_QT = {"QImage", "QTransform", "scaled", "scaledToWidth", "scaledToHeight", "transformed"}


def dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def violations(path: Path) -> list[str]:
    found = []
    in_videoio = CV2_ALLOWED_DIR in path.parents
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        where = f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Import):
            modules, names = [alias.name for alias in node.names], []
        elif isinstance(node, ast.ImportFrom):
            modules, names = [node.module or ""], [alias.name for alias in node.names]
        else:
            modules, names = [], []
        for module in modules:
            top = module.split(".")[0]
            if top in BANNED_MODULES:
                found.append(f"{where}: imports {module}")
            if top == "cv2" and not in_videoio:
                found.append(f"{where}: cv2 may only be used in moshflow/videoio")
            if top == "numpy" and names and set(names) & BANNED_NUMPY:
                found.append(f"{where}: imports {set(names) & BANNED_NUMPY} from numpy")
        for name in names:
            if name in BANNED_QT:
                found.append(f"{where}: imports {name}")
            if modules and modules[0].split(".")[0] == "cv2":
                found.append(f"{where}: use 'import cv2' so the calls can be checked")

        if isinstance(node, ast.Attribute):
            chain = dotted(node)
            if chain:
                head, _, rest = chain.partition(".")
                first = rest.split(".")[0]
                if head == "cv2" and first not in CV2_ALLOWED and not first.startswith("CAP_PROP_"):
                    found.append(f"{where}: cv2.{rest} is not video I/O")
                if head in ("np", "numpy") and first in BANNED_NUMPY:
                    found.append(f"{where}: {chain} is a ready-made algorithm")
            if node.attr in BANNED_QT:
                found.append(f"{where}: .{node.attr} is a Qt image function")
    return found


@pytest.mark.parametrize("path", SOURCES, ids=[str(p.relative_to(ROOT)) for p in SOURCES])
def test_no_forbidden_library_use(path):
    assert not violations(path), "\n".join(violations(path))


def test_the_check_catches_violations(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import cv2\nimport numpy as np\nx = cv2.resize(a, b)\ny = np.fft.fft(a)\n", encoding="utf-8")
    original = globals()["ROOT"]
    globals()["ROOT"] = tmp_path
    try:
        found = violations(bad)
    finally:
        globals()["ROOT"] = original
    assert any("cv2.resize" in v for v in found)
    assert any("np.fft" in v for v in found)
    assert any("may only be used" in v for v in found)


# Visual Studio 2019 installs Python 3.7, so the program must run on it.
OLD_PYTHON = (3, 7)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT.parent)))
def test_runs_on_old_python(path):
    source = path.read_text(encoding="utf-8")
    # No newer syntax. On Python 3.7 itself, parsing is the check.
    tree = ast.parse(source, feature_version=OLD_PYTHON) if sys.version_info >= (3, 8) else ast.parse(source)
    if tree.body:
        # Annotations such as `int | None` or `list[int]` stay unevaluated.
        assert "from __future__ import annotations" in source, f"{path} needs the __future__ import"
    for node in ast.walk(tree):
        assert type(node).__name__ != "NamedExpr", f"{path}:{node.lineno} walrus needs Python 3.8"
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            newer = {a.name for a in node.names} & {"Protocol", "Literal", "TypedDict", "Final", "runtime_checkable"}
            in_try = any(isinstance(t, ast.Try) and node in ast.walk(t) for t in ast.walk(tree))
            assert not newer or in_try, f"{path}:{node.lineno} typing.{newer} needs Python 3.8"
