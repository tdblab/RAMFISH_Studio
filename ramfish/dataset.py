"""Import a whole multi-round dataset from a .zip archive or a folder.

Expected layout (the one the original RAMFISH scripts used)::

    Round_1/DAPI.tif  Round_1/wg.tif  Round_2/DAPI.tif  Round_2/optix.tif ...

Round folders may also be called ``round1``, ``R01``, ``Cycle 3``...; files at
the top level can carry the round in their name instead (``R1_DAPI.tif``,
``round2-wg.tif``). The channel name is the file name without extension and
without legacy prefixes such as ``ALIGNED_``.
"""

from __future__ import annotations

import io
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from .io import MICROSCOPY_EXTS, PLAIN_IMAGE_EXTS, SUPPORTED_EXTS, TIFF_EXTS, _plain_to_czyx, _tiff_to_czyx, file_ext, load_stack, to_16bit_scale
from .project import clean_name

_ROUND_DIR = re.compile(r"^(?:round|rnd|cycle|hyb|r)[\s_-]*0*(\d+)$", re.I)
_ROUND_PREFIX = re.compile(r"^(?:round|rnd|cycle|hyb|r)[\s_-]*0*(\d+)[\s_-]+(.+)$", re.I)
_LEGACY_PREFIXES = ("ALIGNED_", "NORM_", "RAW_")


@dataclass
class Entry:
    name: str          # path inside the archive / folder
    round: int | None  # None = could not be determined
    channel: str


def _strip_ext(filename: str) -> str:
    return re.sub(r"(\.ome)?\.[^.]+$", "", filename, flags=re.I)


def parse_name(relpath: str) -> tuple[int | None, str]:
    parts = PurePosixPath(relpath.replace("\\", "/")).parts
    stem = _strip_ext(parts[-1])
    for prefix in _LEGACY_PREFIXES:
        if stem.upper().startswith(prefix):
            stem = stem[len(prefix):]
    rnd = None
    for d in reversed(parts[:-1]):
        m = _ROUND_DIR.match(d.strip())
        if m:
            rnd = int(m.group(1))
            break
    if rnd is None:
        m = _ROUND_PREFIX.match(stem)
        if m:
            rnd, stem = int(m.group(1)), m.group(2)
    else:
        # "Round_3/3_Dll.tif": drop a leading number that repeats the round.
        m = re.match(r"^0*(\d+)[\s_-]+(.+)$", stem)
        if m and int(m.group(1)) == rnd:
            stem = m.group(2)
    return rnd, clean_name(stem)


def _is_image(name: str) -> bool:
    p = PurePosixPath(name)
    if any(part.startswith(".") or part == "__MACOSX" for part in p.parts):
        return False
    return file_ext(name) in SUPPORTED_EXTS


def scan(names: list[str]) -> list[Entry]:
    """Assign every image to a round and channel.

    Within a round, a single file whose name contains ``dapi`` or ``mask``
    (e.g. ``DAPI_round3``) is renamed to exactly ``DAPI`` / ``mask`` so that
    alignment finds it.
    """
    entries = [Entry(n, *parse_name(n)) for n in sorted(names) if _is_image(n)]
    by_round: dict[int | None, list[Entry]] = {}
    for e in entries:
        by_round.setdefault(e.round, []).append(e)
    for rnd, items in by_round.items():
        if rnd is None:
            continue
        for canonical in ("DAPI", "mask"):
            key = canonical.lower()
            exact = [e for e in items if e.channel.lower() == key]
            if exact:
                for e in exact:
                    e.channel = canonical
                continue
            hits = [e for e in items if key in e.channel.lower()]
            if len(hits) == 1:
                hits[0].channel = canonical
    return entries


class ZipSource:
    def __init__(self, fileobj):
        self.zf = zipfile.ZipFile(fileobj)

    def names(self) -> list[str]:
        return [i.filename for i in self.zf.infolist() if not i.is_dir()]

    def read(self, name: str) -> bytes:
        return self.zf.read(name)


class FolderSource:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Not a folder: {self.root}")

    def names(self) -> list[str]:
        return [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()]

    def read(self, name: str) -> bytes:
        return (self.root / name).read_bytes()


def load_stack_bytes(name: str, data: bytes) -> np.ndarray:
    """Like :func:`ramfish.io.load_stack` for in-memory file contents."""
    ext = file_ext(name)
    if ext in TIFF_EXTS:
        return to_16bit_scale(_tiff_to_czyx(io.BytesIO(data)))
    if ext in PLAIN_IMAGE_EXTS:
        return to_16bit_scale(_plain_to_czyx(data))
    if ext in MICROSCOPY_EXTS:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / Path(name).name
            p.write_bytes(data)
            return load_stack(p)
    raise ValueError(f"Unsupported file type: {name}")
