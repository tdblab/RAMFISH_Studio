"""One RAMFISH project = one folder, ``Output/<project name>/``::

    1_normalized/Round_1/DAPI.tif           step 1: 16-bit images used by the next steps
    1_normalized/exports/Round_1/NORM_...   step 1: the exporter's files (chosen format, Z-slices, MIP)
    2_rough_aligned/Round_2/ALIGNED_DAPI.tif
    3_fine_aligned/R2_aligned_DAPI.tif
    4_spots/data_sheets/, spot_images/, synthetic_spots/
    5_merge/multiplex_composite.jpg
    .ramfish_project.json                   rounds, channels and settings

Round 1 is the coordinate reference, so it only ever has ``normalized`` images.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .io import read_plane, write_plane_u16

STAGES = ("normalized", "rough", "fine")
META = ".ramfish_project.json"
STEP_DIRS = {"normalized": "1_normalized", "rough": "2_rough_aligned", "fine": "3_fine_aligned",
             "spots": "4_spots", "merge": "5_merge"}
# Channels used to align rounds, in order of preference (matched case-insensitively).
ALIGNMENT_CHANNELS = ("dapi", "mask")


def is_alignment_channel(name: str) -> bool:
    return name.lower() in ALIGNMENT_CHANNELS


def is_mask(name: str) -> bool:
    return name.lower() == "mask"


def clean_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name.strip()).strip("._")
    return name or "channel"


class Project:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / META).exists():
            self.meta = json.loads((self.root / META).read_text())
        else:
            self.meta = {
                "name": self.root.name,
                "rounds": {},
                "history": [],
                "created": _now(),
                "software_version": __version__,
            }
            self.save()

    # ------------------------------------------------------------------ meta
    def save(self) -> None:
        (self.root / META).write_text(json.dumps(self.meta, indent=2))

    def rounds(self) -> list[int]:
        return sorted(int(r) for r in self.meta["rounds"])

    def channels(self, rnd: int) -> list[str]:
        return list(self.meta["rounds"].get(str(rnd), []))

    def is_empty(self) -> bool:
        return not any(self.meta["rounds"].values())

    def log(self, step: str, params: dict, **details) -> None:
        self.meta["history"].append({"time": _now(), "step": step, "params": params, **details})
        self.save()

    # ---------------------------------------------------------------- images
    @property
    def name(self) -> str:
        return self.meta.get("name", self.root.name)

    def step_dir(self, step: str) -> Path:
        return self.root / STEP_DIRS[step]

    def path(self, stage: str, rnd: int, channel: str) -> Path:
        if stage == "normalized":
            return self.step_dir(stage) / f"Round_{rnd}" / f"{channel}.tif"
        if stage == "rough":
            return self.step_dir(stage) / f"Round_{rnd}" / f"ALIGNED_{channel}.tif"
        if stage == "fine":
            return self.step_dir(stage) / f"R{rnd}_aligned_{channel}.tif"
        raise ValueError(stage)

    def has(self, stage: str, rnd: int, channel: str) -> bool:
        return self.path(stage, rnd, channel).exists()

    def add_image(self, rnd: int, channel: str, plane: np.ndarray) -> str:
        """Store a normalized image; invalidates later stages of that round."""
        channel = clean_name(channel)
        write_plane_u16(self.path("normalized", rnd, channel), plane)
        chans = self.meta["rounds"].setdefault(str(rnd), [])
        if channel not in chans:
            chans.append(channel)
        self.meta.get("rough_params", {}).pop(str(rnd), None)
        self.save()
        self.clear_stage("rough", rnd)
        self.clear_stage("fine", rnd)
        return channel

    def rename_channel(self, rnd: int, old: str, new: str) -> str:
        new = clean_name(new)
        if new == old or new in self.channels(rnd):
            return old
        for stage in STAGES:
            src = self.path(stage, rnd, old)
            if src.exists():
                src.rename(self.path(stage, rnd, new))
        chans = self.meta["rounds"][str(rnd)]
        chans[chans.index(old)] = new
        self.save()
        return new

    def write_stage(self, stage: str, rnd: int, channel: str, plane: np.ndarray) -> None:
        write_plane_u16(self.path(stage, rnd, channel), plane)

    def clear_stage(self, stage: str, rnd: int) -> None:
        if stage == "fine":
            for p in self.step_dir("fine").glob(f"R{rnd}_aligned_*"):
                p.unlink()
            return
        d = self.step_dir(stage) / f"Round_{rnd}"
        if d.exists():
            shutil.rmtree(d)
        if stage == "rough":
            self.clear_stage("fine", rnd)

    def remove_channel(self, rnd: int, channel: str) -> None:
        for stage in STAGES:
            p = self.path(stage, rnd, channel)
            if p.exists():
                p.unlink()
        chans = self.meta["rounds"].get(str(rnd), [])
        if channel in chans:
            chans.remove(channel)
        if not chans:
            self.meta["rounds"].pop(str(rnd), None)
        self.save()

    def remove_round(self, rnd: int) -> None:
        for stage in STAGES:
            self.clear_stage(stage, rnd)
        self.meta["rounds"].pop(str(rnd), None)
        self.save()

    def stage_of(self, rnd: int, channel: str, upto: str = "fine") -> str | None:
        """Most processed stage available for an image, not beyond ``upto``."""
        allowed = STAGES[: STAGES.index(upto) + 1]
        if rnd == self.reference_round:
            allowed = ("normalized",)
        for stage in reversed(allowed):
            if self.has(stage, rnd, channel):
                return stage
        return None

    def load(self, rnd: int, channel: str, stage: str | None = None, upto: str = "fine") -> np.ndarray:
        stage = stage or self.stage_of(rnd, channel, upto)
        if stage is None:
            raise FileNotFoundError(f"No image for Round {rnd} / {channel}.")
        return read_plane(self.path(stage, rnd, channel))

    @property
    def reference_round(self) -> int | None:
        r = self.rounds()
        return r[0] if r else None

    # ---------------------------------------------------------- alignment
    def alignment_options(self, rnd: int) -> list[str]:
        """Channels of ``rnd`` that can align it to Round 1: DAPI and/or mask.

        A channel qualifies when Round 1 has a channel of the same name
        (case-insensitive). DAPI is listed first.
        """
        r0 = self.reference_round
        if r0 is None:
            return []
        base = {c.lower() for c in self.channels(r0)}
        opts = [c for c in self.channels(rnd) if is_alignment_channel(c) and c.lower() in base]
        return sorted(opts, key=lambda c: ALIGNMENT_CHANNELS.index(c.lower()))

    def alignment_channel(self, rnd: int) -> str | None:
        """Channel used to align ``rnd``: the user's choice, else DAPI, else mask."""
        opts = self.alignment_options(rnd)
        chosen = self.meta.get("align_with", {}).get(str(rnd))
        if chosen in opts:
            return chosen
        return opts[0] if opts else None

    def set_alignment_channel(self, rnd: int, channel: str) -> None:
        self.meta.setdefault("align_with", {})[str(rnd)] = channel
        self.save()

    def background_channel(self) -> str | None:
        """Round 1's DAPI, else its mask: the tissue context for merged images."""
        chans = self.channels(self.reference_round) if self.reference_round is not None else []
        for name in ALIGNMENT_CHANNELS:
            hit = next((c for c in chans if c.lower() == name), None)
            if hit:
                return hit
        return None

    def reference_partner(self, rnd: int, channel: str) -> str:
        """Round 1's channel matching ``channel`` (same name, any case)."""
        return next(c for c in self.channels(self.reference_round) if c.lower() == channel.lower())

    def gene_targets(self) -> list[tuple[int, str, str]]:
        """``(round, channel, label)`` for every gene channel (not DAPI / mask).

        Labels are the channel names, made unique with a ``_R<n>`` suffix if
        the same name appears in more than one round.
        """
        items = [(r, c) for r in self.rounds() for c in self.channels(r) if not is_alignment_channel(c)]
        counts: dict[str, int] = {}
        for _, c in items:
            counts[c] = counts.get(c, 0) + 1
        return [(r, c, c if counts[c] == 1 else f"{c}_R{r}") for r, c in items]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
