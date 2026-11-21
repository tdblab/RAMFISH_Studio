"""Which gene is in which channel of a multi-channel microscope file, and which round the file belongs to.

The file name says which genes were stained and with which dye, for example
``round1_dpp_alexa546__optix_alexa647__omb_alexa488__hh_alexa514_.oir``. The order of the genes in the name
does not have to be the order of the channels in the file (in that example the file stores 488, 514, 546, 647).
So every gene is matched to a channel by its dye, in this order of preference:

1. the dye name the microscope saved for each channel (Olympus .oir files save it, e.g. "Alexa Fluor 546");
2. the excitation wavelength saved for each channel: each dye goes to the channel with the closest wavelength,
   choosing the combination with the smallest total difference;
3. the order of the genes in the file name (only when neither of the above is available; flagged for checking).

The filename rules are those of D-CIPHER: gene and dye tokens are separated by underscores, a dye written as its
own token belongs to the gene before it (``dpp_alexa546``), or it can be written in brackets (``dpp(546)``,
``fz4[A647]``, ``spalt(b1)``). HCR amplifier names (b1, b2, ...) are turned into dyes with a table you can edit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Laser line (nm) used for each dye. Used to match a dye to the excitation wavelength saved for a channel.
DYE_LASER_NM = {
    "dapi": 405, "hoechst": 405, "alexa350": 405, "alexa405": 405, "pacificblue": 405,
    "alexa488": 488, "fitc": 488, "gfp": 488, "fam": 488, "atto488": 488,
    "alexa514": 514, "yfp": 514,
    "alexa532": 532,
    "alexa546": 561, "alexa555": 561, "alexa568": 561, "cy3": 561, "tamra": 561, "tritc": 561, "rfp": 561, "atto565": 561,
    "alexa594": 594, "mcherry": 594, "texasred": 594, "atto594": 594,
    "alexa647": 640, "cy5": 640, "atto647n": 640, "sir": 640,
    "alexa700": 640, "alexa750": 730, "cy7": 730,
}
NUCLEAR_DYES = {"dapi", "hoechst"}
# Amplifier -> dye. Default: the usual HCR set-up of the RAMFISH papers (b1 = Alexa 546, b2 = Alexa 647,
# b3 = Alexa 488). Edit it on the Import page if your amplifiers carry other dyes.
DEFAULT_AMPLIFIER_DYES = {"b1": "alexa546", "b2": "alexa647", "b3": "alexa488", "b4": "alexa514", "b5": "alexa594"}
DEFAULT_OMIT_TOKENS = {"wt", "mut", "mutant", "rnai", "control", "ctrl", "processed"}

_ROUND_TOKEN = re.compile(r"^(?:round|rnd|cycle|hyb|r)0*(\d+)$", re.I)
_MAG_TOKEN = re.compile(r"^(\d{1,3})x(oil|water|w|o|sil|silicone|glyc|glycerol)?$", re.I)
_IMMERSION = {"oil", "water", "silicone", "sil", "glycerol", "glyc"}
_INLINE = re.compile(r"^([^\[\(\-]+)[\[\(\-]([^\]\)]+)[\]\)]?$")
_AMP = re.compile(r"^b(\d)$", re.I)
_JOINED = re.compile(r"^(.+?)((?:alexa|alex|af)\d{3})$", re.I)   # gene and dye written together

HOW_DYE_NAME = "dye name saved in the file"
HOW_WAVELENGTH = "excitation wavelength saved in the file"
HOW_ORDER = "order in the file name (please check)"
HOW_UNNAMED = "not in the file name"
HOW_SINGLE = "single-channel file"


def dye_key(text: str | None) -> str | None:
    """'Alexa Fluor 546', 'alexa546', 'alex546', 'AF546', 'A546', '546', 'Cy3', 'DAPI' -> a key of DYE_LASER_NM."""
    if not text:
        return None
    t = re.sub(r"[\s\-_()\[\]{}]", "", str(text).lower()).replace("fluor", "")
    t = re.sub(r"^hoechst\d*$", "hoechst", t)
    m = re.fullmatch(r"(?:alexa|alex|af|a)?(\d{3})", t)
    if m and f"alexa{m.group(1)}" in DYE_LASER_NM:
        return f"alexa{m.group(1)}"
    return t if t in DYE_LASER_NM else None


def dye_label(key: str | None) -> str:
    if not key:
        return ""
    if key.startswith("alexa"):
        return f"Alexa {key[5:]}"
    return {"dapi": "DAPI", "hoechst": "Hoechst"}.get(key, key.upper() if len(key) <= 4 else key)


@dataclass
class Stain:
    gene: str
    dye: str | None = None        # key of DYE_LASER_NM
    amplifier: str | None = None  # e.g. "b1"


@dataclass
class ParsedName:
    round: int | None
    prefix: str                   # tokens before the round (e.g. "fish1"): the sample
    stains: list[Stain] = field(default_factory=list)


def parse_filename(filename: str, omit_tokens: set[str] | None = None) -> ParsedName:
    """Round, sample prefix and (gene, dye) list from a file name."""
    omit = {t.lower() for t in (DEFAULT_OMIT_TOKENS if omit_tokens is None else omit_tokens)}
    name = Path(filename).name
    name = re.sub(r"(\.ome)?\.[A-Za-z0-9]+$", "", name)
    tokens = [t for t in re.split(r"[_\s]+", name) if t]

    rnd, prefix = None, ""
    for i, t in enumerate(tokens):
        m = _ROUND_TOKEN.match(t)
        if m:
            rnd = int(m.group(1))
            prefix = "_".join(tokens[:i])
            tokens = tokens[i + 1:]
            break

    stains: list[Stain] = []
    skip_next = False
    for k, t in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        low = t.lower()
        if low in omit:
            continue
        mag = _MAG_TOKEN.match(t)
        if mag:
            nxt = tokens[k + 1].lower() if k + 1 < len(tokens) else ""
            skip_next = not mag.group(2) and nxt in _IMMERSION
            continue
        if t.isdigit() and dye_key(t) is None:
            continue                      # "..._1.oir" (a copy number), not a gene
        dk = dye_key(t)
        amp = _AMP.match(t)
        prev = stains[-1] if stains else None
        if dk and dk not in NUCLEAR_DYES:
            if prev and prev.dye is None and prev.amplifier is None:
                prev.dye = dk
                continue
            if dk.startswith("alexa"):
                continue                  # a stray "alexa488" is never a gene
            # otherwise a dye-like word with no gene before it (e.g. "gfp") is the gene itself
        if amp and prev and prev.dye is None and prev.amplifier is None:
            prev.amplifier = f"b{amp.group(1)}"
            continue
        if dk in NUCLEAR_DYES:
            stains.append(Stain("DAPI", dk))
            continue
        j = _JOINED.match(t)
        if j and dye_key(j.group(2)):
            stains.append(Stain(j.group(1), dye_key(j.group(2))))     # "hsp67balexa546" = hsp67b + Alexa 546
            continue
        m = _INLINE.match(t)
        if m:
            gene, hint = m.group(1).strip(), m.group(2).strip()
            ah = _AMP.match(hint)
            if dye_key(gene) in NUCLEAR_DYES:
                stains.append(Stain("DAPI", "dapi"))
            else:
                stains.append(Stain(gene, dye_key(hint), f"b{ah.group(1)}" if ah else None))
            continue
        stains.append(Stain(t))
    return ParsedName(rnd, prefix, stains)


@dataclass
class ChannelMeta:
    index: int
    name: str = ""
    dye: str | None = None            # dye key from the dye name saved in the file
    excitation_nm: float | None = None
    emission_nm: float | None = None


@dataclass
class Assignment:
    index: int          # channel number in the file (0-based)
    name: str           # gene name, "DAPI", or C<n>
    dye: str | None
    how: str


def assign_channels(parsed: ParsedName, meta: list[ChannelMeta], amplifier_dyes: dict | None = None) -> tuple[list[Assignment], list[str]]:
    """Gene name for every channel of one file. Returns (assignments in channel order, warnings)."""
    amp = {k.lower(): v for k, v in (DEFAULT_AMPLIFIER_DYES if amplifier_dyes is None else amplifier_dyes).items()}
    warnings: list[str] = []
    n = len(meta)
    stains = []
    for s in parsed.stains:
        d = s.dye or (dye_key(amp.get(s.amplifier)) if s.amplifier else None)
        if s.amplifier and not d:
            warnings.append(f"Amplifier {s.amplifier} of {s.gene} has no dye in the amplifier table.")
        stains.append((s.gene, d))

    # A nuclear stain is not always written in the name: a 405 nm (or DAPI-named) channel left over is DAPI.
    if not any(d in NUCLEAR_DYES for _, d in stains) and len(stains) < n:
        nuclear = [c for c in meta if c.dye in NUCLEAR_DYES or (c.excitation_nm and c.excitation_nm < 430)]
        if len(nuclear) == 1:
            stains.append(("DAPI", "dapi"))

    result: dict[int, Assignment] = {}
    todo = list(range(len(stains)))

    # 1. dye names saved in the file
    by_dye: dict[str, list[ChannelMeta]] = {}
    for c in meta:
        if c.dye:
            by_dye.setdefault(c.dye, []).append(c)
    for i in list(todo):
        gene, d = stains[i]
        hits = [c for c in by_dye.get(d or "", []) if c.index not in result]
        if d and len(hits) == 1:
            result[hits[0].index] = Assignment(hits[0].index, gene, d, HOW_DYE_NAME)
            todo.remove(i)

    # 2. excitation wavelengths saved in the file (best overall match)
    free = [c for c in meta if c.index not in result and c.excitation_nm]
    with_dye = [i for i in todo if stains[i][1] in DYE_LASER_NM]
    if free and with_dye:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        cost = np.array([[abs(float(c.excitation_nm) - DYE_LASER_NM[stains[i][1]]) for c in free] for i in with_dye])
        rows, cols = linear_sum_assignment(cost)
        for r, k in zip(rows, cols):
            i, c = with_dye[r], free[k]
            gene, d = stains[i]
            result[c.index] = Assignment(c.index, gene, d, HOW_WAVELENGTH)
            todo.remove(i)
            if cost[r, k] > 40:
                warnings.append(f"{gene} ({dye_label(d)}) was put on channel {c.index + 1}, whose saved excitation "
                                f"({c.excitation_nm:.0f} nm) is {cost[r, k]:.0f} nm away. Please check.")

    # 3. the order of the genes in the file name
    free_idx = [c.index for c in meta if c.index not in result]
    for i, ci in zip(list(todo), free_idx):
        gene, d = stains[i]
        result[ci] = Assignment(ci, gene, d, HOW_ORDER)
        todo.remove(i)
    for i in todo:
        warnings.append(f"{stains[i][0]} is in the file name, but the file has no channel left for it ({n} channels).")
    for c in meta:
        if c.index not in result:
            result[c.index] = Assignment(c.index, f"C{c.index + 1}", c.dye, HOW_UNNAMED)

    out = [result[c.index] for c in meta]
    seen: dict[str, int] = {}
    for a in out:                      # one name per channel (a probe in two colors becomes gene, gene_2)
        k = a.name.lower()
        seen[k] = seen.get(k, 0) + 1
        if seen[k] > 1:
            a.name = f"{a.name}_{seen[k]}"
    return out, warnings


# ---------------------------------------------------------------- reading channel metadata from files
_OIR_CHANNEL = re.compile(
    rb'<commonphase:channel xsi:type="lsmimage:NormalChannel"[^>]*order="(\d+)"[^>]*>\s*'
    rb'<commonphase:name>([^<]*)</commonphase:name>(?:(?!</commonphase:channel>).){0,4000}?'
    rb'<lsmimage:dyeName>([^<]*)</lsmimage:dyeName>', re.S)


def oir_dye_names(path: str | Path, chunk: int = 32 << 20) -> dict[str, str]:
    """{channel name ('CH1', ...): dye name} from the settings an Olympus .oir file stores with the image.

    The settings sit near the end of the file, so the end is read first; large files are then read in pieces.
    """
    path = Path(path)
    size = path.stat().st_size
    found: dict[str, str] = {}

    def scan(buf: bytes):
        for m in _OIR_CHANNEL.finditer(buf):
            name, dye = m.group(2).decode("utf-8", "replace").strip(), m.group(3).decode("utf-8", "replace").strip()
            if name and dye and name not in found:
                found[name] = dye

    with open(path, "rb") as f:
        f.seek(max(0, size - 2 * chunk))
        scan(f.read())
        if not found and size > 2 * chunk:
            f.seek(0)
            overlap = b""
            while True:
                block = f.read(chunk)
                if not block:
                    break
                scan(overlap + block)
                overlap = block[-65536:]
    return found


# Olympus software saves a large stack as name.oir plus extra files name_00001, name_00002, ... (no extension)
# that hold the rest of the planes. They must sit next to the .oir for the whole stack to be read.
_OIR_PART = re.compile(r"^(.+)_(\d{5})$")
_OIR_ZSTACK = re.compile(
    rb"<commonparam:axis>ZSTACK</commonparam:axis>(?:(?!</commonimage:axis>).){0,400}?"
    rb"<commonparam:step>([\d.]+)</commonparam:step>\s*<commonparam:maxSize>(\d+)</commonparam:maxSize>", re.S)


def oir_part_stem(name: str) -> str | None:
    """'scan_00001' -> 'scan' (the .oir it belongs to, without '.oir'); None for any other file name."""
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    m = _OIR_PART.match(base)
    return m.group(1) if m else None


def oir_parts_for(oir_name: str, names) -> list[str]:
    """The extra files of ``oir_name`` among ``names`` (same folder, same name plus _00001, ...), in order."""
    oir = str(oir_name).replace("\\", "/")
    folder, _, base = oir.rpartition("/")
    if not base.lower().endswith(".oir"):
        return []
    stem = base[:-4]
    out = []
    for n in names:
        n2 = str(n).replace("\\", "/")
        f, _, b = n2.rpartition("/")
        if f == folder and oir_part_stem(b) == stem:
            out.append(n)
    return sorted(out, key=lambda n: str(n)[-5:])


def oir_planned_z(path: str | Path, chunk: int = 64 << 20) -> tuple[int, float] | None:
    """(number of planes, step in µm) the Z stack was set up with on the microscope, from the .oir settings.

    Compared with the planes actually in the file, this shows whether extra _00001 files are missing.
    """
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - chunk))
        m = _OIR_ZSTACK.search(f.read())
    if not m:
        return None
    return int(m.group(2)), float(m.group(1))


def channels_from_ome_xml(xml: str | None) -> list[ChannelMeta]:
    """Channel names, fluor (dye) and excitation / emission wavelengths of the first image in OME-XML."""
    if not xml:
        return []
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    pixels = next((e for e in root.iter() if e.tag.endswith("}Pixels") or e.tag == "Pixels"), None)
    if pixels is None:
        return []
    out = []
    for i, ch in enumerate(e for e in pixels if e.tag.endswith("Channel")):
        def num(k):
            try:
                v = float(ch.attrib.get(k))
                return v if v > 0 else None
            except (TypeError, ValueError):
                return None
        name = ch.attrib.get("Name", "") or ""
        out.append(ChannelMeta(i, name, dye_key(ch.attrib.get("Fluor")) or dye_key(name),
                               num("ExcitationWavelength"), num("EmissionWavelength")))
    return out
