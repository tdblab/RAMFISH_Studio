"""Reading microscopy files into a common ``(C, Z, Y, X)`` float32 layout.

All pixel data inside a RAMFISH project is kept on a 16-bit intensity scale
(0-65535) regardless of the bit depth of the source file, so that the
normalization, spot-intensity cutoffs and exports behave the same for 8-bit
JPEGs, 12-bit camera TIFFs and 16-bit confocal stacks.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

# The formats ramfish_processor.py accepts.
SUPPORTED_EXTENSIONS = [
    '.ome.tif', '.ome.tiff', '.qptiff', '.dicom', '.mrcs',
    '.tif', '.tiff', '.oir', '.oib', '.czi', '.nd2', '.ims', '.lif', '.vsi',
    '.lsm', '.zvi', '.lei', '.ics', '.ids', '.stk', '.dv', '.r3d', '.im3',
    '.dcm', '.mrc', '.rec', '.png', '.jpg', '.jpeg', '.bmp'
]
PLAIN_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
TIFF_EXTS = {".tif", ".tiff"}
# Everything else (OME-TIFF, QPTIFF and the proprietary formats) goes through AICSImage.
MICROSCOPY_EXTS = {e for e in SUPPORTED_EXTENSIONS if e not in PLAIN_IMAGE_EXTS | TIFF_EXTS}
SUPPORTED_EXTS = PLAIN_IMAGE_EXTS | TIFF_EXTS | MICROSCOPY_EXTS


def file_ext(path: str | Path) -> str:
    name = str(path).lower()
    for ext in (".ome.tif", ".ome.tiff"):
        if name.endswith(ext):
            return ext
    return Path(name).suffix


def to_16bit_scale(arr: np.ndarray) -> np.ndarray:
    """Map any integer/float array onto a float32 0-65535 intensity scale."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) * (65535.0 / 255.0)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32)
    if arr.dtype == bool:
        return arr.astype(np.float32) * 65535.0
    out = arr.astype(np.float32)
    lo, hi = float(np.nanmin(out)), float(np.nanmax(out))
    if np.issubdtype(arr.dtype, np.floating) and lo >= 0.0 and hi <= 1.0:
        return out * 65535.0
    if lo < 0.0 or hi > 65535.0:
        span = hi - lo if hi > lo else 1.0
        return (out - lo) * (65535.0 / span)
    return out


def _collapse_pseudocolor(czyx: np.ndarray) -> np.ndarray:
    """Turn an RGB(A) image into one channel when it only carries one signal.

    Pseudo-coloured exports (a blue mask, a yellow gene on black) or grey
    images saved as RGB become a single channel (max over colours, so the
    signal keeps its full brightness). Genuine composites with signal in
    several colours keep R, G and B as separate channels.
    """
    if czyx.shape[0] == 4:
        czyx = czyx[:3]
    if czyx.shape[0] != 3:
        return czyx
    # A pseudo-coloured export (yellow, magenta, orange, pink, grey, ...) repeats
    # one signal in several colours, possibly at different strengths; only
    # colours that vary independently carry separate signals.
    # A colour far fainter than the main one (e.g. a green trace of at most 3/255
    # under a blue signal) is an export artefact, not a separate signal.
    peaks = [float(czyx[c].max()) for c in range(3)]
    used = [c for c in range(3) if peaks[c] > 0.05 * max(peaks)]
    if all(_same_signal(czyx[used[0]], czyx[c]) for c in used[1:]):
        return czyx.max(axis=0, keepdims=True)
    return czyx


def _same_signal(a: np.ndarray, b: np.ndarray) -> bool:
    """True if ``b`` is (a scaled copy of) ``a``."""
    if np.array_equal(a, b):
        return True
    step = max(1, int(np.sqrt(a.size / 250_000)))   # a few hundred thousand pixels are plenty
    x = a[..., ::step, ::step].astype(np.float64).ravel()
    y = b[..., ::step, ::step].astype(np.float64).ravel()
    if x.std() == 0 or y.std() == 0:
        return False
    return float(np.corrcoef(x, y)[0, 1]) > 0.98


def _tiff_to_czyx(path_or_buf) -> np.ndarray:
    with tifffile.TiffFile(path_or_buf) as tf:
        series = tf.series[0]
        data = series.asarray()
        axes = series.axes.upper()
    rgb = "S" in axes and "C" not in axes and "I" not in axes and "Q" not in axes

    # Drop time / position axes by taking the first index.
    for ax in "TR":
        while ax in axes:
            i = axes.index(ax)
            data = np.take(data, 0, axis=i)
            axes = axes[:i] + axes[i + 1:]

    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"TIFF has no recognizable Y/X axes (axes={axes!r}).")

    # RGB samples and unnamed stack axes are treated as channels; Z stays Z.
    axes = "".join("C" if a in "SIQ" else a for a in axes)
    # Merge repeated C axes (e.g. "CSYX" -> "CCYX").
    while axes.count("C") > 1:
        first = axes.index("C")
        second = axes.index("C", first + 1)
        data = np.moveaxis(data, second, first + 1)
        moved = axes[:second] + axes[second + 1:]
        shape = list(data.shape)
        shape[first:first + 2] = [shape[first] * shape[first + 1]]
        data = data.reshape(shape)
        axes = moved
    for ax in "CZ":
        if ax not in axes:
            data = data[np.newaxis]
            axes = ax + axes
    unknown = [a for a in axes if a not in "CZYX"]
    for a in unknown:
        i = axes.index(a)
        data = np.take(data, 0, axis=i)
        axes = axes[:i] + axes[i + 1:]
    order = [axes.index(a) for a in "CZYX"]
    data = np.transpose(data, order)
    return _collapse_pseudocolor(data) if rgb else data


def _plain_to_czyx(path_or_bytes) -> np.ndarray:
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image (unsupported or corrupted file).")
    if img.ndim == 3:
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        return _collapse_pseudocolor(np.moveaxis(rgb, -1, 0))[:, np.newaxis]
    return img[np.newaxis, np.newaxis]


_AICS = None


def _aicsimageio():
    """Java / Bio-Formats environment set up exactly as in ramfish_processor.py."""
    global _AICS
    if _AICS is None:
        conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
        java_path = Path(conda_prefix) / "Library"
        if not os.environ.get("JAVA_HOME"):
            os.environ["JAVA_HOME"] = str(java_path)
        import scyjava

        scyjava.config.endpoints.append('com.google.guava:guava:31.1-jre')
        # Use the installed Java (JAVA_HOME / PATH). scyjava >= 1.12 otherwise downloads its own, because
        # it cannot read the version of conda-forge's openjdk ("25.0.2-internal").
        java_exe = Path(os.environ["JAVA_HOME"]) / "bin" / ("java.exe" if sys.platform == "win32" else "java")
        if hasattr(scyjava.config, "set_java_constraints") and (java_exe.exists() or shutil.which("java")):
            scyjava.config.set_java_constraints(fetch="never", version="")
        from aicsimageio import AICSImage

        try:
            from aicsimageio.readers.bioformats_reader import BioformatsReader
        except ImportError:
            BioformatsReader = None
        _AICS = (AICSImage, BioformatsReader)
    return _AICS


def _microscopy_to_czyx(path: Path) -> np.ndarray:
    """Read a microscope file as in ramfish_processor.py (AICSImage, Bio-Formats for .oir/.lif/.vsi)."""
    return _microscopy_read(path)[0]


def _microscopy_open(path: Path):
    """The AICSImage object for a microscope file (nothing is read yet), or None if only tifffile can read it."""
    try:
        AICSImage, BioformatsReader = _aicsimageio()
    except ImportError as exc:
        if file_ext(path) in {".ome.tif", ".ome.tiff", ".qptiff"}:
            return None
        raise RuntimeError(
            f"Cannot read {path.name}: aicsimageio / scyjava are not installed. Install the environment "
            "described in the README (conda: bioformats_jar openjdk maven; pip: aicsimageio scyjava)."
        ) from exc
    ext = file_ext(path)
    if ext in ['.oir', '.lif', '.vsi'] and BioformatsReader is not None:
        return AICSImage(path, reader=BioformatsReader)
    return AICSImage(path)


def _microscopy_read(path: Path):
    """(CZYX array, AICSImage object or None) for a microscope file."""
    img = _microscopy_open(path)
    if img is None:
        return _tiff_to_czyx(str(path)), None
    return np.asarray(img.get_image_data("CZYX")), img


def _ome_xml_of(img) -> str | None:
    try:
        meta = img.ome_metadata
        return meta.to_xml() if hasattr(meta, "to_xml") else str(meta)
    except Exception:
        return None


def _pixel_um_from_xml(xml: str | None):
    """(Z, Y, X) pixel size in micrometers from OME-XML, None where it is missing."""
    if not xml:
        return None
    import re as _re
    m = _re.search(r"<(?:\w+:)?Pixels\b[^>]*>", xml)
    if not m:
        return None
    def val(k):
        v = _re.search(rf'\b{k}="([0-9.eE+-]+)"', m.group(0))
        try:
            return float(v.group(1)) if v and float(v.group(1)) > 0 else None
        except ValueError:
            return None
    out = (val("PhysicalSizeZ"), val("PhysicalSizeY"), val("PhysicalSizeX"))
    return out if any(out) else None


def open_with_info(path: str | Path):
    """Open an image file for reading one channel at a time (a large stack never has to fit in memory twice).

    Returns ``(shape, info, get_channel)``: ``shape`` is (C, Z, Y, X); ``info`` = {"channels": [ChannelMeta],
    "pixel_um": (z, y, x) or None}; ``get_channel(c)`` gives that channel as a (Z, Y, X) array on the 0-65535
    scale (16-bit data is returned as it is, without a copy in 32-bit numbers).
    Channel dyes come from the OME metadata and, for Olympus .oir files, from the dye names the microscope saved.
    """
    from .channels import ChannelMeta, channels_from_ome_xml, dye_key, oir_dye_names
    path = Path(path)
    ext = file_ext(path)
    xml, pixel_um, img, data = None, None, None, None
    if ext in PLAIN_IMAGE_EXTS:
        data = _plain_to_czyx(path)
    elif ext in TIFF_EXTS or ext in {".ome.tif", ".ome.tiff"}:
        data = _tiff_to_czyx(str(path))
        try:
            with tifffile.TiffFile(str(path)) as tf:
                xml = tf.ome_metadata
        except Exception:
            xml = None
    elif ext in MICROSCOPY_EXTS:
        img = _microscopy_open(path)
        if img is None:
            data = _tiff_to_czyx(str(path))
        else:
            xml = _ome_xml_of(img)
            try:
                p = img.physical_pixel_sizes
                pixel_um = tuple((float(v) if v and v > 0 else None) for v in (p.Z, p.Y, p.X))
            except Exception:
                pixel_um = None
            if np.dtype(img.dtype) not in (np.dtype(np.uint8), np.dtype(np.uint16)):
                data = np.asarray(img.get_image_data("CZYX"))   # other number types are scaled over the whole file
                img = None
    else:
        raise ValueError(f"Unsupported file type: {ext or path.name}")

    if data is not None:
        if data.ndim != 4:
            raise ValueError(f"Expected a CZYX stack, got shape {data.shape}.")
        data = data if data.dtype == np.uint16 else to_16bit_scale(data)
        shape = data.shape
        get_channel = lambda c: data[c]
    else:
        d = img.dims
        shape = (d.C, d.Z, d.Y, d.X)
        extra = {"T": 0} if "T" in d.order else {}

        def get_channel(c):
            arr = np.asarray(img.get_image_dask_data("ZYX", C=c, **extra).compute())
            return arr if arr.dtype == np.uint16 else to_16bit_scale(arr)

    channels = channels_from_ome_xml(xml)
    if len(channels) != shape[0]:
        channels = [ChannelMeta(i) for i in range(shape[0])]
    if ext == ".oir":
        try:
            dyes = oir_dye_names(path)
            for c in channels:
                key = dye_key(dyes.get(c.name) or dyes.get(f"CH{c.index + 1}"))
                if key:
                    c.dye = key
        except Exception:
            pass
    if pixel_um is None or not any(pixel_um):
        pixel_um = _pixel_um_from_xml(xml)
    return shape, {"channels": channels, "pixel_um": pixel_um}, get_channel


def load_stack_with_info(path: str | Path):
    """Like :func:`load_stack`, plus what the file says about its channels and pixel size (see :func:`open_with_info`)."""
    shape, info, get_channel = open_with_info(path)
    stack = np.stack([np.asarray(get_channel(c), dtype=np.float32) for c in range(shape[0])])
    return stack, info


def load_stack(path: str | Path) -> np.ndarray:
    """Load an image file as a float32 ``(C, Z, Y, X)`` array on a 0-65535 scale."""
    path = Path(path)
    ext = file_ext(path)
    if ext in PLAIN_IMAGE_EXTS:
        data = _plain_to_czyx(path)
    elif ext in TIFF_EXTS:
        data = _tiff_to_czyx(str(path))
    elif ext in MICROSCOPY_EXTS:
        data = _microscopy_to_czyx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext or path.name}")
    if data.ndim != 4:
        raise ValueError(f"Expected a CZYX stack, got shape {data.shape}.")
    return to_16bit_scale(data)


def project_stack(stack_czyx: np.ndarray, channel: int, z_mode: str = "mip", z_index: int = 0) -> np.ndarray:
    """Reduce one channel of a CZYX stack to a 2D plane."""
    vol = stack_czyx[channel]
    if vol.shape[0] == 1:
        return vol[0]
    if z_mode == "mip":
        return vol.max(axis=0)
    if z_mode == "mean":
        return vol.mean(axis=0)
    if z_mode == "slice":
        return vol[int(np.clip(z_index, 0, vol.shape[0] - 1))]
    raise ValueError(f"Unknown z_mode {z_mode!r}")


def read_plane(path: str | Path) -> np.ndarray:
    """Read a single-plane project image as float32 on the 0-65535 scale."""
    stack = load_stack(path)
    return stack[0, 0]


def write_plane_u16(path: str | Path, plane: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(np.rint(plane), 0, 65535).astype(np.uint16)
    tifffile.imwrite(str(path), out, compression="zlib")


def write_stack_u16(path: str | Path, stack_zyx: np.ndarray, pixel_um=None) -> None:
    """Save a (Z, Y, X) stack as a 16-bit TIFF that Fiji opens with the right pixel size and Z step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(stack_zyx)
    if data.dtype != np.uint16:
        data = np.clip(np.rint(data), 0, 65535).astype(np.uint16)
    if data.ndim == 2:
        data = data[np.newaxis]
    kwargs = {"imagej": True, "metadata": {"axes": "ZYX", "unit": "um"}, "compression": "zlib"}
    if pixel_um:
        z, y, x = (list(pixel_um) + [None, None, None])[:3]
        if x and y:
            kwargs["resolution"] = (1.0 / float(x), 1.0 / float(y))
        if z:
            kwargs["metadata"]["spacing"] = float(z)
    tifffile.imwrite(str(path), data, **kwargs)


def read_stack(path: str | Path) -> np.ndarray:
    """A TIFF written by :func:`write_stack_u16` as a float32 (Z, Y, X) array."""
    data = tifffile.imread(str(path)).astype(np.float32)
    return data[np.newaxis] if data.ndim == 2 else data


def to_display_u8(img: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.8) -> np.ndarray:
    """Percentile-stretch an image to uint8 for on-screen previews."""
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def downsample_to(img: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Shrink so the longest side is at most ``max_dim``; returns (image, factor)."""
    h, w = img.shape[:2]
    k = min(1.0, max_dim / float(max(h, w)))
    if k >= 1.0:
        return img, 1.0
    size = (max(1, int(round(w * k))), max(1, int(round(h * k))))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), k
