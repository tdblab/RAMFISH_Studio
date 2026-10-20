# 🔬 RAMFISH Studio

<img width="2816" height="959" alt="ramfish" src="https://github.com/user-attachments/assets/eb67a12b-e8a6-4a91-b31f-ac40a24b4625" />

The five tools of the **RAMFISH Software Suite** in one browser app:

```text
1. Import & normalize → 2. Rough alignment → 3. Fine alignment → 4. Spot counter → 5. Multiplex merge
```

Each step is the corresponding RAMFISH script (`ramfish_processor.py`, `ramfish_rough_aligner.py`,
`ramfish_fine_aligner.py`, `ramfish_spotcounter.py`, `ramfish_merger.html`), with the same processing,
parameters and interface. The app passes the images from one step to the next, so files no longer have to be
moved between `normalized_data/`, `Round_N/`, `aligned_outputs/` and `rawdata/` by hand.

## Installation

### One-click installer (recommended)

Download the installer for your computer from the **Releases** page and double-click it:

| Windows | macOS | Linux |
|---|---|---|
| `RAMFISH-Studio-…-Windows-x86_64.exe` | `RAMFISH-Studio-…-MacOSX-arm64.pkg` (Apple Silicon) or `-x86_64.pkg` (Intel) | `bash RAMFISH-Studio-…-Linux-x86_64.sh` |

Then start **RAMFISH Studio** from the desktop / Start menu, Applications or the app menu. It opens in your web
browser and runs entirely on your computer; results are saved in `Documents/RAMFISH Studio/Output/`. No Python or
conda knowledge needed. See [installer/README.md](installer/README.md) for details (and how the installers are
built).

### Manual installation (conda)

Install Miniconda: https://docs.conda.io/en/latest/, then in the Anaconda Prompt (Windows) or a terminal:

```bash
# 1. Create and activate the conda environment
conda create -n ramfish python=3.12 -y
conda activate ramfish

# 2. Install Java and Bio-Formats bridge libraries (required for .oir, .czi, .lif, .nd2, etc.)
conda install -c conda-forge bioformats_jar openjdk maven -y

# 3. Install RAMFISH Studio's dependencies (from this folder)
pip install -r requirements.txt

# 4. Optional: GPU acceleration (NVIDIA CUDA 12.x; use cupy-cuda11x for CUDA 11.x)
pip install "cupy-cuda12x[ctk]"
```

## Run

```bash
conda activate ramfish
cd path/to/RAMFISH_Studio
streamlit run app.py
```

The app opens in your browser. On the first page, type a **project name**; everything is saved in
`Output/<project name>/` next to the app (pick the name again later to continue the project):

```text
Output/<project name>/
├── 1_normalized/      Round_1/DAPI.tif, wg.tif …   images used by the next steps
│                      exports/Round_1/NORM_…       exported copies (chosen format, Z-slices, MIP)
├── 2_rough_aligned/   Round_2/ALIGNED_DAPI.tif …
├── 3_fine_aligned/    R2_aligned_DAPI.tif …
├── 4_spots/           data_sheets/  spot_images/  synthetic_spots/
└── 5_merge/           multiplex_composite.jpg
```

Round 1 is the reference that all other rounds are aligned to. Each page starts with a short
**How to use this page** guide.

## Steps

1. **Import & normalize.** Upload microscope files for one round (`.oir`, `.czi`, `.lif`, `.nd2`, `.tif`, …), or a
   `.zip` with one folder per round (`Round_1/`, `Round_2/`, …). Z-stacks get a maximum-intensity projection.
   Background subtraction + percentile normalization (GPU if available). Each round needs an image named `DAPI` or `mask`; the other image names are the gene names.
2. **Rough alignment.** The rough-aligner interface: Round 1 is fixed, choose the round to align, then adjust
   opacity, scale, rotation, X/Y translation and horizontal flip (scroll to zoom, drag to pan). **Apply Rough
   Alignment to Round** transforms every image of that round.
3. **Fine alignment.** Affine + non-rigid B-Spline registration (SimpleITK) of each round's `DAPI` — or its `mask`
   if the round has no DAPI (affine only by default) — onto Round 1, applied to every image of the round.
4. **Spot counter.** LoG blob detection with the `config.json` settings (`blob_min_sigma`, `blob_max_sigma`,
   `blob_num_sigma`, `blob_threshold`, `intensity_cutoff`, `synthetic_spot_radius`), each explained on the page.
5. **Multiplex merge.** The RAM-FISH Merger, preloaded with the synthetic spots (or the aligned images) and the
   DAPI backdrop. Generate the composite and download it (also saved in `5_merge/`).

## Hardware

| | Recommended | High performance |
|---|---|---|
| CPU | AMD Ryzen 5 5600X / Intel Core i5-12400F | AMD Ryzen Threadripper 9965WX |
| GPU | NVIDIA RTX 3060 | NVIDIA RTX 5080 / 5090 |
| RAM | 32 GB DDR4 | 128 GB DDR5 |

Tested on Windows 11 and Ubuntu 24.04.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## License

MIT License, © 2026 Tirtha Das Banerjee. See [LICENSE](LICENSE).
