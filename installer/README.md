# RAMFISH Studio installers

One-click installers that put RAMFISH Studio and everything it needs (Python, the image libraries, Java for
Bio-Formats) into a single folder. They do not need administrator rights and leave any other Python / conda on the
computer untouched.

| System | File | How to install |
|---|---|---|
| Windows 10/11 | `RAMFISH-Studio-<version>-Windows-x86_64.exe` | Double-click and follow the steps. |
| macOS (Apple Silicon / Intel) | `RAMFISH-Studio-<version>-MacOSX-arm64.pkg` / `-x86_64.pkg` | Double-click and follow the steps. |
| Linux | `RAMFISH-Studio-<version>-Linux-x86_64.sh` | `bash RAMFISH-Studio-<version>-Linux-x86_64.sh` |

After installation, start **RAMFISH Studio** from the desktop / Start menu (Windows), Applications (macOS) or the
application menu (Linux). A small window opens, then the app appears in the web browser. Everything runs on the
computer itself; nothing is uploaded. Results are saved in `Documents/RAMFISH Studio/Output/<project name>/`.
Close the small window to quit.

Notes for users:
- Reading `.oir`, `.lif`, `.vsi` and other proprietary formats needs internet the first time (Bio-Formats is
  downloaded once and then cached). TIFF, OME-TIFF, PNG and JPEG work offline.
- The installers are not code-signed yet. On Windows click **More info → Run anyway**; on macOS, if it is blocked,
  right-click the `.pkg` → **Open** (or System Settings → Privacy & Security → **Open Anyway**).
- To uninstall: Windows → Settings → Apps → *RAMFISH Studio* → Uninstall. macOS / Linux → delete the
  `RAMFISH-Studio` folder (in your home folder) and the shortcut. Your `Output` folder is kept.

## Building the installers

Windows and Mac installers can only be built on Windows and Mac computers, so GitHub builds them for you
(free for public repositories), using `.github/workflows/installers.yml`:

1. Put the contents of this folder (`app.py`, `installer/`, `.github/`, ...) at the top level of your GitHub
   repository (e.g. `RAMFISH_Studio`). The hidden `.github` folder must be included. (Uploading through the
   GitHub website can skip it: then use **Add file → Create new file**, name it
   `.github/workflows/installers.yml` and paste in that file's contents.)
2. On GitHub, open the repository → **Actions** tab. If asked, click *I understand my workflows, enable them*.
3. Click **installers** (left) → **Run workflow** → **Run workflow**. It takes about 30 minutes.
4. Open the finished run (green tick) and scroll to **Artifacts**: one download per system (Windows,
   macOS Apple Silicon, macOS Intel, Linux). Each download is a .zip holding the installer.
5. To publish them for everyone: **Releases → Draft a new release**, choose a new tag `v2.0.0`, **Publish**.
   This runs the workflow again and, about 30 minutes later, the four installers appear on that release page,
   where users download them.

If a build turns red, open it, copy the error at the bottom of the red step and send it to whoever maintains
the installers.

When releasing a new version, update the version number in `construct.yaml` and
`recipes/ramfish-studio/meta.yaml`.

What is in this folder:
- `recipes/ramfish-studio/`: conda package with the app and its shortcut (`menu/ramfish-studio.json`).
- `recipes/aicsimageio/`: aicsimageio 4.14.0 exactly as on PyPI, with the outdated `lxml<5` cap lifted so it
  installs next to Python 3.12.
- `construct.yaml`: the installer definition (built with conda
  [constructor](https://github.com/conda/constructor)).
- `icons/`: app icon (`.png` Linux, `.ico` Windows, `.icns` macOS).

To build locally (for the system you are on), with conda/miniforge:

```bash
conda create -n builder -c conda-forge conda-build constructor menuinst conda-standalone
conda activate builder
conda build --output-folder ./channel -c conda-forge installer/recipes/aicsimageio
conda build --output-folder ./channel -c ./channel -c conda-forge installer/recipes/ramfish-studio
RAMFISH_LOCAL_CHANNEL=file://$PWD/channel constructor installer --output-dir dist
```
