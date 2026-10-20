"""Start RAMFISH Studio on this computer and open it in the web browser.

This is what the desktop / Start-menu shortcut runs. Everything is computed
locally; the browser is only the window the app is shown in.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def documents_folder() -> Path:
    """The user's Documents folder (falls back to the home folder)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            # CSIDL_PERSONAL = 5 ("My Documents"), also when redirected to OneDrive.
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0 and buf.value:
                return Path(buf.value)
        except Exception:
            pass
    docs = Path.home() / "Documents"
    return docs if docs.is_dir() else Path.home()


def free_port(start: int = 8501) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free network port found between 8501 and 8600.")


def setup_environment() -> None:
    """What `conda activate` would do for the installed app (the installer ships no `conda` command)."""
    prefix = Path(sys.prefix)
    os.environ["CONDA_PREFIX"] = str(prefix)
    if sys.platform == "win32":
        dirs = [prefix, prefix / "Library" / "mingw-w64" / "bin", prefix / "Library" / "usr" / "bin",
                prefix / "Library" / "bin", prefix / "Scripts", prefix / "bin"]
    else:
        dirs = [prefix / "bin"]
    os.environ["PATH"] = os.pathsep.join([str(d) for d in dirs] + [os.environ.get("PATH", "")])
    # Java and Maven for Bio-Formats (.oir, .czi, .lif, .nd2, ...), installed together with the app.
    java = prefix / "Library" if sys.platform == "win32" else prefix / "lib" / "jvm"
    if java.exists():
        os.environ["JAVA_HOME"] = str(java)
    os.environ.setdefault("JGO_CACHE_DIR", str(prefix / "jgo"))
    output = Path(os.environ.get("RAMFISH_OUTPUT_DIR", documents_folder() / "RAMFISH Studio" / "Output"))
    output.mkdir(parents=True, exist_ok=True)
    os.environ["RAMFISH_OUTPUT_DIR"] = str(output)


def wait_until_ready(url: str, timeout: float = 120.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url + "/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def running_instance(lock: Path) -> str | None:
    """URL of a RAMFISH Studio that is already running (started by an earlier double-click)."""
    try:
        url = lock.read_text().strip()
        with urllib.request.urlopen(url + "/_stcore/health", timeout=2) as r:
            return url if r.status == 200 else None
    except Exception:
        return None


def main() -> int:
    setup_environment()
    lock = Path(os.environ["RAMFISH_OUTPUT_DIR"]).parent / ".running"
    url = running_instance(lock)
    if url:
        print(f"RAMFISH Studio is already running: {url}  (opening it in your browser)", flush=True)
        webbrowser.open(url)
        time.sleep(3)
        return 0
    port = free_port()
    url = f"http://localhost:{port}"
    line = "=" * 64
    print(line)
    print("  RAMFISH Studio")
    print(line)
    print(f"  Starting...  The app will open in your web browser at {url}")
    print(f"  Results are saved in: {os.environ['RAMFISH_OUTPUT_DIR']}")
    print()
    print("  Keep this window open while you work.")
    print("  To quit RAMFISH Studio, close this window.")
    print(line, flush=True)

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP_DIR / "app.py"),
        "--server.port", str(port),
        "--server.address", "localhost",
        "--server.headless", "true",
        "--server.maxUploadSize", "16384",
        "--browser.gatherUsageStats", "false",
        "--client.toolbarMode", "viewer",
        "--global.developmentMode", "false",
    ]
    proc = subprocess.Popen(cmd, cwd=APP_DIR)
    if wait_until_ready(url):
        try:
            lock.write_text(url)
        except OSError:
            pass
        webbrowser.open(url)
        print(f"\n  RAMFISH Studio is running: {url}\n  (If the browser did not open, copy this address into it.)\n",
              flush=True)
    else:
        print("\n  RAMFISH Studio did not start. Please send the messages above to the RAMFISH team.\n", flush=True)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0
    finally:
        try:
            if lock.read_text().strip() == url:
                lock.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
