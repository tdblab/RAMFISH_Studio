#!/bin/bash
set -euo pipefail
dest="$PREFIX/share/ramfish-studio"
mkdir -p "$dest" "$PREFIX/Menu"
cp -R app.py ramfish_launcher.py ui_common.py ui_components.py app_pages assets components ramfish .streamlit "$dest/"
find "$dest" -name "__pycache__" -type d -prune -exec rm -rf {} +
cp installer/menu/ramfish-studio.json "$PREFIX/Menu/"
cp installer/icons/ramfish-studio.png installer/icons/ramfish-studio.ico installer/icons/ramfish-studio.icns "$PREFIX/Menu/"
