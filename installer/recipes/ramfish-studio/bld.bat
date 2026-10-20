set "DEST=%PREFIX%\share\ramfish-studio"
mkdir "%DEST%"
mkdir "%PREFIX%\Menu"
copy app.py "%DEST%\" || exit 1
copy ramfish_launcher.py "%DEST%\" || exit 1
copy ui_common.py "%DEST%\" || exit 1
copy ui_components.py "%DEST%\" || exit 1
for %%D in (app_pages assets components ramfish .streamlit) do (
  robocopy "%%D" "%DEST%\%%D" /E /XD __pycache__ /NFL /NDL /NJH /NJS
  if errorlevel 8 exit 1
)
copy installer\menu\ramfish-studio.json "%PREFIX%\Menu\" || exit 1
copy installer\icons\ramfish-studio.* "%PREFIX%\Menu\" || exit 1
exit 0
