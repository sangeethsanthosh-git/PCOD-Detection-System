# Build Desktop Executable

This project can be packaged into a single Windows desktop executable that starts the local Django server internally and opens the PCOS AI platform inside a native `pywebview` window.

## 1. Install desktop runtime dependencies

From `C:\pcod\pcos_project`:

```powershell
C:\pcod\venv\Scripts\python.exe -m pip install pywebview pyinstaller
```

## 2. Build the desktop executable

Run this from `C:\pcod\pcos_project`:

```powershell
C:\pcod\venv\Scripts\pyinstaller.exe --noconfirm --clean launch_app.spec
```

Notes:
- `..\assets\favicon.ico` is embedded into the packaged executable icon.
- Replace `assets\favicon.ico` if you want to swap in a different desktop icon later.
- `templates`, `static`, `models`, and `db.sqlite3` are bundled into the executable.
- `dataset;dataset` includes the in-project dataset folder used by the analytics and default feature services.
- The packaged runtime now validates the bundled dataset schema and ignores incompatible CSVs such as `dataset\data3.csv` when selecting the clinical reference dataset.
- `--collect-submodules webview` and `--collect-data webview` are important because `pywebview` loads platform backends dynamically.
- `--collect-all xgboost` is required because the bundled clinical model depends on `xgboost.dll` at unpickle time.
- The launcher uses `8000` by default, falls back to `8001`, and continues scanning local ports if both are already busy.
- The launcher copies `db.sqlite3` into `%LOCALAPPDATA%\PCOSAIClinicalSupportPlatform\db.sqlite3` so the database survives across runs.
- Prediction failures are logged to `%LOCALAPPDATA%\PCOSAIClinicalSupportPlatform\logs\prediction_errors.log`.

## 3. Clean old packaging artifacts

Before rebuilding, remove old packaging outputs from the project root:

```powershell
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
```

## 4. Output

After the build completes, the executable will be created at:

```text
dist\pcod.exe
```

## 5. Runtime behavior

When the user double-clicks the executable:

1. The local Django server starts in the background.
2. The launcher checks for an available local port.
3. A native desktop window opens with title `PCOS AI Clinical Support Platform`.
4. The full platform loads inside that window, without opening an external browser.

## 6. Desktop window settings

The launcher creates a desktop window with:

- Title: `PCOS AI Clinical Support Platform`
- Width: `1200`
- Height: `800`
- Resizable: `True`

## 7. Smoke test

After building:

1. Open `dist`.
2. Double-click `pcod.exe`.
3. Confirm a native app window opens.
4. Verify:
   - prediction works
   - result gauge renders
   - analysis charts load
   - doctor finder works
   - education videos render
   - search assistant responds

## 8. Troubleshooting

- If the app starts but the window is blank, rebuild with the same `--collect-submodules webview --collect-data webview` flags.
- If the desktop window fails to initialize, the launcher shows a Windows error dialog with the local URL and error message.
- If analysis charts fail, rebuild and confirm the dataset folder was included with `--add-data "..\dataset;dataset"`.
- If the database does not retain changes, confirm the app can write to `%LOCALAPPDATA%\PCOSAIClinicalSupportPlatform`.
