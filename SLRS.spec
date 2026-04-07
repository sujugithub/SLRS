# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for SLRS — Sign Language Recognition System
#
# Build with:
#   pyinstaller SLRS.spec
#
# Output: dist/SLRS.app  (macOS)  |  dist/SLRS/SLRS.exe  (Windows)

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

APP_NAME = "SLRS"
APP_DIR  = Path(SPECPATH)   # directory that contains this .spec file

# ── Collect third-party packages that carry binary/data files ──────────────────
mp_datas,  mp_binaries,  mp_hidden  = collect_all("mediapipe")
cv2_datas, cv2_binaries, cv2_hidden = collect_all("cv2")

# ── Icon (optional — omit gracefully if not present) ──────────────────────────
_icns = str(APP_DIR / "assets" / "SLRS.icns")
_ico  = str(APP_DIR / "assets" / "SLRS.ico")
_icon = _icns if sys.platform == "darwin" and os.path.isfile(_icns) else \
        _ico  if sys.platform == "win32"  and os.path.isfile(_ico)  else None

# ──────────────────────────────────────────────────────────────────────────────

a = Analysis(
    [str(APP_DIR / "main.py")],
    pathex=[str(APP_DIR)],

    # ── Binary shared libraries ──────────────────────────────────────────────
    binaries=mp_binaries + cv2_binaries,

    # ── Data files bundled into the executable ───────────────────────────────
    datas=(
        mp_datas
        + cv2_datas
        + [
            # MediaPipe .task model files (read-only)
            (str(APP_DIR / "models"), "models"),
            # Training data: pretrained + user-created signs (seeded on first launch)
            (str(APP_DIR / "data"),   "data"),
        ]
    ),

    # ── Hidden imports not discovered automatically ──────────────────────────
    hiddenimports=(
        mp_hidden
        + cv2_hidden
        + [
            # PyQt6
            "PyQt6",
            "PyQt6.QtCore",
            "PyQt6.QtGui",
            "PyQt6.QtWidgets",
            # scikit-learn (PyInstaller hook often misses these)
            "sklearn",
            "sklearn.ensemble",
            "sklearn.ensemble._forest",
            "sklearn.tree",
            "sklearn.tree._classes",
            "sklearn.utils",
            "sklearn.utils._typedefs",
            "sklearn.utils._heap",
            "sklearn.utils._sorting",
            "sklearn.utils._vector_sentinel",
            "sklearn.utils._cython_blas",
            "sklearn.neighbors._partition_nodes",
            "sklearn.neural_network._multilayer_perceptron",
            "sklearn.neural_network._base",
            "sklearn.preprocessing._encoders",
            "sklearn.pipeline",
            # joblib
            "joblib",
            "joblib.externals.loky",
            "joblib.externals.loky.backend.contexts",
            # numpy / pillow
            "numpy",
            "PIL",
            "PIL.Image",
            # pyttsx3 — all platform drivers bundled; only the right one is used at runtime
            "pyttsx3",
            "pyttsx3.drivers",
            "pyttsx3.drivers.nsss",    # macOS (NSSpeechSynthesizer)
            "pyttsx3.drivers.sapi5",   # Windows
            "pyttsx3.drivers.espeak",  # Linux
        ]
    ),

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    # Exclude heavy optional packages not used in this app
    excludes=[
        "tkinter", "tk", "_tkinter",
        "matplotlib", "IPython", "jupyter",
        "transformers", "torch",   # optional T5 grammar (not bundled)
    ],

    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,           # UPX can break some native libs; keep off for stability
    console=False,       # windowed mode — no terminal window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    **({"icon": _icon} if _icon else {}),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

# ── macOS .app bundle ──────────────────────────────────────────────────────────
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=_icon,
        bundle_identifier="com.slrs.app",
        info_plist={
            # Camera access — required for macOS Transparency, Consent & Control
            "NSCameraUsageDescription":
                "SLRS needs camera access to detect hand gestures in real time.",
            "NSMicrophoneUsageDescription":
                "SLRS does not use the microphone.",
            # App metadata
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": "SLRS",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1",
            # macOS display
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "NSAppleScriptEnabled": False,
            "LSMinimumSystemVersion": "10.15",
            # Hide from Dock Expose while loading (optional cosmetic)
            "LSUIElement": False,
        },
    )
