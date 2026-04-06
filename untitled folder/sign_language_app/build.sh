#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build.sh  —  macOS build script for SLRS
#
# Usage:
#   cd sign_language_app
#   chmod +x build.sh
#   ./build.sh
#
# Output:
#   dist/SLRS.app   ← double-click to launch
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SLRS — Build Script (macOS)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Python check ────────────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON" ]]; then
    echo "[ERROR] Python not found. Install Python 3.9+ and try again."
    exit 1
fi
echo "[✓] Python: $($PYTHON --version)"

# ── 2. Install / upgrade pip dependencies ─────────────────────────────────────
echo ""
echo "[→] Installing requirements..."
$PYTHON -m pip install --quiet --upgrade pip
$PYTHON -m pip install --quiet -r requirements.txt

# ── 3. Install PyInstaller if missing ─────────────────────────────────────────
if ! $PYTHON -m PyInstaller --version &>/dev/null; then
    echo "[→] Installing PyInstaller..."
    $PYTHON -m pip install --quiet pyinstaller pyinstaller-hooks-contrib
else
    echo "[✓] PyInstaller: $($PYTHON -m PyInstaller --version)"
fi

# ── 4. Optional: generate icon if assets/SLRS.icns is missing ─────────────────
if [[ ! -f "assets/SLRS.icns" ]]; then
    echo ""
    echo "[i] No icon found at assets/SLRS.icns — building without icon."
    echo "    Run:  python assets/create_icon.py  to generate one."
fi

# ── 5. Clean previous build artefacts ─────────────────────────────────────────
echo ""
echo "[→] Cleaning previous build..."
rm -rf build dist __pycache__

# ── 6. Build ───────────────────────────────────────────────────────────────────
echo ""
echo "[→] Running PyInstaller..."
$PYTHON -m PyInstaller SLRS.spec --noconfirm

# ── 7. Done ────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -d "dist/SLRS.app" ]]; then
    echo "  Build SUCCEEDED"
    echo ""
    echo "  App location : $(pwd)/dist/SLRS.app"
    echo ""
    echo "  To run       : open dist/SLRS.app"
    echo "               or double-click it in Finder"
    echo ""
    echo "  NOTE: On first launch macOS will ask for camera permission."
    echo "        Grant it — the app cannot function without it."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "  Build FAILED — check the output above for errors."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
fi
