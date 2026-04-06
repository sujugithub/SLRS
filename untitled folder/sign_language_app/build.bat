@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM build.bat  —  Windows build script for SLRS
REM
REM Usage:
REM   cd sign_language_app
REM   build.bat
REM
REM Output:
REM   dist\SLRS\SLRS.exe   (folder-based build)
REM ─────────────────────────────────────────────────────────────────────────────

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ==================================================
echo   SLRS -- Build Script (Windows)
echo ==================================================

REM ── 1. Python check ──────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo [OK] %%v

REM ── 2. Install requirements ───────────────────────────────────────────────────
echo.
echo [->] Installing requirements...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

REM ── 3. Install PyInstaller if missing ─────────────────────────────────────────
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo [->] Installing PyInstaller...
    python -m pip install --quiet pyinstaller pyinstaller-hooks-contrib
) else (
    for /f "tokens=*" %%v in ('python -m PyInstaller --version') do echo [OK] PyInstaller %%v
)

REM ── 4. Clean previous build ───────────────────────────────────────────────────
echo.
echo [->] Cleaning previous build...
if exist build   rmdir /s /q build
if exist dist    rmdir /s /q dist

REM ── 5. Build ──────────────────────────────────────────────────────────────────
echo.
echo [->] Running PyInstaller...
python -m PyInstaller SLRS.spec --noconfirm

REM ── 6. Done ───────────────────────────────────────────────────────────────────
echo.
echo ==================================================
if exist "dist\SLRS\SLRS.exe" (
    echo   Build SUCCEEDED
    echo.
    echo   App location : %cd%\dist\SLRS\SLRS.exe
    echo.
    echo   To run: double-click dist\SLRS\SLRS.exe
    echo         or run it from Explorer
    echo.
    echo   NOTE: On first launch Windows may show a SmartScreen
    echo         warning. Click "More info" then "Run anyway".
    echo         Grant camera access when prompted.
) else (
    echo   Build FAILED -- check the output above for errors.
    pause
    exit /b 1
)
echo ==================================================
pause
