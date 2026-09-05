@echo off
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Install Python 3.11 or 3.12 from https://www.python.org/downloads/
    echo         and tick "Add Python to PATH" during setup, then run this again.
    pause
    exit /b 1
)

echo Creating a private Python environment for PAIMANA...
echo This does NOT touch your other Python installations or projects.
echo.
python -m venv .venv
if not exist .venv\Scripts\python.exe (
    echo [ERROR] Environment creation failed.
    pause
    exit /b 1
)

echo Installing dependencies - needs internet, may take a few minutes...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

echo.
echo ============================================================
echo  Done. Now double-click  run_windows.bat
echo  It will automatically use this private environment.
echo ============================================================
pause
