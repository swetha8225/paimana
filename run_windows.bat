@echo off
cd /d "%~dp0"
setlocal

rem Prefer the private .venv created by make_venv_windows.bat, if present
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe

echo ============================================================
echo   PAIMANA - Pro-active Analytics for Infrastructure
echo   Monitoring and Assessment (National Analytics)
echo ============================================================
echo.

%PY% --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Install Python 3.11 or 3.12 from https://www.python.org/downloads/
    echo         and tick "Add Python to PATH" during setup, then run this again.
    pause
    exit /b 1
)

echo [1/4] Python runtime:
%PY% --version
echo.

echo [2/4] Checking that pandas imports correctly on this machine...
%PY% -c "import pandas" >nul 2>nul
if errorlevel 1 (
    echo   pandas fails to import. Your Python environment has a NumPy 1.x
    echo   binary clash - pandas was built for NumPy 1.x but NumPy 2.2.6 is
    echo   installed. Fixing it by upgrading pandas and pyarrow...
    %PY% -m pip install --upgrade "pandas>=2.2.3" pyarrow
    %PY% -c "import pandas" >nul 2>nul
    if errorlevel 1 (
        echo.
        echo [ERROR] Could not fix automatically in this environment.
        echo         Double-click  make_venv_windows.bat  to create a private
        echo         environment for PAIMANA, then run this file again.
        pause
        exit /b 1
    )
)
echo   OK
echo.

echo [3/4] Checking dependencies - first run installs them, needs internet...
%PY% -c "import fastapi, uvicorn, catboost, lifelines, shap" >nul 2>nul
if errorlevel 1 %PY% -m pip install -r backend\requirements.txt
echo.

echo [4/4] Starting PAIMANA...
cd backend

rem If the bundled models were pickled with different library versions,
rem retrain them from the bundled dataset - about 1-2 minutes.
%PY% -c "import joblib; joblib.load('artifacts/models.joblib')" >nul 2>nul
if errorlevel 1 (
    echo   Bundled model artifacts are not loadable with your installed
    echo   library versions. Retraining from the bundled dataset...
    %PY% -m app.ml.pipeline
)

echo.
echo   Browser:  http://localhost:8000
echo   API docs: http://localhost:8000/docs
echo   Keep this window open. Press Ctrl+C to stop.
echo.
%PY% -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
