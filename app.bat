@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "VENV_DIR=.venv"
set "PY=%VENV_DIR%\Scripts\python.exe"

if exist "%PY%" (
    echo [app.bat] Virtual environment found at %VENV_DIR%, skipping creation and dependency check.
) else (
    echo [app.bat] Creating virtual environment at %VENV_DIR% ...
    where python >nul 2>nul
    if errorlevel 1 (
        echo [app.bat] ERROR: python was not found on PATH. Install Python 3.10+ and try again.
        exit /b 1
    )
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [app.bat] ERROR: failed to create the virtual environment.
        exit /b 1
    )

    echo [app.bat] Installing dependencies from requirements.txt ...
    "%PY%" -m pip install --upgrade pip >nul
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [app.bat] ERROR: dependency installation failed.
        exit /b 1
    )
)

set "MODELS_MISSING=0"
if not exist "outputs\electricity_model.joblib" set "MODELS_MISSING=1"
if not exist "outputs\hydrogen_model.joblib" set "MODELS_MISSING=1"

if "%MODELS_MISSING%"=="1" (
    echo [app.bat] Trained models missing, training from committed sample data ...
    echo [app.bat] This can take ~45-50 minutes ^(electricity ~40 min, hydrogen ~7 min^). Please wait ...
    "%PY%" train_model.py
    if errorlevel 1 (
        echo [app.bat] ERROR: training failed.
        exit /b 1
    )
    echo [app.bat] Training complete.
) else (
    echo [app.bat] Trained models found in outputs\, skipping training.
)

echo [app.bat] Starting Django app at http://127.0.0.1:8000/ (also reachable on your LAN/Tailscale IP) ...
start "" "http://127.0.0.1:8000/"
"%PY%" webui\manage.py runserver 0.0.0.0:8000

endlocal
