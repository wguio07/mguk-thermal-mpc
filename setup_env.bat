@echo off
REM ═══════════════════════════════════════════════════════════
REM  MGU-K Thermal MPC — Environment Setup (Windows)
REM  Run this once from the mguk_thermal_mpc\ directory
REM ═══════════════════════════════════════════════════════════

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

echo [4/4] Installing dependencies...
pip install -r requirements.txt

echo.
echo ═══════════════════════════════════════════════════════════
echo  Setup complete! To activate the environment in future:
echo    venv\Scripts\activate
echo.
echo  To run the project:
echo    python src\track_model.py
echo ═══════════════════════════════════════════════════════════
pause
