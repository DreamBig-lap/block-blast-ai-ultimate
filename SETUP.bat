@echo off
setlocal EnableDelayedExpansion
title Block Blast AI Ultimate - First-Time Setup

echo ==========================================================================
echo   BLOCK BLAST AI ULTIMATE - FIRST-TIME ENVIRONMENT SETUP
echo ==========================================================================
echo.
echo This script will configure your Python environment and dependencies
echo so you can run Block Blast AI with one click.
echo.

:: -------------------------------------------------------------------------
:: 1. Check Python
:: -------------------------------------------------------------------------
echo [1/4] Checking Python installation...

set "SYS_PYTHON="
where python >nul 2>&1
if %errorlevel% equ 0 (
    set "SYS_PYTHON=python"
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "SYS_PYTHON=py -3"
    )
)

if "%SYS_PYTHON%"=="" (
    echo.
    echo [ERROR] Python is not installed or not found in system PATH!
    echo.
    echo Please install Python 3.9 or higher:
    echo   1. Download Python from: https://www.python.org/downloads/
    echo   2. Run installer and CHECK "Add Python to PATH" on the first screen.
    echo   3. Restart this setup script after installation.
    echo.
    pause
    exit /b 1
)

%SYS_PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python version is below 3.9! Block Blast AI requires Python 3.9+.
    echo Please upgrade Python from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%V in ('%SYS_PYTHON% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do (
    echo       Python %%V detected. [OK]
)

:: -------------------------------------------------------------------------
:: 2. Check ADB (Android Debug Bridge)
:: -------------------------------------------------------------------------
echo.
echo [2/4] Checking ADB (Android Debug Bridge)...

set "ADB_EXE="
set "FOUND_ADB=0"

where adb >nul 2>&1
if %errorlevel% equ 0 (
    set "ADB_EXE=adb"
    set "FOUND_ADB=1"
    echo       ADB found in system PATH. [OK]
)

if %FOUND_ADB% equ 0 if exist "C:\platform-tools\adb.exe" (
    set "ADB_EXE=C:\platform-tools\adb.exe"
    set "PATH=C:\platform-tools;!PATH!"
    set "FOUND_ADB=1"
    echo       ADB found at C:\platform-tools\adb.exe. [OK]
)

if %FOUND_ADB% equ 0 if exist "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" (
    set "ADB_EXE=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"
    set "PATH=%LOCALAPPDATA%\Android\Sdk\platform-tools;!PATH!"
    set "FOUND_ADB=1"
    echo       ADB found in Android SDK platform-tools. [OK]
)

if %FOUND_ADB% equ 0 (
    echo       [NOTICE] ADB not found on this computer.
    echo       To play on a physical Android phone, download Google Platform-Tools:
    echo         https://developer.android.com/tools/releases/platform-tools
    echo       Extract the ZIP to C:\platform-tools
    echo       Note: You can still run the Offline Simulator without ADB.
)

:: -------------------------------------------------------------------------
:: 3. Create or verify Python Virtual Environment (.venv)
:: -------------------------------------------------------------------------
echo.
echo [3/4] Setting up Python virtual environment (.venv)...

cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo       Creating virtual environment in .venv...
    %SYS_PYTHON% -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo       Virtual environment created successfully. [OK]
) else (
    echo       Existing virtual environment found in .venv. [OK]
)

:: -------------------------------------------------------------------------
:: 4. Install dependencies into .venv
:: -------------------------------------------------------------------------
echo.
echo [4/4] Installing required dependencies (NumPy, OpenCV)...

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "VENV_PIP=%~dp0.venv\Scripts\pip.exe"

"%VENV_PYTHON%" -m pip install --upgrade pip >nul 2>&1
"%VENV_PIP%" install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies from requirements.txt!
    echo Please verify your internet connection and try again.
    echo.
    pause
    exit /b 1
)

echo       Installing package in editable mode...
"%VENV_PIP%" install -e . >nul 2>&1

:: Verify imports
"%VENV_PYTHON%" -c "import numpy, cv2; print(f'       Verified: NumPy {numpy.__version__}, OpenCV {cv2.__version__} [OK]')"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dependency verification failed!
    pause
    exit /b 1
)

echo.
echo ==========================================================================
echo   SETUP COMPLETED SUCCESSFULLY!
echo ==========================================================================
echo   You are now ready to launch Block Blast AI.
echo   Simply double-click START.bat to begin!
echo ==========================================================================
echo.
pause
