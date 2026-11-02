@echo off
setlocal EnableDelayedExpansion
title Block Blast AI Ultimate - Launcher

cd /d "%~dp0"

echo ==========================================================================
echo   BLOCK BLAST AI ULTIMATE - AUTONOMOUS BOT LAUNCHER
echo ==========================================================================
echo.

:: -------------------------------------------------------------------------
:: 1. Detect Python
:: -------------------------------------------------------------------------
set "PYTHON_EXE="

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    echo [OK] Using virtual environment Python: .venv\Scripts\python.exe
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python"
        echo [OK] Using system Python
    ) else (
        where py >nul 2>&1
        if %errorlevel% equ 0 (
            set "PYTHON_EXE=py -3"
            echo [OK] Using Python Launcher (py -3)
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo.
    echo ==========================================================================
    echo   [ERROR] PYTHON NOT FOUND!
    echo ==========================================================================
    echo   Python 3.9 or higher is required to run Block Blast AI.
    echo.
    echo   HOW TO FIX:
    echo   1. Download Python from: https://www.python.org/downloads/
    echo   2. Run installer and CHECK "Add Python to PATH" on the first screen.
    echo   3. After installing, double-click SETUP.bat first!
    echo ==========================================================================
    echo.
    pause
    exit /b 1
)

:: Verify Python version >= 3.9
"%PYTHON_EXE%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ==========================================================================
    echo   [ERROR] PYTHON VERSION TOO OLD!
    echo ==========================================================================
    echo   Block Blast AI requires Python 3.9 or higher.
    echo   Please upgrade Python from: https://www.python.org/downloads/
    echo ==========================================================================
    echo.
    pause
    exit /b 1
)

:: -------------------------------------------------------------------------
:: 2. Check Python Dependencies
:: -------------------------------------------------------------------------
"%PYTHON_EXE%" -c "import numpy, cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ==========================================================================
    echo   [NOTICE] REQUIRED PACKAGES MISSING!
    echo ==========================================================================
    echo   NumPy and/or OpenCV are not installed in the active environment.
    echo.
    echo   Please run SETUP.bat to install all required dependencies automatically.
    echo ==========================================================================
    echo.
    set /p RUN_SETUP="Would you like to run SETUP.bat now? (Y/N): "
    if /i "!RUN_SETUP!"=="Y" (
        call SETUP.bat
        goto restart_launcher
    )
    pause
    exit /b 1
)

:restart_launcher

:: -------------------------------------------------------------------------
:: 3. Detect ADB (Android Debug Bridge)
:: -------------------------------------------------------------------------
set "ADB_EXE="
set "FOUND_ADB=0"

where adb >nul 2>&1
if %errorlevel% equ 0 (
    set "ADB_EXE=adb"
    set "FOUND_ADB=1"
)

if %FOUND_ADB% equ 0 if exist "C:\platform-tools\adb.exe" (
    set "ADB_EXE=C:\platform-tools\adb.exe"
    set "PATH=C:\platform-tools;!PATH!"
    set "FOUND_ADB=1"
)

if %FOUND_ADB% equ 0 if exist "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" (
    set "ADB_EXE=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"
    set "PATH=%LOCALAPPDATA%\Android\Sdk\platform-tools;!PATH!"
    set "FOUND_ADB=1"
)

if %FOUND_ADB% equ 1 (
    echo [OK] ADB Bridge detected: !ADB_EXE!
) else (
    echo [WARN] ADB Bridge NOT found in system PATH or standard folders.
    echo        Live play requires ADB. Offline Simulator is still available.
)

:: -------------------------------------------------------------------------
:: 4. Check Connected Android Devices
:: -------------------------------------------------------------------------
set "DEVICE_COUNT=0"
set "READY_COUNT=0"
set "UNAUTH_COUNT=0"
set "OFFLINE_COUNT=0"
set "FIRST_DEVICE="

if %FOUND_ADB% equ 1 (
    for /f "skip=1 tokens=1,2" %%A in ('adb devices 2^>nul') do (
        if not "%%A"=="" (
            set /a DEVICE_COUNT+=1
            if "%%B"=="device" (
                set /a READY_COUNT+=1
                if "!FIRST_DEVICE!"=="" set "FIRST_DEVICE=%%A"
                echo [OK] Connected Device: %%A [READY]
            ) else if "%%B"=="unauthorized" (
                set /a UNAUTH_COUNT+=1
                echo [WARN] Connected Device: %%A [UNAUTHORIZED]
            ) else if "%%B"=="offline" (
                set /a OFFLINE_COUNT+=1
                echo [WARN] Connected Device: %%A [OFFLINE]
            )
        )
    )
)

if %DEVICE_COUNT% equ 0 (
    if %FOUND_ADB% equ 1 (
        echo [INFO] No Android device currently detected via USB.
    )
)

if %UNAUTH_COUNT% gtr 0 (
    echo.
    echo --------------------------------------------------------------------------
    echo   [ACTION REQUIRED: AUTHORIZE USB DEBUGGING]
    echo   Your phone is plugged in but USB debugging is not yet authorized!
    echo   1. Unlock your phone screen.
    echo   2. Look for the popup: "Allow USB debugging?".
    echo   3. Check "Always allow from this computer" and tap "Allow".
    echo --------------------------------------------------------------------------
)

:: -------------------------------------------------------------------------
:: 5. Interactive Menu
:: -------------------------------------------------------------------------
echo.
echo ==========================================================================
echo   SELECT LAUNCH MODE:
echo ==========================================================================
echo   [1] Start Autonomous Bot (Live Play on Phone)
echo   [2] Start in Dry-Run Mode (Safe Vision Test - No Screen Swipes)
echo   [3] Start Offline CLI Simulator (No Phone Required)
echo   [4] Run Full System Preflight Audit and Diagnostics
echo   [5] Run Complete Self-Test Suite (246 Unit Tests)
echo   [6] First-Time Setup / Reinstall Packages (SETUP.bat)
echo   [7] Exit
echo ==========================================================================
echo.

if not "%~1"=="" (
    set "USER_CHOICE=%~1"
    goto dispatch_choice
)

set "USER_CHOICE=1"
set /p "USER_CHOICE=Enter your choice [1-7] (Default is 1): "

:dispatch_choice
echo.
if "%USER_CHOICE%"=="1" goto launch_live
if "%USER_CHOICE%"=="2" goto launch_dry_run
if "%USER_CHOICE%"=="3" goto launch_cli
if "%USER_CHOICE%"=="4" goto launch_preflight
if "%USER_CHOICE%"=="5" goto launch_tests
if "%USER_CHOICE%"=="6" goto launch_setup
if "%USER_CHOICE%"=="7" goto launch_exit

echo [ERROR] Invalid choice "%USER_CHOICE%". Please enter a number between 1 and 7.
echo.
pause
goto show_menu_again

:launch_live
if %READY_COUNT% equ 0 (
    echo [WARNING] No authorized device is currently ready for Live Play!
    echo Please ensure your phone is connected, unlocked, and authorized.
    echo.
    set /p CONFIRM_LIVE="Do you still want to attempt live launch? (Y/N): "
    if /i not "!CONFIRM_LIVE!"=="Y" goto show_menu_again
)
echo ==========================================================================
echo   LAUNCHING AUTONOMOUS BOT IN LIVE MODE...
echo   KEY CONTROLS IN TERMINAL:
echo     [S] Start / Resume     [P] Pause
echo     [Space] Single Move    [Q] Stop / Exit
echo ==========================================================================
echo.
"%PYTHON_EXE%" run_autonomous_game.py
goto handle_exit

:launch_dry_run
echo ==========================================================================
echo   LAUNCHING AUTONOMOUS BOT IN DRY-RUN MODE...
echo   (Perception and AI will run, but NO physical touch events are sent)
echo ==========================================================================
echo.
"%PYTHON_EXE%" run_autonomous_game.py --dry-run
goto handle_exit

:launch_cli
echo ==========================================================================
echo   LAUNCHING OFFLINE CLI SIMULATOR...
echo   (Play Block Blast with the AI directly in your terminal)
echo ==========================================================================
echo.
"%PYTHON_EXE%" cli_demo.py
goto handle_exit

:launch_preflight
echo ==========================================================================
echo   RUNNING ENVIRONMENT PREFLIGHT AUDIT...
echo ==========================================================================
echo.
"%PYTHON_EXE%" action_engine/preflight_check.py
echo.
pause
goto show_menu_again

:launch_tests
echo ==========================================================================
echo   RUNNING FULL AUTOMATED UNIT TEST SUITE (246 TESTS)...
echo ==========================================================================
echo.
"%PYTHON_EXE%" -m unittest discover tests
echo.
pause
goto show_menu_again

:launch_setup
call SETUP.bat
goto show_menu_again

:launch_exit
echo Exiting Block Blast AI. Goodbye!
exit /b 0

:show_menu_again
cls
goto restart_launcher

:handle_exit
set "EXIT_CODE=%errorlevel%"
echo.
if %EXIT_CODE% neq 0 (
    echo ==========================================================================
    echo   [NOTICE] APPLICATION EXITED WITH CODE %EXIT_CODE%
    echo ==========================================================================
    echo   TROUBLESHOOTING TIPS:
    echo   1. If perception failed, check that phone screen is ON and unlocked.
    echo   2. Ensure Block Blast game board is currently visible on phone screen.
    echo   3. If touch events failed, verify USB Debugging permission is granted.
    echo      On Xiaomi/MIUI devices, also enable USB debugging Security settings.
    echo   4. Run Option 4 from START.bat to audit your system preflight state.
    echo ==========================================================================
) else (
    echo ==========================================================================
    echo   Block Blast AI session completed successfully.
    echo ==========================================================================
)
echo.
pause
