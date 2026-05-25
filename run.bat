@echo off
title Object Detection & Tracking System - Enterprise Vision
color 0A
cls

echo =========================================================================
echo  OBJECT DETECTION & TRACKING SYSTEM
echo  Enterprise Vision System v1.0 // Antigravity Core
echo =========================================================================
echo.

:: 1. Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.8+ (64-bit recommended) from python.org.
    echo.
    pause
    exit /b 1
)

:: 2. Check virtual environment
if not exist .venv (
    echo [INFO] Python virtual environment not detected. Creating one...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment '.venv' created.
    echo.
)

:: 3. Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: 4. Verify/Install dependencies
echo [INFO] Inspecting dependencies in virtual environment...
echo [INFO] This might take a few moments on the first launch...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dependency installation encountered problems.
    echo Please check internet connection and pip configuration.
    echo.
    pause
    exit /b 1
)
echo [SUCCESS] Dependencies verified.
echo.

:: 5. Boot Application
echo [INFO] Launching vision dashboard window...
echo.
python main.py
if %errorlevel% neq 0 (
    echo.
    echo [WARN] Application terminated with exit code %errorlevel%.
)

pause
