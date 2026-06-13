@echo off
echo ============================
echo  TurtleNeckDetector Build
echo ============================

set PYTHON=.venv\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=python

echo [1/4] Cleaning previous build...
if exist "dist\TurtleNeckDetector" rmdir /s /q "dist\TurtleNeckDetector"
if exist "build" rmdir /s /q "build"
if exist "TurtleNeckDetector.spec" del /q "TurtleNeckDetector.spec"

echo [2/4] Running PyInstaller...
%PYTHON% -m PyInstaller ^
  --noconsole ^
  --onedir ^
  --name TurtleNeckDetector ^
  --collect-all mediapipe ^
  --hidden-import pystray._win32 ^
  --hidden-import firebase_admin ^
  --hidden-import google.cloud.firestore ^
  --hidden-import customtkinter ^
  --hidden-import google_auth_oauthlib ^
  --add-data "config.json;." ^
  --add-data "assets;assets" ^
  --add-data ".env;." ^
  --add-data "client_secret.json;." ^
  --add-data "firebase_key.json;." ^
  turtle_neck.py

if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller failed.
    pause
    exit /b 1
)

echo [3/4] Copying runtime files...
set DIST=dist\TurtleNeckDetector

if exist ".env" (
    copy /y ".env" "%DIST%\.env" > nul
    echo      .env included.
) else (
    echo      WARNING: .env not found (FIREBASE_API_KEY missing).
)

if exist "firebase_key.json" (
    copy /y "firebase_key.json" "%DIST%\firebase_key.json" > nul
    echo      firebase_key.json included.
) else (
    echo      firebase_key.json not found (upload disabled).
)

if exist "client_secret.json" (
    copy /y "client_secret.json" "%DIST%\client_secret.json" > nul
    echo      client_secret.json included.
) else (
    echo      WARNING: client_secret.json not found (Google login disabled).
)

echo [4/4] Done.
echo.
echo Output: %DIST%\TurtleNeckDetector.exe
echo.
echo Checklist before distribution:
echo   1. Set FIREBASE_API_KEY in %DIST%\.env
echo   2. Confirm firebase_key.json is included
echo   3. Test on a clean machine
echo.
pause
