@echo off
chcp 65001 >nul
title SmartCanvasCropper CLI Setup
echo SmartCanvasCropper environment check, setup, and launch
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup_and_run.ps1" -Action prepare-run -Mode auto
set "SC_EXIT=%ERRORLEVEL%"
echo.
if not "%SC_EXIT%"=="0" echo Setup returned exit code %SC_EXIT%. See the log path above.
pause
exit /b %SC_EXIT%
