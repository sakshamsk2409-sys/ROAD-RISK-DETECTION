@echo off
title AI ADAS Live Camera Detection
echo ========================================================
echo   Launching AI ADAS Live Device Camera Detection
echo ========================================================
echo   - Camera Switching : Press 'C' or Space, or click Cam Badge
echo   - Fullscreen       : Press 'F'
echo   - Reset Tracks     : Press 'R'
echo   - Quit             : Press 'Q' or Esc
echo ========================================================
echo.

cd /d "%~dp0\.."
python live_camera\live_detector.py

pause
