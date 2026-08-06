@echo off
REM Double-click this to launch the live training dashboard.
REM Opens a console window running the server + your browser to it.

cd /d "%~dp0"

echo Starting Health Risk Model Zoo dashboard...
start "Health Risk Dashboard - python -m src.dashboard" cmd /k python -m src.dashboard

timeout /t 2 /nobreak >nul
start "" http://localhost:8765

echo.
echo Dashboard launched in a separate window (close that window to stop it).
echo Browser opening at http://localhost:8765
echo.
pause
