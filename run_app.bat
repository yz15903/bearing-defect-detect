@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found.
  echo Open "Anaconda Prompt", run: conda activate bearing
  pause
  exit /b 1
)
python app\main.py
if errorlevel 1 pause
