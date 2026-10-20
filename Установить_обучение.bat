@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run the app launcher to create its Python environment.
  goto pause_error
)
".venv\Scripts\python.exe" -m pip install -r requirements-learning.txt
if errorlevel 1 (
  echo Learning dependencies could not be installed. Check internet connection and Python version.
  goto pause_error
)
echo Learning installed. Open the app, then PROJECT - LEARNING.
echo The first training run downloads X-CLIP weights. Your videos stay local.
pause
exit /b 0
:pause_error
pause
exit /b 1
