@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run Запустить.bat to create the app environment.
  goto pause_error
)
".venv\Scripts\python.exe" -m pip install -r requirements-ai.txt
if errorlevel 1 (
  echo AI dependencies could not be installed. Check internet connection and Python version.
  goto pause_error
)
echo AI dependencies installed. Open Запустить.bat and enable the desired analysis switches.
pause
exit /b 0
:pause_error
pause
exit /b 1
