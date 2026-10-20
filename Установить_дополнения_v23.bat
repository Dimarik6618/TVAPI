@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run the app launcher once to create the Python environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements-v23.txt
if errorlevel 1 (
  echo Installation failed. Check internet connection and Python version.
  pause
  exit /b 1
)
echo All v23 features installed. Open the app.
pause
exit /b 0
