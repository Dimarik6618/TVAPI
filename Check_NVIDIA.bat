@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run the app launcher to create its Python environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" check_nvidia.py
set "MOTO_CHECK_STATUS=%errorlevel%"
pause
exit /b %MOTO_CHECK_STATUS%
