@echo off
setlocal
cd /d "%~dp0"

set "PYTHON="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul
if not errorlevel 1 set "PYTHON=py -3"
if not defined PYTHON (
  python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul
  if not errorlevel 1 set "PYTHON=python"
)

if not defined PYTHON goto missing_python
if not exist ".venv\Scripts\python.exe" (
  %PYTHON% -m venv .venv
  if errorlevel 1 goto create_error
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto pip_error
".venv\Scripts\python.exe" moto_shorts.py
if errorlevel 1 goto app_error
exit /b 0

:missing_python
echo Python 3.10 or newer was not found.
echo Install it from https://www.python.org/downloads/windows/
echo During installation check "Add python.exe to PATH".
echo Close this window and run this file again.
goto pause_error
:create_error
echo Could not create the Python environment in this folder.
echo Move the extracted folder to Documents and try again.
goto pause_error
:pip_error
echo Could not install dependencies. Check the internet connection.
goto pause_error
:app_error
echo The app stopped with an error. See the message above.
:pause_error
pause
exit /b 1
