@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run the app launcher to create its Python environment.
  goto install_error
)
echo Close Moto Shorts before installation.
echo Installing CUDA 12.8 PyTorch for NVIDIA RTX 50-series from the official index.
echo This download can be large. The NVIDIA driver must already be installed.
".venv\Scripts\python.exe" -m pip install --upgrade "torch==2.9.1" "torchvision==0.24.1" --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto install_error
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto install_error
".venv\Scripts\python.exe" check_nvidia.py --require-cuda
if errorlevel 1 (
  echo Packages were installed, but CUDA computation did not pass. See diagnostics above.
  echo Check the NVIDIA driver and free GPU memory. Restart the app after correcting the problem.
  goto install_error
)
echo CUDA computation passed. Restart Moto Shorts to use the updated runtime.
pause
exit /b 0
:install_error
echo NVIDIA setup did not complete successfully. Read the error above.
pause
exit /b 1
