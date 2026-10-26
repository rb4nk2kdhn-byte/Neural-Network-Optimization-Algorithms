@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First create .venv and install dependencies. See README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m unittest discover -s tests -v
pause
