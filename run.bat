@echo off
echo Starting Physics Simulation...
call .venv\Scripts\activate.bat
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo Simulation crashed or exited with error.
    pause
)
