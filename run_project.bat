@echo off
set VENV_DIR=.venv

REM Step 1: Create virtual environment if it doesn't exist
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

REM Step 2: Activate virtual environment
call %VENV_DIR%\Scripts\activate.bat

REM Step 3: Install requirements
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Step 4: Run the main Python script
echo Running main script...
python scripts\main.py

REM Keep terminal open
pause
