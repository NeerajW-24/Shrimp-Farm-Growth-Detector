@echo off
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment - first run only...
    py -3.13 -m venv .venv || python -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python -m streamlit run app.py
pause
