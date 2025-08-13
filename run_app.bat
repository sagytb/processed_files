@echo off
chcp 65001 > nul

TITLE Financial Analyzer AI - User Interface

TYPE app_hebrew.txt

cd /d "%~dp0"

IF NOT EXIST "env\Scripts\activate.bat" (
    ECHO ERROR: Virtual environment 'env' not found!
    PAUSE
    EXIT /B 1
)

ECHO Activating virtual environment...
CALL env\Scripts\activate.bat

ECHO Launching Streamlit application...
ECHO A new browser window should open shortly.
ECHO.

streamlit run analyzer.py

ECHO.
ECHO Streamlit server has been stopped. Press any key to close this window.
PAUSE