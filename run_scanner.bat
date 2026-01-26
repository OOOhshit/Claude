@echo off
echo Oil Companies Market & Technology Scanner
echo ==========================================
echo.

echo Checking Python installation...
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

echo Installing required packages...
py -m pip install -r requirements.txt

echo.
echo Starting the scanner...
echo This may take 10-15 minutes depending on network speed
echo.

py main.py

echo.
echo Analysis complete! Check the generated files:
echo - analysis_report.html (open in browser)
echo - company_summary.csv (open in Excel)
echo - ai_analysis_results.json (detailed data)
echo.

pause