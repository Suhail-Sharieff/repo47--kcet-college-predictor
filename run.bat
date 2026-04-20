@echo off
title KCET College Predictor
echo.
echo  ================================================
echo    KCET College Predictor - Starting Server
echo  ================================================
echo.

cd /d "%~dp0"

REM Check if data is parsed
if not exist "data\cutoffs.csv" (
    echo  [STEP 1/3] Parsing PDFs - this may take 5-10 minutes...
    venv\Scripts\python src\parse_pdfs.py
    if errorlevel 1 (
        echo  [ERROR] PDF parsing failed. Check your PDF files.
        pause
        exit /b 1
    )
) else (
    echo  [OK] Cutoff data found: data\cutoffs.csv
)

REM Check if model is trained
if not exist "models\model_bundle.pkl" (
    echo  [STEP 2/3] Training ML model...
    venv\Scripts\python src\train_model.py
    if errorlevel 1 (
        echo  [ERROR] Model training failed.
        pause
        exit /b 1
    )
) else (
    echo  [OK] ML model found: models\model_bundle.pkl
)

echo.
echo  [STEP 3/3] Starting web server...
echo  Open your browser at: http://localhost:5000
echo  Press Ctrl+C to stop.
echo.

venv\Scripts\python src\app.py
pause
