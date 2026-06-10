@echo off
echo ============================================================
echo   DL Project — GitHub Setup Script
echo ============================================================
echo.

cd /d "%~dp0"

REM Remove stale git lock if present
if exist ".git\config.lock" (
    del /f ".git\config.lock"
    echo [OK] Removed stale git lock
)

REM Remove incomplete .git and start fresh
if exist ".git" (
    rmdir /s /q ".git"
    echo [OK] Cleared old .git folder
)

REM Initialize repo
git init
git config user.email "israelagyekum@gmail.com"
git config user.name "Israel Agyekum"
echo [OK] Git repo initialized

REM Stage all files
git add .
echo [OK] All files staged

REM First commit
git commit -m "Initial commit: Explanation-Supervised Attention CNN for ChestX-ray14

- src/: Full model code (AttentionModel, BaselineModel, GradCAM, losses, metrics)
- notebooks/: Colab training + inference server notebooks
- dashboard.html: Interactive results dashboard (WHO color scheme)
- app.py: Streamlit inference app
- config.yaml: Training configuration
- report/: Project reports (PDF + DOCX) and all result figures
- data/: Dataset metadata CSVs (BBox_List_2017, Data_Entry_2017)
- outputs/figures/: EDA and training visualisation plots

Note: Model checkpoints (.pt files) are stored on Google Drive, not GitHub."

echo.
echo ============================================================
echo   Commit done!
echo ============================================================
echo.
echo NEXT STEP — Push to GitHub:
echo   1. Go to https://github.com/new
echo   2. Create a NEW repository named:  dl-chest-xray-attention
echo   3. Set it to Public (or Private — your choice)
echo   4. Do NOT tick "Add README" or any other option
echo   5. Click "Create repository"
echo   6. Copy the repo URL (e.g. https://github.com/YourUsername/dl-chest-xray-attention.git)
echo   7. Run this in Git Bash inside this folder:
echo.
echo      git remote add origin ^<paste-your-repo-url^>
echo      git branch -M main
echo      git push -u origin main
echo.
pause
