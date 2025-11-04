@echo off

cd /d "%~dp0"
::call .venv\Scripts\activate
::python train_test_ML_models.py
::deactivate
.venv\Scripts\python.exe train_test_ML_models.py

echo.
echo Done. Press any key to exit.
pause > nul
