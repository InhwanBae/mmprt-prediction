@echo off

cd /d "%~dp0"
echo Running Deep Learning Model Training and Testing...
call .venv\Scripts\activate
python train_test_DL_models.py
::deactivate
::.venv\Scripts\python.exe train_test_DL_models.py

echo.
echo Done. Press any key to exit.
pause > nul
