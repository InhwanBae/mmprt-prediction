@echo off
setlocal

pushd "%~dp0"

:: ensure python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found in PATH. Install Python and ensure 'python' is on PATH.
    popd
    echo.
    echo Error occurred. Press any key to exit.
    pause > nul
    exit /b 1
)

:: create virtual environment if it doesn't exist
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment ".venv"...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        popd
        echo.
        echo Error occurred. Press any key to exit.
        pause > nul
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

:: activate the virtual environment in this session
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    popd
    echo.
    echo Error occurred. Press any key to exit.
    pause > nul
    exit /b 1
)

:: install required packages
if exist "utils\requirements.txt" (
    echo Installing packages from utils\requirements.txt...
    python -m pip install --upgrade pip
    python -m pip install -r "utils\requirements.txt"
    if errorlevel 1 (
        echo Failed to install requirements.
        popd
        echo.
        echo Error occurred. Press any key to exit.
        pause > nul
        exit /b 1
    )
) else (
    echo requirements file not found: utils\requirements.txt
    popd
    echo.
    echo Error occurred. Press any key to exit.
    pause > nul
    exit /b 1
)

echo Setup complete.
echo Waiting 10 seconds before exit...
timeout /t 10 /nobreak >nul
popd
endlocal
