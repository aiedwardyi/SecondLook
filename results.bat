@echo off
REM Re-render a BCC run's saved training and test-eval tables (read-only).
setlocal
if "%~1"=="" (
    echo Usage: results ^<run-dir^>
    echo Example: results experiments\bcc
    exit /b 1
)
set "RUN_DIR=%~f1"
pushd "%~dp0"
if defined PYTHONPATH (
    set "PYTHONPATH=%~dp0;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%~dp0"
)
if exist .\.venv\Scripts\Activate.bat (
    call .\.venv\Scripts\Activate.bat
) else (
    echo INFO: .venv not found at .\.venv; using current python on PATH.
)
python -m scripts.show_results --run-dir "%RUN_DIR%"
set "EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXIT_CODE%
