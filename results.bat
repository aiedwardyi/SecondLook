@echo off
REM Re-render a BCC run's saved training and test-eval tables (read-only).
setlocal
if "%~1"=="before" (
    set "MODEL=before"
) else if "%~1"=="after" (
    set "MODEL=after"
) else (
    echo Usage: results.bat ^<before^|after^>   ^(before = baseline, no correction; after = corrected^)
    exit /b 1
)
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
if not exist experiments\%MODEL%\best.pth (
    echo No trained %MODEL% model found at experiments\%MODEL%. Run train.bat %MODEL% first.
    exit /b 1
)
python -m scripts.show_results --run-dir experiments\%MODEL%
set "EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXIT_CODE%
