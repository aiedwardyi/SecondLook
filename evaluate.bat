@echo off
REM Evaluate one side (before = baseline, after = corrected) on the test split with deterministic env.
setlocal
if "%~1"=="before" (
    set "MODEL=before"
) else if "%~1"=="after" (
    set "MODEL=after"
) else (
    echo Usage: evaluate.bat ^<before^|after^>   ^(before = baseline, no correction; after = corrected^)
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

set PYTHONHASHSEED=42
set CUBLAS_WORKSPACE_CONFIG=:4096:8

if not exist experiments\%MODEL%\best.pth (
    echo No trained %MODEL% model found at experiments\%MODEL%. Run train.bat %MODEL% first.
    exit /b 1
)

python -m scripts.evaluate_bcc --run-dir experiments\%MODEL% --csv-path splits\heidelberg_bcc.csv --data-root data || exit /b

popd
endlocal
