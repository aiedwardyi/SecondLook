@echo off
REM Evaluate the trained BCC detector on the test split with deterministic env.
setlocal
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

python -m scripts.evaluate_bcc --run-dir experiments\bcc --csv-path splits\heidelberg_bcc.csv --data-root . || exit /b

popd
endlocal
