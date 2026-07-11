@echo off
REM Train one side (before = baseline, after = corrected) with a fixed seed and deterministic env.
setlocal
if "%~1"=="before" (
    set "MODEL=before"
    set "CORRECTION=--no-correction"
) else if "%~1"=="after" (
    set "MODEL=after"
    set "CORRECTION=--correction"
) else (
    echo Usage: train.bat ^<before^|after^>   ^(before = baseline, no correction; after = corrected^)
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

python -m scripts.train_bcc %CORRECTION% --output-dir experiments\%MODEL% ^
  --csv-path splits\heidelberg_bcc.csv --data-root data --seed 42 --batch-size 16 ^
  --lr 1e-4 --epochs-phase1 12 --epochs-phase2 40 --early-stopping-patience 10 || exit /b

popd
endlocal
