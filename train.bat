@echo off
REM Train the BCC detector with a fixed seed and deterministic env.
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

python -m scripts.train_bcc --output-dir experiments\bcc ^
  --csv-path splits\heidelberg_bcc.csv --data-root . --seed 42 --batch-size 16 ^
  --lr 1e-4 --epochs-phase1 12 --epochs-phase2 40 --early-stopping-patience 10 || exit /b

popd
endlocal
