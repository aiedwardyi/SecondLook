@echo off
REM Launch the BCC Streamlit app via the repo venv so torch and src are importable.
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
if not defined NO_ALBUMENTATIONS_UPDATE set "NO_ALBUMENTATIONS_UPDATE=1"
streamlit run app\streamlit_app_bcc.py --server.address 127.0.0.1
set "EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXIT_CODE%
