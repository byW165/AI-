@echo off
setlocal
set PYTHONDONTWRITEBYTECODE=1
pushd "%~dp0"
if errorlevel 1 goto path_error
if not exist ".venv\Scripts\python.exe" goto missing_environment
echo Opening the literature dashboard...
".venv\Scripts\python.exe" scripts\launch_dashboard.py %*
set "dashboard_exit=%errorlevel%"
popd
if "%dashboard_exit%"=="0" exit /b 0
echo.
echo Startup failed. Please keep this error message for troubleshooting.
pause
exit /b %dashboard_exit%
:missing_environment
echo Python environment is missing. Follow README.md to install dependencies.
popd
pause
exit /b 1
:path_error
echo Cannot open the project folder.
pause
exit /b 1
