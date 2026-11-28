@echo off
rem MoshFlow: double-click to run on Windows.
rem The first run creates .venv with Python 3.7 or newer and installs
rem requirements.txt into it (needs internet). Later runs start at once.
setlocal
cd /d "%~dp0"
set "VENV_PY=.venv\Scripts\python.exe"
set "READY=.venv\moshflow_ready.txt"
if exist "%READY%" goto run

if exist "%VENV_PY%" goto install

set PY=
for %%V in (3.12 3.11 3.10 3.13 3.9 3.8 3.7 3) do (
  if not defined PY (
    py -%%V -c "import sys; sys.exit(sys.version_info < (3, 7))" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(sys.version_info < (3, 7))" >nul 2>&1 && set "PY=python"
)
rem Python that the Visual Studio installer puts in (VS 2019: 3.7, VS 2022: 3.9).
for %%P in ("%ProgramFiles(x86)%\Microsoft Visual Studio\Shared\Python37_64\python.exe"
            "%ProgramFiles%\Python39\python.exe"
            "%ProgramFiles(x86)%\Microsoft Visual Studio\Shared\Python39_64\python.exe") do (
  if not defined PY if exist %%P set PY=%%P
)
if not defined PY goto nopython

echo Creating .venv with %PY% ...
%PY% -m venv .venv
if errorlevel 1 goto fail

:install
echo Installing requirements (first run only, takes a few minutes) ...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo ok> "%READY%"

:run
"%VENV_PY%" run.py %*
if errorlevel 1 pause
exit /b

:nopython
echo No Python 3.7 or newer was found.
echo Install Python 3.12 from https://www.python.org/downloads/ (tick "Add python.exe to PATH"),
echo or run:  winget install Python.Python.3.12
echo Then double-click run_windows.bat again.
pause
exit /b 1

:fail
echo Setup failed (see the messages above). Check the internet connection and run this file again.
pause
exit /b 1
