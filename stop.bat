@echo off
setlocal
set "FOUND="

for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":6446" ^| findstr "LISTENING"') do (
    set "FOUND=1"
    echo Stopping process %%P on port 6446...
    taskkill /PID %%P /F >nul 2>&1
)

if not defined FOUND echo No process is listening on port 6446.
