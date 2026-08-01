@echo off
setlocal
cd /d "%~dp0"

for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":6446" ^| findstr "LISTENING"') do (
    echo Port 6446 is already in use by PID %%P.
    echo Run stop.bat first, or use the existing server.
    exit /b 1
)

set OPENCODE_ENABLE_EXA=1
python server.py
