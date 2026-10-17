@echo off
setlocal
cd /d "%~dp0"

rem Free the port first. A surviving server makes this one die with "bind 10048"
rem while the old code keeps serving, so the restart would silently do nothing.
call "%~dp0stop.bat"
if errorlevel 1 echo WARNING: port 6446 was not released; the new server may fail to bind.

set OPENCODE_ENABLE_EXA=1
python server.py
