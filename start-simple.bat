@echo off
cd /d "%~dp0"

rem Same port-free step as start.bat: without it this variant dies on "bind 10048"
rem whenever a server already runs.
call "%~dp0stop.bat"

python server.py --no-proxy-pool
