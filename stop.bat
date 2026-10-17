@echo off
rem Stop whatever is listening on port 6446, then wait until the port is really free.
rem
rem The old netstat/token parse was unreliable: it could resolve the PID as 0 (and
rem sometimes matched no line at all), so taskkill did nothing, the running server
rem stayed alive, and the next start died with "bind 10048" while the OLD code kept
rem serving - you would think you had restarted. Get-NetTCPConnection is
rem authoritative here, and "RemotePort -eq 0" isolates the listening socket so a
rem client connection is never killed.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Get-NetTCPConnection -LocalPort 6446 -ErrorAction SilentlyContinue | Where-Object RemotePort -eq 0 | Select-Object -ExpandProperty OwningProcess -Unique; if (-not $p) { Write-Host 'No process is listening on port 6446.'; exit 0 }; foreach ($x in $p) { Write-Host ('Stopping process ' + $x + ' on port 6446...'); Stop-Process -Id $x -Force -ErrorAction SilentlyContinue }; for ($i=0; $i -lt 40; $i++) { if (-not (Get-NetTCPConnection -LocalPort 6446 -ErrorAction SilentlyContinue | Where-Object RemotePort -eq 0)) { Write-Host 'Port 6446 is free.'; exit 0 }; Start-Sleep -Milliseconds 250 }; Write-Host 'WARNING: port 6446 is still held after 10s.'; exit 1"
exit /b %ERRORLEVEL%
