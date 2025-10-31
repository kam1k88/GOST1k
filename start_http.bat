@echo off
cd /d %~dp0
echo [🌐] Запуск локального веб-сервера на http://localhost:5500
start "" python -m http.server 5500
timeout /t 2 >nul
start "" http://localhost:5500/index.html
