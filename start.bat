@echo off
cd /d C:\Users\kam1k88\GOST1k

echo [*] Запуск RAG-сервера...
start "" python rag_api.py

echo [*] Ожидание, пока сервер запустится на http://localhost:8080 ...
setlocal enabledelayedexpansion

set "timeout=40"
set "count=0"

:waitloop
powershell -Command "(Invoke-WebRequest -Uri http://localhost:8080 -UseBasicParsing -TimeoutSec 1).StatusCode" >nul 2>&1
if %errorlevel%==0 (
    echo [*] ✅ Сервер готов, открываю браузер...
    start "" http://localhost:8080/docs
    goto end
)

set /a count+=1
if %count% geq %timeout% (
    echo [*] ❌ Сервер не ответил за %timeout% секунд.
    goto end
)

timeout /t 1 >nul
goto waitloop

:end
echo [*] Готово.
pause
