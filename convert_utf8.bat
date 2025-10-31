@echo off
cd /d "%~dp0"
echo ===============================================
echo    Перекодировка .RTF и .DOCX файлов в UTF-8
echo ===============================================

REM Запуск PowerShell-скрипта
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0convert_utf8.ps1"

echo.
echo ✅ Завершено. Окно можно закрыть.
pause
