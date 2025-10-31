Write-Host "`n Запускаем ГОСТ1k локально..." -ForegroundColor Cyan

# Проверяем и поднимаем Ollama
$ollama = Get-Process ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Host "[+] Ollama не запущена  стартуем службу..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve"
} else {
    Write-Host "[] Ollama уже работает" -ForegroundColor Green
}

# Поднимаем API (порт 8080)
Write-Host "[+] Запуск API (FastAPI на 8080)" -ForegroundColor Yellow
Start-Process "python" -ArgumentList "rag_api.py"

# Поднимаем UI (порт 5500)
Write-Host "[+] Запуск локального UI (5500)" -ForegroundColor Yellow
Start-Process "python" -ArgumentList "-m http.server 5500"

# Финальное сообщение
Write-Host "`n Всё готово. Открой браузер:" -ForegroundColor Green
Write-Host "   http://localhost:5500/index.html" -ForegroundColor Cyan
Write-Host "   API Swagger: http://localhost:8080/docs`n" -ForegroundColor Cyan
