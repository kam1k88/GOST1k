Write-Host "[🔁] Индексация новых документов..."
cd $PSScriptRoot
python ingest.py
Write-Host "[✓] Готово. Новые документы добавлены в Chroma."
