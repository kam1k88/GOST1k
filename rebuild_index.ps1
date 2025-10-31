Write-Host "[🔥] Полная пересборка индекса..."
cd $PSScriptRoot

if (Test-Path "chroma_db") {
    Remove-Item -Recurse -Force "chroma_db"
    Write-Host "[🧹] Удалена старая база chroma_db"
}

python ingest.py
Write-Host "[✓] Новая база успешно создана."
