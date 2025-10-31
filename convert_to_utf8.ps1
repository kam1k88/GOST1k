# Папка с документами
$path = "C:\Users\kam1k88\GOST1k\docs"

# Убедимся, что папка существует
if (!(Test-Path $path)) {
    Write-Host "Папка не найдена: $path"
    exit
}

# Получаем все RTF и DOCX
$files = Get-ChildItem -Path $path -Recurse -Include *.rtf, *.docx

foreach ($file in $files) {
    $ext = $file.Extension.ToLower()
    $newFile = [System.IO.Path]::ChangeExtension($file.FullName, ".utf8$ext")

    if ($ext -eq ".rtf") {
        # Читаем как ANSI / CP1251
        $content = Get-Content -Path $file.FullName -Encoding Default -Raw
        # Перекодируем в UTF-8 и сохраняем
        [System.IO.File]::WriteAllText($newFile, $content, (New-Object System.Text.UTF8Encoding($false)))
        Write-Host "Преобразован RTF: $file -> $newFile"
    }
    elseif ($ext -eq ".docx") {
        # Для DOCX — это ZIP, нужно пройти через Word COM
        $word = New-Object -ComObject Word.Application
        $doc = $word.Documents.Open($file.FullName)
        $utf8Path = $newFile
        $doc.SaveAs([ref] $utf8Path, [ref] 0)  # 0 = wdFormatDocumentDefault (UTF-8)
        $doc.Close()
        $word.Quit()
        Write-Host "Преобразован DOCX: $file -> $newFile"
    }
}
