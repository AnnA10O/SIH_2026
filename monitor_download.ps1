while ($true) {
    Clear-Host
    $count = (Get-ChildItem -Path d:\SIH\data\raw\satellite -Recurse -Filter *.h5 | Measure-Object).Count
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host "   MOSDAC REAL-TIME PROGRESS         " -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host " Expected Total:   ~12,144 files"
    Write-Host " Downloaded:       $count files" -ForegroundColor Green
    
    if ($count -gt 0) {
        $percent = [math]::Round(($count / 12144) * 100, 1)
        Write-Host " Progress:         $percent %" -ForegroundColor Yellow
    }
    
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host " (Auto-refreshing every 5 seconds. Press Ctrl+C to stop)"
    Start-Sleep -Seconds 5
}
