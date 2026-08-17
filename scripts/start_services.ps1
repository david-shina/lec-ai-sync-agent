param(
    [string]$Scenario = "conflict"
)

$root = "C:\Users\Dell\Desktop\Python Projects\Interview"

$env:SCENARIO = $Scenario

$caption = Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","services.caption_service:app","--port","8001" -WorkingDirectory $root -PassThru -WindowStyle Hidden
$audio   = Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","services.audio_service:app","--port","8002" -WorkingDirectory $root -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 2

try {
    $c = Invoke-WebRequest -Uri "http://localhost:8001/health" -TimeoutSec 3 -UseBasicParsing
    Write-Output "caption_service: $($c.Content)"
} catch {
    Write-Output "caption_service: OFFLINE (expected for source_offline scenarios)"
}
$a = Invoke-WebRequest -Uri "http://localhost:8002/health" -TimeoutSec 3 -UseBasicParsing
Write-Output "audio_service: $($a.Content)"

Write-Output ""
Write-Output "PIDs: caption=$($caption.Id) audio=$($audio.Id)"
Write-Output "Set-Content -LiteralPath $root\.cache\.pids -Value @($caption.Id, $audio.Id)"
Set-Content -LiteralPath "$root\.cache\.pids" -Value "@($caption.Id, $audio.Id)" -Encoding utf8
