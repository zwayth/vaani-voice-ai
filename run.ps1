# Start Vaani Voice AI on http://localhost:8800
Set-Location $PSScriptRoot
if (-not (Test-Path .venv)) {
  uv venv .venv --python 3.12
  uv pip install --python .venv\Scripts\python.exe -r requirements.txt
}
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# local Kokoro-82M voice (fast offline TTS, English + Hindi)
New-Item -ItemType Directory models -Force | Out-Null
$base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
foreach ($f in "kokoro-v1.0.int8.onnx", "voices-v1.0.bin") {
  if (-not (Test-Path "models\$f")) { Write-Host "Downloading $f ..."; curl.exe -sL -o "models\$f" "$base/$f" }
}
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8800 --ws-max-size 4194304
