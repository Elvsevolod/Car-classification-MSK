$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $ProjectRoot
py -3.11 -m venv .venv-rtx4060
if ($LASTEXITCODE -ne 0) { throw "Install Python 3.11 x64 first" }
$ResearchPython = Join-Path $ProjectRoot ".venv-rtx4060/Scripts/python.exe"
& $ResearchPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip setup failed" }
& $ResearchPython -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }
& $ResearchPython -m pip install -r (Join-Path $PSScriptRoot "requirements-windows.txt")
if ($LASTEXITCODE -ne 0) { throw "Research dependencies installation failed" }
& $ResearchPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: check NVIDIA driver'; print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA preflight failed; CPU fallback is forbidden" }
& $ResearchPython -m ipykernel install --user --name car-reid-rtx4060 --display-name "Car ReID RTX 4060"
if ($LASTEXITCODE -ne 0) { throw "Kernel registration failed" }
Write-Host "Ready. Open train_windows_queues.ipynb, choose Car ReID RTX 4060, then Run All."
