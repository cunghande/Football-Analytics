# Launch the verified CUDA Python, even if a CPU-only venv is active.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8501 is occupied. Stop the old Streamlit process with Ctrl+C, then run this script again.'
}

$pythonPath = (& py -3.14 -c 'import sys; print(sys.executable)' | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Python 3.14 was not found. Check: py --list-paths'
}

& $pythonPath -c 'import torch, streamlit; assert torch.cuda.is_available(), "Python 3.14 has no CUDA"; print(torch.cuda.get_device_name(0))'
if ($LASTEXITCODE -ne 0) {
    throw 'The dashboard cannot start on GPU. Check PyTorch/CUDA in Python 3.14.'
}

& $pythonPath -m streamlit run (Join-Path $PSScriptRoot 'dashboard.py') --server.port 8501
