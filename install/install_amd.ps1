# install_amd.ps1 -- CEX Orchestrator: AMD Radeon RX 6000+ setup (Windows)
# Installs: DirectML, torch-directml, Vulkan check, COLMAP, all NeRF deps
# Run: Set-ExecutionPolicy Bypass -Scope Process; .\install\install_amd.ps1

param(
    [string]$PythonVenv = "venv_amd",
    [switch]$SkipVulkan,
    [switch]$TestOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$SCRIPT_DIR  = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = Split-Path -Parent $SCRIPT_DIR

function Write-Info { param($m) Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Write-Ok   { param($m) Write-Host "[OK]    $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Write-Err  { param($m) Write-Host "[ERR]   $m" -ForegroundColor Red }

Write-Info "CEX Orchestrator -- AMD Radeon RX 6000+ installer (Windows)"
Write-Info "Project:  $PROJECT_DIR"
Write-Info "Strategy: torch-directml (NeRF) + onnxruntime-directml (ONNX inference)"
""

# ---------------------------------------------------------------
# 1. Python venv
# ---------------------------------------------------------------
$venvPath = Join-Path $PROJECT_DIR $PythonVenv
python -m venv $venvPath | Out-Null
$pip    = Join-Path $venvPath "Scripts\pip.exe"
$python = Join-Path $venvPath "Scripts\python.exe"
& $pip install --quiet --upgrade pip

Write-Info "Installing Python packages (requirements_amd.txt)..."
& $pip install --quiet -r "$PROJECT_DIR\requirements_amd.txt"
Write-Ok "Base packages installed"

# ---------------------------------------------------------------
# 2. torch + torch-directml (AMD on Windows)
# ---------------------------------------------------------------
Write-Info "Installing PyTorch + torch-directml for AMD..."
# CPU wheel of torch (torch-directml does NOT need CUDA torch)
& $pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $pip install --quiet torch-directml
Write-Ok "torch-directml installed"

# Verify
$dmlCheck = & $python -c "import torch_directml; print(torch_directml.device_count())" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Ok "torch-directml: $dmlCheck DirectML device(s) found"
} else {
    Write-Warn "torch-directml import failed: $dmlCheck"
    Write-Warn "Ensure AMD Adrenalin driver 23.7.1+ is installed"
}

# ---------------------------------------------------------------
# 3. onnxruntime-directml (ONNX inference via DirectML)
# ---------------------------------------------------------------
Write-Info "Verifying onnxruntime-directml..."
$dmlOrt = & $python -c "import onnxruntime as ort; print('DmlExecutionProvider' in ort.get_available_providers())" 2>&1
if ($dmlOrt -eq "True") {
    Write-Ok "onnxruntime-directml: DmlExecutionProvider available"
} else {
    Write-Warn "DmlExecutionProvider not available. Reinstalling..."
    & $pip install --quiet --force-reinstall onnxruntime-directml
}

# ---------------------------------------------------------------
# 4. OpenEXR wheel (Windows)
# ---------------------------------------------------------------
$pyVer   = & $python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"
$exrWheel = Get-ChildItem "$PROJECT_DIR\wheels\" -Filter "OpenEXR-*-cp${pyVer}-*-win_amd64.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($exrWheel) {
    & $pip install --quiet $exrWheel.FullName
    Write-Ok "OpenEXR installed from: $($exrWheel.Name)"
} else {
    Write-Warn "OpenEXR wheel not found in wheels\"
    Write-Warn "Download: https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr"
    Write-Warn "Place in: $PROJECT_DIR\wheels\OpenEXR-1.3.2-cp${pyVer}-cp${pyVer}-win_amd64.whl"
}

# ---------------------------------------------------------------
# 5. Vulkan SDK check (AMD RDNA2+ has native Vulkan 1.3)
# ---------------------------------------------------------------
if (-not $SkipVulkan) {
    Write-Info "Checking Vulkan SDK..."
    if ($env:VULKAN_SDK -and (Test-Path $env:VULKAN_SDK)) {
        Write-Ok "Vulkan SDK: $env:VULKAN_SDK"
    } else {
        Write-Warn "Vulkan SDK not detected (VULKAN_SDK env var missing)."
        Write-Warn "Install from: https://vulkan.lunarg.com/sdk/home#windows"
        Write-Warn "AMD RDNA2+ GPUs have full Vulkan 1.3 support via Adrenalin driver."
    }
    # Verify via vulkaninfo (ships with Vulkan SDK)
    $vkInfo = Get-Command vulkaninfo -ErrorAction SilentlyContinue
    if ($vkInfo) {
        $gpuLine = & vulkaninfo 2>&1 | Select-String "deviceName" | Select-Object -First 1
        Write-Ok "Vulkan device: $gpuLine"
    }
}

# ---------------------------------------------------------------
# 6. COLMAP
# ---------------------------------------------------------------
$colmap = Get-Command colmap -ErrorAction SilentlyContinue
if ($colmap) {
    Write-Ok "COLMAP: $($colmap.Source)"
} else {
    Write-Warn "COLMAP not found. Required for camera pose estimation."
    Write-Warn "Download: https://github.com/colmap/colmap/releases"
}

# ---------------------------------------------------------------
# 7. AMD driver check
# ---------------------------------------------------------------
Write-Info "Checking AMD GPU..."
try {
    $dxDiag = & $python -c @"
import torch_directml
n = torch_directml.device_count()
for i in range(n):
    d = torch_directml.device(i)
    print(f'  Device {i}: {d}')
"@ 2>&1
    Write-Ok "AMD DirectML devices:$dxDiag"
} catch {
    Write-Warn "Could not enumerate DirectML devices: $_"
    Write-Warn "Ensure AMD Adrenalin Software 23.7.1+ is installed:"
    Write-Warn "https://www.amd.com/en/support"
}

# ---------------------------------------------------------------
# 8. torch-ngp clone guidance
# ---------------------------------------------------------------
Write-Info ""
Write-Info "=== torch-ngp setup (AMD NeRF) ==="
Write-Info "torch-ngp runs on AMD via torch-directml. Steps:"
Write-Info ""
Write-Info "  git clone https://github.com/ashawkey/torch-ngp"
Write-Info "  cd torch-ngp"
Write-Info "  pip install -r requirements.txt"
Write-Info "  # No CUDA build needed with torch-directml"
Write-Info "  python main_nerf.py data/fox --workspace trial_fox -O"
Write-Info ""
Write-Info "Note: torch-ngp custom CUDA kernels (raymarching etc.) will fall back to"
Write-Info "PyTorch native ops on DirectML -- slower than RTX but functional."
Write-Info "For best AMD performance, prefer the Vulkan NeRF path (future update)."
Write-Info ""

# ---------------------------------------------------------------
# 9. Write local-only config
# ---------------------------------------------------------------
$cfgPath = Join-Path $PROJECT_DIR "config\cex_config.yaml"
if (-not (Test-Path $cfgPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $cfgPath) | Out-Null
    Write-Info "Config already present at $cfgPath"
}

""
Write-Ok "AMD setup complete."
Write-Info "Test ONNX inference: $python gpu\cex_gpu_server.py"
Write-Info "Test NGP (AMD):      $python gpu\cex_ngp_amd.py"
Write-Info "Config (local only): config\cex_config.yaml"
Write-Info ""
Write-Info "Remember: local_only=true by default. Cloud inference is disabled."
Write-Info "To enable cloud: edit config\cex_config.yaml -> cloud.enabled: true"
