# install_rtx.ps1 -- CEX Orchestrator: NVIDIA RTX setup
# Installs: Python deps, Vulkan SDK check, CUDA check, instant-ngp guidance
# Run as Administrator: Set-ExecutionPolicy Bypass -Scope Process; .\install\install_rtx.ps1

param(
    [string]$PythonVenv = "venv_rtx",
    [switch]$SkipVulkan,
    [switch]$SkipCuda,
    [switch]$TestOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = Split-Path -Parent $SCRIPT_DIR

function Write-Info { param($m) Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Write-Ok   { param($m) Write-Host "[OK]    $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Write-Err  { param($m) Write-Host "[ERR]   $m" -ForegroundColor Red }

Write-Info "CEX Orchestrator -- NVIDIA RTX installer"
Write-Info "Project: $PROJECT_DIR"
Write-Info "Venv:    $PythonVenv"
""

# ---------------------------------------------------------------
# 1. Python venv
# ---------------------------------------------------------------
Write-Info "Setting up Python virtual environment..."
$venvPath = Join-Path $PROJECT_DIR $PythonVenv
python -m venv $venvPath | Out-Null
$pip = Join-Path $venvPath "Scripts\pip.exe"
& $pip install --quiet --upgrade pip

Write-Info "Installing Python packages (requirements_rtx.txt)..."
& $pip install --quiet -r "$PROJECT_DIR\requirements_rtx.txt"
Write-Ok "Python packages installed"

# ---------------------------------------------------------------
# 2. OpenEXR wheel (must be installed separately on Windows)
# ---------------------------------------------------------------
Write-Info "Checking OpenEXR..."
$pyVer = & (Join-Path $venvPath "Scripts\python.exe") -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"
$exrWheel = Get-ChildItem "$PROJECT_DIR\wheels\" -Filter "OpenEXR-*-cp${pyVer}-*-win_amd64.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($exrWheel) {
    & $pip install --quiet $exrWheel.FullName
    Write-Ok "OpenEXR installed from local wheel"
} else {
    Write-Warn "OpenEXR wheel not found in wheels\"
    Write-Warn "Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr"
    Write-Warn "Place in: $PROJECT_DIR\wheels\OpenEXR-1.3.2-cp${pyVer}-cp${pyVer}-win_amd64.whl"
}

# ---------------------------------------------------------------
# 3. CUDA check
# ---------------------------------------------------------------
if (-not $SkipCuda) {
    Write-Info "Checking CUDA toolkit..."
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        $cudaVer = (& nvcc --version 2>&1) | Select-String "release" | Select-Object -First 1
        Write-Ok "CUDA: $cudaVer"
    } else {
        Write-Warn "CUDA toolkit not found."
        Write-Warn "Download CUDA 12.2+: https://developer.nvidia.com/cuda-downloads"
        Write-Warn "Required for instant-ngp build."
    }
}

# ---------------------------------------------------------------
# 4. Vulkan SDK check
# ---------------------------------------------------------------
if (-not $SkipVulkan) {
    Write-Info "Checking Vulkan SDK..."
    $vkPath = $env:VULKAN_SDK
    if ($vkPath -and (Test-Path $vkPath)) {
        $vkVer = Split-Path $vkPath -Leaf
        Write-Ok "Vulkan SDK: $vkVer at $vkPath"
    } else {
        Write-Warn "Vulkan SDK not found (VULKAN_SDK env var not set)."
        Write-Warn "Install from: https://vulkan.lunarg.com/sdk/home#windows"
        Write-Warn "Required for instant-ngp rendering."
    }
}

# ---------------------------------------------------------------
# 5. NVIDIA driver check
# ---------------------------------------------------------------
Write-Info "Checking NVIDIA driver..."
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvsmi) {
    $drvInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    Write-Ok "GPU: $drvInfo"

    # Persistent mode (optional -- keeps GPU warm between inference calls)
    Write-Info "Note: enable persistent mode for faster repeated inference:"
    Write-Info "  nvidia-smi -pm 1  (requires Admin)"
} else {
    Write-Warn "nvidia-smi not found. Install NVIDIA Game Ready or Studio driver 537+"
    Write-Warn "Studio driver recommended: less aggressive GPU clock scaling under load"
    Write-Warn "Download: https://www.nvidia.com/en-us/drivers/"
}

# ---------------------------------------------------------------
# 6. COLMAP check
# ---------------------------------------------------------------
Write-Info "Checking COLMAP (Structure-from-Motion)..."
$colmap = Get-Command colmap -ErrorAction SilentlyContinue
if ($colmap) {
    Write-Ok "COLMAP: $($colmap.Source)"
} else {
    Write-Warn "COLMAP not found. Required to convert photos to NeRF training data."
    Write-Warn "Download binary: https://github.com/colmap/colmap/releases"
    Write-Warn "Add colmap.exe directory to PATH."
}

# ---------------------------------------------------------------
# 7. instant-ngp guidance
# ---------------------------------------------------------------
Write-Info ""
Write-Info "=== instant-ngp build instructions ==="
Write-Info "instant-ngp (NVIDIA NeRF) is not installed via pip -- must build from source."
Write-Info ""
Write-Info "Prerequisites (must be installed first):"
Write-Info "  - Visual Studio 2022 (Desktop C++ workload)"
Write-Info "  - CUDA Toolkit 12.2+"
Write-Info "  - Vulkan SDK 1.3+"
Write-Info "  - CMake 3.20+"
Write-Info ""
Write-Info "Build steps:"
Write-Info '  git clone --recursive https://github.com/NVlabs/instant-ngp'
Write-Info '  cd instant-ngp'
Write-Info '  cmake -B build -DCMAKE_BUILD_TYPE=Release'
Write-Info '  cmake --build build --config Release -j'
Write-Info '  # Output: build\Release\pyngp.pyd'
Write-Info '  # Add to PYTHONPATH:'
Write-Info '  $env:PYTHONPATH += ";C:\path\to\instant-ngp\build\Release"'
Write-Info ""

# ---------------------------------------------------------------
# 8. Write local-only config (if not present)
# ---------------------------------------------------------------
$cfgPath = Join-Path $PROJECT_DIR "config\cex_config.yaml"
if (-not (Test-Path $cfgPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $cfgPath) | Out-Null
    Copy-Item "$PROJECT_DIR\config\cex_config.yaml" $cfgPath -ErrorAction SilentlyContinue
    Write-Info "Default config written (local_only=true): $cfgPath"
} else {
    Write-Ok "Config exists: $cfgPath"
}

""
Write-Ok "RTX setup complete."
Write-Info "Test inference:    $venvPath\Scripts\python.exe gpu\cex_gpu_server.py"
Write-Info "Test NGP:          $venvPath\Scripts\python.exe gpu\cex_ngp_executor.py"
Write-Info "Config:            config\cex_config.yaml  (local_only=true by default)"
