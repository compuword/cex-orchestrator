# install_llama_rpc_windows.ps1 -- Build llama.cpp with CUDA + RPC on Windows
#
# Requires:
#   - Git             (winget install Git.Git)
#   - CMake >= 3.21   (winget install Kitware.CMake)
#   - CUDA Toolkit    (installed -- CUDA 12.8 detected on this machine)
#   - Visual Studio 2022 with C++ workload  OR  Build Tools
#
# Output:
#   C:\llama.cpp\build\bin\Release\llama-server.exe
#   C:\llama.cpp\build\bin\Release\llama-rpc-server.exe
#
# Usage:
#   Set-ExecutionPolicy Bypass -Scope Process
#   .\install\install_llama_rpc_windows.ps1

param(
    [string]$InstallDir = "C:\llama.cpp",
    [string]$CudaArch   = "89",       # 89 = RTX 40xx (Ada Lovelace)
    [switch]$SkipClone,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

function Write-Step { param([string]$msg) Write-Host "`n[>>] $msg" -ForegroundColor Cyan }
function Write-Ok   { param([string]$msg) Write-Host "  [OK] $msg"   -ForegroundColor Green }
function Write-Fail { param([string]$msg) Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Write-Warn { param([string]$msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------
# Step 1: Check prerequisites
# ---------------------------------------------------------------
Write-Step "Checking prerequisites"

$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) { Write-Fail "git not found. Install: winget install Git.Git"; exit 1 }
Write-Ok "git: $($gitCmd.Source)"

$cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmakeCmd) { Write-Fail "cmake not found. Install: winget install Kitware.CMake"; exit 1 }
Write-Ok "cmake: $($cmakeCmd.Source)"

$nvccCmd = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvccCmd) {
    Write-Warn "nvcc not found -- building without CUDA (CPU only)"
    $useCuda = $false
} else {
    $nvccVer = & nvcc --version 2>&1 | Select-String "release" | Select-Object -First 1
    Write-Ok "nvcc: $nvccVer"
    $useCuda = $true
}

# ---------------------------------------------------------------
# Step 2: Clone llama.cpp
# ---------------------------------------------------------------
Write-Step "Cloning llama.cpp"

if (-not $SkipClone) {
    if (Test-Path "$InstallDir\.git") {
        Write-Ok "Already cloned -- pulling latest"
        git -C $InstallDir pull --ff-only
    } else {
        New-Item -ItemType Directory -Force $InstallDir | Out-Null
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git $InstallDir
        Write-Ok "Cloned to $InstallDir"
    }
} else {
    Write-Ok "Skipping clone (--SkipClone)"
}

# ---------------------------------------------------------------
# Step 3: CMake configure
# ---------------------------------------------------------------
Write-Step "Configuring CMake (CUDA=$useCuda, RPC=ON, arch=$CudaArch)"

$buildDir = "$InstallDir\build"
New-Item -ItemType Directory -Force $buildDir | Out-Null

$cmakeArgs = @(
    "-S", $InstallDir,
    "-B", $buildDir,
    "-DGGML_RPC=ON",
    "-DCMAKE_BUILD_TYPE=Release"
)

if ($useCuda) {
    $cmakeArgs += @(
        "-DGGML_CUDA=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
    )
}

if (-not $SkipBuild) {
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-Fail "CMake configure failed"; exit 1 }
    Write-Ok "CMake configured"

    # ---------------------------------------------------------------
    # Step 4: Build
    # ---------------------------------------------------------------
    Write-Step "Building (this may take 10-20 minutes on first run)"

    $cores = [Environment]::ProcessorCount
    & cmake --build $buildDir --config Release --parallel $cores `
            --target llama-server rpc-server

    if ($LASTEXITCODE -ne 0) { Write-Fail "Build failed"; exit 1 }
    Write-Ok "Build complete"
} else {
    Write-Ok "Skipping build (--SkipBuild)"
}

# ---------------------------------------------------------------
# Step 5: Verify binaries
# ---------------------------------------------------------------
Write-Step "Verifying binaries"

$binDir   = "$buildDir\bin\Release"
$serverEx = "$binDir\llama-server.exe"
# rpc-server was renamed from llama-rpc-server in llama.cpp >= b4000
$rpcEx = if (Test-Path "$binDir\rpc-server.exe") { "$binDir\rpc-server.exe" } else { "$binDir\llama-rpc-server.exe" }

if (Test-Path $serverEx) {
    Write-Ok "llama-server.exe: $serverEx"
} else {
    Write-Fail "llama-server.exe not found at $serverEx"
    exit 1
}

if (Test-Path $rpcEx) {
    Write-Ok "rpc-server.exe: $rpcEx"
} else {
    Write-Warn "rpc-server.exe not found -- RPC offload disabled"
}

# ---------------------------------------------------------------
# Step 6: Add to PATH (current session + permanent)
# ---------------------------------------------------------------
Write-Step "Adding to PATH"

$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$binDir*") {
    [Environment]::SetEnvironmentVariable(
        "PATH", "$binDir;$currentPath", "User"
    )
    $env:PATH = "$binDir;$env:PATH"
    Write-Ok "Added $binDir to User PATH"
} else {
    Write-Ok "Already in PATH"
}

# ---------------------------------------------------------------
# Step 7: Quick smoke test
# ---------------------------------------------------------------
Write-Step "Smoke test"

try {
    $ver = & "$serverEx" --version 2>&1 | Select-Object -First 1
    Write-Ok "Version: $ver"
} catch {
    Write-Warn "Could not run version check: $_"
}

# ---------------------------------------------------------------
# Done
# ---------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  [OK] llama.cpp installed with CUDA + RPC" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Run as RPC compute node (this machine):" -ForegroundColor White
Write-Host "  rpc-server --host 0.0.0.0 --port 50052" -ForegroundColor Yellow
Write-Host ""
Write-Host "Run as inference server (with RPC backends):" -ForegroundColor White
Write-Host "  llama-server --model models\llama-3.1-8b-Q4_K_M.gguf \`" -ForegroundColor Yellow
Write-Host "    --rpc 192.168.1.100:50052 \`" -ForegroundColor Yellow
Write-Host "    --n-gpu-layers 99 --port 8080" -ForegroundColor Yellow
Write-Host ""
Write-Host "Download a model (example):" -ForegroundColor White
Write-Host "  pip install huggingface-hub" -ForegroundColor Yellow
Write-Host "  huggingface-cli download bartowski/Llama-3.1-8B-Instruct-GGUF \" -ForegroundColor Yellow
Write-Host "    --include 'Llama-3.1-8B-Instruct-Q4_K_M.gguf' --local-dir models\" -ForegroundColor Yellow
Write-Host ""
Write-Host "Run orchestrator test:" -ForegroundColor White
Write-Host "  python test_connection.py" -ForegroundColor Yellow
