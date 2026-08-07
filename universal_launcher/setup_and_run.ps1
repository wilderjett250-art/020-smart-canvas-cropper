<#
.SYNOPSIS
    SmartCanvasCropper command-line environment checker, installer, and launcher.

.DESCRIPTION
    The script uses the portable Python bundled with SmartCanvasCropper. It does
    not install or modify system Python and does not require administrator rights.

.PARAMETER Action
    doctor      Read-only checks for the package, computer, GPU, disk, and sources.
    prepare     Install and validate the selected inference backend.
    run         Run a validated cached backend; if none exists, prepare CPU.
    prepare-run Prepare the backend and then launch SmartCanvasCropper.

.PARAMETER Mode
    auto, nvidia, directml, or cpu. Auto selects CUDA for NVIDIA, DirectML for
    AMD/Intel, and downloaded CPU runtime when no supported GPU is detected.

.PARAMETER LocalRoot
    Optional local cache root. The default is
    %LOCALAPPDATA%\SmartCanvasCropper\v1.9.2.

.PARAMETER ForceReinstall
    Rebuild the selected GPU runtime. The existing runtime is renamed as a
    timestamped backup instead of being deleted.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\tools\setup_and_run.ps1 -Action doctor

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\tools\setup_and_run.ps1 -Action prepare-run -Mode auto
#>
[CmdletBinding()]
param(
    [ValidateSet('doctor', 'prepare', 'run', 'prepare-run')]
    [string]$Action = 'prepare-run',

    [ValidateSet('auto', 'nvidia', 'directml', 'cpu')]
    [string]$Mode = 'auto',

    [string]$LocalRoot,

    [switch]$ForceReinstall
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$script:ProductVersion = '1.9.2'
$script:RuntimeSchema = 'portable-py3119-cuda271-dml241-cpu271-ultralytics8437-r6'
$script:PrimaryDetectorSha256 = '946E70DE8D586E84F1BB98CAAB8C094E3A8CF501E4807888166299439A67CBA3'
$script:RecoveryDetectorSha256 = '9915A40C0AF95E45B38FAB9A20678E7A089914959205239A713A34958599455B'
$script:SegmenterSha256 = '6DBB90523A35330FEDD7F1D3DFC66F995213D81B29A5CA8108DBCDD4E37D6C2F'
$script:CommonIndexes = @(
    'https://pypi.org/simple',
    'https://mirrors.aliyun.com/pypi/simple',
    'https://pypi.tuna.tsinghua.edu.cn/simple'
)
$script:NvidiaIndexes = @(
    'https://download.pytorch.org/whl/cu126',
    'https://mirrors.nju.edu.cn/pytorch/whl/cu126',
    'https://mirror.sjtu.edu.cn/pytorch-wheels/cu126'
)
$script:CpuIndexes = @(
    'https://download.pytorch.org/whl/cpu',
    'https://mirrors.nju.edu.cn/pytorch/whl/cpu',
    'https://mirror.sjtu.edu.cn/pytorch-wheels/cpu'
)

$script:PackageRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$script:AppRoot = Join-Path $script:PackageRoot 'app'
$script:PythonBase = Join-Path $script:PackageRoot 'python-base'
$script:MainExe = Join-Path $script:PackageRoot 'SmartCanvasCropper.exe'
$script:AppScript = Join-Path $script:AppRoot 'smart_canvas_cropper.py'

if ([string]::IsNullOrWhiteSpace($LocalRoot)) {
    if (-not [string]::IsNullOrWhiteSpace($env:SMART_CROPPER_LOCAL_ROOT)) {
        $LocalRoot = [Environment]::ExpandEnvironmentVariables($env:SMART_CROPPER_LOCAL_ROOT.Trim())
    }
    else {
        $LocalRoot = Join-Path $env:LOCALAPPDATA 'SmartCanvasCropper\v1.9.2'
    }
}
$script:LocalRoot = [IO.Path]::GetFullPath($LocalRoot)
$script:LogPath = $null
$script:TranscriptStarted = $false

function Write-Section([string]$Text) {
    Write-Host ''
    Write-Host ('=== ' + $Text + ' ===') -ForegroundColor Cyan
}

function Write-Okay([string]$Text) {
    Write-Host ('[OK] ' + $Text) -ForegroundColor Green
}

function Write-WarningLine([string]$Text) {
    Write-Host ('[WARN] ' + $Text) -ForegroundColor Yellow
}

function Write-Failure([string]$Text) {
    Write-Host ('[ERROR] ' + $Text) -ForegroundColor Red
}

function Format-ByteSize([Int64]$Bytes) {
    $units = @('B', 'KB', 'MB', 'GB', 'TB')
    $value = [double]$Bytes
    $unit = 0
    while ($value -ge 1024 -and $unit -lt ($units.Count - 1)) {
        $value /= 1024
        $unit++
    }
    return ('{0:N2} {1}' -f $value, $units[$unit])
}

function Test-VirtualAdapterName([string]$Name) {
    return $Name -match '(?i)virtual|todesk|gameviewer|oray|idd.*driver|remote|mirror|indirect|basic display|parsec'
}

function Get-PhysicalMemoryBytes {
    try {
        return [Int64](Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
    }
    catch {
        try {
            return [Int64](Get-WmiObject Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
        }
        catch {
            return [Int64]0
        }
    }
}

function Get-HardwareInfo {
    $adapters = @()
    try {
        $adapters = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | ForEach-Object {
            [pscustomobject]@{
                Name = [string]$_.Name
                DriverVersion = [string]$_.DriverVersion
                IsVirtual = Test-VirtualAdapterName ([string]$_.Name)
            }
        })
    }
    catch {
        try {
            $adapters = @(Get-WmiObject Win32_VideoController -ErrorAction Stop | ForEach-Object {
                [pscustomobject]@{
                    Name = [string]$_.Name
                    DriverVersion = [string]$_.DriverVersion
                    IsVirtual = Test-VirtualAdapterName ([string]$_.Name)
                }
            })
        }
        catch {
            Write-WarningLine ('无法通过系统接口读取显卡：' + $_.Exception.Message)
        }
    }

    $physicalNames = @($adapters | Where-Object { -not $_.IsVirtual } | ForEach-Object { $_.Name })
    $hasNvidia = @($physicalNames | Where-Object { $_ -match '(?i)NVIDIA' }).Count -gt 0
    $hasAmd = @($physicalNames | Where-Object { $_ -match '(?i)AMD|Radeon|Advanced Micro Devices' }).Count -gt 0
    $hasIntel = @($physicalNames | Where-Object { $_ -match '(?i)Intel.*(Graphics|UHD|Iris|Arc|HD)' }).Count -gt 0

    if (-not [string]::IsNullOrWhiteSpace($env:SMART_CROPPER_TEST_GPU)) {
        $override = $env:SMART_CROPPER_TEST_GPU.Trim().ToLowerInvariant()
        Write-WarningLine ('检测到测试显卡覆盖值：' + $override)
        $hasNvidia = $override -eq 'nvidia'
        $hasAmd = $override -eq 'amd'
        $hasIntel = $override -eq 'intel'
    }

    $preferred = 'cpu'
    if ($hasNvidia) { $preferred = 'nvidia' }
    elseif ($hasAmd -or $hasIntel) { $preferred = 'directml' }

    return [pscustomobject]@{
        Adapters = $adapters
        HasNvidia = $hasNvidia
        HasAmd = $hasAmd
        HasIntel = $hasIntel
        PreferredMode = $preferred
    }
}

function Resolve-Mode($Hardware) {
    if ($Mode -ne 'auto') { return $Mode }
    return $Hardware.PreferredMode
}

function Get-RuntimeRoot([string]$RuntimeMode) {
    return Join-Path $script:LocalRoot ('runtime-' + $RuntimeMode)
}

function Get-MarkerPath([string]$RuntimeMode) {
    return Join-Path (Get-RuntimeRoot $RuntimeMode) 'runtime-ready.txt'
}

function Get-ExpectedMarker([string]$RuntimeMode) {
    return $script:RuntimeSchema + '|' + $RuntimeMode
}

function Test-Payload {
    Write-Section '发布包完整性'
    $required = @(
        $script:MainExe,
        (Join-Path $script:PythonBase 'python.exe'),
        (Join-Path $script:PythonBase 'pythonw.exe'),
        (Join-Path $script:PythonBase 'DLLs\_tkinter.pyd'),
        (Join-Path $script:PythonBase 'Lib\site-packages\pip'),
        $script:AppScript,
        (Join-Path $script:AppRoot 'yolov8l-worldv2-canvas.pt'),
        (Join-Path $script:AppRoot 'yolov8s-worldv2-canvas.pt'),
        (Join-Path $script:AppRoot 'mobile_sam.pt')
    )
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($missing.Count -gt 0) {
        foreach ($path in $missing) { Write-Failure ('缺少：' + $path) }
        throw '发布包不完整。请完整解压 ZIP，不要只复制 EXE。'
    }

    $primaryDetector = Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $script:AppRoot 'yolov8l-worldv2-canvas.pt')
    $recoveryDetector = Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $script:AppRoot 'yolov8s-worldv2-canvas.pt')
    $segmenter = Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $script:AppRoot 'mobile_sam.pt')
    if ($primaryDetector.Hash -ne $script:PrimaryDetectorSha256 -or
        $recoveryDetector.Hash -ne $script:RecoveryDetectorSha256 -or
        $segmenter.Hash -ne $script:SegmenterSha256) {
        throw '模型权重校验失败。请重新解压原始 ZIP。'
    }
    Write-Okay ('发布包完整，模型权重校验通过：' + $script:PackageRoot)
}

function Get-FreeSpaceInfo {
    $root = [IO.Path]::GetPathRoot($script:LocalRoot)
    try {
        $drive = New-Object IO.DriveInfo($root)
        if ($drive.IsReady) {
            return [pscustomobject]@{ Known = $true; Free = [Int64]$drive.AvailableFreeSpace; Root = $root }
        }
    }
    catch { }
    return [pscustomobject]@{ Known = $false; Free = [Int64]0; Root = $root }
}

function Test-SourceEndpoint([string]$Url, [int]$TimeoutMilliseconds = 5000) {
    $uri = New-Object Uri($Url)
    $client = New-Object Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($uri.Host, 443, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne($TimeoutMilliseconds, $false)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

function Show-Doctor($Hardware, [string]$SelectedMode, [switch]$SkipNetwork) {
    Write-Section '电脑环境'
    Write-Host ('产品版本：' + $script:ProductVersion)
    Write-Host ('Windows：' + [Environment]::OSVersion.VersionString)
    Write-Host ('64 位系统：' + [Environment]::Is64BitOperatingSystem)
    Write-Host ('进程架构：' + $env:PROCESSOR_ARCHITECTURE)
    $physicalMemory = Get-PhysicalMemoryBytes
    if ($physicalMemory -gt 0) { Write-Host ('内存：' + (Format-ByteSize $physicalMemory)) }
    else { Write-WarningLine '无法读取物理内存容量。' }
    if ($Hardware.Adapters.Count -eq 0) {
        Write-WarningLine '系统未返回显卡信息。'
    }
    else {
        foreach ($adapter in $Hardware.Adapters) {
            $kind = if ($adapter.IsVirtual) { '虚拟/远程，忽略' } else { '物理显卡' }
            Write-Host ('显卡：{0} | 驱动：{1} | {2}' -f $adapter.Name, $adapter.DriverVersion, $kind)
        }
    }
    Write-Host ('请求模式：' + $Mode)
    Write-Host ('选定后端：' + $SelectedMode)
    Write-Host ('缓存目录：' + $script:LocalRoot)

    $space = Get-FreeSpaceInfo
    if ($space.Known) {
        Write-Host ('可用空间：' + (Format-ByteSize $space.Free) + ' @ ' + $space.Root)
    }
    else {
        Write-WarningLine ('无法读取缓存盘可用空间：' + $space.Root)
    }

    Write-Section '缓存状态'
    foreach ($runtimeMode in @('nvidia', 'directml', 'cpu')) {
        $root = Get-RuntimeRoot $runtimeMode
        $marker = Get-MarkerPath $runtimeMode
        $markerOkay = (Test-Path -LiteralPath $marker) -and ((Get-Content -LiteralPath $marker -Raw -Encoding UTF8).Trim() -eq (Get-ExpectedMarker $runtimeMode))
        if ($markerOkay -and (Test-Path -LiteralPath (Join-Path $root 'python.exe'))) {
            Write-Okay ($runtimeMode + ' 缓存标记有效：' + $root)
        }
        elseif (Test-Path -LiteralPath $root) {
            Write-WarningLine ($runtimeMode + ' 缓存存在但未通过标记检查：' + $root)
        }
        else {
            Write-Host ('[INFO] ' + $runtimeMode + ' 尚未安装。')
        }
    }
    Write-Section '依赖源连通性'
    if ($SkipNetwork) {
        Write-Host '[INFO] 当前操作不需要下载，已跳过依赖源探测。'
        return
    }
    Write-Host '每个源最多等待 5 秒。'
    $sources = @($script:CommonIndexes)
    if ($SelectedMode -eq 'nvidia') { $sources = @($script:NvidiaIndexes) + $sources }
    elseif ($SelectedMode -eq 'cpu') { $sources = @($script:CpuIndexes) + $sources }
    foreach ($source in $sources) {
        if (Test-SourceEndpoint $source) { Write-Okay $source }
        else { Write-WarningLine ('无法连接：' + $source) }
    }
}

function Start-SetupTranscript {
    New-Item -ItemType Directory -Force -Path (Join-Path $script:LocalRoot 'logs') | Out-Null
    $script:LogPath = Join-Path (Join-Path $script:LocalRoot 'logs') ('setup-cli-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
    try {
        Start-Transcript -LiteralPath $script:LogPath -Force | Out-Null
        $script:TranscriptStarted = $true
        Write-Host ('日志：' + $script:LogPath)
    }
    catch {
        Write-WarningLine ('无法启动命令行日志记录：' + $_.Exception.Message)
    }
}

function Set-IsolatedEnvironment {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONNOUSERSITE = '1'
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
    $env:PIP_CONFIG_FILE = 'nul'
    $env:PIP_DEFAULT_TIMEOUT = '30'
    $env:PYTHONPATH = $script:AppRoot
    $env:SMART_CROPPER_LOG_DIR = Join-Path $script:LocalRoot 'logs'
    $env:TORCH_HOME = Join-Path $script:LocalRoot 'torch-cache'
    $env:MPLCONFIGDIR = Join-Path $script:LocalRoot 'matplotlib-cache'
    $env:ULTRALYTICS_SETTINGS_DIR = Join-Path $script:LocalRoot 'ultralytics-settings'
    $env:SMART_CROPPER_MODEL_DIR = $script:AppRoot
    Remove-Item Env:PIP_INDEX_URL -ErrorAction SilentlyContinue
    Remove-Item Env:PIP_EXTRA_INDEX_URL -ErrorAction SilentlyContinue
    Remove-Item Env:PIP_NO_INDEX -ErrorAction SilentlyContinue
    Remove-Item Env:PIP_FIND_LINKS -ErrorAction SilentlyContinue
}

function Invoke-PythonChecked([string]$Python, [string[]]$Arguments, [string]$Description) {
    Write-Host ('[RUN] ' + $Description) -ForegroundColor Cyan
    & $Python @Arguments | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw ($Description + '失败，退出代码：' + $exitCode)
    }
}

function Invoke-PipWithFallback(
    [string]$Python,
    [string[]]$Packages,
    [string[]]$Indexes,
    [string]$Description,
    [switch]$PreRelease
) {
    $pipExitCode = -1
    for ($index = 0; $index -lt $Indexes.Count; $index++) {
        $source = $Indexes[$index]
        Write-Section ($Description + '（源 ' + ($index + 1) + '/' + $Indexes.Count + '）')
        Write-Host ('依赖源：' + $source)
        Write-Host '网络连接/读取超时：30 秒；失败后自动切换下一个源。'
        Write-Host ('安装包：' + ($Packages -join ', '))

        $arguments = @(
            '-m', 'pip', '--isolated', 'install',
            '--disable-pip-version-check', '--no-cache-dir',
            '--progress-bar', 'on', '--only-binary=:all:',
            '--no-warn-script-location', '--retries', '1', '--timeout', '30'
        )
        if ($PreRelease) { $arguments += '--pre' }
        $arguments += $Packages
        $arguments += @('--index-url', $source)

        & $Python @arguments | Out-Host
        $pipExitCode = $LASTEXITCODE
        if ($pipExitCode -eq 0) {
            Write-Okay ($Description + '完成。')
            return
        }
        Write-WarningLine ('当前依赖源失败，退出代码 ' + $pipExitCode + '，正在切换。')
    }
    throw ($Description + '失败，所有依赖源均不可用；最后退出代码：' + $pipExitCode)
}

function Backup-ExistingRuntime([string]$RuntimeRoot) {
    if (-not (Test-Path -LiteralPath $RuntimeRoot)) { return }
    $parent = Split-Path -Parent $RuntimeRoot
    $leaf = Split-Path -Leaf $RuntimeRoot
    $backup = Join-Path $parent ($leaf + '-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    Write-WarningLine ('现有环境将保留为备份：' + $backup)
    Move-Item -LiteralPath $RuntimeRoot -Destination $backup
}

function Assert-FreeSpace([string]$RuntimeMode) {
    $space = Get-FreeSpaceInfo
    if (-not $space.Known) {
        Write-WarningLine '无法读取可用空间，将继续安装。'
        return
    }
    $required = if ($RuntimeMode -eq 'nvidia') { [Int64]6GB } else { [Int64]2GB }
    if ($space.Free -lt $required) {
        throw ('磁盘空间不足。' + $RuntimeMode + ' 模式至少需要 ' + (Format-ByteSize $required) + '，当前可用 ' + (Format-ByteSize $space.Free) + '。')
    }
    Write-Okay ('空间检查通过：可用 ' + (Format-ByteSize $space.Free))
}

function Get-RuntimeValidationCode([string]$RuntimeMode) {
    $expectedDevice = if ($RuntimeMode -eq 'nvidia') { 'cuda' } elseif ($RuntimeMode -eq 'directml') { 'privateuseone' } else { 'cpu' }
    return "import os,sys,torch,cv2,PIL,ultralytics; sys.stderr=None; os.environ['SMART_CROPPER_RUNTIME_MODE']='$RuntimeMode'; import smart_canvas_cropper as app; from ultralytics import YOLO,SAM; d,label=app.choose_compute_device(); tensor_device='cuda:0' if d==0 else d; v=torch.ones(1).to(tensor_device).cpu(); root=os.environ['SMART_CROPPER_MODEL_DIR']; primary=YOLO(os.path.join(root,'yolov8l-worldv2-canvas.pt')); recovery=YOLO(os.path.join(root,'yolov8s-worldv2-canvas.pt')); seg=SAM(os.path.join(root,'mobile_sam.pt')); assert getattr(primary.model,'txt_feats',None) is not None; assert getattr(recovery.model,'txt_feats',None) is not None; actual=str(tensor_device); print('python='+sys.version.split()[0]); print('torch='+torch.__version__); print('device='+actual); print('label='+label); print('opencv='+cv2.__version__); print('ultralytics='+ultralytics.__version__); print('models=preloaded'); ok=(actual.startswith('$expectedDevice') and float(v[0])==1.0); raise SystemExit(0 if ok else 4)"
}

function Test-Runtime([string]$RuntimeMode, [switch]$Quiet) {
    $runtimeRoot = Get-RuntimeRoot $RuntimeMode
    $python = Join-Path $runtimeRoot 'python.exe'
    $marker = Get-MarkerPath $RuntimeMode
    if (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $marker)) { return $false }
    if ((Get-Content -LiteralPath $marker -Raw -Encoding UTF8).Trim() -ne (Get-ExpectedMarker $RuntimeMode)) { return $false }

    Set-IsolatedEnvironment
    $code = Get-RuntimeValidationCode $RuntimeMode
    if (-not $Quiet) { Write-Host ('正在实际验证 ' + $RuntimeMode + ' 推理设备与模型预加载…') }
    & $python '-c' $code | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        if (-not $Quiet) { Write-Okay ($RuntimeMode + ' 环境、推理设备与模型预加载验证通过。') }
        return $true
    }
    if (-not $Quiet) { Write-WarningLine ($RuntimeMode + ' 环境验证失败，退出代码：' + $exitCode) }
    return $false
}

function Prepare-Runtime([string]$RuntimeMode) {
    Write-Section ('准备 ' + $RuntimeMode + ' 推理环境')
    Assert-FreeSpace $RuntimeMode
    $runtimeRoot = Get-RuntimeRoot $RuntimeMode

    if (-not $ForceReinstall -and (Test-Runtime $RuntimeMode)) {
        Write-Okay ('复用已验证环境：' + $runtimeRoot)
        return $RuntimeMode
    }

    $existingPython = Join-Path $runtimeRoot 'python.exe'
    if (-not $ForceReinstall -and (Test-Path -LiteralPath $existingPython)) {
        Write-Host ('发现未写入就绪标记的现有环境，正在先验证并恢复：' + $runtimeRoot)
        if (Test-RuntimeWithoutMarker $RuntimeMode $existingPython) {
            [IO.File]::WriteAllText((Get-MarkerPath $RuntimeMode), (Get-ExpectedMarker $RuntimeMode), (New-Object Text.UTF8Encoding($false)))
            if (-not (Test-Runtime $RuntimeMode -Quiet)) { throw ($RuntimeMode + ' 恢复后的缓存验证失败。') }
            Write-Okay ('现有环境验证通过并已恢复就绪标记：' + $runtimeRoot)
            return $RuntimeMode
        }
        Write-WarningLine ($RuntimeMode + ' 现有环境未通过验证，将重新构建。')
    }

    if (Test-Path -LiteralPath $runtimeRoot) {
        Backup-ExistingRuntime $runtimeRoot
    }
    New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
    Write-Host ('正在从发布包复制便携 Python：' + $runtimeRoot)
    Get-ChildItem -LiteralPath $script:PythonBase -Force | Copy-Item -Destination $runtimeRoot -Recurse -Force
    $python = Join-Path $runtimeRoot 'python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw '便携 Python 复制失败。' }

    Set-IsolatedEnvironment
    Invoke-PythonChecked $python @('-c', "import sys,pip,tkinter; print('python='+sys.version.split()[0]); print('pip='+pip.__version__); print('tk='+str(tkinter.TkVersion))") '检查便携 Python'
    Invoke-PipWithFallback $python @('pip==25.2') $script:CommonIndexes '更新安装组件'

    if ($RuntimeMode -eq 'nvidia') {
        Invoke-PipWithFallback $python @('torch==2.7.1', 'torchvision==0.22.1') $script:NvidiaIndexes '安装 NVIDIA CUDA 推理组件'
    }
    elseif ($RuntimeMode -eq 'directml') {
        Invoke-PipWithFallback $python @('torch-directml==0.2.5.dev240914') $script:CommonIndexes '安装 AMD / Intel DirectML 推理组件' -PreRelease
    }
    else {
        Invoke-PipWithFallback $python @('torch==2.7.1', 'torchvision==0.22.1') $script:CpuIndexes '安装 CPU 推理组件'
    }
    Invoke-PipWithFallback $python @(
        'ultralytics==8.4.37',
        'opencv-python==4.11.0.86',
        'pillow==11.3.0',
        'tqdm==4.67.1'
    ) $script:CommonIndexes '安装图像处理组件'

    if (-not (Test-RuntimeWithoutMarker $RuntimeMode $python)) {
        throw ($RuntimeMode + ' 组件已安装，但实际推理设备验证失败。')
    }
    [IO.File]::WriteAllText((Get-MarkerPath $RuntimeMode), (Get-ExpectedMarker $RuntimeMode), (New-Object Text.UTF8Encoding($false)))
    if (-not (Test-Runtime $RuntimeMode -Quiet)) { throw ($RuntimeMode + ' 最终缓存验证失败。') }
    Write-Okay ('环境准备并验证完成：' + $runtimeRoot)
    return $RuntimeMode
}

function Test-RuntimeWithoutMarker([string]$RuntimeMode, [string]$Python) {
    Set-IsolatedEnvironment
    $code = Get-RuntimeValidationCode $RuntimeMode
    Write-Host ('正在执行 ' + $RuntimeMode + ' 张量设备与模型预加载验证…')
    & $Python '-c' $code | Out-Host
    return $LASTEXITCODE -eq 0
}

function Start-Cropper([string]$RuntimeMode) {
    Set-IsolatedEnvironment
    $env:SMART_CROPPER_LOCAL_ROOT = $script:LocalRoot
    $env:SMART_CROPPER_RUNTIME_MODE = $RuntimeMode
    $env:SMART_CROPPER_MODEL_DIR = $script:AppRoot

    $pythonw = Join-Path (Get-RuntimeRoot $RuntimeMode) 'pythonw.exe'
    if (-not (Test-Path -LiteralPath $pythonw)) { throw ($RuntimeMode + ' 的 pythonw.exe 不存在。') }
    Write-Okay ('正在启动，推理后端：' + $RuntimeMode)
    Start-Process -FilePath $pythonw -ArgumentList ('"' + $script:AppScript + '"') -WorkingDirectory $script:AppRoot
}

function Get-AutoFallbackModes([string]$SelectedMode) {
    if ($Mode -ne 'auto') { return @($SelectedMode) }
    if ($SelectedMode -eq 'nvidia') { return @('nvidia', 'directml', 'cpu') }
    if ($SelectedMode -eq 'directml') { return @('directml', 'cpu') }
    return @('cpu')
}

$exitCode = 0
try {
    [Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
    Write-Host ('SmartCanvasCropper v' + $script:ProductVersion + ' 命令行配置工具') -ForegroundColor White
    Write-Host ('操作：' + $Action + ' | 模式：' + $Mode)
    Test-Payload
    $hardware = Get-HardwareInfo
    $selectedMode = Resolve-Mode $hardware

    if ($Action -eq 'doctor') {
        Show-Doctor $hardware $selectedMode
        Write-Section '结论'
        Write-Okay ('建议后端：' + $selectedMode + '。doctor 只做检查，没有修改电脑环境。')
    }
    else {
        Start-SetupTranscript
        $skipNetwork = $Action -eq 'run'
        if ($Action -eq 'prepare-run') {
            $cachedPython = Join-Path (Get-RuntimeRoot $selectedMode) 'python.exe'
            $cachedMarker = Get-MarkerPath $selectedMode
            $skipNetwork = (Test-Path -LiteralPath $cachedPython) -and
                (Test-Path -LiteralPath $cachedMarker) -and
                ((Get-Content -LiteralPath $cachedMarker -Raw -Encoding UTF8).Trim() -eq (Get-ExpectedMarker $selectedMode))
        }
        Show-Doctor $hardware $selectedMode -SkipNetwork:$skipNetwork
        $preparedMode = $null

        if ($Action -eq 'run') {
            foreach ($candidate in (Get-AutoFallbackModes $selectedMode)) {
                if (Test-Runtime $candidate) {
                    $preparedMode = $candidate
                    break
                }
            }
            if ($null -eq $preparedMode) {
                Write-WarningLine '没有找到已验证缓存，正在准备 CPU 兼容环境。'
                $preparedMode = Prepare-Runtime 'cpu'
            }
        }
        else {
            foreach ($candidate in (Get-AutoFallbackModes $selectedMode)) {
                try {
                    $preparedMode = Prepare-Runtime $candidate
                    break
                }
                catch {
                    Write-Failure ($candidate + ' 环境准备失败：' + $_.Exception.Message)
                    if ($Mode -ne 'auto') { throw }
                    Write-WarningLine '自动模式将尝试下一种后端。'
                }
            }
            if ($null -eq $preparedMode) {
                throw '所有候选推理环境均未能准备完成，请查看上方日志。'
            }
        }

        if ($Action -eq 'run' -or $Action -eq 'prepare-run') {
            Start-Cropper $preparedMode
            Write-Okay ('程序已启动，实际后端：' + $preparedMode)
        }
        else {
            Write-Okay ('配置完成，已验证后端：' + $preparedMode)
        }
        if ($script:LogPath) { Write-Host ('详细日志：' + $script:LogPath) }
    }
}
catch {
    $exitCode = 1
    Write-Failure $_.Exception.Message
    if ($script:LogPath) { Write-Host ('详细日志：' + $script:LogPath) }
}
finally {
    if ($script:TranscriptStarted) {
        try { Stop-Transcript | Out-Null } catch { }
    }
}

exit $exitCode





