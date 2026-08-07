$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$taskRoot = Join-Path (Split-Path -Parent $projectRoot) 'SmartCanvasCropper-final-20260802'
$packageName = 'SmartCanvasCropper-Windows-v1.9.2-Universal'
$stageRoot = Join-Path $taskRoot ($packageName + '-stage')
$finalRoot = Join-Path $taskRoot $packageName
$zipPath = Join-Path $taskRoot ($packageName + '.zip')
$seedRoot = if (Test-Path -LiteralPath $finalRoot) {
    $finalRoot
} else {
    Join-Path $taskRoot 'SmartCanvasCropper-Windows-v1.8.1-Universal'
}
$launcherSource = Join-Path $projectRoot 'universal_launcher\Program.cs'
$cliSource = Join-Path $projectRoot 'universal_launcher\setup_and_run.ps1'
$cmdSource = Join-Path $projectRoot 'universal_launcher\Setup-And-Run.cmd'
$appSource = Join-Path $projectRoot 'smart_canvas_cropper.py'
$csc = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'

function Assert-TaskPath([string]$Path) {
    $taskFull = [IO.Path]::GetFullPath($taskRoot).TrimEnd('\') + '\'
    $pathFull = [IO.Path]::GetFullPath($Path).TrimEnd('\') + '\'
    if (-not $pathFull.StartsWith($taskFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the confirmed task root: $Path"
    }
}

function Reset-GeneratedPath([string]$Path) {
    Assert-TaskPath $Path
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

foreach ($required in @(
    $seedRoot,
    $launcherSource,
    $cliSource,
    $cmdSource,
    $csc,
    $appSource,
    (Join-Path $projectRoot 'models\yolov8l-worldv2-canvas.pt'),
    (Join-Path $projectRoot 'models\yolov8s-worldv2-canvas.pt'),
    (Join-Path $projectRoot 'models\mobile_sam.pt'),
    (Join-Path $projectRoot 'universal_launcher\README-zh-CN.txt'),
    (Join-Path $projectRoot 'universal_launcher\runtime-manifest.txt'),
    (Join-Path $projectRoot 'universal_launcher\THIRD_PARTY_NOTICES.txt')
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing required build input: $required"
    }
}

$appSourceText = [IO.File]::ReadAllText($appSource, [Text.Encoding]::UTF8)
foreach ($forbidden in @('self.edge_trim', 'ttk.Spinbox', 'on_pointer_move', 'SMART_CROPPER_SEGMENT_REFINEMENT_MODE')) {
    if ($appSourceText.Contains($forbidden)) {
        throw "Expert tuning is not locked in the application source: $forbidden"
    }
}
foreach ($requiredSetting in @(
    'APP_VERSION = "1.9.2"',
    'EDGE_TRIM_PERCENT = 0.8',
    'SEGMENT_CACHE_POLICY = "reset"',
    'SEGMENT_REFINEMENT_MODE = "ranked"'
)) {
    if (-not $appSourceText.Contains($requiredSetting)) {
        throw "Locked production setting is missing: $requiredSetting"
    }
}

New-Item -ItemType Directory -Force -Path $taskRoot | Out-Null
Reset-GeneratedPath $stageRoot
Copy-Item -LiteralPath $seedRoot -Destination $stageRoot -Recurse

$entryCommands = @(Get-ChildItem -LiteralPath $stageRoot -Filter '*.cmd' -File)
if ($entryCommands.Count -ne 1) {
    throw ('Expected exactly one command entry in seed package, found ' + $entryCommands.Count + '.')
}
$entryCommand = $entryCommands[0].FullName
$entryCommandName = $entryCommands[0].Name

$appRoot = Join-Path $stageRoot 'app'
$sourceRoot = Join-Path $stageRoot 'source'
$toolsRoot = Join-Path $stageRoot 'tools'
Reset-GeneratedPath (Join-Path $stageRoot 'cpu-fallback')
Reset-GeneratedPath $appRoot
Reset-GeneratedPath $sourceRoot
Reset-GeneratedPath $toolsRoot
New-Item -ItemType Directory -Force -Path $appRoot,$sourceRoot,$toolsRoot | Out-Null

Copy-Item -LiteralPath $appSource -Destination $appRoot -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'models\yolov8l-worldv2-canvas.pt') -Destination $appRoot -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'models\yolov8s-worldv2-canvas.pt') -Destination $appRoot -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'models\mobile_sam.pt') -Destination $appRoot -Force

$launcherExe = Join-Path $stageRoot 'SmartCanvasCropper.exe'
& $csc /nologo /target:winexe /optimize+ /platform:x64 /out:$launcherExe `
    /reference:System.dll /reference:System.Core.dll /reference:System.Drawing.dll `
    /reference:System.Windows.Forms.dll /reference:System.Management.dll `
    /reference:System.Net.Http.dll $launcherSource
if ($LASTEXITCODE -ne 0) {
    throw "Launcher compilation failed with exit code $LASTEXITCODE."
}

Copy-Item -LiteralPath $cliSource -Destination (Join-Path $toolsRoot 'setup_and_run.ps1') -Force
Copy-Item -LiteralPath $cmdSource -Destination $entryCommand -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'universal_launcher\README-zh-CN.txt') -Destination $stageRoot -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'universal_launcher\runtime-manifest.txt') -Destination $stageRoot -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'universal_launcher\THIRD_PARTY_NOTICES.txt') -Destination $stageRoot -Force
Copy-Item -LiteralPath $launcherSource -Destination (Join-Path $sourceRoot 'Program.cs') -Force
Copy-Item -LiteralPath $cliSource -Destination (Join-Path $sourceRoot 'setup_and_run.ps1') -Force
Copy-Item -LiteralPath $MyInvocation.MyCommand.Path -Destination (Join-Path $sourceRoot 'build_final_v192.ps1') -Force

$primaryDetectorHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $appRoot 'yolov8l-worldv2-canvas.pt')).Hash
$recoveryDetectorHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $appRoot 'yolov8s-worldv2-canvas.pt')).Hash
$segmenterHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $appRoot 'mobile_sam.pt')).Hash
if ($primaryDetectorHash -ne '946E70DE8D586E84F1BB98CAAB8C094E3A8CF501E4807888166299439A67CBA3') {
    throw 'Primary detector weight hash mismatch in final stage.'
}
if ($recoveryDetectorHash -ne '9915A40C0AF95E45B38FAB9A20678E7A089914959205239A713A34958599455B') {
    throw 'Recovery detector weight hash mismatch in final stage.'
}
if ($segmenterHash -ne '6DBB90523A35330FEDD7F1D3DFC66F995213D81B29A5CA8108DBCDD4E37D6C2F') {
    throw 'Segmenter weight hash mismatch in final stage.'
}

$files = Get-ChildItem -LiteralPath $stageRoot -File -Recurse
$expandedBytes = ($files | Measure-Object -Property Length -Sum).Sum
$buildInfo = @(
    'Product=SmartCanvasCropper 1.9.2',
    ('PreferredEntry=' + $entryCommandName),
    'ReleaseChannel=Formal',
    'BackendPlan=NVIDIA CUDA | AMD/Intel DirectML | CPU on-demand fallback',
    'SetupFlow=doctor | prepare | run | prepare-run',
    'PythonDeployment=Bundled portable Python 3.11.9',
    'ExpertParameters=Locked',
    'EdgeTrimPercent=0.8',
    'SegmentRefinement=Ranked',
    'ManualCornerAdjustment=Disabled',
    ('FileCount=' + $files.Count),
    ('ExpandedBytes=' + $expandedBytes),
    ('PrimaryDetectorSHA256=' + $primaryDetectorHash),
    ('RecoveryDetectorSHA256=' + $recoveryDetectorHash),
    ('SegmenterSHA256=' + $segmenterHash),
    ('BuiltAt=' + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz'))
)
$buildInfoPath = Join-Path $stageRoot 'BUILD-INFO.txt'
$buildInfo | Set-Content -LiteralPath $buildInfoPath -Encoding utf8

$files = Get-ChildItem -LiteralPath $stageRoot -File -Recurse
$expandedBytes = ($files | Measure-Object -Property Length -Sum).Sum
$buildInfo = Get-Content -LiteralPath $buildInfoPath -Encoding utf8 | ForEach-Object {
    if ($_ -like 'FileCount=*') { 'FileCount=' + $files.Count }
    elseif ($_ -like 'ExpandedBytes=*') { 'ExpandedBytes=' + $expandedBytes }
    else { $_ }
}
$buildInfo | Set-Content -LiteralPath $buildInfoPath -Encoding utf8
$files = Get-ChildItem -LiteralPath $stageRoot -File -Recurse
$expandedBytes = ($files | Measure-Object -Property Length -Sum).Sum

Reset-GeneratedPath $finalRoot
Move-Item -LiteralPath $stageRoot -Destination $finalRoot
Reset-GeneratedPath $zipPath
Compress-Archive -LiteralPath $finalRoot -DestinationPath $zipPath -CompressionLevel Optimal

$zip = Get-Item -LiteralPath $zipPath
$zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash
[pscustomobject]@{
    Folder = $finalRoot
    Zip = $zip.FullName
    ZipBytes = $zip.Length
    ZipSHA256 = $zipHash
    ExpandedBytes = $expandedBytes
    Files = $files.Count
}
