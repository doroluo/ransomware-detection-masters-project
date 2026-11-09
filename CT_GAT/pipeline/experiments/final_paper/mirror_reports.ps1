param([Parameter(Mandatory=$true)][string]$SourceRoot,
      [Parameter(Mandatory=$true)][string]$DestinationRoot)
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $SourceRoot).Path
$destination = [IO.Path]::GetFullPath($DestinationRoot)
if (-not $destination.Contains('\outputs\')) { throw 'Report destination must be an outputs directory' }
New-Item -ItemType Directory -Path $destination -Force | Out-Null
foreach ($name in @('STATUS.md','STATUS.json','EXPERIMENT_PLAN.md','PLAN.json','SUMMARY.md','RESULTS.csv','PER_FOLD_SUMMARY.csv','OOF_SUMMARY.csv','CI_SUMMARY.csv')) {
    $path = Join-Path $source $name
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        Copy-Item -LiteralPath $path -Destination (Join-Path $destination $name) -Force
    }
}
$figures = Join-Path $source 'figures'
if (Test-Path -LiteralPath $figures -PathType Container) {
    Copy-Item -LiteralPath $figures -Destination $destination -Recurse -Force
}
