param(
    [string]$Destination = (Join-Path $PSScriptRoot '../../reports/ida_cfg_20261006/full'),
    [string]$VmOut = '/home/seed/work/out/ida_cfg_20261006',
    [int]$MaxHours = 168
)
$ErrorActionPreference = 'Stop'
$key = Join-Path $env:USERPROFILE '.ssh/vm_transfer'
$ssh = (Get-Command ssh.exe).Source
$scp = (Get-Command scp.exe).Source
$python = (Get-Command python.exe).Source
$exporter = Join-Path $PSScriptRoot 'vm_ida_cfg.py'
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$Destination = (Resolve-Path -LiteralPath $Destination).Path
$statePath = Join-Path $Destination 'transfer_status.json'
$deadline = (Get-Date).AddHours($MaxHours)
$connectionFailures = 0
function Save-State([string]$State, [string]$Detail) {
    @{ state=$State; detail=$Detail; updated=(Get-Date).ToString('o'); watcher_pid=$PID } |
        ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
}
try {
    Save-State 'waiting' 'Waiting for the VM to finish extraction and validate its export package.'
    while ((Get-Date) -lt $deadline) {
        $receiptText = & $ssh -i $key -o BatchMode=yes -o ConnectTimeout=10 -p 2222 seed@127.0.0.1 "if test -f $VmOut/pipeline_failure.json; then echo PIPELINE_FAILED; elif test -f $VmOut/transfer_ready.json; then cat $VmOut/transfer_ready.json; else echo PENDING; fi" 2>&1
        if ($LASTEXITCODE -ne 0) {
            $connectionFailures++
            if ($connectionFailures -ge 60) { throw 'VM unavailable for 60 consecutive polls; restart receiver after restoring access.' }
            Start-Sleep -Seconds 60
            continue
        }
        $connectionFailures = 0
        if (($receiptText -join '').Trim() -eq 'PIPELINE_FAILED') { throw 'VM pipeline failed; inspect pipeline_failure.json and batch.log in the VM task folder.' }
        if (($receiptText -join '').Trim() -eq 'PENDING') {
            Start-Sleep -Seconds 60
            continue
        }
        $receipt = ($receiptText -join "`n") | ConvertFrom-Json
        if ($receipt.sha256 -notmatch '^[0-9a-f]{64}$' -or $receipt.package -ne "$VmOut/safe_cfg_export.tar") {
            throw 'Unexpected VM transfer receipt.'
        }
        Save-State 'transferring' "Receiving $($receipt.graphs) validated CFG files."
        $archive = Join-Path $Destination 'safe_cfg_export.tar'
        $partial = "$archive.partial"
        & $scp -i $key -o BatchMode=yes -o ConnectTimeout=10 -P 2222 "seed@127.0.0.1:$VmOut/safe_cfg_export.tar" $partial
        if ($LASTEXITCODE -ne 0) { throw 'SCP failed; no partial package was opened.' }
        if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $receipt.sha256) {
            throw 'Package SHA-256 verification failed.'
        }
        Move-Item -LiteralPath $partial -Destination $archive -Force
        Save-State 'validating' 'Validating every checksum, graph schema and group path on the desktop.'
        & $python $exporter verify --archive $archive --destination $Destination
        if ($LASTEXITCODE -ne 0) { throw 'Desktop package validation failed.' }
        $receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Destination 'transfer_receipt.json') -Encoding utf8
        Save-State 'complete' "Verified and grouped $($receipt.graphs) CFG files. See summary.json for counts and failures."
        exit 0
    }
    throw 'Seven-day waiting limit reached; restart receiver if the VM job is still running.'
} catch {
    Save-State 'error' $_.Exception.Message
    throw
}
