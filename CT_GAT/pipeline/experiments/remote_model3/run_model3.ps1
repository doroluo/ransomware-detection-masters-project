[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('qualify', 'all', '3')]
    [string]$Mode = 'qualify',

    [Parameter(Position = 1)]
    [ValidateSet('signal250', 'signal500', 'signal750', 'signal1000',
                 'callwalk250', 'callwalk500', 'callwalk750', 'callwalk1000')]
    [string]$Dataset = 'signal250',

    [Parameter(Position = 2)]
    [ValidateSet('none', 'no_api_identity', 'no_behavior', 'shuffle_instructions', 'edge_free')]
    [string]$Ablation = 'none',

    [string]$HandoffRoot = $PSScriptRoot,
    [string]$RepoPath = $env:MODEL3_REPO,
    [string]$Python = $env:MODEL3_PYTHON
)

$ErrorActionPreference = 'Stop'
if (-not $RepoPath) { $RepoPath = Join-Path $HOME 'ransomware-diagnostics' }
if (-not $Python) { $Python = 'python' }
$expectedCodeCommit = '10a4e11a653c7d8ee8d632ddb91c511b9dfc91d6'
& git -C $RepoPath merge-base --is-ancestor $expectedCodeCommit HEAD
if ($LASTEXITCODE -ne 0) {
    throw 'Update the separate ransomware-diagnostics checkout to the revised Radeon handoff. The old paper checkout cannot run this launcher.'
}
# Pin the entire pipeline except this launcher and its documentation. This also
# rejects changes from later commits until a revised handoff explicitly allows them.
& git -C $RepoPath diff --quiet $expectedCodeCommit -- CT_GAT/pipeline ':(exclude)CT_GAT/pipeline/experiments/remote_model3/run_model3.ps1' ':(exclude)CT_GAT/pipeline/experiments/remote_model3/README.md'
if ($LASTEXITCODE -ne 0) { throw 'Pipeline sources differ from the qualified handoff code. Use a separate clean checkout.' }
& git -C $RepoPath diff --quiet HEAD -- CT_GAT/pipeline
if ($LASTEXITCODE -ne 0) { throw 'The pipeline has local edits. Use a clean checkout.' }
& $Python -c 'import torch, torch_geometric; assert torch.cuda.is_available(), "GPU-enabled PyTorch is required"; print("PyTorch", torch.__version__, "HIP", torch.version.hip, "GPU", torch.cuda.get_device_name(0), "PyG", torch_geometric.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'Activate the same GPU Python environment that passed the diagnostic.' }

$jobs = @()
if ($Mode -eq 'all') {
    # The original 12 reserved fold-3 jobs, with a full qualification per recipe.
    foreach ($name in @('signal250', 'signal500', 'signal750', 'signal1000',
                         'callwalk250', 'callwalk500', 'callwalk750', 'callwalk1000')) {
        $jobs += [pscustomobject]@{ Dataset = $name; Ablation = 'none' }
    }
    foreach ($name in @('no_api_identity', 'no_behavior', 'shuffle_instructions', 'edge_free')) {
        $jobs += [pscustomobject]@{ Dataset = 'signal500'; Ablation = $name }
    }
}
else {
    if ($Ablation -ne 'none' -and $Dataset -ne 'signal500') { throw 'Assigned ablations use signal500.' }
    $jobs += [pscustomobject]@{ Dataset = $Dataset; Ablation = $Ablation }
}

Push-Location (Join-Path $RepoPath 'CT_GAT/pipeline')
try {
    foreach ($job in $jobs) {
        $runArgs = @('-u', '-m', 'experiments.remote_model3.run',
                     '--handoff-root', $HandoffRoot, '--variant', $job.Dataset,
                     '--ablation', $job.Ablation)
        if ($Mode -eq 'qualify' -or $Mode -eq 'all') {
            & $Python @runArgs '--qualification'
            if ($LASTEXITCODE -ne 0) { throw "Full-epoch qualification failed (code $LASTEXITCODE). No full job was started for this recipe. Review its progress and log output." }
        }
        if ($Mode -ne 'qualify') {
            & $Python @runArgs
            if ($LASTEXITCODE -ne 0) { throw "Model 3 exited with code $LASTEXITCODE. Review its status before restarting. Saved epoch checkpoints remain available." }
        }
    }
}
finally { Pop-Location }
