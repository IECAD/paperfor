Param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)

$python = "python"
$phases = @("phase1_apple", "phase2_native", "phase3_ablation")
$commonArgs = if ($RunArgs) { $RunArgs } else { @() }

foreach ($phase in $phases) {
    Write-Host "=== Running $phase ==="
    & $python "-m" "new_bench_02.runner.main" "--phase" $phase @commonArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "=== Generating plots and statistics ==="
& $python "generate_all_plots.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python "analyze_statistics.py"
exit $LASTEXITCODE
