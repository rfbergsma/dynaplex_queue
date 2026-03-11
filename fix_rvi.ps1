$filepath = 'C:\Users\yc0127198\source\repos\dynaplex_queue\DynaPlexPrivate\src\executables\queue_rvi\queue_rvi.cpp'
$bytes = [System.IO.File]::ReadAllBytes($filepath)
$content = [System.Text.Encoding]::UTF8.GetString($bytes)
$lines = $content -split "`r`n"

# Replace lines 357-372 (0-indexed): old delta block
$t4 = "`t`t`t`t"
$t5 = "`t`t`t`t`t"
$t6 = "`t`t`t`t`t`t"

$newLines = @(
    "${t4}double delta = std::abs(g_star - g_prev);",
    "${t4}g_prev = g_star;",
    "",
    "${t4}h = std::move(h_new);",
    "",
    "${t4}if (iter % 200 == 0)",
    "${t5}std::cout << `"  iter `" << std::setw(5) << iter",
    "${t5}<< `"  g*=`" << std::setprecision(10) << g_star",
    "${t5}<< `"  delta_g=`" << std::setprecision(4) << delta << `"`n`";",
    "",
    "${t4}if (iter > 0 && delta < eps) {",
    "${t5}std::cout << `"  Converged at iter `" << iter",
    "${t6}<< `"  g* = `" << std::setprecision(12) << g_star << `"`n`";",
    "${t5}break;",
    "${t4}}"
)

# Verify lines 357-358 look right before replacing
Write-Host "Line 357: $($lines[357])"
Write-Host "Line 358: $($lines[358])"
Write-Host "Line 368: $($lines[368])"

$before = $lines[0..356]
$after = $lines[373..($lines.Length - 1)]
$result = ($before + $newLines + $after) -join "`r`n"

[System.IO.File]::WriteAllText($filepath, $result, [System.Text.Encoding]::UTF8)
Write-Host "Done. Total lines now: $(($result -split '`r`n').Length)"
