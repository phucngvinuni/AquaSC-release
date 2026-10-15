# Network-latency sender (run on the laptop, Windows PowerShell). One persistent TCP connection to the PC receiver.
# Measures time from start of send until the 1-byte ack arrives (== uplink transfer + one return packet).
#   powershell -ExecutionPolicy Bypass -File net_latency_sender.ps1 -Server PC_IP -Trials 20
param([string]$Server = "PC_IP", [int]$Port = 5201, [int]$Trials = 20, [int[]]$Sizes = @(1, 385, 49152, 1048576))
$c = New-Object System.Net.Sockets.TcpClient; $c.NoDelay = $true; $c.Connect($Server, $Port)
$st = $c.GetStream(); $ack = New-Object byte[] 1; $rng = New-Object System.Random 42
$out = @()
foreach ($n in $Sizes) {
  $p = New-Object byte[] $n; $rng.NextBytes($p)
  $h = [BitConverter]::GetBytes([uint32]$n); [Array]::Reverse($h)
  for ($w = 0; $w -lt 3; $w++) { $st.Write($h, 0, 4); $st.Write($p, 0, $n); [void]$st.Read($ack, 0, 1) }   # warmup
  $t = @()
  for ($i = 0; $i -lt $Trials; $i++) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $st.Write($h, 0, 4); $st.Write($p, 0, $n); [void]$st.Read($ack, 0, 1)
    $sw.Stop(); $t += $sw.Elapsed.TotalMilliseconds
  }
  $s = $t | Sort-Object
  $mean = ($t | Measure-Object -Average).Average
  $sd = [Math]::Sqrt((($t | ForEach-Object { ($_ - $mean) * ($_ - $mean) }) | Measure-Object -Sum).Sum / ($t.Count - 1))
  $row = [pscustomobject]@{bytes=$n; trials=$Trials; mean_ms=[Math]::Round($mean,2); std_ms=[Math]::Round($sd,2);
    p50_ms=[Math]::Round($s[[int][Math]::Floor(($Trials-1)*0.5)],2); p95_ms=[Math]::Round($s[[int][Math]::Ceiling(($Trials-1)*0.95)],2);
    min_ms=[Math]::Round($s[0],2); max_ms=[Math]::Round($s[-1],2)}
  $out += $row; $row | Format-Table -HideTableHeaders -AutoSize | Out-String | Write-Host
}
$c.Close(); $out | Format-Table -AutoSize; $out | Export-Csv -NoTypeInformation net_latency_laptop_to_pc.csv
Write-Host "wrote net_latency_laptop_to_pc.csv"
