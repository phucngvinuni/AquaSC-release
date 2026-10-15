#!/usr/bin/env bash
# scp latency over ONE persistent SSH connection (ssh -M / ControlPath), as in RESULTS_OF_RECORD.md §6af.
# Run from the sender machine (WSL on the laptop). Password is asked once, when the master connection opens.
#   bash net_latency_scp.sh USER@HOST [TRIALS=20]
# Timed per trial: `scp payload user@host:/tmp/nl_scp/payload.bin` through the control socket (wall clock, ms).
# Floor: `ssh -S sock host true` (one command round trip through the same socket).
set -euo pipefail
DEST=${1:?usage: net_latency_scp.sh user@host [trials]}; TRIALS=${2:-20}
SOCK=/tmp/nl_scp_ctl.$$; WORK=$(mktemp -d); trap 'ssh -S "$SOCK" -O exit "$DEST" 2>/dev/null || true; rm -rf "$WORK"' EXIT
ssh -o ConnectTimeout=10 -M -S "$SOCK" -fN "$DEST"
ssh -S "$SOCK" "$DEST" 'mkdir -p /tmp/nl_scp'
now() { date +%s%N; }
stats() { sort -n | awk '{a[NR]=$1; s+=$1} END {m=s/NR; for(i=1;i<=NR;i++) v+=(a[i]-m)^2; sd=(NR>1)?sqrt(v/(NR-1)):0;
  printf "n=%d mean=%.2f std=%.2f p50=%.2f p95=%.2f min=%.2f max=%.2f\n", NR, m, sd, a[int((NR-1)*0.5)+1], a[int((NR-1)*0.95+0.999)+1], a[1], a[NR]}'; }
echo "ssh_true (floor, through control socket):"
for i in $(seq 1 $TRIALS); do t0=$(now); ssh -S "$SOCK" "$DEST" true; t1=$(now); awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.2f\n",(b-a)/1e6}'; done | stats
for N in 385 49152 1048576; do
  head -c $N /dev/urandom > "$WORK/p$N.bin"
  for i in 1 2 3; do scp -q -o ControlPath="$SOCK" "$WORK/p$N.bin" "$DEST:/tmp/nl_scp/p.bin"; done   # warm-up
  echo "scp $N B:"
  for i in $(seq 1 $TRIALS); do t0=$(now); scp -q -o ControlPath="$SOCK" "$WORK/p$N.bin" "$DEST:/tmp/nl_scp/p.bin"; t1=$(now); awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.2f\n",(b-a)/1e6}'; done | stats
done
ssh -S "$SOCK" "$DEST" 'rm -rf /tmp/nl_scp'
