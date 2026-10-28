#!/usr/bin/env bash
# Part 3 formal sweep runner (NOT executed as part of this handoff).
# Usage: bash runner_part3.sh <attempt-root> <n_workers<=4>
#
# - refuses ANY existing attempt root (interrupted attempts are preserved)
# - runs the 18 emitted worker commands (widths 8..128 x seeds 1,2) with at
#   most n_workers concurrent processes, each pinned to one distinct core
# - monitors TOTAL worker RSS live; records max observed into
#   <attempt-root>/rss-monitor.jsonl and aborts the whole sweep if >= 6000 MB
set -euo pipefail
cd "$(dirname "$0")"

ATTEMPT_ROOT="${1:?usage: runner_part3.sh <attempt-root> [n_workers]}"
WORKERS="${2:-4}"
CAP_MB=6000

if [ -e "$ATTEMPT_ROOT" ]; then
  echo "REFUSING: attempt root already exists: $ATTEMPT_ROOT" >&2
  exit 2
fi
mkdir -p "$ATTEMPT_ROOT"

uv run python -c "
import sweep, pathlib
cmds = sweep.build_worker_cmds([8,12,16,24,32,48,64,96,128],[1,2],
                               pathlib.Path('$ATTEMPT_ROOT'), n_workers=$WORKERS)
print('\n'.join(cmds))" > "$ATTEMPT_ROOT/worker-commands.txt"

monitor() {
  local max=0
  while pgrep -f "sweep.py --width" > /dev/null; do
    total=$(pgrep -f "sweep.py --width" | while read -r p; do
              awk '/VmRSS/{printf "%f\n", $2/1024}' /proc/$p/status 2>/dev/null
            done | awk '{s+=$1} END{print s+0}')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $total" >> "$ATTEMPT_ROOT/rss-monitor.jsonl"
    if awk -v t="$total" -v c="$CAP_MB" 'BEGIN{exit !(t>=c)}'; then
      echo "RSS CAP EXCEEDED: ${total} MB >= ${CAP_MB} MB - aborting sweep" >&2
      pkill -f "sweep.py --width" || true
      exit 3
    fi
    if awk -v t="$total" -v m="$max" 'BEGIN{exit !(t>m)}'; then max=$total; fi
    sleep 2
  done
  echo "$max"
}
monitor &
MON_PID=$!

# shellcheck disable=SC2013
cat "$ATTEMPT_ROOT/worker-commands.txt" | xargs -P "$WORKERS" -I{} bash -c '{}'
STATUS=$?

kill "$MON_PID" 2>/dev/null || true
echo "sweep finished with xargs status $STATUS; max total RSS:" \
     "$(tail -1 "$ATTEMPT_ROOT/rss-monitor.jsonl" 2>/dev/null || echo 'n/a')"
exit "$STATUS"
