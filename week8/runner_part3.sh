#!/usr/bin/env bash
# Codex reviewed dispatch wrapper. Models/workers are authored through WSL Pi.
# Replaces the unused unsafe pgrep/pkill runner; prior version remains in Git.
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -lt 1 ]; then
  echo "Usage: bash runner_part3.sh <new-attempt-root> [workers<=4]" >&2
  exit 2
fi
exec python3 coordinate_part3.py --out "$1" --widths 8,12,16,24,32,48,64,96,128 --steps 1500 --workers "${2:-4}"
