#!/usr/bin/env bash
# Usage:  bash analysis/run.sh [baseline|v2|probe] [dev|holdout|full]
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-v2}" in
  baseline) cp starter/agent_baseline.py starter/agent.py ;;
  v2)       cp analysis/agent_v2.py      starter/agent.py ;;
  probe)    cp analysis/agent_v1_probe.py starter/agent.py ;;
  *) echo "unknown agent: $1"; exit 1 ;;
esac
case "${2:-full}" in
  dev)     DS=analysis/dev.jsonl ;;
  holdout) DS=analysis/holdout.jsonl ;;
  *)       DS=data/public_set.jsonl ;;
esac
python3 -m evaluator.local_evaluator --dataset "$DS" --output "analysis/results_${1:-v2}_${2:-full}.json"
