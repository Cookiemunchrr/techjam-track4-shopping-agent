#!/usr/bin/env bash
# Usage: bash run_tests.sh [fast|all]
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-all}" = "fast" ]; then
  echo "== fast suite (no catalog load) =="
  python3 -m unittest tests.test_text tests.test_agent_contract tests.test_agent_robustness -v
else
  echo "== full suite =="
  python3 -m unittest discover -s tests -p 'test_*.py' -v
fi
