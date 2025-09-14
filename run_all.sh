#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%Y%m%d_%H%M%S)"
out="results_${ts}.out"

{
  echo "===== ENV INFO ====="
  echo "DATE: $(date)"
  echo "HOST: $(hostname)"
  echo "PYTHON: $(python -V 2>&1)"
  python - <<'PY'
import torch, triton, sys
print("TORCH:", torch.__version__)
print("TRITON:", getattr(triton, "__version__", "unknown"))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY
  echo "GIT HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
  echo

  echo "===== RUNNING AUTOGRADER (P1-P7) ====="
  python autograder.py --p1 --p2 --p3 --p4 --p5 --p6 --p7 || true
  echo

  echo "===== RUNNING OPTIONAL AUTOGRADER (P8-P9) ====="
  python autograder_optional.py --p8 --p9 || true
  echo

  echo "===== DONE ====="
} 2>&1 | tee "$out"

echo
echo "All done: output saved to $out"