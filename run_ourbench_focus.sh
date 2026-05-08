#!/bin/bash
# Run inference.py on every OurBench folder's .json and save under outputs/FOCUS/<folder>/
# Usage: bash run_ourbench_focus.sh [extra inference args...]
# Example with custom model: bash run_ourbench_focus.sh --model vis

set -euo pipefail

OURBENCH="/workspace/OurBench"
OUTPUT_ROOT="/workspace/outputs/Short_FOCUS"
PYTHON="/workspace/.venv/bin/python"
SCRIPT="/workspace/examples/inference.py"
LOG_FILE="${OUTPUT_ROOT}/run_summary.log"

mkdir -p "$OUTPUT_ROOT"

# Add folder names to skip here (one per line)
SKIP_LIST=(
    Car-Ball-Soccer
    Cycle-Ball-Match
)

ok=0
fail=0
fail_list=()
skip=0

for folder in "$OURBENCH"/*/; do
    [[ -d "$folder" ]] || continue
    name=$(basename "$folder")
    json="${folder}${name}.json"

    # Check skip list
    for s in "${SKIP_LIST[@]}"; do
        if [[ "$name" == "$s" ]]; then
            echo "[SKIP] $name — in skip list"
            ((skip++)) || true
            continue 2
        fi
    done

    if [[ ! -f "$json" ]]; then
        echo "[SKIP] $name — no json found"
        continue
    fi

    out_dir="${OUTPUT_ROOT}/${name}"
    mkdir -p "$out_dir"

    echo "========================================"
    echo "[RUN ] $name"
    echo "  json : $json"
    echo "  out  : $out_dir"

    if "$PYTHON" "$SCRIPT" \
            --input-files "$json" \
            --output-dir  "$out_dir" \
            --model vis \
            "$@" \
            2>&1 | tee "${out_dir}/inference.log"; then
        echo "[OK  ] $name"
        ((ok++)) || true
    else
        echo "[FAIL] $name  (exit $?)"
        fail_list+=("$name")
        ((fail++)) || true
    fi
done

echo ""
echo "========================================"
echo "DONE  success=${ok}  failure=${fail}  skipped=${skip}"
echo "========================================"

{
    echo "Run completed at $(date)"
    echo "Success : $ok"
    echo "Failure : $fail"
    echo "Skipped : $skip"
    if [[ ${#fail_list[@]} -gt 0 ]]; then
        echo "Failed cases:"
        for f in "${fail_list[@]}"; do
            echo "  - $f"
        done
    fi
} | tee "$LOG_FILE"
