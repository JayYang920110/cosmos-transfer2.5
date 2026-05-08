#!/usr/bin/env python3
"""Sweep script for cat-in-the-sun: grid over control_weight x creg x p."""

import json
import subprocess
import tempfile
from pathlib import Path

BASE_JSON = Path("/workspace/test/cat/cat-in-the-sun/cat.json")
OUTPUT_BASE = Path("outputs/mytest/cat/transfer")

CONTROL_WEIGHTS = [0.2]
CREGS = [3, 5, 7, 10]
PS = [5, 2]

with open(BASE_JSON) as f:
    base_config = json.load(f)

jobs = [
    (cw, creg, p)
    for cw in CONTROL_WEIGHTS
    for creg in CREGS
    for p in PS
]

print(f"Total combinations: {len(jobs)}")

for i, (cw, creg, p) in enumerate(jobs, 1):
    tag = f"cw{cw}_creg{creg}_p{p}"
    output_dir = OUTPUT_BASE / tag
    print(f"\n[{i}/{len(jobs)}] {tag}")

    config = json.loads(json.dumps(base_config))  # deep copy
    config["vis"]["control_weight"] = cw
    config["crossattn_logit_boost"]["creg"] = float(creg)
    config["crossattn_logit_boost"]["p"] = float(p)

    tmp_path = BASE_JSON.parent / f"_sweep_{tag}.json"
    try:
        with open(tmp_path, "w") as f:
            json.dump(config, f, indent=4)

        cmd = [
            "python", "examples/inference.py",
            "-i", str(tmp_path),
            "-o", str(output_dir),
        ]
        print("  cmd:", " ".join(cmd))
        result = subprocess.run(cmd, cwd="/workspace")
        if result.returncode != 0:
            print(f"  [FAILED] returncode={result.returncode}")
        else:
            print(f"  [OK] -> {output_dir}")
    finally:
        tmp_path.unlink(missing_ok=True)

print("\nSweep complete.")
