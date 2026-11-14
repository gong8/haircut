#!/bin/sh
set -eu
mkdir -p /logs/verifier
python3 - <<'PY'
import json
from pathlib import Path

score = Path('/var/lib/haircut/run-6.json')
if score.exists():
    metrics = json.loads(score.read_text())
    reward = {name: float(metrics[name]) for name in ('reward', 'nicks', 'steps', 'on_target', 'done')}
else:
    reward = {'reward': 0.0, 'nicks': 0.0, 'steps': 0.0, 'on_target': 0.0, 'done': 0.0}
Path('/logs/verifier/reward.json').write_text(json.dumps(reward) + '\n')
PY
