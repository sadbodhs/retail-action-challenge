#!/usr/bin/env bash
# Local scoreboard: final local val mAP of every training run, then our Kaggle submissions
# and the top of the public leaderboard.
#   scripts/scoreboard.sh            # runs + Kaggle
#   scripts/scoreboard.sh --local    # runs only (no network)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
COMP=retail-action-challenge-praw-workshop-wacv-2027

echo "== local runs (val mAP, our metric) =="
scripts/docker_run.sh python - <<'EOF'
import glob, json, os
rows = []
for f in sorted(glob.glob("runs/*/log.json")):
    log = json.load(open(f))
    vals = [r["val_map"] for r in log if "val_map" in r]
    rows.append((vals[-1] if vals else float("nan"), max(vals) if vals else float("nan"), os.path.basename(os.path.dirname(f)), len(log)))
print(f"{'run':<14}{'epochs':>7}{'final':>8}{'best':>8}")
for final, best, name, ep in sorted(rows, reverse=True):
    print(f"{name:<14}{ep:>7}{final:>8.4f}{best:>8.4f}")
EOF

[ "${1:-}" = "--local" ] && exit 0
echo
echo "== our Kaggle submissions =="
scripts/docker_run.sh kaggle competitions submissions -c "$COMP" < /dev/null 2>/dev/null | cut -c1-200
echo
echo "== public leaderboard (top 10) =="
scripts/docker_run.sh kaggle competitions leaderboard -c "$COMP" --show < /dev/null 2>/dev/null | head -12
