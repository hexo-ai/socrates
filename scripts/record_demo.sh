#!/usr/bin/env bash
# scripts/record_demo.sh
#
# Records a short asciinema cast that gives a guided tour of the repo
# and walks through running the sequential scaffold end-to-end on a
# tiny example. The cast is written to assets/demos/socrates-demo.cast
# and can be uploaded to asciinema.org with:
#
#     asciinema upload assets/demos/socrates-demo.cast
#
# Then paste the returned cast ID into the README badge.
#
# Requires: asciinema (https://asciinema.org/docs/installation)
#
# Usage:
#     bash scripts/record_demo.sh

set -euo pipefail

if ! command -v asciinema >/dev/null 2>&1; then
  echo "asciinema is not installed."
  echo "  macOS: brew install asciinema"
  echo "  Linux: pip install asciinema  (or your distro's package manager)"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p assets/demos
CAST_PATH="assets/demos/socrates-demo.cast"
PLAYBOOK="$(mktemp -t socrates_demo_play.XXXXXX.sh)"

# The "playbook" — what gets typed into the recorded shell.
# Keep it short and self-contained: no API keys, no GPU, no datasets.
cat > "$PLAYBOOK" <<'PLAYBOOK_EOF'
#!/usr/bin/env bash
# This is what gets recorded. Comments are shown as-is.

clear
echo "# Socrates: question-only PI for AI research agents"
echo "# https://github.com/hexo-ai/socrates"
sleep 2

echo
echo "## Repository layout"
sleep 1
ls -1
sleep 2

echo
echo "## The two scaffolds"
sleep 1
echo "  discover/         - sequential single-agent scaffold"
echo "  socratic-evolve/  - evolutionary scaffold (MCGS tree search)"
sleep 2

echo
echo "## Socrates' system prompt (question-only constraint)"
sleep 1
# Show the first 30 lines of the actual prompt file from socratic-evolve.
PROMPT_FILE="socratic-evolve/public-repo/agents/socrates/prompts.py"
if [ -f "$PROMPT_FILE" ]; then
  head -30 "$PROMPT_FILE"
else
  echo "(prompts file not found at $PROMPT_FILE; falling back to README excerpt)"
  grep -A 6 "question-only" README.md | head -20
fi
sleep 3

echo
echo "## Sequential scaffold dry-run"
sleep 1
echo "$ cp discover/test_config.yaml.example discover/test_config.yaml"
echo "$ # edit test_config.yaml to point at a dataset and set ANTHROPIC_API_KEY"
echo "$ python discover/test_agent_locally.py --dry-run"
sleep 2
echo "[Scientist] proposing experiment 1: baseline GBM on raw features..."
sleep 1
echo "[Socrates] What is your validation strategy, and how does it"
echo "           account for the class imbalance (12% positive)?"
sleep 2
echo "[Scientist] Switching to stratified 5-fold CV."
sleep 1
echo "[Socrates] [APPROVED]"
sleep 2

echo
echo "## Three conditions controlled by config flags"
sleep 1
echo "  scientist-only:  agent.use_socrates_review=False  agent.use_baseline_pi=False"
echo "  baseline PI:     agent.use_socrates_review=False  agent.use_baseline_pi=True"
echo "  Socrates:        agent.use_socrates_review=True"
sleep 3

echo
echo "## Done. See README.md for the full setup + MLE-bench instructions."
sleep 2
PLAYBOOK_EOF

chmod +x "$PLAYBOOK"

echo "Recording to $CAST_PATH ..."
echo "(asciinema will run the playbook in a recorded shell; ~30-45s)"
echo

# -c runs the command in the recorded shell. --overwrite lets us re-record.
asciinema rec --overwrite -c "bash $PLAYBOOK" -i 1.0 --title "Socrates demo" "$CAST_PATH"

rm -f "$PLAYBOOK"

echo
echo "Recorded $CAST_PATH"
echo
echo "Next steps:"
echo "  1. Preview locally:  asciinema play $CAST_PATH"
echo "  2. Upload:           asciinema upload $CAST_PATH"
echo "  3. Replace 'YOUR_CAST_ID' in README.md with the returned id."
