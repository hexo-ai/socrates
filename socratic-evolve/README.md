# SocraticEvolve — Evolutionary Scaffold Agent

Evolutionary (MLevolve) scaffold for the Socrates paper. Wraps MLevolve — an agentic ML system that solves Kaggle-style competitions via Monte Carlo Graph Search (MCGS) with parallel branches, evolution stages (paradigm-shift mutations), and fusion stages (cross-branch solution merging).

## Structure

- `base_agent.py` — Base class with webhook protocol. Do not modify.
- `models.py` — Pydantic models for webhook payloads. Do not modify.
- `custom_agent.py` — `SocraticEvolveAgent` extends `BaseAgent` with a 4-step pipeline:
  1. `_prepare_input()` — resolves `data_dir`/`desc_file` from `agent_config`
  2. `_run_search()` — launches `public-repo/run.py` as a subprocess with OmegaConf CLI overrides
  3. `_collect_artifacts()` — copies `best_solution.py`, `journal.json`, `submission.csv` into `src/`
  4. Reports result via `send_experiment_completed()`
- `schema.json` — Default `agent_config` values.
- `test_agent_locally.py` — Local testing harness.
- `test_config.yaml.example` — Template for local testing.
- `public-repo/` — MLevolve core (see below).

## MLevolve core (`public-repo/`)

- `run.py` — Main entry point. Uses OmegaConf to merge `config/config.yaml` with CLI overrides.
- `config/config.yaml` — Default MLevolve config (search parameters, Socrates toggles, model settings).
- `engine/agent_search.py` — MCGS tree search: `step()`, `execute_deferred_node()`, cross-branch fusion.
- `engine/executor.py` — Code execution sandbox (Interpreter).
- `agents/` — Multi-agent subsystem:
  - `socrates/` — Question-only PI (system prompt + approval loop)
  - `evolution_agent.py` — Paradigm-shift mutations within a branch
  - `fusion_agent.py` — Merges solutions across branches
  - `coder.py`, `planner.py`, `debug.py`, `memory/` — core Scientist agents
- `llm/` — LLM client wrappers (`claude.py`, `gemini.py`).

## Setup

```bash
# Install agent-level deps
pip install -r requirements.txt

# Install MLevolve deps (three files, use --no-deps to avoid version conflicts)
pip install --no-deps -r public-repo/requirements_base.txt
pip install --no-deps -r public-repo/requirements_ml.txt
pip install --no-deps -r public-repo/requirements_domain.txt

# Copy and edit test config
cp test_config.yaml.example test_config.yaml

# Run locally
python test_agent_locally.py
```

## Configuration Flow

`agent_config` → `_get_config()` merges with defaults → passed as CLI args to `public-repo/run.py` → OmegaConf merges with `config/config.yaml`.

Key config fields: `exp_id`, `dataset_dir`, `model`, `steps`, `time_limit`, `initial_drafts`, `parallel_search_num`, `num_gpus`, `use_coldstart`, `use_diff_mode`, `use_socrates_review`, `socrates_max_rounds`.

## Switching Conditions

Three conditions from the paper are controlled via config flags:

- **Scientist-only**: `use_socrates_review: false` and `ENABLE_PI_A: false`
- **Baseline PI**: `ENABLE_PI_A: true`, standard (non-question-only) PI prompt
- **Socrates**: `use_socrates_review: true` and `ENABLE_PI_A: true`

## Output Directories

- `src/` — Generated artifacts (`best_solution.py`, `journal.json`, `submission.csv`)
- `webhooks/` — Local testing webhook JSON dumps (when `webhook_url` is `None`)
- `runs/` — MLevolve run directories (timestamped)
