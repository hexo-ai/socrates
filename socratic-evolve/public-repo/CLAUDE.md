# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

MLevolve — an agentic ML system that solves Kaggle-style competitions via Monte Carlo Graph Search (MCGS). Used as the evolutionary scaffold in the Socrates paper.

## Key Commands

```bash
# Local testing (uses test_config.yaml if present, otherwise interactive)
python3 ../test_agent_locally.py
python3 ../test_agent_locally.py --dataset data/my_dataset.csv
python3 ../test_agent_locally.py --interactive

# Install MLevolve deps (three separate files, use --no-deps)
pip3 install --no-deps -r requirements_base.txt
pip3 install --no-deps -r requirements_ml.txt
pip3 install --no-deps -r requirements_domain.txt

# Run directly
python run.py exp_id="statoil-iceberg-classifier-challenge" \
  agent.use_socrates_review=True \
  agent.steps=50
```

## Architecture

### Entry points

- `run.py` — Main entry point. Uses OmegaConf to merge `config/config.yaml` with CLI overrides. Orchestrates AgentSearch + Interpreter.
- `config/config.yaml` — Default MLevolve config. Override via CLI args.

### Core engine

- `engine/agent_search.py` — MCGS tree search agent: `step()`, `execute_deferred_node()`, cross-branch fusion.
- `engine/executor.py` — Code execution sandbox (Interpreter).

### Agent subsystem (`agents/`)

- `socrates/` — Question-only PI implementation
  - `prompts.py` — System prompt enforcing question-only constraint
  - `approval_loop.py` — Multi-round question → defense → `[APPROVED]` loop
  - `config.py` — Toggle flags (`ENABLE_PI_A`)
- `evolution_agent.py` — Paradigm-shift mutations within a branch
- `fusion_agent.py` — Cross-branch solution merging
- `planner.py`, `coder.py`, `debug.py`, `memory/` — core Scientist components

### LLM layer (`llm/`)

- `claude.py`, `gemini.py` — LLM client wrappers. API keys read from environment variables (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).

## Configuration Flow

CLI args → OmegaConf merge with `config/config.yaml` → passed to AgentSearch.

Key config fields:
- Data: `exp_id`, `dataset_dir`, `data_dir`, `desc_file`
- Model: `agent.code.model`, `agent.feedback.model`
- Search: `agent.search.parallel_search_num`, `agent.search.num_drafts`, `agent.search.max_debug_depth`
- Socrates: `agent.use_socrates_review`, `agent.socrates_max_rounds`
- Limits: `agent.steps`, `agent.time_limit`, `exec.timeout`

## Output

- `runs/<timestamp>_<exp_name>/` — timestamped run directory with journal, best solution, submissions.

## Rules

- API keys are injected via environment variables; never hardcode them.
- All generated artifacts go under `runs/`.
