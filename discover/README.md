# Discover — Sequential Scaffold Agent

Sequential single-agent implementation for the Socrates paper. A single agent writes and executes experiments one at a time with no built-in exploration mechanism. The agent must decide what to try next based solely on its own reasoning (or, with PI enabled, on the output of the Socratic review loop).

## Files

- `base_agent.py` — Base class providing the webhook protocol (INITIAL_MESSAGES, ACTION_RECEIVED, STEP_FINISHED, EXPERIMENT_COMPLETED, etc.). Do not modify.
- `models.py` — Pydantic models for webhook payloads. Do not modify.
- `custom_agent.py` — Agent implementation (`DiscoverAgent`). Runs TTT training via subprocess, reports progress via webhooks.
- `schema_ttt.json` — Default `agent_config` values for the platform.
- `test_agent_locally.py` — Local testing harness; reads `test_config.yaml` if present.
- `test_config.yaml.example` — Template for local testing (copy to `test_config.yaml` and fill in values).

## Setup

```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and edit test config
cp test_config.yaml.example test_config.yaml
# Set ANTHROPIC_API_KEY and other fields

# Run locally
python test_agent_locally.py
```

## Agent Structure

The agent implements three required methods on `BaseAgent`:

- `start()` — main entry point; spawns training subprocess, reports steps
- `continue_agent()` — handles user feedback mid-experiment
- `abort()` — stops the agent cleanly

Factory function `create_agent()` is the runtime entry point.

## Webhook Protocol

Events sent during a run (saved to `webhooks/*.json` in local testing):

1. `INITIAL_MESSAGES` — system/user prompts at start
2. `ACTION_RECEIVED` — before each action executes
3. `STEP_FINISHED` — after each action completes
4. `EXPERIMENT_COMPLETED` — on success
5. `EXPERIMENT_FAILED` — on failure
6. `EXPERIMENT_ABORTED` — on user abort

See `base_agent.py` for detailed method signatures.

## Available `self.*` Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.experiment_id` | `str` | Unique experiment identifier |
| `self.project_id` | `str` | Project identifier |
| `self.problem_statement` | `str` | Task description |
| `self.max_steps` | `int` | Maximum steps allowed |
| `self.api_keys` | `dict` | API keys (auto-populated from env vars) |
| `self.agent_config` | `dict` | Hyperparameters from schema |
| `self.current_step` | `int` | Current step number |
| `self.is_aborted` | `bool` | Whether abort was requested |
