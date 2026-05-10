"""
Custom Agent: TTT Training with JSONL Step Logging

Runs `python3 -m tinker_cookbook.recipes.ttt.train` as a subprocess,
tails the resulting actions.jsonl file, and reports each action as
a structured step via webhooks.
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from base_agent import BaseAgent


class TTTJSONLProcessor:
    def __init__(self, jsonl_path):
        self.path = Path(jsonl_path)
        self._offset = 0

    def get_new_actions(self, process_finished=False):
        actions = []
        with open(self.path, "r") as f:
            f.seek(self._offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                actions.append(json.loads(line))
            self._offset = f.tell()
        return actions


class DiscoverAgent(BaseAgent):
    """Discover agent that runs TTT training and logs steps from actions.jsonl."""

    def _prepare_dataset(self, problem_idx: str, env: str, src_dir: str, python_bin: str):
        """Pull and prepare dataset for the selected MLE problem if not already set up."""
        if env != "mle_bench":
            print(f"Skipping dataset prep for env={env} (only mle_bench needs it)")
            return

        setup_script = Path(src_dir) / "tasks" / "mle_bench" / "setup_competition.py"
        if not setup_script.exists():
            print(f"WARNING: setup_competition.py not found at {setup_script}")
            return

        # Check if competition data is already prepared by inspecting the registry
        check_cmd = [
            python_bin, "-c",
            f"from mlebench.registry import registry; "
            f"c = registry.get_competition('{problem_idx}'); "
            f"pub = list(c.public_dir.iterdir()) if c.public_dir.exists() else []; "
            f"priv = list(c.private_dir.iterdir()) if c.private_dir.exists() else []; "
            f"print(f'public={{len(pub)}} private={{len(priv)}}')"
        ]
        result = subprocess.run(
            check_cmd, capture_output=True, text=True, cwd=src_dir
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"Dataset check for '{problem_idx}': {output}")
            # If both dirs have files, data is already prepared
            if "public=0" not in output and "private=0" not in output:
                print(f"Dataset for '{problem_idx}' already prepared, skipping download")
                return
        else:
            print(f"Dataset check failed (mlebench may not be installed yet): {result.stderr.strip()}")

        # Download from Kaggle
        print(f"Downloading dataset for '{problem_idx}' from Kaggle...")
        tmp_data_dir = Path(tempfile.mkdtemp(prefix=f"kaggle_{problem_idx}_"))

        kaggle_cmd = [
            "kaggle", "competitions", "download",
            "-c", problem_idx,
            "-p", str(tmp_data_dir),
        ]
        dl_result = subprocess.run(kaggle_cmd, capture_output=True, text=True)

        if dl_result.returncode != 0:
            print(f"ERROR: Kaggle download failed: {dl_result.stderr.strip()}")
            print("Make sure 'kaggle' CLI is installed and ~/.kaggle/kaggle.json is configured")
            shutil.rmtree(tmp_data_dir, ignore_errors=True)
            return

        print(f"Downloaded to {tmp_data_dir}")

        # Unzip if kaggle downloaded a zip file
        for zf in tmp_data_dir.glob("*.zip"):
            print(f"Unzipping {zf.name}...")
            subprocess.run(["unzip", "-o", "-q", str(zf), "-d", str(tmp_data_dir)])
            zf.unlink()

        # Run setup_competition.py to prepare public/private splits
        setup_cmd = [
            python_bin, str(setup_script),
            "-c", problem_idx,
            "--data-dir", str(tmp_data_dir),
            "--skip-checksums",
        ]
        print(f"Running setup: {' '.join(setup_cmd)}")
        setup_result = subprocess.run(
            setup_cmd, capture_output=True, text=True, cwd=src_dir
        )

        if setup_result.returncode == 0:
            print(f"Dataset setup complete for '{problem_idx}'")
            if setup_result.stdout:
                print(setup_result.stdout.strip())
        else:
            print(f"ERROR: Dataset setup failed: {setup_result.stderr.strip()}")
            if setup_result.stdout:
                print(setup_result.stdout.strip())

        # Cleanup temp dir
        shutil.rmtree(tmp_data_dir, ignore_errors=True)

    async def start(self):
        print(f"Discover agent starting for experiment {self.experiment_id}")

        config = self.agent_config
        env = config.get("env", "mle_bench")
        problem_idx = config.get("problem_idx", "ventilator-pressure-prediction")
        model_name = config.get("model_name", "openai/gpt-oss-120b")
        sampler_type = config.get("sampler_type", "greedy")
        initial_exp_type = config.get("initial_exp_type", "random")
        group_size = config.get("group_size", 8)
        groups_per_batch = config.get("groups_per_batch", 16)
        learning_rate = config.get("learning_rate", "4e-5")
        num_epochs = config.get("num_epochs", 25)
        eval_timeout = config.get("eval_timeout", 300)
        num_cpus_per_task = config.get("num_cpus_per_task", 2)
        wandb_project = config.get("wandb_project", "discover-ttt")
        wandb_name = config.get("wandb_name", "mle")
        load_checkpoint_path = config.get("load_checkpoint_path")

        # Resolve src dir and python binary
        src_dir = config.get("src_dir", "/app/src")
        if not Path(src_dir).exists():
            src_dir = str(Path(__file__).parent / "src")

        venv_python = Path(src_dir) / ".venv" / "bin" / "python3"
        python_bin = str(venv_python) if venv_python.exists() else "python3"

        # Pull and prepare dataset if needed
        self._prepare_dataset(problem_idx, env, src_dir, python_bin)

        log_path = config.get("log_path")
        if not log_path:
            log_path = str(Path(src_dir).parent / "logs" / wandb_name)

        cmd = [
            python_bin, "-m", "tinker_cookbook.recipes.ttt.train",
            f"env={env}",
            f"problem_idx={problem_idx}",
            f"model_name={model_name}",
            f"sampler_type={sampler_type}",
            f"initial_exp_type={initial_exp_type}",
            f"group_size={group_size}",
            f"groups_per_batch={groups_per_batch}",
            f"learning_rate={learning_rate}",
            f"num_epochs={num_epochs}",
            f"eval_timeout={eval_timeout}",
            f"num_cpus_per_task={num_cpus_per_task}",
            f"wandb_project={wandb_project}",
            f"wandb_name={wandb_name}",
            f"log_path={log_path}",
        ]

        if load_checkpoint_path:
            cmd.append(f"load_checkpoint_path={load_checkpoint_path}")

        for k, v in config.get("extra_overrides", {}).items():
            cmd.append(f"{k}={v}")

        actions_jsonl = Path(log_path) / "actions.jsonl"

        print(f"Configuration:")
        print(f"  env: {env}")
        print(f"  problem_idx: {problem_idx}")
        print(f"  model_name: {model_name}")
        print(f"  num_epochs: {num_epochs}")
        print(f"  group_size: {group_size}")
        print(f"  groups_per_batch: {groups_per_batch}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  wandb: {wandb_project}/{wandb_name}")
        print(f"  src_dir: {src_dir}")
        print(f"  python: {python_bin}")
        print(f"  actions_jsonl: {actions_jsonl}")

        await self.send_initial_messages(
            system_message=f"Running TTT training: {env}/{problem_idx}",
            user_message=f"Model: {model_name}, Epochs: {num_epochs}, LR: {learning_rate}",
            step_number=0,
        )

        print(f"\nRunning: {' '.join(cmd)}")
        print(f"Working dir: {src_dir}\n")

        proc_env = {**os.environ}
        tinker_key = self.api_keys.get("TINKER_API_KEY", "")
        if tinker_key:
            proc_env["TINKER_API_KEY"] = tinker_key

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=src_dir,
            env=proc_env,
        )

        step_counter = 1
        processor = None

        while True:
            if self.is_aborted:
                print("Aborting training...")
                process.terminate()
                break

            # Read stdout (non-blocking: readline returns "" at EOF)
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())

            # Create processor once actions.jsonl appears
            if processor is None and actions_jsonl.exists():
                processor = TTTJSONLProcessor(actions_jsonl)
                print(f"Found {actions_jsonl}, tailing for steps...")

            # Poll for new actions from the JSONL file
            if processor:
                new_actions = processor.get_new_actions(process_finished=False)
                for action in new_actions:
                    action_id = action["action_id"] or f"action_{self.experiment_id}_{step_counter}"
                    thought = action["full_response"] or action["parsed_code"] or ""
                    reward = action["reward"]
                    status = action["status"]
                    result = action["result"] or ""
                    verify_out = action["verify_output"] or ""

                    observation = f"reward={reward} status={status}"
                    if result:
                        observation += f"\nresult: {result}"
                    if verify_out:
                        observation += f"\nverify: {verify_out}"

                    best = action["best_value_so_far"]
                    if best is not None:
                        observation += f"\nbest_so_far: {best}"
                    if action["is_new_best"]:
                        observation += "\n** NEW BEST **"

                    await self.send_action_received(
                        step_number=step_counter,
                        action_id=action_id,
                        action_type="ttt_action",
                        action_message={
                            "role": "assistant",
                            "content": thought[:10000],
                            "tool_calls": None,
                            "completion_details": None,
                        },
                    )
                    await self.send_step_finished(
                        step_number=step_counter,
                        action_id=action_id,
                        action_type="ttt_action",
                        observation_content=observation[:5000],
                        tool_call_id=f"call_{step_counter}",
                    )

                    step_counter += 1
                    self.current_step = step_counter

            await asyncio.sleep(0.5)

        # Process remaining lines after subprocess exits
        if processor:
            remaining = processor.get_new_actions(process_finished=True)
            for action in remaining:
                action_id = action["action_id"] or f"action_{self.experiment_id}_{step_counter}"
                thought = action["full_response"] or action["parsed_code"] or ""
                reward = action["reward"]
                status = action["status"]
                result = action["result"] or ""
                verify_out = action["verify_output"] or ""

                observation = f"reward={reward} status={status}"
                if result:
                    observation += f"\nresult: {result}"
                if verify_out:
                    observation += f"\nverify: {verify_out}"

                best = action["best_value_so_far"]
                if best is not None:
                    observation += f"\nbest_so_far: {best}"
                if action["is_new_best"]:
                    observation += "\n** NEW BEST **"

                await self.send_action_received(
                    step_number=step_counter,
                    action_id=action_id,
                    action_type="ttt_action",
                    action_message={
                        "role": "assistant",
                        "content": thought[:10000],
                        "tool_calls": None,
                        "completion_details": None,
                    },
                )
                await self.send_step_finished(
                    step_number=step_counter,
                    action_id=action_id,
                    action_type="ttt_action",
                    observation_content=observation[:5000],
                    tool_call_id=f"call_{step_counter}",
                )

                step_counter += 1
                self.current_step = step_counter

        return_code = process.wait()
        success = return_code == 0
        summary = f"TTT training {'completed' if success else 'failed'} for {env}/{problem_idx}"
        print(f"\n{summary} (exit code: {return_code})")

        if success:
            await self.send_experiment_completed(
                success=True,
                summary=summary,
                approach=f"model={model_name}, epochs={num_epochs}, lr={learning_rate}",
                step_number=self.current_step,
            )
        else:
            await self.send_experiment_failed(
                error_message=f"Training exited with code {return_code}",
                step_number=self.current_step,
            )

        print("Discover agent finished!")

    async def continue_agent(
        self,
        user_message: str,
        new_max_steps: int,
        step_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ):
        print(f"Received user message: {user_message}")
        self.max_steps = new_max_steps

        if step_number is not None:
            self.current_step = step_number
        else:
            self.current_step += 1

        await self.send_step_finished(
            step_number=self.current_step,
            user_message=user_message,
            git_branch=branch_name,
        )

        print("Continuation not implemented, marking experiment as completed")
        await self.send_experiment_completed(
            success=True,
            summary="Agent continuation not implemented",
            approach="TTT training",
            step_number=self.current_step,
        )

    async def abort(self):
        print("Aborting discover agent...")
        self.is_aborted = True
        await self.send_experiment_aborted(
            reason="Aborted by user",
            last_step=self.current_step,
        )


def create_agent(
    experiment_id: str,
    project_id: str,
    problem_statement: str,
    max_steps: int,
    api_keys: Dict[str, str],
    webhook_url: Optional[str] = None,
    agent_config: Dict[str, Any] = None,
    jwt_token: str = None,
) -> BaseAgent:
    """Create and return the agent instance."""
    return DiscoverAgent(
        experiment_id=experiment_id,
        project_id=project_id,
        problem_statement=problem_statement,
        max_steps=max_steps,
        api_keys=api_keys,
        webhook_url=webhook_url,
        agent_config=agent_config,
        jwt_token=jwt_token,
    )
