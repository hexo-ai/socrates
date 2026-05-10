"""
Custom Agent Implementation: socratic-evolve

Wraps MLEvolve (public-repo/) to run autonomous ML algorithm search.

Pipeline:
  1. Parse task payload
  2. Prepare MLEvolve config and workspace
  3. Run MLEvolve search via subprocess
  4. Collect results and save artifacts to src/
  5. Report results via webhooks
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from base_agent import BaseAgent

# Resolve paths relative to this file
AGENT_DIR = Path(__file__).resolve().parent
PUBLIC_REPO = AGENT_DIR / "public-repo"


def resolve_src_dir():
    """Return the output directory based on runtime environment."""
    home_src = Path.home() / "src"
    if home_src.exists() or os.environ.get("AGENT_RUNTIME"):
        return home_src
    return AGENT_DIR / "src"


class SocraticEvolveAgent(BaseAgent):
    """Runs MLEvolve search for a given ML task."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.src_dir = resolve_src_dir()
        self.src_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = None
        self.process = None

        print(f"📦 SocraticEvolveAgent initialized:")
        print(f"   experiment_id: {self.experiment_id}")
        print(f"   project_id: {self.project_id}")
        print(f"   max_steps: {self.max_steps}")
        print(f"   src_dir: {self.src_dir}")
        print(f"   AGENT_DIR: {AGENT_DIR}")
        print(f"   PUBLIC_REPO: {PUBLIC_REPO}")
        print(f"   PUBLIC_REPO exists: {PUBLIC_REPO.exists()}")
        print(f"   webhook_url: {self.webhook_url or '(None - saving locally)'}")
        print(f"   problem_statement: {self.problem_statement[:100]}...")
        api_key_status = {k: ('set' if v else 'empty') for k, v in self.api_keys.items()}
        print(f"   api_keys: {api_key_status}")
        print(f"   agent_config: {json.dumps(self.agent_config, indent=2, default=str)}")

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    async def start(self):
        print(f"\n🚀 SocraticEvolve agent starting for experiment {self.experiment_id}")
        print(f"=" * 70)

        # Dump resolved config upfront
        cfg = self._get_config()
        print(f"📊 Resolved MLEvolve configuration:")
        print(f"   Search: steps={cfg['steps']}, time_limit={cfg['time_limit']}, initial_drafts={cfg['initial_drafts']}, seed={cfg['seed']}")
        print(f"   Model: {cfg['model']} (feedback: {cfg['feedback_model']})")
        print(f"   Execution: parallel={cfg['parallel_search_num']}, timeout={cfg['exec_timeout']}, cpus={cfg['cpu_number']}, gpus={cfg['num_gpus']}")
        print(f"   Features: coldstart={cfg['use_coldstart']}, diff_mode={cfg['use_diff_mode']}, data_leakage_check={cfg['check_data_leakage']}, global_memory={cfg['use_global_memory']}")
        print(f"   Task: exp_id='{cfg['exp_id']}', dataset_dir='{cfg['dataset_dir']}', data_dir='{cfg['data_dir']}', desc_file='{cfg['desc_file']}'")
        print(f"   Goal: '{cfg['goal'][:80]}'" if cfg['goal'] else "   Goal: (empty)")
        print(f"   Eval: '{cfg['eval'][:80]}'" if cfg['eval'] else "   Eval: (empty)")
        if cfg['base_url']:
            print(f"   base_url: {cfg['base_url']}")
        if cfg['feedback_base_url']:
            print(f"   feedback_base_url: {cfg['feedback_base_url']}")
        if cfg['extra_overrides']:
            print(f"   extra_overrides: {cfg['extra_overrides']}")

        print(f"\n📡 Sending INITIAL_MESSAGES webhook...")
        await self.send_initial_messages(
            system_message=(
                "You are a Socratic-Evolve agent that autonomously solves "
                "ML competitions using progressive search and experience-driven memory."
            ),
            user_message=f"Task: {self.problem_statement}",
            step_number=0,
        )
        print(f"✅ INITIAL_MESSAGES sent")

        # Step 1 — Prepare input
        print(f"\n{'=' * 70}")
        print(f"📋 STEP 1: Prepare input")
        print(f"{'=' * 70}")
        step = 1
        action_id = f"action_{self.experiment_id}_{step}"
        print(f"📡 Sending ACTION_RECEIVED (step={step}, action_id={action_id})...")
        await self.send_action_received(
            step_number=step,
            action_id=action_id,
            action_type="prepare_input",
            action_message={
                "role": "assistant",
                "content": "Preparing MLEvolve workspace and config.",
                "tool_calls": None,
                "completion_details": None,
            },
        )
        print(f"✅ ACTION_RECEIVED sent")

        print(f"🔧 Running _prepare_input()...")
        data_dir, desc_file = self._prepare_input()
        print(f"✅ _prepare_input() returned:")
        print(f"   data_dir: {data_dir} (exists={data_dir.exists()})")
        print(f"   desc_file: {desc_file} (exists={desc_file.exists()})")
        if desc_file.exists():
            desc_size = desc_file.stat().st_size
            print(f"   desc_file size: {desc_size} bytes")
        if data_dir.exists():
            data_contents = list(data_dir.iterdir())[:20]
            print(f"   data_dir contents ({len(list(data_dir.iterdir()))} items): {[p.name for p in data_contents]}")

        print(f"📡 Sending STEP_FINISHED (step={step})...")
        await self.send_step_finished(
            step_number=step,
            action_id=action_id,
            action_type="prepare_input",
            observation_content=f"Workspace ready. data_dir={data_dir}, desc_file={desc_file}",
            tool_call_id=f"call_{step}",
        )
        print(f"✅ STEP_FINISHED sent")
        self.current_step = step

        # Step 2 — Run MLEvolve search
        print(f"\n{'=' * 70}")
        print(f"📋 STEP 2: Run MLEvolve search")
        print(f"{'=' * 70}")
        step = 2
        action_id = f"action_{self.experiment_id}_{step}"
        print(f"📡 Sending ACTION_RECEIVED (step={step}, action_id={action_id})...")
        await self.send_action_received(
            step_number=step,
            action_id=action_id,
            action_type="run_search",
            action_message={
                "role": "assistant",
                "content": "Launching MLEvolve progressive search.",
                "tool_calls": None,
                "completion_details": None,
            },
        )
        print(f"✅ ACTION_RECEIVED sent")

        print(f"🔧 Running _run_search()...")
        run_ok, run_summary = await self._run_search(data_dir, desc_file)
        print(f"\n✅ _run_search() returned:")
        print(f"   success: {run_ok}")
        print(f"   summary length: {len(run_summary)} chars")
        print(f"   summary (last 200 chars): ...{run_summary[-200:]}")

        print(f"📡 Sending STEP_FINISHED (step={step})...")
        await self.send_step_finished(
            step_number=step,
            action_id=action_id,
            action_type="run_search",
            observation_content=run_summary,
            tool_call_id=f"call_{step}",
        )
        print(f"✅ STEP_FINISHED sent")
        self.current_step = step

        # Step 3 — Collect artifacts
        print(f"\n{'=' * 70}")
        print(f"📋 STEP 3: Collect artifacts")
        print(f"{'=' * 70}")
        step = 3
        action_id = f"action_{self.experiment_id}_{step}"
        print(f"📡 Sending ACTION_RECEIVED (step={step}, action_id={action_id})...")
        await self.send_action_received(
            step_number=step,
            action_id=action_id,
            action_type="collect_artifacts",
            action_message={
                "role": "assistant",
                "content": "Collecting results and saving to src/.",
                "tool_calls": None,
                "completion_details": None,
            },
        )
        print(f"✅ ACTION_RECEIVED sent")

        print(f"🔧 Running _collect_artifacts()...")
        score, approach = self._collect_artifacts()
        print(f"✅ _collect_artifacts() returned:")
        print(f"   score: {score}")
        print(f"   approach: {approach[:100]}...")

        # List what's in src_dir now
        if self.src_dir.exists():
            src_contents = list(self.src_dir.iterdir())
            print(f"   src_dir contents ({len(src_contents)} items):")
            for p in src_contents:
                size = p.stat().st_size if p.is_file() else "dir"
                print(f"      {p.name} ({size})")

        print(f"📡 Sending STEP_FINISHED (step={step})...")
        await self.send_step_finished(
            step_number=step,
            action_id=action_id,
            action_type="collect_artifacts",
            observation_content=f"Artifacts saved. score={score}, approach={approach}",
            tool_call_id=f"call_{step}",
        )
        print(f"✅ STEP_FINISHED sent")
        self.current_step = step

        # Step 4 — Report result
        print(f"\n{'=' * 70}")
        print(f"📋 STEP 4: Report final result")
        print(f"{'=' * 70}")
        if run_ok:
            print(f"📡 Sending EXPERIMENT_COMPLETED (score={score})...")
            await self.send_experiment_completed(
                success=True,
                summary=run_summary,
                score=score,
                approach=approach,
                step_number=self.current_step,
            )
            print(f"✅ EXPERIMENT_COMPLETED sent")
        else:
            print(f"📡 Sending EXPERIMENT_FAILED...")
            await self.send_experiment_failed(
                error_message=run_summary,
                step_number=self.current_step,
            )
            print(f"✅ EXPERIMENT_FAILED sent")
        print(f"\n🏁 SocraticEvolve agent finished!")

    # ------------------------------------------------------------------
    # Continue / Abort
    # ------------------------------------------------------------------
    async def continue_agent(
        self,
        user_message: str,
        new_max_steps: int,
        step_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ):
        print(f"\n📨 continue_agent() called:")
        print(f"   user_message: {user_message}")
        print(f"   new_max_steps: {new_max_steps}")
        print(f"   step_number: {step_number}")
        print(f"   branch_name: {branch_name}")
        self.max_steps = new_max_steps
        if step_number is not None:
            self.current_step = step_number
        else:
            self.current_step += 1
        print(f"   resolved current_step: {self.current_step}")

        print(f"📡 Sending STEP_FINISHED with user message...")
        await self.send_step_finished(
            step_number=self.current_step,
            user_message=user_message,
            git_branch=branch_name,
        )
        print(f"✅ STEP_FINISHED sent")

        await self.send_experiment_completed(
            success=True,
            summary="Continuation acknowledged",
            step_number=self.current_step,
        )
        print(f"✅ EXPERIMENT_COMPLETED sent")

    async def abort(self):
        print(f"\n🛑 abort() called!")
        print(f"   current_step: {self.current_step}")
        print(f"   process running: {self.process and self.process.returncode is None}")
        self.is_aborted = True
        if self.process and self.process.returncode is None:
            print(f"   Terminating subprocess (pid={self.process.pid})...")
            self.process.terminate()
        print(f"📡 Sending EXPERIMENT_ABORTED...")
        await self.send_experiment_aborted(
            reason="Aborted by user",
            last_step=self.current_step,
        )
        print(f"✅ EXPERIMENT_ABORTED sent")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_config(self) -> Dict[str, Any]:
        """Build full MLEvolve config from agent_config + schema defaults."""
        print(f"\n🔧 _get_config(): building config from agent_config + defaults")
        cfg = {
            # Search
            "steps": 500,
            "time_limit": 43200,
            "initial_drafts": 3,
            "seed": 42,
            # Model
            "model": "claude-opus-4-6",
            "feedback_model": None,  # defaults to model
            "base_url": "",
            "feedback_base_url": "",
            # Execution
            "parallel_search_num": 3,
            "exec_timeout": 32400,
            "start_cpu_id": 0,
            "cpu_number": 21,
            "num_gpus": 1,
            # Features
            "use_coldstart": True,
            "use_diff_mode": True,
            "check_data_leakage": True,
            "use_global_memory": False,
            # Task resolution
            "exp_id": "",
            "dataset_dir": "",
            "data_dir": "",
            "desc_file": "",
            "goal": "",
            "eval": "",
            # Extra
            "extra_overrides": {},
        }
        # Flat keys in agent_config override defaults
        overridden_flat = []
        for k in list(cfg.keys()):
            if k in self.agent_config:
                overridden_flat.append(f"{k}={self.agent_config[k]}")
                cfg[k] = self.agent_config[k]
        if overridden_flat:
            print(f"   Flat overrides from agent_config: {', '.join(overridden_flat)}")
        # Nested mlevolve dict also overrides (backwards compat)
        if "mlevolve" in self.agent_config:
            print(f"   Nested 'mlevolve' dict found, merging: {list(self.agent_config['mlevolve'].keys())}")
            cfg.update(self.agent_config["mlevolve"])
        if not cfg["feedback_model"]:
            cfg["feedback_model"] = cfg["model"]
            print(f"   feedback_model defaulted to model: {cfg['model']}")
        return cfg

    def _resolve_python_bin(self) -> str:
        """Find the best python binary in the container."""
        print(f"🐍 _resolve_python_bin(): searching for python binary...")
        for candidate in [
            Path("/app/src/.venv/bin/python3"),
            Path("/workspace/.venv/bin/python3"),
        ]:
            print(f"   Checking {candidate}... {'EXISTS' if candidate.exists() else 'not found'}")
            if candidate.exists():
                print(f"   Using: {candidate}")
                return str(candidate)
        print(f"   Using fallback: {sys.executable}")
        return sys.executable

    def _prepare_dataset(self, exp_id: str, dataset_dir: str):
        """Download and prepare MLE-bench dataset if not already present.

        Mirrors the working pattern from discover/custom_agent.py.
        """
        print(f"\n📂 _prepare_dataset(exp_id='{exp_id}', dataset_dir='{dataset_dir}')")
        prepared_path = Path(dataset_dir) / exp_id / "prepared" / "public"
        print(f"   Checking prepared_path: {prepared_path}")
        print(f"   prepared_path exists: {prepared_path.exists()}")
        if prepared_path.exists() and any(prepared_path.iterdir()):
            contents = list(prepared_path.iterdir())[:10]
            print(f"   ✅ Dataset for '{exp_id}' already prepared at {prepared_path} ({len(list(prepared_path.iterdir()))} items)")
            print(f"   First items: {[p.name for p in contents]}")
            return

        python_bin = self._resolve_python_bin()

        # Check via mlebench registry
        print(f"   Checking mlebench registry for '{exp_id}'...")
        check_cmd = [
            python_bin, "-c",
            f"from mlebench.registry import registry; "
            f"c = registry.get_competition('{exp_id}'); "
            f"pub = list(c.public_dir.iterdir()) if c.public_dir.exists() else []; "
            f"priv = list(c.private_dir.iterdir()) if c.private_dir.exists() else []; "
            f"print(f'public={{len(pub)}} private={{len(priv)}}')"
        ]
        print(f"   Running: {' '.join(check_cmd)}")
        result = subprocess.run(check_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"   Dataset check for '{exp_id}': {output}")
            if "public=0" not in output and "private=0" not in output:
                print(f"   ✅ Dataset for '{exp_id}' already prepared, skipping download")
                return
        else:
            print(f"   ❌ mlebench registry check failed (rc={result.returncode})")
            print(f"   stdout: {result.stdout.strip()}")
            print(f"   stderr: {result.stderr.strip()}")

        # Try mlebench prepare command first
        print(f"   Attempting mlebench prepare for '{exp_id}'...")
        prep_cmd = [python_bin, "-m", "mlebench.prepare", "-c", exp_id]
        print(f"   Running: {' '.join(prep_cmd)}")
        prep_result = subprocess.run(prep_cmd, capture_output=True, text=True)
        if prep_result.returncode == 0:
            print(f"   ✅ mlebench prepare succeeded for '{exp_id}'")
            print(f"   stdout: {prep_result.stdout.strip()[:200]}")
            return

        print(f"   ❌ mlebench prepare failed (rc={prep_result.returncode})")
        print(f"   stdout: {prep_result.stdout.strip()[:200]}")
        print(f"   stderr: {prep_result.stderr.strip()[:200]}")
        print(f"   Falling back to Kaggle download...")

        # Download from Kaggle
        tmp_data_dir = Path(tempfile.mkdtemp(prefix=f"kaggle_{exp_id}_"))
        print(f"   tmp_data_dir: {tmp_data_dir}")

        kaggle_cmd = [
            "kaggle", "competitions", "download",
            "-c", exp_id,
            "-p", str(tmp_data_dir),
        ]
        print(f"   Running: {' '.join(kaggle_cmd)}")
        dl_result = subprocess.run(kaggle_cmd, capture_output=True, text=True)

        if dl_result.returncode != 0:
            print(f"   ❌ Kaggle download failed (rc={dl_result.returncode})")
            print(f"   stdout: {dl_result.stdout.strip()[:200]}")
            print(f"   stderr: {dl_result.stderr.strip()[:200]}")
            print("   Make sure 'kaggle' CLI is installed and ~/.kaggle/kaggle.json is configured")
            shutil.rmtree(tmp_data_dir, ignore_errors=True)
            return

        print(f"   ✅ Downloaded to {tmp_data_dir}")
        downloaded_files = list(tmp_data_dir.iterdir())
        print(f"   Downloaded files: {[p.name for p in downloaded_files]}")

        # Unzip downloaded archives
        for zf in tmp_data_dir.glob("*.zip"):
            print(f"   Unzipping {zf.name}...")
            subprocess.run(["unzip", "-o", "-q", str(zf), "-d", str(tmp_data_dir)])
            zf.unlink()

        after_unzip = list(tmp_data_dir.iterdir())
        print(f"   After unzip: {[p.name for p in after_unzip]}")

        # Run setup via mlebench if available
        setup_cmd = [
            python_bin, "-m", "mlebench.prepare",
            "-c", exp_id,
            "--data-dir", str(tmp_data_dir),
        ]
        print(f"   Running setup: {' '.join(setup_cmd)}")
        setup_result = subprocess.run(setup_cmd, capture_output=True, text=True)

        if setup_result.returncode == 0:
            print(f"   ✅ Dataset setup complete for '{exp_id}'")
        else:
            print(f"   ⚠️  Dataset setup returned errors (rc={setup_result.returncode})")
            print(f"   stderr: {setup_result.stderr.strip()[:300]}")
            # Copy raw data as fallback so run.py has something to work with
            fallback_dir = Path(dataset_dir) / exp_id / "prepared" / "public"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            for item in tmp_data_dir.iterdir():
                dest = fallback_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
            print(f"   Copied raw data to {fallback_dir}")

        shutil.rmtree(tmp_data_dir, ignore_errors=True)

    def _prepare_input(self) -> tuple[Path, Path]:
        """Resolve data_dir and desc_file for MLEvolve.

        Resolution order:
        1. exp_id + dataset_dir  → auto-resolve paths (prepare data if missing)
        2. explicit data_dir     → use directly
        3. fallback              → write problem_statement to workspace
        """
        print(f"\n📂 _prepare_input(): resolving data_dir and desc_file")
        cfg = self._get_config()
        exp_id = cfg["exp_id"]
        dataset_dir = cfg["dataset_dir"]
        print(f"   exp_id='{exp_id}', dataset_dir='{dataset_dir}', data_dir='{cfg['data_dir']}', desc_file='{cfg['desc_file']}'")

        # Option 1: resolve from exp_id + dataset_dir
        if exp_id and dataset_dir:
            print(f"   → Option 1: resolve from exp_id + dataset_dir")
            # Prepare dataset if not already present
            dataset_dir_resolved = str(Path(dataset_dir).resolve())
            try:
                self._prepare_dataset(exp_id, dataset_dir_resolved)
            except OSError as e:
                print(f"   ⚠️  _prepare_dataset failed: {e}")

            data_dir = Path(dataset_dir_resolved) / exp_id / "prepared" / "public"
            if not data_dir.exists():
                print(f"   ⚠️  Data dir {data_dir} still missing after prep, falling through to next option")
            else:
                desc_file = data_dir / "description.md"
                if not desc_file.exists():
                    print(f"   description.md not in data_dir, writing problem_statement to workspace")
                    work = self.src_dir / "workspace"
                    work.mkdir(parents=True, exist_ok=True)
                    desc_file = work / "description.md"
                    desc_file.write_text(self.problem_statement)
                print(f"   ✅ Resolved: data_dir={data_dir}, desc_file={desc_file}")
                return data_dir, desc_file

        # Option 2: explicit data_dir
        data_dir_str = cfg["data_dir"]
        if data_dir_str:
            print(f"   → Option 2: explicit data_dir='{data_dir_str}'")
            data_dir = Path(data_dir_str)
            print(f"   data_dir exists: {data_dir.exists()}")
            desc_file_str = cfg["desc_file"]
            if desc_file_str:
                desc_file = Path(desc_file_str)
                print(f"   Using explicit desc_file='{desc_file}' (exists={desc_file.exists()})")
            elif (data_dir / "description.md").exists():
                desc_file = data_dir / "description.md"
                print(f"   Found description.md in data_dir")
            else:
                work = self.src_dir / "workspace"
                work.mkdir(parents=True, exist_ok=True)
                desc_file = work / "description.md"
                desc_file.write_text(self.problem_statement)
                print(f"   No description.md found, wrote problem_statement to {desc_file}")
            print(f"   ✅ Resolved: data_dir={data_dir}, desc_file={desc_file}")
            return data_dir, desc_file

        # Option 3: fallback — use problem_statement only
        print(f"   → Option 3: fallback (no exp_id or data_dir)")
        work = self.src_dir / "workspace"
        work.mkdir(parents=True, exist_ok=True)
        desc_file = work / "description.md"
        desc_file.write_text(self.problem_statement)
        print(f"   ✅ Resolved (fallback): data_dir={work}, desc_file={desc_file}")
        return work, desc_file

    async def _run_search(self, data_dir: Path, desc_file: Path) -> tuple[bool, str]:
        """Run MLEvolve's run.py as a subprocess."""
        print(f"\n🔬 _run_search(data_dir={data_dir}, desc_file={desc_file})")
        cfg = self._get_config()
        exp_id = cfg["exp_id"] or self.experiment_id
        exp_name = f"run_{exp_id}"
        run_dir = self.src_dir / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        print(f"   exp_name: {exp_name}")
        print(f"   run_dir: {run_dir}")

        # Inject API keys into environment
        env = os.environ.copy()
        injected_keys = []
        for key, val in self.api_keys.items():
            if val:
                env[key] = val
                injected_keys.append(key)
        print(f"   Injected API keys into env: {injected_keys}")

        python_bin = self._resolve_python_bin()
        run_py = PUBLIC_REPO / "run.py"
        print(f"   run.py path: {run_py} (exists={run_py.exists()})")
        print(f"   python binary: {python_bin}")

        cmd = [
            python_bin,
            str(run_py),
            f"data_dir={data_dir}",
            f"desc_file={desc_file}",
            f"exp_name={exp_name}",
            f"log_dir={run_dir}",
            f"workspace_dir={run_dir}",
            f"agent.steps={cfg['steps']}",
            f"agent.time_limit={cfg['time_limit']}",
            f"agent.initial_drafts={cfg['initial_drafts']}",
            f"agent.seed={cfg['seed']}",
            f"agent.code.model={cfg['model']}",
            f"agent.feedback.model={cfg['feedback_model']}",
            f"agent.search.parallel_search_num={cfg['parallel_search_num']}",
            f"agent.search.num_gpus={cfg['num_gpus']}",
            f"agent.use_diff_mode={cfg['use_diff_mode']}",
            f"agent.check_data_leakage={cfg['check_data_leakage']}",
            f"agent.use_global_memory={cfg['use_global_memory']}",
            f"exec.timeout={cfg['exec_timeout']}",
            f"start_cpu_id={cfg['start_cpu_id']}",
            f"cpu_number={cfg['cpu_number']}",
            f"coldstart.use_coldstart={cfg['use_coldstart']}",
        ]

        # Only pass optional string fields when non-empty (OmegaConf rejects empty)
        if cfg["base_url"]:
            cmd.append(f"agent.code.base_url={cfg['base_url']}")
        if cfg["feedback_base_url"]:
            cmd.append(f"agent.feedback.base_url={cfg['feedback_base_url']}")
        if cfg["exp_id"]:
            cmd.append(f"exp_id={cfg['exp_id']}")
        if cfg["dataset_dir"]:
            cmd.append(f"dataset_dir={cfg['dataset_dir']}")
        if cfg["goal"]:
            cmd.append(f"goal={cfg['goal']}")
        if cfg["eval"]:
            cmd.append(f"eval={cfg['eval']}")

        # Apply any extra overrides
        for k, v in cfg.get("extra_overrides", {}).items():
            cmd.append(f"{k}={v}")

        print(f"\n   Full command:")
        for i, arg in enumerate(cmd):
            print(f"      [{i}] {arg}")

        print(f"\n   Launching subprocess (cwd={PUBLIC_REPO})...")
        start_time = time.time()

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PUBLIC_REPO),
            env=env,
        )
        print(f"   Subprocess started (pid={self.process.pid})")

        output_lines = []
        line_count = 0
        print(f"\n   --- MLEvolve subprocess output ---")
        while True:
            if self.is_aborted:
                print(f"   🛑 Abort flag detected, terminating subprocess...")
                self.process.terminate()
                break
            line = await self.process.stdout.readline()
            if not line:
                break
            decoded = line.decode(errors="replace").rstrip()
            output_lines.append(decoded)
            line_count += 1
            print(decoded)

        await self.process.wait()
        elapsed = time.time() - start_time
        success = self.process.returncode == 0
        summary = "\n".join(output_lines[-30:]) if output_lines else "No output"

        print(f"   --- End MLEvolve output ---")
        print(f"\n   Subprocess finished:")
        print(f"   return_code: {self.process.returncode}")
        print(f"   success: {success}")
        print(f"   total output lines: {line_count}")
        print(f"   elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")

        if not success:
            summary = f"MLEvolve exited with code {self.process.returncode}.\n{summary}"
            print(f"   ❌ FAILED! Last 5 lines:")
            for ln in output_lines[-5:]:
                print(f"      {ln}")

        # Check what's in run_dir after search
        if run_dir.exists():
            run_contents = list(run_dir.iterdir())
            print(f"   run_dir contents ({len(run_contents)} items): {[p.name for p in run_contents[:20]]}")

        return success, summary

    def _collect_artifacts(self) -> tuple[Optional[float], str]:
        """Copy results from run_dir to src/ and extract score."""
        print(f"\n📦 _collect_artifacts()")
        score = None
        approach = "MLEvolve progressive search"

        if not self.run_dir:
            print(f"   ⚠️  run_dir is None, skipping artifact collection")
            return score, approach

        print(f"   run_dir: {self.run_dir}")
        print(f"   src_dir: {self.src_dir}")

        # Find the most recent run directory (timestamped)
        all_items = list(self.run_dir.iterdir()) if self.run_dir.exists() else []
        run_dirs = sorted(
            [d for d in all_items if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        print(f"   Found {len(run_dirs)} run subdirectories: {[d.name for d in run_dirs]}")

        if not run_dirs:
            print(f"   ⚠️  No run subdirectories found!")
            print(f"   All items in run_dir: {[p.name for p in all_items]}")
            return score, approach

        for rd in run_dirs:
            print(f"\n   Processing latest run: {rd.name}")
            rd_contents = list(rd.iterdir())
            print(f"   Run dir contents: {[p.name for p in rd_contents]}")

            logs_dir = rd / "logs" if (rd / "logs").exists() else rd
            print(f"   logs_dir: {logs_dir} (is 'logs' subdir: {(rd / 'logs').exists()})")
            if logs_dir.exists():
                logs_contents = list(logs_dir.iterdir())
                print(f"   logs_dir contents: {[p.name for p in logs_contents]}")

            # Copy best_solution.py
            best = logs_dir / "best_solution.py"
            if best.exists():
                dest = self.src_dir / "best_solution.py"
                shutil.copy2(best, dest)
                print(f"   ✅ Copied best_solution.py ({best.stat().st_size} bytes) -> {dest}")
            else:
                print(f"   ⚠️  best_solution.py not found at {best}")

            # Copy journal
            journal_file = logs_dir / "journal.json"
            if journal_file.exists():
                dest = self.src_dir / "journal.json"
                shutil.copy2(journal_file, dest)
                print(f"   ✅ Copied journal.json ({journal_file.stat().st_size} bytes) -> {dest}")
                score, approach = self._parse_journal(journal_file)
                print(f"   Parsed journal: score={score}, approach={approach[:80]}...")
            else:
                print(f"   ⚠️  journal.json not found at {journal_file}")

            # Copy submission if exists
            submission = rd / "workspace" / "submission" / "submission.csv"
            if submission.exists():
                dest = self.src_dir / "submission.csv"
                shutil.copy2(submission, dest)
                print(f"   ✅ Copied submission.csv ({submission.stat().st_size} bytes) -> {dest}")
            else:
                print(f"   ⚠️  submission.csv not found at {submission}")
                # Check what's in workspace
                workspace = rd / "workspace"
                if workspace.exists():
                    ws_contents = list(workspace.iterdir())
                    print(f"   workspace contents: {[p.name for p in ws_contents]}")

            break  # only process the latest run

        print(f"\n   Final: score={score}, approach={approach[:80]}...")
        return score, approach

    def _parse_journal(self, journal_file: Path) -> tuple[Optional[float], str]:
        """Extract best score and approach from journal.json."""
        print(f"\n📖 _parse_journal({journal_file.name})")
        data = json.loads(journal_file.read_text())
        nodes = data.get("nodes", [])
        print(f"   Total nodes in journal: {len(nodes)}")

        best_score = None
        best_plan = "MLEvolve progressive search"
        scores_found = []

        for i, node in enumerate(nodes):
            metric = node.get("metric")
            if metric and metric.get("value") is not None:
                val = metric["value"]
                scores_found.append(val)
                if best_score is None or val > best_score:
                    best_score = val
                    if node.get("plan"):
                        best_plan = node["plan"][:500]
                    print(f"   Node {i}: new best score={val}")

        print(f"   All scores found: {scores_found}")
        print(f"   Best score: {best_score}")
        print(f"   Best plan: {best_plan[:80]}...")
        return best_score, best_plan


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
    """Entry point called by the agent runtime."""
    return SocraticEvolveAgent(
        experiment_id=experiment_id,
        project_id=project_id,
        problem_statement=problem_statement,
        max_steps=max_steps,
        api_keys=api_keys,
        webhook_url=webhook_url,
        agent_config=agent_config,
        jwt_token=jwt_token,
    )
