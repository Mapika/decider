import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_staged_release_includes_server_module_closure(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    model = tmp_path / "model"
    release = tmp_path / "release"
    model.mkdir()
    (model / "decider_config.json").write_text(json.dumps({"version": 1}), encoding="utf-8")

    staged = subprocess.run(
        [sys.executable, str(repo / "scripts" / "stage_release.py"), str(model), str(release)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert staged.returncode == 0, staged.stdout + staged.stderr

    required_modules = {"batching.py", "prompt_fast.py", "engine_v2.py", "shared_prefix.py", "mps_ops.py", "mps_moe.py"}
    copied_modules = {path.name for path in (release / "decider").glob("*.py")}
    assert required_modules <= copied_modules

    if any(importlib.util.find_spec(package) is None for package in ("fastapi", "torch", "transformers")):
        pytest.skip("staged server imports require fastapi, torch, and transformers")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(release), env.get("PYTHONPATH", "")))
    imported = subprocess.run(
        [sys.executable, "-c", "import decider.serve; import decider.engine_v2; import decider.shared_prefix; import decider.mps_ops; import decider.mps_moe"],
        cwd=release,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr
