"""Smoke tests that run every example script and check that it exits cleanly.

Each script under ``examples/`` is executed in a separate Python process with
its own directory as working directory, so relative paths inside the examples
(e.g. for exports or meshes) resolve as intended. Matplotlib is forced to the
non-interactive ``Agg`` backend so that ``plt.show()`` does not block.

Run a single example with e.g.
    pytest test/test_examples.py -k double_pendulum
"""

import os
import pathlib
import subprocess
import sys

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"

# maximum wall-clock time per example in seconds
TIMEOUT = 600

# scripts (relative to EXAMPLES_DIR, POSIX style) that should not be run,
# mapped to the reason for skipping them
SKIP = {
    "cable_vis/plot_frf.py": "Postprocessing script to generare plots.",
    "frf_clamped_cable/frf_clamped_cable.py": "Long showcase example, not intended to be a test.",
    "motion_stage/motion_stage_02.py": "long showcase example, not inteded to be a test!",
    "urdf_double_pendulum/double_pendulum.py": "Example's not running (Problem with the renderer).",
    "urdf_panda/panda.py": "Example's not running (missing `pip install pycollada`).",
    "urdf_unitree_go1/go1.py": "Example's not running (Problem with the renderer).",
    "wilberforce2p0/wilberforce2p0.py": "Long showcase example, not intended to be a test.",
}


def collect_example_scripts():
    return sorted(
        p
        for p in EXAMPLES_DIR.rglob("*.py")
        if "__pycache__" not in p.parts and not p.name.startswith("_")
    )


def example_id(script_path):
    return script_path.relative_to(EXAMPLES_DIR).as_posix()


def example_params():
    for script_path in collect_example_scripts():
        rel = example_id(script_path)
        marks = [pytest.mark.skip(reason=SKIP[rel])] if rel in SKIP else []
        yield pytest.param(script_path, id=rel, marks=marks)


@pytest.mark.parametrize("script_path", example_params())
def test_example_runs_without_error(script_path):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    try:
        result = subprocess.run(
            [sys.executable, script_path.name],
            cwd=script_path.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{example_id(script_path)} did not finish within {TIMEOUT} s")

    assert result.returncode == 0, (
        f"{example_id(script_path)} exited with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
