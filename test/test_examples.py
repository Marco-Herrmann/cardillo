import subprocess
import pathlib
import pytest


example_base = pathlib.Path("examples")

# Alle Beispielordner unter examples/
example_dirs = [d for d in example_base.iterdir() if d.is_dir()]

# In jedem Beispielordner ein .py-File zum Ausführen finden
example_scripts = []
for example_dir in example_dirs:
    # Suche nach einer .py-Datei (z. B. run.py, main.py, etc.)
    candidates = list(example_dir.glob("*.py"))
    if candidates:
        example_scripts.append(candidates[0])  # Nimm die erste gefundene Datei

from pprint import pprint

pprint(example_scripts)
# example_scripts = [example_scripts[1]] # marker import
example_scripts = [example_scripts[3]]


@pytest.mark.parametrize("script_path", example_scripts, ids=lambda p: p.parent.name)
def test_example_runs_without_error(script_path):
    """Führt das gefundene Beispiel-Skript aus und prüft, ob es ohne Fehler durchläuft."""
    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Fehler beim Ausführen von {script_path}:\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )
