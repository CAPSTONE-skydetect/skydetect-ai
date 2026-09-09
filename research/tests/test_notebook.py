import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def test_notebook_executes_with_project_interpreter(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    notebook = nbformat.read(root / "research" / "notebooks" / "research_lab.ipynb", as_version=4)
    nbformat.validate(notebook)
    monkeypatch.setenv("RESEARCH_OUTPUT_DIR", str(tmp_path))
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=180,
                            resources={"metadata": {"path": str(root)}})
    executed = client.execute(cleanup_kc=True)
    assert client.km is None
    assert all(cell.execution_count is not None for cell in executed.cells if cell.cell_type == "code")
    assert (tmp_path / "simulation_features_v3.csv").exists()
    assert (tmp_path / "feature_ecdf.png").stat().st_size > 10000
