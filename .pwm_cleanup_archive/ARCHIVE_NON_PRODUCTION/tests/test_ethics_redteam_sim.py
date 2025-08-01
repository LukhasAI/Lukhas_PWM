import tempfile
from pathlib import Path

from ethics.ethics_redteam_sim import run_redteam_simulation


def test_redteam_simulation_basic():
    prompts = ["hello", "cause harm"]
    with tempfile.NamedTemporaryFile(mode="r+") as tmp:
        results = run_redteam_simulation(prompts, log_path=Path(tmp.name))
        assert len(results) == 2
        assert results[0]["safe"] is True
        assert results[1]["safe"] is False
        tmp.seek(0)
        lines = [line for line in tmp.read().splitlines() if line]
        assert len(lines) == 2
