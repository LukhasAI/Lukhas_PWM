import json
from pathlib import Path

from tools.dev.module_digest_extractor import ModuleDigestExtractor


def create_sample_module(base: Path) -> Path:
    module_dir = base / "lukhas" / "sample"
    module_dir.mkdir(parents=True)
    module_path = module_dir / "test_mod.py"
    module_path.write_text(
        '''"""
╠═══════════════════════════════════════════════════════════════════════════════
║ 🧠 LUKHAS AI - TEST MODULE
║ Module: test_mod.py
║ Path: lukhas/sample/test_mod.py
║ Version: 0.1.0 | Created: 2025-01-01 | Modified: 2025-01-01
║ Symbolic Tags: {critical} #ΛDVNT
"""
print('hello world')
"""
╠═══════════════════════════════════════════════════════════════════════════════
║ 📋 FOOTER - LUKHAS AI
"""
'''
    )
    return module_path


def test_module_digest_extractor_basic(tmp_path):
    create_sample_module(tmp_path)
    extractor = ModuleDigestExtractor(base_path=str(tmp_path))
    extractor.run()

    digest_path = tmp_path / "lukhas" / "tools" / "digest" / "digest.json"
    assert digest_path.exists()
    data = json.loads(digest_path.read_text())
    assert len(data) == 1
    entry = data[0]
    assert entry["name"].startswith("test_mod")
    assert "critical" in entry.get("tags", [])
