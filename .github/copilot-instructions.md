# 🤖 Copilot Instructions for LUKHAS AGI Codebase

Welcome, AI agent! This guide will help you be productive in the LUKHAS AGI codebase. Focus on these project-specific conventions, workflows, and architectural patterns:

## 🧠 Big Picture Architecture
- **LUKHAS** is a modular AGI platform blending quantum, biological, and ethical systems.
- Major domains: `core/`, `identity/`, `memory/`, `quantum/`, `bio/`, `consciousness/`, `emotion/`, `orchestration/`, `ethics/`, `security/`, `api/`.
- Each domain is a semi-autonomous subsystem. Cross-domain communication is via orchestrators and bridge modules.
- **Key pattern:** "Dream-state" and "bio-inspired" modules often use multi-stage pipelines and symbolic data flows.
- **Integration focus:** Many files are legacy or partially integrated—see `project-docs/agents/AGENT.md` for active integration tasks.

## 🛠️ Developer Workflows
- **Build:**
  - Python: `pip install -r requirements.txt` (core), `pip install -r requirements-test.txt` (tests)
  - Node.js: `npm install` (for UI/frontend)
  - Docker: `docker-compose up` (optional, for full stack)
- **Test:**
  - Run all: `python -m pytest tests/`
  - Integration: `python tests/run_comprehensive_test_suite.py`
  - Benchmarks: `python benchmarks/run_all_benchmarks.py`
- **CLI Tools:**
  - `lukhas_db/cli.py` for database and audit operations
  - Emergency scripts: `emergency_kill_analysis.sh`, `reset_copilot_chat.sh`, `nuclear_vscode_reset.sh`
- **Documentation:**
  - Main: `README.md`, `project-docs/README.md`, `project-docs/agents/AGENT.md`
  - Each major directory has its own `README.md` with domain-specific details

## 🏗️ Project Conventions & Patterns
- **File/Module Naming:**
  - Use descriptive, domain-specific names (e.g., `bio_optimization_adapter.py`, `dream_adapter.py`)
  - Integration/adapters are named as `*_adapter.py` or `*_hub.py`
- **Testing:**
  - Place tests in `tests/` or `*/tests/` subfolders
  - Use documented mocks for identity/emotion/quantum modules
- **Data Flows:**
  - Symbolic and quantum data often use multi-stage transformation (see `quantum/bio_components.py`)
  - Orchestration modules coordinate cross-domain actions (see `orchestration/`)
- **Legacy/Integration:**
  - Many files are in transition; check `AGENT.md` and `AGENT_SPECIFIC_TASKS.md` for current integration priorities

## 🔗 Integration & External Dependencies
- **External APIs:**
  - REST endpoints in `api/`
  - Some modules use external quantum or bio-simulation libraries (see `requirements.txt`)
- **Database:**
  - Custom DB logic in `lukhas_db/` (see CLI)
- **Audit/Trace:**
  - System-wide audit trail and drift detection (see `orchestration/monitoring/`, `lukhas_db/`)

## 📝 Examples
- **Quantum bio-encoding:** `quantum/bio_components.py` (multi-layer encoding)
- **Dream adapter:** `orchestration/brain/unified_integration/adapters/dream_adapter.py` (state tracking)
- **Emotion engine:** `emotion/dreamseed_upgrade.py` (tier validation)
- **Human-in-the-loop:** `orchestration/integration/human_in_the_loop_orchestrator.py` (email notifications)

## ⚠️ Special Notes
- **Do not assume all files are integrated**—always check `AGENT.md` for current status.
- **Follow domain-specific README.md** for conventions unique to each subsystem.
- **Emergency scripts** are provided for VS Code and extension issues—see root directory.

---

For more, see `README.md`, `project-docs/README.md`, and `project-docs/agents/AGENT.md`.

---

*Last updated: 2025-07-31. Please suggest improvements if you find missing or outdated info!*
