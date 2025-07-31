# Test Matrix

This file tracks the status of all tests in the LUKHAS AGI system.

| Status | Test File | Linked Module | Last Successful Commit | Assigned Agent |
| --- | --- | --- | --- | --- |
| ✅ | `tests/active/basic_test.py` | `core/lukhas_dast/engine.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ✅ | `tests/active/simple_test.py` | `core/lukhas_dast/engine.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ✅ | `tests/active/test_compliance_engine.py` | `core/governance/compliance_engine.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ✅ | `tests/active/test_compliance_engine_2.py` | `core/governance/compliance_engine.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ⚠️ | `tests/active/test_context_analyzer.py` | `core/advanced/brain/context_analyzer.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ⚠️ | `tests/active/test_emotion_mapper_alt.py` | `core/advanced/brain/emotion_mapper_alt.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ⚠️ | `tests/active/test_healix_mapper.py` | `core/advanced/brain/healix_mapper.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ⚠️ | `tests/active/test_intent_node.py` | `nodes/intent_node.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ✅ | `tests/active/test_logger.py` | `orchestration/brain/trace_memoria_logger.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ✅ | `tests/active/test_quantum_consensus.py` | `core/advanced/brain/test_quantum_consensus.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ✅ | `tests/active/test_symptom_reporter.py` | `core/interaction/symptom_reporter.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ⚠️ | `tests/active/test_voice_processing.py` | `core/advanced/brain/test_voice_processing.py` | `jules-01-orchestration-phase-3.1` | Jules-01 |
| ❌ | `tests/hold/dreams/test_dream_convergence_tester.py` | `creativity/dream_systems/dream_convergence_tester.py` | --- | Jules-04 |
| ❌ | `tests/hold/emotion/test_emotion_recursion.py` | `memory/core_memory/emotional_memory.py` | --- | Jules-02 |
| ❌ | `tests/hold/genesis/test_boot.py` | `symbolic_boot.py` | --- | Jules-01 |
| ❌ | `tests/hold/orchestration/test_orchestration.py` | `orchestration/inter_agent_simulation.py` | --- | Jules-01 |
| ❌ | `tests/hold/orchestration/test_orchestration_plugins.py` | `core_orchestration/agi_brain_orchestrator.py` | --- | Jules-01 |
| ❌ | `tests/hold/simple_security_test.py` | `N/A` | --- | Codex-Z |
| ❌ | `tests/hold/test_security_fixes.py` | `N/A` | --- | Codex-Z |
| ❌ | `tests/hold/test_security_fixes_validation.py` | `N/A` | --- | Codex-Z |
| ❌ | `tests/hold/test_security_fixes_verification.py` | `N/A` | --- | Codex-Z |
