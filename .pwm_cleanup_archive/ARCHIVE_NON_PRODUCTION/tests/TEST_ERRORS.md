# Test Error Log

This file tracks the errors found while running the test suite.

## 🔥 Top 5 Missing Modules

1.  `fastapi`
2.  `bcrypt`
3.  `psycopg`
4.  `pydantic_settings`
5.  `qrcode`

## 🧹 Phase Cleanup Candidates

*   `tests/orchestration/test_orchestration_plugins.py`: This test is failing due to a `NameError`.
*   `orchestration/brain/test_logger.py`: This test is failing due to a `ModuleNotFoundError`.
*   `orchestration/test_orchestrator_demo.py`: This test is failing due to a `ModuleNotFoundError`.
*   `quantum/test_phase_quantum_integration.py`: This test is failing due to a `ModuleNotFoundError`.
*   `quantum/test_quantum_core.py`: This test is failing due to a `ModuleNotFoundError`.
*   `quantum/test_quantum_inspired_layer.py`: This test is failing due to a `ModuleNotFoundError`.
*   `temporary-test/test_dream_convergence_tester.py`: This test is failing due to a `ModuleNotFoundError`.
*   `temporary-test/Λ_dependency_connectivity_test.py`: This test is failing due to a file mismatch.
*   `tests/dreams/test_dream_convergence_tester.py`: This test is failing due to a `ModuleNotFoundError`.
*   `tests/emotion/test_emotion_recursion.py`: This test is failing due to a `ModuleNotFoundError`.

## Test Error Log

*   **File**: `oneiric/tests/conftest.py`
    *   **Error**: `ModuleNotFoundError: No module named 'fastapi'`
*   **File**: `oneiric/oneiric_core/identity/lukhas_id.py`
    *   **Error**: `ModuleNotFoundError: No module named 'bcrypt'`
*   **File**: `oneiric/oneiric_core/db/user_repository.py`
    *   **Error**: `ModuleNotFoundError: No module named 'psycopg'`
*   **File**: `oneiric/oneiric_core/settings.py`
    *   **Error**: `ModuleNotFoundError: No module named 'pydantic_settings'`
*   **File**: `lukhas/identity/backend/verifold/visual/glyph_stego_encoder.py`
    *   **Error**: `ModuleNotFoundError: No module named 'qrcode'`
*   **File**: `lukhas/identity/backend/qrglyphs/qrglymph_public.py`
    *   **Error**: `ModuleNotFoundError: No module named 'segno'`
*   **File**: `lukhas/identity/backend/verifold/tests/test_zkNarrativeProof_adapter.py`
    *   **Error**: `ModuleNotFoundError: No module named 'cryptography.zkNarrativeProof_adapter'`
*   **File**: `learning/reinforcement_learning_rpc_test.py`
    *   **Error**: `NameError: name 'Tuple' is not defined`
*   **File**: `core/lukhas_dast/engine_fixed.py`
    *   **Error**: `SyntaxError: invalid syntax`
*   **File**: `core/governance/compliance_engine.py`
    *   **Error**: `IndentationError: unexpected indent`
*   **File**: `core/advanced/brain/tests/test_intent_node.py`
    *   **Error**: `IndentationError: expected an indented block after 'except' statement on line 141`
*   **File**: `nodes/intent_node.py`
    *   **Error**: `IndentationError: expected an indented block after 'except' statement on line 141`
*   **File**: `creativity/neuro_haiku_generator.py`
    *   **Error**: `SyntaxError: invalid syntax`
*   **File**: `orchestration/brain/BIO_SYMBOLIC/mito_quantum_attention.py`
    *   **Error**: `NameError: name 'quantum_attention' is not defined`
*   **File**: `core/adaptive_systems/crista_optimizer/crista_optimizer.py`
    *   **Error**: `NameError: name 'OptimizationMode' is not defined`
*   **File**: `core/adaptive_systems/crista_optimizer/symbolic_network.py`
    *   **Error**: `NameError: name 'NodeType' is not defined`
*   **File**: `core/adaptive_systems/crista_optimizer/topology_manager.py`
    *   **Error**: `NameError: name 'NetworkHealth' is not defined`
*   **File**: `core/bio_orchestrator/orchestrator.py`
    *   **Error**: `ModuleNotFoundError: No module named 'core.quantum'`
*   **File**: `quantum/quantum_inspired_layer.py`
    *   **Error**: `ModuleNotFoundError: No module named 'quantum.base_oscillator'`
*   **File**: `core/bio_core/dream/test_dream.py`
    *   **Error**: `ModuleNotFoundError: No module named 'core.advanced.brain.dream_engine.enhanced_dream_engine'`
*   **File**: `core/interaction/test_symptom_reporter.py`
    *   **Error**: `ModuleNotFoundError: No module named 'symptom_reporter'`
*   **File**: `core/test_logger.py`
    *   **Error**: `ModuleNotFoundError: No module named 'trace.memoria_logger'`
*   **File**: `lukhas/identity/backend/qrglyphs/tests/test_create_qrglyph.py`
    *   **Error**: `ModuleNotFoundError: No module named 'qrglymph_public'`
*   **File**: `lukhas/identity/backend/verifold/tests/test_consent_scope_validator.py`
    *   **Error**: `ModuleNotFoundError: No module named 'identity.consent_scope_validator'`
*   **File**: `lukhas/identity/backend/verifold/tests/test_crypto_router.py`
    *   **Error**: `ModuleNotFoundError: No module named 'cryptography'`
*   **File**: `lukhas/identity/backend/verifold/tests/test_ethics_verifier.py`
    *   **Error**: `ModuleNotFoundError: No module named 'identity.ethics_verifier'`
*   **File**: `lukhas/identity/backend/verifold/tests/test_glyph_stego_encoder.py`
    *   **Error**: `ModuleNotFoundError: No module named 'visual'`
*   **File**: `lukhas/identity/backend/verifold/tests/test_symbolic_audit_mode.py`
    *   **Error**: `ModuleNotFoundError: No module named 'symbolic_audit_mode'`
*   **File**: `lukhas/identity/tests/test_authentication_server.py`
    *   **Error**: `ModuleNotFoundError: No module named 'tests.test_authentication_server'`
*   **File**: `lukhas/identity/tests/test_comprehensive.py`
    *   **Error**: `ModuleNotFoundError: No module named 'tests.test_comprehensive'`
*   **File**: `lukhas/identity/tests/test_constitutional_enforcer.py`
    *   **Error**: `ModuleNotFoundError: No module named 'backend.audit_logger'`
*   **File**: `lukhas/identity/tests/test_constitutional_gatekeeper.py`
    *   **Error**: `ImportError: attempted relative import with no known parent package`
*   **File**: `lukhas/identity/tests/test_core_components.py`
    *   **Error**: `ModuleNotFoundError: No module named 'core.constitutional_gatekeeper'`
*   **File**: `lukhas/identity/tests/test_cultural_safety.py`
    *   **Error**: `ModuleNotFoundError: No module named 'core.constitutional_gatekeeper'`
*   **File**: `lukhas/identity/tests/test_emoji_grid_sizing.py`
    *   **Error**: `ModuleNotFoundError: No module named 'core.constitutional_gatekeeper'`
*   **File**: `lukhas/identity/tests/test_entropy_synchronizer.py`
    *   **Error**: `ModuleNotFoundError: No module named 'mobile'`
*   **File**: `lukhas/identity/tests/test_integration.py`
    *   **Error**: `ModuleNotFoundError: No module named 'core.constitutional_gatekeeper'`
*   **File**: `lukhas/identity/tests/test_mobile_qr_refresh.py`
    *   **Error**: `ModuleNotFoundError: No module named 'mobile'`
*   **File**: `memory/core_memory/test_memory_engines.py`
    *   **Error**: `ModuleNotFoundError: No module named 'memory.adaptive_memory'`
*   **File**: `memory/core_memory/test_memory_system.py`
    *   **Error**: `ModuleNotFoundError: No module named 'CORE'`
*   **File**: `orchestration/brain/personality/test_cretivity.py`
    *   **Error**: `ModuleNotFoundError: No module named 'neuro_haiku_generator'`
*   **File**: `orchestration/brain/test_batch_efficiency.py`
    *   **Error**: `ModuleNotFoundError: No module named 'github_vulnerability_manager'`
*   **File**: `orchestration/brain/test_integration.py`
    *   **Error**: `AttributeError: module 'threading' has no attribute 'Future'`
*   **File**: `orchestration/test_orchestrator_demo.py`
    *   **Error**: `ModuleNotFoundError: No module named 'orchestration.brain.prime_oscillator'`
*   **File**: `quantum/test_phase_quantum_integration.py`
    *   **Error**: `ModuleNotFoundError: No module named 'lukhas'`
*   **File**: `quantum/test_quantum_core.py`
    *   **Error**: `ModuleNotFoundError: No module named 'src'`
*   **File**: `quantum/test_quantum_inspired_layer.py`
    *   **Error**: `ModuleNotFoundError: No module named 'CORE'`
*   **File**: `temporary-test/test_dream_convergence_tester.py`
    *   **Error**: `ModuleNotFoundError: No module named 'dream_convergence_tester'`
*   **File**: `temporary-test/Λ_dependency_connectivity_test.py`
    *   **Error**: `import file mismatch`
*   **File**: `tests/dreams/test_dream_convergence_tester.py`
    *   **Error**: `ModuleNotFoundError: No module named 'core.docututor.memory_evolution.memory_evolution'`
*   **File**: `tests/emotion/test_emotion_recursion.py`
    *   **Error**: `ModuleNotFoundError: No module named 'symbolic_drift_tracker'`
*   **File**: `tests/orchestration/test_orchestration_plugins.py`
    *   **Error**: `ModuleNotFoundError: No module named 'core_orchestration.agi_brain_orchestrator'`
