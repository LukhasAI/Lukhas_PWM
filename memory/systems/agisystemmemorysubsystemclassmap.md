# AGI SYSTEM: MEMORY SUBSYSTEM CLASS & INTEGRATION MAP (HIGH PRIORITY)

## Overview
This map documents all major memory subsystem classes, their roles, and integration points, based on the latest code review. It is intended for future audit, onboarding, and architectural planning.

---

## 1. memory/CompatibilityMemoryManager.py
- **Class:** `MemoryManager`
  - **Role:** Central manager for storage, retrieval, and organization of system memories. Integrates trauma lock, identity, and dream reflection systems.
  - **Integration Points:**
    - `memory_folds` (folded memory structures)
    - `TraumaLock` (security/lockdown)
    - `MemoryIdentityIntegration` (identity/consent)
    - `DreamReflectionLoop` (dream-based memory processing)
  - **Key Methods:** `store`, `retrieve`, `forget`, `process_dream_cycle`

---

## 2. memory/adaptive_memory/AdaptiveMemoryEngine.py
- **Class:** `AdaptiveMemoryEngine`
  - **Role:** Self-adapting memory architecture, ready for integration with broader AGI system.
  - **Integration Points:**
    - Configurable for future adaptive/learning logic
    - Designed for async operation and system stats
  - **Key Methods:** `initialize`, `process`, `get_stats`, `shutdown`

---

## 3. memory/bio_symbolic_memory/BioSymbolicMemory.py
- **Class:** `BioSymbolicMemory`
  - **Role:** Bio-inspired memory system with working, episodic, semantic, and procedural memory.
  - **Integration Points:**
    - `MemoryConsolidationEngine` (pattern extraction/consolidation)
    - Used by learning and adaptation modules
  - **Key Methods:** `store_interaction`, `_compute_importance`, `_find_related_memories`

---

## 4. brain/memory/enhanced_memory_manager.py
- **Class:** `EnhancedMemoryManager`
  - **Role:** Advanced manager integrating emotional context, quantum attention, and enhanced retrieval.
  - **Integration Points:**
    - Emotional oscillator (emotion context)
    - Quantum attention (prioritized recall)
    - Clustering and consolidation logic
  - **Key Methods:** `store_with_emotional_context`, `retrieve_with_emotional_context`, `consolidate_memories`, `find_emotionally_similar_memories`

---

## 5. Cross-Subsystem Integration
- **Dream Reflection:** Memory managers integrate with dream engines for pattern recognition and memory evolution.
- **Identity & Consent:** Memory access and storage are gated by identity and access tier systems.
- **Emotional & Quantum Context:** Advanced managers use emotional and quantum signals for memory prioritization and retrieval.
- **Learning & Adaptation:** Bio-symbolic and adaptive engines are designed for integration with learning, adaptation, and creative modules.

---

## Actionable Notes
- **Audit all integration points regularly** for breaking changes or undocumented dependencies.
- **Expand test coverage** for all advanced/experimental memory managers.
- **Document new memory classes and methods** as they are added.
- **Maintain this map as a living document** for onboarding and architectural review. 