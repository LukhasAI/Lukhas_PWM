# Connectivity Index for lukhas/emotion

Generated: 2025-07-28T17:40:40.359779

## Summary

- **Total Modules:** 13
- **Total Symbols:** 224
- **Total Dependencies:** 90
- **Missed Opportunities:** 4

## 🔍 Missed Opportunities

### 🔴 Unused Exports
**Description:** Module lukhas.emotion.dreamseed_upgrade has 45 unused public symbols
**Affected Files:** lukhas.emotion.dreamseed_upgrade
**Suggestion:** Consider making these symbols private or removing them: EmotionalTier, SymbolicEmotionTag, EmotionalSafetyLevel, EmotionalAccessContext, SymbolicEmotionState...

### 🟡 Unused Exports
**Description:** Module lukhas.emotion.symbolic_user_intent has 8 unused public symbols
**Affected Files:** lukhas.emotion.symbolic_user_intent
**Suggestion:** Consider making these symbols private or removing them: intent_type, confidence, entities, raw_input, sid...

### 🔴 Unused Exports
**Description:** Module lukhas.emotion.dreamseed_unified has 34 unused public symbols
**Affected Files:** lukhas.emotion.dreamseed_unified
**Suggestion:** Consider making these symbols private or removing them: EmotionalTier, SymbolicEmotionTag, EmotionalSafetyLevel, UnifiedEmotionalAccessContext, UnifiedSymbolicEmotionState...

### 🔴 Unused Exports
**Description:** Module lukhas.emotion.tools.emotional_echo_detector has 45 unused public symbols
**Affected Files:** lukhas.emotion.tools.emotional_echo_detector
**Suggestion:** Consider making these symbols private or removing them: ArchetypePattern, EchoSeverity, EmotionalSequence, RecurringMotif, LoopReport...

## Module Details

### lukhas.emotion.affect_stagnation_detector

**Metrics:**
- Connectivity Score: 20.00%
- Cohesion Score: 0.00%
- Coupling Score: 33.33%
- Used/Total Symbols: 1/5

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| AffectStagnationDetector | class | True | N/A | ✅ |
| __init__ | function | False | 1 | ❌ |
| check_for_stagnation | function | False | 4 | ✅ |
| EmotionalMemory | class | False | N/A | ❌ |
| affect_vector_velocity | function | False | 1 | ❌ |

### lukhas.emotion.dreamseed_upgrade

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 91.67%
- Used/Total Symbols: 0/48

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| EmotionalTier | class | False | N/A | ✅ |
| SymbolicEmotionTag | class | False | N/A | ✅ |
| EmotionalSafetyLevel | class | False | N/A | ✅ |
| EmotionalAccessContext | dataclass | False | N/A | ✅ |
| SymbolicEmotionState | dataclass | False | N/A | ✅ |
| DriftRegulationResult | dataclass | False | N/A | ✅ |
| CodreamerIsolationResult | dataclass | False | N/A | ✅ |
| DreamSeedEmotionEngine | class | False | N/A | ✅ |
| create_dreamseed_emotion_engine | function | False | 1 | ✅ |
| user_id | constant | False | N/A | ❌ |
| session_id | constant | False | N/A | ❌ |
| tier_level | constant | False | N/A | ❌ |
| trust_score | constant | False | N/A | ❌ |
| dream_phase | constant | False | N/A | ❌ |
| codreamer_ids | constant | False | N/A | ❌ |
| safety_override | constant | False | N/A | ❌ |
| timestamp | constant | False | N/A | ❌ |
| emotion_vector | constant | False | N/A | ❌ |
| symbolic_tags | constant | False | N/A | ❌ |
| safety_level | constant | False | N/A | ❌ |
| drift_score | constant | False | N/A | ❌ |
| harmony_score | constant | False | N/A | ❌ |
| empathy_resonance | constant | False | N/A | ❌ |
| codreamer_isolation | constant | False | N/A | ❌ |
| ethical_flags | constant | False | N/A | ❌ |
| original_emotion | constant | False | N/A | ❌ |
| regulated_emotion | constant | False | N/A | ❌ |
| regulation_applied | constant | False | N/A | ❌ |
| safety_intervention | constant | False | N/A | ❌ |
| symbolic_tags_added | constant | False | N/A | ❌ |
| regulation_strength | constant | False | N/A | ❌ |
| user_emotion | constant | False | N/A | ❌ |
| codreamer_signatures | constant | False | N/A | ❌ |
| isolation_strength | constant | False | N/A | ❌ |
| bleed_through_detected | constant | False | N/A | ❌ |
| cross_contamination_risk | constant | False | N/A | ❌ |
| isolation_tags | constant | False | N/A | ❌ |
| __init__ | function | False | 2 | ❌ |
| assign_emotional_tier | function | False | 12 | ✅ |
| inject_symbolic_tags | function | False | 12 | ✅ |
| _calculate_harmony_score | function | False | 5 | ✅ |
| regulate_drift_feedback | function | False | 10 | ✅ |
| isolate_codreamer_affect | function | False | 8 | ✅ |
| enforce_emotional_safety | function | False | 10 | ✅ |
| process_dreamseed_emotion | function | False | 4 | ✅ |
| _log_to_file | function | False | 2 | ✅ |
| get_session_metrics | function | False | 2 | ✅ |
| get_system_health_report | function | False | 1 | ✅ |

### lukhas.emotion.multimodal_sentiment

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 16.67%
- Used/Total Symbols: 0/1

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| analyze_multimodal_sentiment | function | False | 7 | ✅ |

### lukhas.emotion.symbolic_user_intent

**Metrics:**
- Connectivity Score: 20.00%
- Cohesion Score: 0.00%
- Coupling Score: 41.67%
- Used/Total Symbols: 2/10

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| UserIntent | dataclass | True | N/A | ✅ |
| IntentEncoder | class | True | N/A | ✅ |
| intent_type | constant | False | N/A | ❌ |
| confidence | constant | False | N/A | ❌ |
| entities | constant | False | N/A | ❌ |
| raw_input | constant | False | N/A | ❌ |
| sid | constant | False | N/A | ❌ |
| drift_score | constant | False | N/A | ❌ |
| affect_delta | constant | False | N/A | ❌ |
| encode | function | False | 3 | ✅ |

### lukhas.emotion.cycler

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 25.00%
- Used/Total Symbols: 0/3

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| EmotionCycler | class | False | N/A | ✅ |
| __init__ | function | False | 2 | ❌ |
| next_emotion | function | False | 1 | ❌ |

### lukhas.emotion.mood_regulator

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 25.00%
- Used/Total Symbols: 0/3

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| MoodRegulator | class | False | N/A | ✅ |
| __init__ | function | False | 2 | ❌ |
| adjust_baseline_from_drift | function | False | 2 | ✅ |

### lukhas.emotion.dreamseed_unified

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 116.67%
- Used/Total Symbols: 0/52

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| EmotionalTier | class | False | N/A | ✅ |
| SymbolicEmotionTag | class | False | N/A | ✅ |
| EmotionalSafetyLevel | class | False | N/A | ✅ |
| UnifiedEmotionalAccessContext | dataclass | False | N/A | ✅ |
| UnifiedSymbolicEmotionState | dataclass | False | N/A | ✅ |
| UnifiedDreamSeedEmotionEngine | class | False | N/A | ✅ |
| create_unified_dreamseed_emotion_engine | function | False | 1 | ✅ |
| user_id | constant | False | N/A | ❌ |
| session_id | constant | False | N/A | ❌ |
| lambda_tier | constant | False | N/A | ❌ |
| legacy_tier | constant | False | N/A | ❌ |
| trust_score | constant | False | N/A | ❌ |
| dream_phase | constant | False | N/A | ❌ |
| codreamer_ids | constant | False | N/A | ❌ |
| safety_override | constant | False | N/A | ❌ |
| consent_grants | constant | False | N/A | ❌ |
| timestamp | constant | False | N/A | ❌ |
| emotion_vector | constant | False | N/A | ❌ |
| symbolic_tags | constant | False | N/A | ❌ |
| safety_level | constant | False | N/A | ❌ |
| drift_score | constant | False | N/A | ❌ |
| harmony_score | constant | False | N/A | ❌ |
| empathy_resonance | constant | False | N/A | ❌ |
| codreamer_isolation | constant | False | N/A | ❌ |
| ethical_flags | constant | False | N/A | ❌ |
| consent_required | constant | False | N/A | ❌ |
| __init__ | function | False | 2 | ❌ |
| assign_unified_emotional_tier | function | False | 13 | ✅ |
| process_unified_dreamseed_emotion | function | False | 10 | ✅ |
| analyze_emotional_patterns_unified | function | False | 4 | ✅ |
| modulate_emotional_state_unified | function | False | 4 | ✅ |
| _lambda_to_emotional_tier | function | False | 1 | ✅ |
| _get_unified_tier_features | function | False | 4 | ✅ |
| _inject_symbolic_tags_unified | function | False | 3 | ✅ |
| _isolate_codreamer_affect_unified | function | False | 2 | ✅ |
| _enforce_emotional_safety_unified | function | False | 3 | ✅ |
| _regulate_drift_feedback_unified | function | False | 2 | ✅ |
| inject_symbolic_tags | function | False | 1 | ✅ |
| isolate_codreamer_affect | function | False | 1 | ✅ |
| enforce_emotional_safety | function | False | 1 | ✅ |
| regulate_drift_feedback | function | False | 1 | ✅ |
| _calculate_harmony_score | function | False | 1 | ✅ |
| _get_user_emotional_memories | function | False | 1 | ❌ |
| _analyze_dominant_emotions | function | False | 1 | ❌ |
| _analyze_transitions | function | False | 1 | ❌ |
| _analyze_valence_trends | function | False | 1 | ❌ |
| _analyze_symbolic_patterns | function | False | 1 | ❌ |
| _verify_emotion_ownership | function | False | 1 | ❌ |
| _get_emotional_state | function | False | 1 | ❌ |
| _calculate_modulation_strength | function | False | 1 | ❌ |
| _apply_modulation_limits_unified | function | False | 1 | ❌ |
| _update_emotional_state | function | False | 1 | ❌ |

### lukhas.emotion.recurring_emotion_tracker

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 50.00%
- Used/Total Symbols: 0/6

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| RecurringEmotionTracker | class | False | N/A | ✅ |
| __init__ | function | False | 2 | ❌ |
| check_for_recurrence | function | False | 4 | ✅ |
| _check_recurrence | function | False | 6 | ✅ |
| _find_origin_dream | function | False | 1 | ✅ |
| update_bio_oscillator | function | False | 3 | ✅ |

### lukhas.emotion.mood_regulation.mood_entropy_tracker

**Metrics:**
- Connectivity Score: 14.29%
- Cohesion Score: 0.00%
- Coupling Score: 25.00%
- Used/Total Symbols: 1/7

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| MoodEntropyTracker | class | True | N/A | ❌ |
| __init__ | function | False | 1 | ❌ |
| add_mood_vector | function | False | 1 | ✅ |
| calculate_entropy | function | False | 2 | ✅ |
| get_mood_harmonics | function | False | 3 | ✅ |
| log_mood | function | False | 1 | ✅ |
| get_entropy | function | False | 10 | ✅ |

### lukhas.emotion.mood_regulation.mood_regulator

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 58.33%
- Used/Total Symbols: 0/6

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| MoodRegulator | class | False | N/A | ✅ |
| __init__ | function | False | 1 | ❌ |
| adjust_baseline_from_drift | function | False | 4 | ✅ |
| DriftAlignmentController | class | False | N/A | ❌ |
| align_drift | function | False | 1 | ❌ |
| suggest_modulation | function | False | 3 | ❌ |

### lukhas.emotion.affect_detection.affect_stagnation_detector

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 41.67%
- Used/Total Symbols: 0/3

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| AffectStagnationDetector | class | False | N/A | ✅ |
| __init__ | function | False | 2 | ❌ |
| check_for_stagnation | function | False | 6 | ✅ |

### lukhas.emotion.affect_detection.recurring_emotion_tracker

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 50.00%
- Used/Total Symbols: 0/7

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| RecurringEmotionTracker | class | False | N/A | ✅ |
| __init__ | function | False | 2 | ❌ |
| check_for_recurrence | function | False | 4 | ✅ |
| _check_recurrence | function | False | 6 | ✅ |
| _find_origin_dream | function | False | 1 | ✅ |
| update_bio_oscillator | function | False | 3 | ✅ |
| inject_dream_snapshot | function | False | 2 | ✅ |

### lukhas.emotion.tools.emotional_echo_detector

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 166.67%
- Used/Total Symbols: 0/73

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| ArchetypePattern | class | False | N/A | ✅ |
| EchoSeverity | class | False | N/A | ✅ |
| EmotionalSequence | dataclass | False | N/A | ✅ |
| RecurringMotif | dataclass | False | N/A | ✅ |
| LoopReport | dataclass | False | N/A | ✅ |
| ArchetypeDetector | class | False | N/A | ✅ |
| EmotionalEchoDetector | class | False | N/A | ✅ |
| main | function | False | 20 | ✅ |
| _load_sample_data | function | False | 4 | ✅ |
| _generate_synthetic_emotional_data | function | False | 7 | ✅ |
| sequence_id | constant | False | N/A | ❌ |
| timestamp | constant | False | N/A | ❌ |
| source | constant | False | N/A | ❌ |
| emotions | constant | False | N/A | ❌ |
| symbols | constant | False | N/A | ❌ |
| intensity | constant | False | N/A | ❌ |
| duration_minutes | constant | False | N/A | ❌ |
| context | constant | False | N/A | ❌ |
| motif_id | constant | False | N/A | ❌ |
| pattern | constant | False | N/A | ❌ |
| occurrences | constant | False | N/A | ❌ |
| first_seen | constant | False | N/A | ❌ |
| last_seen | constant | False | N/A | ❌ |
| frequency | constant | False | N/A | ❌ |
| intensity_trend | constant | False | N/A | ❌ |
| archetype_match | constant | False | N/A | ❌ |
| archetype_score | constant | False | N/A | ❌ |
| report_id | constant | False | N/A | ❌ |
| analysis_window | constant | False | N/A | ❌ |
| sequences_analyzed | constant | False | N/A | ❌ |
| motifs_detected | constant | False | N/A | ❌ |
| high_risk_motifs | constant | False | N/A | ❌ |
| eli_score | constant | False | N/A | ❌ |
| ris_score | constant | False | N/A | ❌ |
| severity | constant | False | N/A | ❌ |
| motifs | constant | False | N/A | ❌ |
| archetype_alerts | constant | False | N/A | ❌ |
| recommendations | constant | False | N/A | ❌ |
| __init__ | function | False | 1 | ✅ |
| _compile_patterns | function | False | 3 | ✅ |
| detect_archetype | function | False | 6 | ✅ |
| _calculate_pattern_match | function | False | 1 | ✅ |
| _direct_sequence_match | function | False | 5 | ✅ |
| _order_sensitive_match | function | False | 3 | ✅ |
| _lcs_length | function | False | 4 | ✅ |
| _simple_semantic_match | function | False | 5 | ✅ |
| extract_emotional_sequence | function | False | 6 | ✅ |
| _identify_source_type | function | False | 12 | ✅ |
| _extract_from_dream | function | False | 2 | ✅ |
| _extract_from_memory | function | False | 5 | ✅ |
| _extract_from_drift_log | function | False | 7 | ✅ |
| _extract_from_generic | function | False | 2 | ✅ |
| _extract_emotions_from_text | function | False | 3 | ✅ |
| _extract_symbols_from_text | function | False | 3 | ✅ |
| _extract_emotions_from_vector | function | False | 5 | ✅ |
| _extract_emotions_from_memory_log | function | False | 4 | ✅ |
| detect_recurring_motifs | function | False | 6 | ✅ |
| _generate_ngrams | function | False | 1 | ✅ |
| compute_loop_score | function | False | 5 | ✅ |
| _calculate_time_span | function | False | 2 | ✅ |
| generate_loop_report | function | False | 5 | ✅ |
| _parse_timestamp | function | False | 2 | ✅ |
| _determine_severity | function | False | 14 | ✅ |
| _generate_archetype_alerts | function | False | 3 | ✅ |
| _generate_recommendations | function | False | 8 | ✅ |
| _format_report_json | function | False | 1 | ✅ |
| _format_report_markdown | function | False | 7 | ✅ |
| emit_symbolic_echo_alert | function | False | 6 | ✅ |
| _generate_alert_description | function | False | 4 | ✅ |
| _get_alert_actions | function | False | 4 | ✅ |
| integrate_with_tuner | function | False | 1 | ✅ |
| integrate_with_governor | function | False | 1 | ✅ |
| get_semantic_group | function | False | 4 | ❌ |

