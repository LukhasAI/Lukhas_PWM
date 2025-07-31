# Connectivity Index for lukhas/learning

Generated: 2025-07-28T17:40:41.571424

## Summary

- **Total Modules:** 28
- **Total Symbols:** 537
- **Total Dependencies:** 176
- **Missed Opportunities:** 16

## 🔍 Missed Opportunities

### 🔴 Unused Exports
**Description:** Module lukhas.learning.federated_meta_learning has 22 unused public symbols
**Affected Files:** lukhas.learning.federated_meta_learning
**Suggestion:** Consider making these symbols private or removing them: FederatedModel, FederatedLearningManager, ReflectiveIntrospectionSystem, MetaLearningSystem, update_with_gradients...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.plugin_learning_engine has 11 unused public symbols
**Affected Files:** lukhas.learning.plugin_learning_engine
**Suggestion:** Consider making these symbols private or removing them: ContentType, UserLevel, GenerationConfig, PluginLearningEngine, content_type...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.service has 10 unused public symbols
**Affected Files:** lukhas.learning.service
**Suggestion:** Consider making these symbols private or removing them: LearningService, learn_from_data, adapt_behavior, synthesize_knowledge, transfer_learning...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.usage_learning has 11 unused public symbols
**Affected Files:** lukhas.learning.usage_learning
**Suggestion:** Consider making these symbols private or removing them: UserInteraction, InteractionPattern, UsageBasedLearning, update, record_interaction...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.federated_learning_system has 13 unused public symbols
**Affected Files:** lukhas.learning.federated_learning_system
**Suggestion:** Consider making these symbols private or removing them: LukhasFederatedModel, LukhasFederatedLearningManager, initialize_lukhas_federated_learning, update_with_gradients, get_parameters...

### 🟡 Unused Exports
**Description:** Module lukhas.learning.meta_learning has 9 unused public symbols
**Affected Files:** lukhas.learning.meta_learning
**Suggestion:** Consider making these symbols private or removing them: LearningMetrics, MetaLearningSystem, accuracy, loss, insights_gained...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.meta_learning_adapter has 31 unused public symbols
**Affected Files:** lukhas.learning.meta_learning_adapter
**Suggestion:** Consider making these symbols private or removing them: LearningPhase, FederatedState, MetaLearningMetrics, LearningRateBounds, MetaLearningEnhancementAdapter...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.metalearningenhancementsystem has 11 unused public symbols
**Affected Files:** lukhas.learning.metalearningenhancementsystem
**Suggestion:** Consider making these symbols private or removing them: Enhancementmode, Systemintegrationstatus, MetaLearningEnhancementsystem, meta_learning_systems_found, systems_enhanced...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.meta_learning_recovery has 10 unused public symbols
**Affected Files:** lukhas.learning.meta_learning_recovery
**Suggestion:** Consider making these symbols private or removing them: MetaLearningRecovery, main, explore_meta_learning_directory, convert_to_lukhas_format, determine_target_directory...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.system has 30 unused public symbols
**Affected Files:** lukhas.learning.system
**Suggestion:** Consider making these symbols private or removing them: LearningType, LearningStrategy, LearningEpisode, MetaLearningResult, BaseMetaLearner...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.tutor has 27 unused public symbols
**Affected Files:** lukhas.learning.tutor
**Suggestion:** Consider making these symbols private or removing them: LearningStyle, DifficultyLevel, TutorMessageType, LearningObjective, TutorMessage...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.meta_adaptive.meta_learning has 22 unused public symbols
**Affected Files:** lukhas.learning.meta_adaptive.meta_learning
**Suggestion:** Consider making these symbols private or removing them: FederatedModel, FederatedLearningManager, ReflectiveIntrospectionSystem, MetaLearningSystem, update_with_gradients...

### 🔴 Unused Exports
**Description:** Module lukhas.learning._dict_learning has 13 unused public symbols
**Affected Files:** lukhas.learning._dict_learning
**Suggestion:** Consider making these symbols private or removing them: sparse_encode, dict_learning_online, dict_learning, SparseCoder, DictionaryLearning...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.meta_learning.federated_integration has 29 unused public symbols
**Affected Files:** lukhas.learning.meta_learning.federated_integration
**Suggestion:** Consider making these symbols private or removing them: Federationstrategy, Privacylevel, Federatednode, Federatedlearningupdate, Federatedlearningintegration...

### 🔴 Unused Exports
**Description:** Module lukhas.learning.meta_learning.symbolic_feedback has 45 unused public symbols
**Affected Files:** lukhas.learning.meta_learning.symbolic_feedback
**Suggestion:** Consider making these symbols private or removing them: Feedbacktype, Symboliccontext, Intentnodehistory, Memoriasnapshot, Dreamreplayrecord...

### 🟢 Isolated Module
**Description:** Module lukhas.learning.systems.lukhas_duet_conductor is isolated (no imports or exports)
**Affected Files:** lukhas.learning.systems.lukhas_duet_conductor
**Suggestion:** Consider if this module should be integrated with others or removed

## Module Details

### lukhas.learning.federated_meta_learning

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 29.63%
- Used/Total Symbols: 0/45

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| FederatedModel | class | False | N/A | ✅ |
| FederatedLearningManager | class | False | N/A | ✅ |
| ReflectiveIntrospectionSystem | class | False | N/A | ✅ |
| MetaLearningSystem | class | False | N/A | ✅ |
| __init__ | function | False | 2 | ❌ |
| update_with_gradients | function | False | 4 | ✅ |
| get_parameters | function | False | 1 | ✅ |
| serialize | function | False | 1 | ✅ |
| deserialize | function | False | 1 | ✅ |
| register_model | function | False | 2 | ✅ |
| get_model | function | False | 3 | ✅ |
| contribute_gradients | function | False | 5 | ✅ |
| _aggregate_model | function | False | 2 | ✅ |
| _update_metrics | function | False | 4 | ✅ |
| save_model | function | False | 2 | ✅ |
| load_models | function | False | 4 | ✅ |
| get_client_status | function | False | 5 | ✅ |
| log_interaction | function | False | 6 | ✅ |
| reflect | function | False | 2 | ✅ |
| _analyze_interactions | function | False | 7 | ✅ |
| _detect_user_patterns | function | False | 6 | ✅ |
| _detect_error_patterns | function | False | 5 | ✅ |
| _calculate_trend | function | False | 5 | ✅ |
| _generate_improvement_plans | function | False | 10 | ✅ |
| _implement_improvements | function | False | 2 | ✅ |
| get_status_report | function | False | 1 | ✅ |
| _register_core_models | function | False | 1 | ✅ |
| optimize_learning_approach | function | False | 2 | ✅ |
| incorporate_feedback | function | False | 9 | ✅ |
| generate_learning_report | function | False | 1 | ✅ |
| get_federated_model | function | False | 1 | ✅ |
| trigger_reflection | function | False | 1 | ✅ |
| _initialize_strategies | function | False | 1 | ✅ |
| _extract_learning_features | function | False | 1 | ✅ |
| _select_strategy | function | False | 4 | ✅ |
| _apply_strategy | function | False | 7 | ✅ |
| _evaluate_performance | function | False | 2 | ✅ |
| _update_strategy_performance | function | False | 6 | ✅ |
| _update_meta_parameters | function | False | 2 | ✅ |
| _adjust_strategy_parameters | function | False | 4 | ✅ |
| _calculate_adaptation_progress | function | False | 5 | ✅ |
| _calculate_sparsity | function | False | 1 | ✅ |
| _estimate_complexity | function | False | 1 | ✅ |
| _calculate_strategy_match | function | False | 10 | ✅ |
| _generate_meta_insights | function | False | 8 | ✅ |

### lukhas.learning.plugin_learning_engine

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 33.33%
- Used/Total Symbols: 0/12

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| ContentType | class | False | N/A | ✅ |
| UserLevel | class | False | N/A | ✅ |
| GenerationConfig | class | False | N/A | ✅ |
| PluginLearningEngine | class | False | N/A | ✅ |
| content_type | type_alias | False | N/A | ❌ |
| user_level | constant | False | N/A | ❌ |
| voice_enabled | constant | False | N/A | ❌ |
| bio_oscillator_aware | constant | False | N/A | ❌ |
| max_complexity | constant | False | N/A | ❌ |
| cultural_context | constant | False | N/A | ❌ |
| __init__ | function | False | 1 | ✅ |
| get_optimal_complexity | function | False | 1 | ✅ |

### lukhas.learning.service

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 29.63%
- Used/Total Symbols: 0/21

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| LearningService | class | False | N/A | ✅ |
| learn_from_data | function | False | 7 | ✅ |
| adapt_behavior | function | False | 4 | ✅ |
| synthesize_knowledge | function | False | 4 | ✅ |
| __init__ | function | False | 1 | ✅ |
| transfer_learning | function | False | 4 | ✅ |
| get_learning_metrics | function | False | 6 | ✅ |
| _process_learning_data | function | False | 1 | ✅ |
| _update_knowledge_base | function | False | 1 | ✅ |
| _get_knowledge_updates | function | False | 1 | ✅ |
| _process_behavior_adaptation | function | False | 1 | ✅ |
| _synthesize_knowledge_sources | function | False | 1 | ✅ |
| _update_knowledge_graph | function | False | 2 | ✅ |
| _process_transfer_learning | function | False | 1 | ✅ |
| _get_detailed_learning_patterns | function | False | 1 | ✅ |
| _analyze_adaptation_trends | function | False | 1 | ✅ |
| _track_knowledge_evolution | function | False | 1 | ✅ |
| IdentityClient | class | False | N/A | ❌ |
| verify_user_access | function | False | 1 | ❌ |
| check_consent | function | False | 1 | ❌ |
| log_activity | function | False | 1 | ❌ |

### lukhas.learning.tutor_learning_engine

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 18.52%
- Used/Total Symbols: 0/4

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| TestTutorLearningEngine | class | False | N/A | ❌ |
| skg | function | False | 1 | ✅ |
| tutor_engine | function | False | 1 | ✅ |
| sample_config | function | False | 1 | ✅ |

### lukhas.learning.usage_learning

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 18.52%
- Used/Total Symbols: 0/12

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| UserInteraction | class | False | N/A | ✅ |
| InteractionPattern | class | False | N/A | ✅ |
| UsageBasedLearning | class | False | N/A | ✅ |
| __init__ | function | False | 1 | ❌ |
| update | function | False | 1 | ✅ |
| record_interaction | function | False | 5 | ✅ |
| identify_patterns | function | False | 4 | ✅ |
| update_user_preferences | function | False | 1 | ✅ |
| get_document_effectiveness | function | False | 2 | ✅ |
| get_popular_sequences | function | False | 1 | ✅ |
| recommend_next_docs | function | False | 7 | ✅ |
| recommendations_list | constant | False | N/A | ❌ |

### lukhas.learning.federated_learning_system

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 33.33%
- Used/Total Symbols: 0/16

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| LukhasFederatedModel | class | False | N/A | ✅ |
| LukhasFederatedLearningManager | class | False | N/A | ✅ |
| initialize_lukhas_federated_learning | function | False | 2 | ✅ |
| __init__ | function | False | 2 | ❌ |
| update_with_gradients | function | False | 8 | ✅ |
| get_parameters | function | False | 1 | ✅ |
| serialize | function | False | 1 | ✅ |
| deserialize | function | False | 1 | ✅ |
| register_model | function | False | 2 | ✅ |
| get_model | function | False | 3 | ✅ |
| contribute_gradients | function | False | 5 | ✅ |
| _calculate_client_weight | function | False | 4 | ✅ |
| _trigger_aggregation | function | False | 1 | ✅ |
| save_model | function | False | 2 | ✅ |
| load_models | function | False | 5 | ✅ |
| get_system_status | function | False | 1 | ✅ |

### lukhas.learning.federated_learning

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 14.81%
- Used/Total Symbols: 0/10

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| FederatedLearningManager | class | False | N/A | ✅ |
| __init__ | function | False | 1 | ❌ |
| register_model | function | False | 1 | ✅ |
| get_model | function | False | 3 | ✅ |
| contribute_gradients | function | False | 5 | ✅ |
| _weighted_update | function | False | 3 | ✅ |
| _ensure_storage_exists | function | False | 1 | ✅ |
| _get_model_path | function | False | 1 | ✅ |
| _persist_model | function | False | 1 | ✅ |
| _load_model | function | False | 2 | ✅ |

### lukhas.learning.exponential_learning

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 3.70%
- Used/Total Symbols: 0/6

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| ExponentialLearningSystem | class | False | N/A | ✅ |
| __init__ | function | False | 2 | ❌ |
| incorporate_experience | function | False | 2 | ✅ |
| _extract_patterns | function | False | 3 | ✅ |
| _update_knowledge | function | False | 4 | ✅ |
| _consolidate_knowledge | function | False | 2 | ✅ |

### lukhas.learning.meta_learning

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 14.81%
- Used/Total Symbols: 0/17

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| LearningMetrics | dataclass | False | N/A | ✅ |
| MetaLearningSystem | class | False | N/A | ✅ |
| accuracy | constant | False | N/A | ❌ |
| loss | constant | False | N/A | ❌ |
| insights_gained | constant | False | N/A | ❌ |
| adaptations_made | constant | False | N/A | ❌ |
| confidence_score | constant | False | N/A | ❌ |
| learning_efficiency | constant | False | N/A | ❌ |
| __init__ | function | False | 1 | ❌ |
| incorporate_feedback | function | False | 1 | ✅ |
| _select_learning_strategy | function | False | 3 | ✅ |
| _apply_federated_knowledge | function | False | 3 | ✅ |
| _generate_learning_plan | function | False | 1 | ✅ |
| _update_metrics | function | False | 1 | ✅ |
| _update_federated_models | function | False | 2 | ✅ |
| _update_symbolic_db | function | False | 6 | ✅ |
| _adapt_learning_strategies | function | False | 2 | ✅ |

### lukhas.learning.embodied_thought.generative_reflex

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 7.41%
- Used/Total Symbols: 0/4

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| GenerativeReflex | class | False | N/A | ✅ |
| __init__ | function | False | 1 | ❌ |
| load_reflex | function | False | 1 | ✅ |
| generate_response | function | False | 1 | ✅ |

### lukhas.learning.meta_learning_adapter

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 25.93%
- Used/Total Symbols: 0/39

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| LearningPhase | class | False | N/A | ✅ |
| FederatedState | class | False | N/A | ✅ |
| MetaLearningMetrics | dataclass | False | N/A | ✅ |
| LearningRateBounds | dataclass | False | N/A | ✅ |
| MetaLearningEnhancementAdapter | class | False | N/A | ✅ |
| current_learning_rate | constant | False | N/A | ❌ |
| learning_rate_adaptation | constant | False | N/A | ❌ |
| learning_rate_stability | constant | False | N/A | ❌ |
| optimal_rate_distance | constant | False | N/A | ❌ |
| federated_nodes_active | constant | False | N/A | ❌ |
| federated_convergence | constant | False | N/A | ❌ |
| consensus_quality | constant | False | N/A | ❌ |
| communication_efficiency | constant | False | N/A | ❌ |
| symbolic_feedback_quality | constant | False | N/A | ❌ |
| intent_node_integration | constant | False | N/A | ❌ |
| memoria_coupling | constant | False | N/A | ❌ |
| dream_system_coherence | constant | False | N/A | ❌ |
| overall_performance | constant | False | N/A | ❌ |
| adaptation_speed | constant | False | N/A | ❌ |
| stability_score | constant | False | N/A | ❌ |
| enhancement_factor | constant | False | N/A | ❌ |
| timestamp | constant | False | N/A | ❌ |
| current_phase | constant | False | N/A | ❌ |
| federated_state | constant | False | N/A | ❌ |
| min_rate | constant | False | N/A | ❌ |
| max_rate | constant | False | N/A | ❌ |
| optimal_rate | constant | False | N/A | ❌ |
| adaptation_factor | constant | False | N/A | ❌ |
| decay_factor | constant | False | N/A | ❌ |
| momentum | constant | False | N/A | ❌ |
| __init__ | function | False | 1 | ✅ |
| _normalize_membrane_potential | function | False | 1 | ❌ |
| _calculate_rate_stability | function | False | 3 | ❌ |
| _calculate_adaptation_speed | function | False | 2 | ❌ |
| _calculate_overall_stability | function | False | 1 | ❌ |
| _calculate_enhancement_factor | function | False | 1 | ❌ |
| _calculate_overall_enhancement | function | False | 1 | ❌ |
| _get_default_metrics | function | False | 1 | ❌ |
| enhancement_results | constant | False | N/A | ❌ |

### lukhas.learning.metalearningenhancementsystem

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 40.74%
- Used/Total Symbols: 0/16

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| Enhancementmode | class | False | N/A | ✅ |
| Systemintegrationstatus | dataclass | False | N/A | ✅ |
| MetaLearningEnhancementsystem | class | False | N/A | ✅ |
| meta_learning_systems_found | constant | False | N/A | ❌ |
| systems_enhanced | constant | False | N/A | ❌ |
| monitoring_active | constant | False | N/A | ❌ |
| rate_optimization_active | constant | False | N/A | ❌ |
| symbolic_feedback_active | constant | False | N/A | ❌ |
| federation_enabled | constant | False | N/A | ❌ |
| last_health_check | constant | False | N/A | ❌ |
| integration_errors | constant | False | N/A | ❌ |
| __init__ | function | False | 2 | ❌ |
| _simulate_meta_learning_system_discovery | function | False | 1 | ✅ |
| _create_mock_system | function | False | 1 | ✅ |
| _synchronize_quantum_signatures | function | False | 2 | ✅ |
| _analyze_common_ethical_issues | function | False | 3 | ✅ |

### lukhas.learning.meta_learning_recovery

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 25.93%
- Used/Total Symbols: 0/11

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| MetaLearningRecovery | class | False | N/A | ✅ |
| main | function | False | 3 | ✅ |
| __init__ | function | False | 1 | ❌ |
| explore_meta_learning_directory | function | False | 9 | ✅ |
| convert_to_lukhas_format | function | False | 15 | ✅ |
| determine_target_directory | function | False | 3 | ✅ |
| recover_meta_learning_components | function | False | 6 | ✅ |
| execute_recovery | function | False | 5 | ✅ |
| exploration_result | constant | False | N/A | ❌ |
| recovery_result | constant | False | N/A | ❌ |
| target_counts | constant | False | N/A | ❌ |

### lukhas.learning.system

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 40.74%
- Used/Total Symbols: 0/36

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| LearningType | class | False | N/A | ✅ |
| LearningStrategy | class | False | N/A | ✅ |
| LearningEpisode | dataclass | False | N/A | ✅ |
| MetaLearningResult | dataclass | False | N/A | ✅ |
| BaseMetaLearner | class | False | N/A | ✅ |
| ModelAgnosticMetaLearner | class | False | N/A | ✅ |
| FewShotLearner | class | False | N/A | ✅ |
| ContinualLearner | class | False | N/A | ✅ |
| AdvancedLearningSystem | class | False | N/A | ✅ |
| episode_id | constant | False | N/A | ❌ |
| task_type | constant | False | N/A | ❌ |
| support_set | constant | False | N/A | ❌ |
| query_set | constant | False | N/A | ❌ |
| learning_objective | constant | False | N/A | ❌ |
| metadata | constant | False | N/A | ❌ |
| timestamp | constant | False | N/A | ❌ |
| performance_metrics | constant | False | N/A | ❌ |
| learned_strategy | constant | False | N/A | ❌ |
| adaptation_speed | constant | False | N/A | ❌ |
| generalization_score | constant | False | N/A | ❌ |
| memory_efficiency | constant | False | N/A | ❌ |
| confidence | constant | False | N/A | ❌ |
| applicable_domains | constant | False | N/A | ❌ |
| __init__ | function | False | 1 | ❌ |
| _calculate_generalization_score | function | False | 2 | ✅ |
| _calculate_memory_efficiency | function | False | 1 | ✅ |
| _extract_applicable_domains | function | False | 4 | ✅ |
| _setup_logging | function | False | 1 | ✅ |
| _analyze_learning_history | function | False | 4 | ✅ |
| domains | constant | False | N/A | ❌ |
| class_prototypes | constant | False | N/A | ❌ |
| class_groups | constant | False | N/A | ❌ |
| prototype | constant | False | N/A | ❌ |
| all_keys | constant | False | N/A | ❌ |
| result | constant | False | N/A | ❌ |
| adaptation_results | constant | False | N/A | ❌ |

### lukhas.learning.tutor

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 22.22%
- Used/Total Symbols: 0/34

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| LearningStyle | class | False | N/A | ❌ |
| DifficultyLevel | class | False | N/A | ❌ |
| TutorMessageType | class | False | N/A | ❌ |
| LearningObjective | class | False | N/A | ✅ |
| TutorMessage | class | False | N/A | ✅ |
| LearningSession | class | False | N/A | ✅ |
| TutorEngine | class | False | N/A | ✅ |
| id | constant | False | N/A | ❌ |
| description | constant | False | N/A | ❌ |
| required_concepts | constant | False | N/A | ❌ |
| difficulty | constant | False | N/A | ❌ |
| estimated_time_minutes | constant | False | N/A | ❌ |
| content | constant | False | N/A | ❌ |
| message_type | type_alias | False | N/A | ❌ |
| voice_style | constant | False | N/A | ❌ |
| cultural_context | constant | False | N/A | ❌ |
| visual_aids | constant | False | N/A | ❌ |
| session_id | constant | False | N/A | ❌ |
| user_id | constant | False | N/A | ❌ |
| topic | constant | False | N/A | ❌ |
| objectives | constant | False | N/A | ❌ |
| current_objective_index | constant | False | N/A | ❌ |
| start_time | constant | False | N/A | ❌ |
| messages | constant | False | N/A | ❌ |
| bio_metrics | constant | False | N/A | ❌ |
| voice_enabled | constant | False | N/A | ❌ |
| Config | class | False | N/A | ❌ |
| __init__ | function | False | 1 | ❌ |
| _generate_learning_objectives | function | False | 3 | ✅ |
| _estimate_learning_time | function | False | 1 | ✅ |
| _generate_welcome_message | function | False | 1 | ✅ |
| _should_adjust_difficulty | function | False | 3 | ✅ |
| _analyze_understanding | function | False | 1 | ✅ |
| _generate_hint | function | False | 1 | ✅ |

### lukhas.learning.meta_adaptive.adaptive_interface_generator

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 7.41%
- Used/Total Symbols: 0/21

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| AdaptiveInterfaceGenerator | class | False | N/A | ✅ |
| __init__ | function | False | 2 | ❌ |
| generate_interface | function | False | 1 | ✅ |
| _get_user_profile | function | False | 1 | ✅ |
| _analyze_context_needs | function | False | 1 | ✅ |
| _get_device_layout | function | False | 4 | ✅ |
| _select_components | function | False | 7 | ✅ |
| _arrange_components | function | False | 4 | ✅ |
| _apply_styling | function | False | 1 | ✅ |
| _define_interactions | function | False | 2 | ✅ |
| _define_animations | function | False | 2 | ✅ |
| _enhance_accessibility | function | False | 2 | ✅ |
| _load_device_profiles | function | False | 1 | ✅ |
| _load_components | function | False | 1 | ✅ |
| _component_addresses_need | function | False | 2 | ✅ |
| _get_component_spec | function | False | 1 | ✅ |
| _find_optimal_placement | function | False | 1 | ✅ |
| _get_base_style | function | False | 1 | ✅ |
| _adjust_for_user | function | False | 2 | ✅ |
| _rotate_grid | function | False | 1 | ✅ |
| _get_standard_interactions | function | False | 1 | ✅ |

### lukhas.learning.meta_learning.meta_core

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 11.11%
- Used/Total Symbols: 0/2

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| MetaCore | class | False | N/A | ✅ |
| __init__ | function | False | 1 | ❌ |

### lukhas.learning.systems.lukhas_core

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 44.44%
- Used/Total Symbols: 0/1

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| process_user_input | function | False | 6 | ❌ |

### lukhas.learning.systems.lukhas_voice_duet

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 3.70%
- Used/Total Symbols: 0/1

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| synthesize_voice | function | False | 2 | ✅ |

### lukhas.learning.adaptive_meta_learning

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 18.52%
- Used/Total Symbols: 0/23

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| AdaptiveMetaLearningSystem | class | False | N/A | ✅ |
| __init__ | function | False | 2 | ❌ |
| optimize_learning_approach | function | False | 2 | ✅ |
| incorporate_feedback | function | False | 5 | ✅ |
| generate_learning_report | function | False | 1 | ✅ |
| _initialize_strategies | function | False | 1 | ✅ |
| _extract_learning_features | function | False | 1 | ✅ |
| _select_strategy | function | False | 4 | ✅ |
| _apply_strategy | function | False | 7 | ✅ |
| _evaluate_performance | function | False | 4 | ✅ |
| _update_strategy_performance | function | False | 6 | ✅ |
| _update_meta_parameters | function | False | 4 | ✅ |
| _adjust_strategy_parameters | function | False | 4 | ✅ |
| _calculate_adaptation_progress | function | False | 5 | ✅ |
| _analyze_performance_trends | function | False | 3 | ✅ |
| _calculate_sparsity | function | False | 1 | ❌ |
| _estimate_complexity | function | False | 1 | ❌ |
| _estimate_noise_level | function | False | 1 | ❌ |
| _check_label_availability | function | False | 3 | ❌ |
| _calculate_strategy_match | function | False | 11 | ✅ |
| _calculate_confidence | function | False | 7 | ✅ |
| _generate_meta_insights | function | False | 9 | ✅ |
| demo_meta_learning | function | False | 2 | ✅ |

### lukhas.learning.systems.intent_language

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 11.11%
- Used/Total Symbols: 0/2

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| interpret_intent | function | False | 3 | ❌ |
| log_interpretation | function | False | 1 | ❌ |

### lukhas.learning.systems.lukhas_duet_conductor

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 0.00%
- Used/Total Symbols: 0/1

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| manage_voice_handoff | function | False | 6 | ✅ |

### lukhas.learning.systems.symbolic_voice_loop

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 25.93%
- Used/Total Symbols: 0/5

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| speak | function | False | 2 | ❌ |
| reflect_with_lukhas | function | False | 1 | ❌ |
| listen_and_log_feedback | function | False | 3 | ❌ |
| generate_dream_outcomes | function | False | 1 | ❌ |
| lukhas_emotional_response | function | False | 2 | ❌ |

### lukhas.learning.aid.dream_engine.narration_controller

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 7.41%
- Used/Total Symbols: 0/3

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| fetch_narration_entries | function | False | 2 | ❌ |
| load_user_settings | function | False | 6 | ❌ |
| filter_narration_queue | function | False | 1 | ❌ |

### lukhas.learning.meta_adaptive.meta_learning

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 29.63%
- Used/Total Symbols: 0/45

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| FederatedModel | class | False | N/A | ✅ |
| FederatedLearningManager | class | False | N/A | ✅ |
| ReflectiveIntrospectionSystem | class | False | N/A | ✅ |
| MetaLearningSystem | class | False | N/A | ✅ |
| __init__ | function | False | 2 | ❌ |
| update_with_gradients | function | False | 4 | ✅ |
| get_parameters | function | False | 1 | ✅ |
| serialize | function | False | 1 | ✅ |
| deserialize | function | False | 1 | ✅ |
| register_model | function | False | 2 | ✅ |
| get_model | function | False | 3 | ✅ |
| contribute_gradients | function | False | 5 | ✅ |
| _aggregate_model | function | False | 2 | ✅ |
| _update_metrics | function | False | 4 | ✅ |
| save_model | function | False | 2 | ✅ |
| load_models | function | False | 4 | ✅ |
| get_client_status | function | False | 5 | ✅ |
| log_interaction | function | False | 6 | ✅ |
| reflect | function | False | 2 | ✅ |
| _analyze_interactions | function | False | 7 | ✅ |
| _detect_user_patterns | function | False | 6 | ✅ |
| _detect_error_patterns | function | False | 5 | ✅ |
| _calculate_trend | function | False | 5 | ✅ |
| _generate_improvement_plans | function | False | 10 | ✅ |
| _implement_improvements | function | False | 2 | ✅ |
| get_status_report | function | False | 1 | ✅ |
| _register_core_models | function | False | 1 | ✅ |
| optimize_learning_approach | function | False | 2 | ✅ |
| incorporate_feedback | function | False | 9 | ✅ |
| generate_learning_report | function | False | 1 | ✅ |
| get_federated_model | function | False | 1 | ✅ |
| trigger_reflection | function | False | 1 | ✅ |
| _initialize_strategies | function | False | 1 | ✅ |
| _extract_learning_features | function | False | 1 | ✅ |
| _select_strategy | function | False | 4 | ✅ |
| _apply_strategy | function | False | 7 | ✅ |
| _evaluate_performance | function | False | 2 | ✅ |
| _update_strategy_performance | function | False | 6 | ✅ |
| _update_meta_parameters | function | False | 2 | ✅ |
| _adjust_strategy_parameters | function | False | 4 | ✅ |
| _calculate_adaptation_progress | function | False | 5 | ✅ |
| _calculate_sparsity | function | False | 1 | ✅ |
| _estimate_complexity | function | False | 1 | ✅ |
| _calculate_strategy_match | function | False | 10 | ✅ |
| _generate_meta_insights | function | False | 8 | ✅ |

### lukhas.learning._dict_learning

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 55.56%
- Used/Total Symbols: 0/30

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| _check_positive_coding | function | False | 3 | ❌ |
| _sparse_encode_precomputed | function | False | 9 | ❌ |
| sparse_encode | function | False | 4 | ❌ |
| _sparse_encode | function | False | 11 | ❌ |
| _update_dict | function | False | 9 | ❌ |
| _dict_learning | function | False | 14 | ❌ |
| dict_learning_online | function | False | 2 | ❌ |
| dict_learning | function | False | 2 | ❌ |
| _BaseSparseCoding | class | False | N/A | ❌ |
| SparseCoder | class | False | N/A | ❌ |
| DictionaryLearning | class | False | N/A | ❌ |
| MiniBatchDictionaryLearning | class | False | N/A | ❌ |
| __init__ | function | False | 1 | ❌ |
| _transform | function | False | 4 | ❌ |
| transform | function | False | 1 | ❌ |
| _inverse_transform | function | False | 4 | ❌ |
| inverse_transform | function | False | 1 | ❌ |
| fit | function | False | 6 | ❌ |
| __sklearn_tags__ | function | False | 1 | ❌ |
| n_components_ | function | False | 1 | ❌ |
| n_features_in_ | function | False | 1 | ❌ |
| _n_features_out | function | False | 1 | ❌ |
| _parameter_constraints | constant | False | N/A | ❌ |
| fit_transform | function | False | 2 | ❌ |
| _check_params | function | False | 2 | ❌ |
| _initialize_dict | function | False | 3 | ❌ |
| _update_inner_stats | function | False | 2 | ❌ |
| _minibatch_step | function | False | 1 | ❌ |
| _check_convergence | function | False | 13 | ❌ |
| partial_fit | function | False | 2 | ❌ |

### lukhas.learning.meta_learning.federated_integration

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 33.33%
- Used/Total Symbols: 0/53

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| Federationstrategy | class | False | N/A | ✅ |
| Privacylevel | class | False | N/A | ✅ |
| Federatednode | dataclass | False | N/A | ✅ |
| Federatedlearningupdate | dataclass | False | N/A | ✅ |
| Federatedlearningintegration | class | False | N/A | ✅ |
| enhance_meta_learning_with_federation | function | False | 4 | ✅ |
| node_id | constant | False | N/A | ❌ |
| node_type | constant | False | N/A | ❌ |
| ethical_compliance_score | constant | False | N/A | ❌ |
| last_sync | constant | False | N/A | ❌ |
| capabilities | constant | False | N/A | ❌ |
| trust_score | constant | False | N/A | ❌ |
| privacy_level | constant | False | N/A | ❌ |
| quantum_signature | constant | False | N/A | ❌ |
| __post_init__ | function | False | 2 | ❌ |
| _generate_quantum_signature | function | False | 1 | ✅ |
| source_node_id | constant | False | N/A | ❌ |
| update_type | constant | False | N/A | ❌ |
| content | constant | False | N/A | ❌ |
| privacy_preserving | constant | False | N/A | ❌ |
| ethical_audit_passed | constant | False | N/A | ❌ |
| timestamp | constant | False | N/A | ❌ |
| __init__ | function | False | 1 | ❌ |
| integrate_with_enhancement_system | function | False | 1 | ✅ |
| register_node | function | False | 2 | ✅ |
| share_learning_insight | function | False | 4 | ✅ |
| receive_federation_updates | function | False | 5 | ✅ |
| coordinate_learning_rates | function | False | 4 | ✅ |
| enhance_symbolic_reasoning_federation | function | False | 5 | ✅ |
| synchronize_federation | function | False | 4 | ✅ |
| get_federation_status | function | False | 1 | ✅ |
| enhance_existing_meta_learning_system | function | False | 7 | ✅ |
| _apply_privacy_filter | function | False | 6 | ✅ |
| _ethical_audit_insight | function | False | 6 | ✅ |
| _generate_update_signature | function | False | 1 | ✅ |
| _process_federation_update | function | False | 3 | ✅ |
| _update_node_trust | function | False | 3 | ✅ |
| _gather_federation_convergence_signals | function | False | 1 | ✅ |
| _calculate_coordinated_rate | function | False | 4 | ✅ |
| _gather_symbolic_patterns | function | False | 1 | ✅ |
| _analyze_cross_node_patterns | function | False | 1 | ✅ |
| _extract_federation_wisdom | function | False | 1 | ✅ |
| _generate_collaborative_reasoning_insights | function | False | 3 | ✅ |
| _should_sync_with_node | function | False | 4 | ✅ |
| _synchronize_with_node | function | False | 3 | ✅ |
| _discover_federation_patterns | function | False | 5 | ✅ |
| _federation_ethical_audit | function | False | 4 | ✅ |
| _generate_coordination_signature | function | False | 1 | ✅ |
| _anonymize_data | function | False | 4 | ✅ |
| _extract_learning_insights | function | False | 4 | ✅ |
| _extract_symbolic_insights | function | False | 3 | ✅ |
| _apply_update_to_meta_learning_system | function | False | 3 | ✅ |
| _calculate_ethical_variance | function | False | 2 | ✅ |

### lukhas.learning.meta_learning.symbolic_feedback

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 40.74%
- Used/Total Symbols: 0/67

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| Feedbacktype | class | False | N/A | ✅ |
| Symboliccontext | class | False | N/A | ✅ |
| Intentnodehistory | dataclass | False | N/A | ✅ |
| Memoriasnapshot | dataclass | False | N/A | ✅ |
| Dreamreplayrecord | dataclass | False | N/A | ✅ |
| Symbolicfeedbackloop | dataclass | False | N/A | ✅ |
| Symbolicfeedbacksystem | class | False | N/A | ✅ |
| create_integrated_symbolic_feedback_system | function | False | 1 | ✅ |
| simulate_intent_node_integration | function | False | 2 | ✅ |
| timestamp | constant | False | N/A | ❌ |
| intent_id | constant | False | N/A | ❌ |
| intent_type | constant | False | N/A | ❌ |
| resolution_success | constant | False | N/A | ❌ |
| confidence_score | constant | False | N/A | ❌ |
| reasoning_steps | constant | False | N/A | ❌ |
| memory_references | constant | False | N/A | ❌ |
| emotional_context | constant | False | N/A | ❌ |
| quantum_signature | constant | False | N/A | ❌ |
| snapshot_id | constant | False | N/A | ❌ |
| coherence_score | constant | False | N/A | ❌ |
| memory_fragments | constant | False | N/A | ❌ |
| retrieval_success_rate | constant | False | N/A | ❌ |
| consolidation_quality | constant | False | N/A | ❌ |
| symbolic_links | constant | False | N/A | ❌ |
| replay_id | constant | False | N/A | ❌ |
| scenario_type | constant | False | N/A | ❌ |
| replay_success | constant | False | N/A | ❌ |
| learning_outcome | type_alias | False | N/A | ❌ |
| performance_delta | constant | False | N/A | ❌ |
| symbolic_insights | constant | False | N/A | ❌ |
| emotional_resonance | constant | False | N/A | ❌ |
| loop_id | constant | False | N/A | ❌ |
| context | constant | False | N/A | ❌ |
| feedback_type | type_alias | False | N/A | ❌ |
| success_metrics | constant | False | N/A | ❌ |
| failure_patterns | constant | False | N/A | ❌ |
| optimization_suggestions | constant | False | N/A | ❌ |
| rehearsal_opportunities | constant | False | N/A | ❌ |
| confidence_adjustment | constant | False | N/A | ❌ |
| __init__ | function | False | 1 | ❌ |
| log_intent_node_interaction | function | False | 4 | ✅ |
| log_memoria_snapshot | function | False | 3 | ✅ |
| log_dream_replay | function | False | 3 | ✅ |
| create_symbolic_feedback_loop | function | False | 4 | ✅ |
| execute_symbolic_rehearsal | function | False | 7 | ✅ |
| get_optimization_insights | function | False | 6 | ✅ |
| _analyze_intent_patterns | function | False | 2 | ✅ |
| _analyze_memoria_patterns | function | False | 2 | ✅ |
| _analyze_dream_patterns | function | False | 2 | ✅ |
| _determine_learning_outcome | function | False | 6 | ✅ |
| _schedule_rehearsal_if_needed | function | False | 3 | ✅ |
| _update_dashboard_symbolic_feedback | function | False | 2 | ✅ |
| _calculate_symbolic_reasoning_confidence | function | False | 2 | ✅ |
| _calculate_emotional_tone_vector | function | False | 3 | ✅ |
| _generate_quantum_signature | function | False | 1 | ✅ |
| _generate_loop_id | function | False | 1 | ✅ |
| _analyze_performance_patterns | function | False | 1 | ✅ |
| _generate_optimization_suggestions | function | False | 4 | ✅ |
| _identify_rehearsal_opportunities | function | False | 3 | ✅ |
| _calculate_confidence_adjustment | function | False | 1 | ✅ |
| _determine_feedback_type | function | False | 4 | ✅ |
| _apply_symbolic_optimizations | function | False | 5 | ✅ |
| _update_pattern_confidence | function | False | 1 | ✅ |
| _find_relevant_success_patterns | function | False | 3 | ✅ |
| _simulate_rehearsal_iteration | function | False | 1 | ✅ |
| _extract_common_suggestions | function | False | 3 | ✅ |
| _generate_system_optimization_recommendations | function | False | 5 | ✅ |

