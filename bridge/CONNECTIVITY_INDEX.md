# Connectivity Index for lukhas/bridge

Generated: 2025-07-28T17:40:41.744155

## Summary

- **Total Modules:** 16
- **Total Symbols:** 270
- **Total Dependencies:** 119
- **Missed Opportunities:** 10

## 🔍 Missed Opportunities

### 🔴 Unused Exports
**Description:** Module lukhas.bridge.symbolic_memory_mapper has 13 unused public symbols
**Affected Files:** lukhas.bridge.symbolic_memory_mapper
**Suggestion:** Consider making these symbols private or removing them: MemoryMapType, SymbolicMemoryNode, SymbolicMemoryMapper, map_symbolic_payload_to_memory, node_id...

### 🔴 Unused Exports
**Description:** Module lukhas.bridge.message_bus has 23 unused public symbols
**Affected Files:** lukhas.bridge.message_bus
**Suggestion:** Consider making these symbols private or removing them: MessagePriority, MessageType, Message, MessageBus, type...

### 🔴 Unused Exports
**Description:** Module lukhas.bridge.trace_logger has 17 unused public symbols
**Affected Files:** lukhas.bridge.trace_logger
**Suggestion:** Consider making these symbols private or removing them: TraceLevel, TraceCategory, BridgeTraceEvent, BridgeTraceLogger, log_symbolic_event...

### 🔴 Unused Exports
**Description:** Module lukhas.bridge.symbolic_dream_bridge has 12 unused public symbols
**Affected Files:** lukhas.bridge.symbolic_dream_bridge
**Suggestion:** Consider making these symbols private or removing them: SymbolicDreamContext, SymbolicDreamBridge, bridge_dream_to_memory, dream_id, phase_state...

### 🔴 Unused Exports
**Description:** Module lukhas.bridge.integration_bridge has 15 unused public symbols
**Affected Files:** lukhas.bridge.integration_bridge
**Suggestion:** Consider making these symbols private or removing them: lukhas_tier_required, PluginModuleAdapter, IntegrationBridge, decorator, BaseLucasModule...

### 🔴 Unused Exports
**Description:** Module lukhas.bridge.symbolic_reasoning_adapter has 12 unused public symbols
**Affected Files:** lukhas.bridge.symbolic_reasoning_adapter
**Suggestion:** Consider making these symbols private or removing them: ReasoningMode, ReasoningContext, SymbolicReasoningAdapter, context_id, mode...

### 🔴 Unused Exports
**Description:** Module lukhas.bridge.explainability_interface_layer has 38 unused public symbols
**Affected Files:** lukhas.bridge.explainability_interface_layer
**Suggestion:** Consider making these symbols private or removing them: ExplanationType, ExplanationAudience, ExplanationDepth, ExplanationRequest, ExplanationProof...

### 🔴 Unused Exports
**Description:** Module lukhas.bridge.llm_wrappers.unified_openai_client has 22 unused public symbols
**Affected Files:** lukhas.bridge.llm_wrappers.unified_openai_client
**Suggestion:** Consider making these symbols private or removing them: ConversationMessage, ConversationState, role, content, timestamp...

### 🔴 Unused Exports
**Description:** Module lukhas.bridge.shared_state has 37 unused public symbols
**Affected Files:** lukhas.bridge.shared_state
**Suggestion:** Consider making these symbols private or removing them: StateAccessLevel, StateOperation, StateValue, StateChange, ConflictResolutionStrategy...

### 🔴 Unused Exports
**Description:** Module lukhas.bridge.model_communication_engine has 24 unused public symbols
**Affected Files:** lukhas.bridge.model_communication_engine
**Suggestion:** Consider making these symbols private or removing them: ModelCommunicationEngine, sinusoids, disable_sdpa, n_mels, n_audio_ctx...

## Module Details

### lukhas.bridge.symbolic_memory_mapper

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 26.67%
- Used/Total Symbols: 0/14

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| MemoryMapType | class | False | N/A | ✅ |
| SymbolicMemoryNode | dataclass | False | N/A | ✅ |
| SymbolicMemoryMapper | class | False | N/A | ✅ |
| map_symbolic_payload_to_memory | function | False | 1 | ✅ |
| node_id | constant | False | N/A | ❌ |
| map_type | type_alias | False | N/A | ❌ |
| symbolic_data | constant | False | N/A | ❌ |
| bridge_metadata | constant | False | N/A | ❌ |
| access_timestamp | constant | False | N/A | ❌ |
| __init__ | function | False | 1 | ❌ |
| create_memory_map | function | False | 1 | ✅ |
| map_to_core_structures | function | False | 1 | ✅ |
| maintain_memory_coherence | function | False | 1 | ✅ |
| archive_memory_map | function | False | 2 | ✅ |

### lukhas.bridge.message_bus

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 73.33%
- Used/Total Symbols: 0/27

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| MessagePriority | class | False | N/A | ✅ |
| MessageType | class | False | N/A | ✅ |
| Message | dataclass | False | N/A | ✅ |
| MessageBus | class | False | N/A | ✅ |
| type | type_alias | False | N/A | ❌ |
| source_module | constant | False | N/A | ❌ |
| target_module | constant | False | N/A | ❌ |
| payload | constant | False | N/A | ❌ |
| id | constant | False | N/A | ❌ |
| priority | constant | False | N/A | ❌ |
| user_id | constant | False | N/A | ❌ |
| tier | constant | False | N/A | ❌ |
| timestamp | constant | False | N/A | ❌ |
| correlation_id | constant | False | N/A | ❌ |
| response_required | constant | False | N/A | ❌ |
| ttl | constant | False | N/A | ❌ |
| __init__ | function | False | 4 | ❌ |
| register_module | function | False | 6 | ✅ |
| unregister_module | function | False | 3 | ✅ |
| subscribe | function | False | 2 | ✅ |
| unsubscribe | function | False | 3 | ✅ |
| _is_circuit_closed | function | False | 6 | ✅ |
| _record_circuit_failure | function | False | 4 | ✅ |
| get_stats | function | False | 1 | ✅ |
| get_message_history | function | False | 1 | ✅ |
| _DummyIdentityClient | class | False | N/A | ❌ |
| get_user_info | function | False | 1 | ❌ |

### lukhas.bridge.trace_logger

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 40.00%
- Used/Total Symbols: 0/19

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| TraceLevel | class | False | N/A | ✅ |
| TraceCategory | class | False | N/A | ✅ |
| BridgeTraceEvent | dataclass | False | N/A | ✅ |
| BridgeTraceLogger | class | False | N/A | ✅ |
| log_symbolic_event | function | False | 1 | ✅ |
| event_id | constant | False | N/A | ❌ |
| timestamp | constant | False | N/A | ❌ |
| category | constant | False | N/A | ❌ |
| level | constant | False | N/A | ❌ |
| component | constant | False | N/A | ❌ |
| message | constant | False | N/A | ❌ |
| metadata | constant | False | N/A | ❌ |
| __init__ | function | False | 1 | ❌ |
| _setup_file_logging | function | False | 1 | ✅ |
| log_bridge_event | function | False | 2 | ✅ |
| trace_symbolic_handshake | function | False | 2 | ✅ |
| trace_memory_mapping | function | False | 2 | ✅ |
| get_trace_summary | function | False | 1 | ✅ |
| export_trace_data | function | False | 2 | ✅ |

### lukhas.bridge.symbolic_dream_bridge

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 33.33%
- Used/Total Symbols: 0/13

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| SymbolicDreamContext | dataclass | False | N/A | ✅ |
| SymbolicDreamBridge | class | False | N/A | ✅ |
| bridge_dream_to_memory | function | False | 1 | ✅ |
| dream_id | constant | False | N/A | ❌ |
| phase_state | constant | False | N/A | ❌ |
| symbolic_map | constant | False | N/A | ❌ |
| resonance_level | constant | False | N/A | ❌ |
| bridge_timestamp | constant | False | N/A | ❌ |
| __init__ | function | False | 1 | ❌ |
| establish_symbolic_handshake | function | False | 1 | ✅ |
| translate_dream_symbols | function | False | 1 | ✅ |
| maintain_phase_resonance | function | False | 1 | ✅ |
| close_bridge | function | False | 2 | ✅ |

### lukhas.bridge.integration_bridge

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 80.00%
- Used/Total Symbols: 0/16

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| lukhas_tier_required | function | False | 1 | ❌ |
| PluginModuleAdapter | class | False | N/A | ✅ |
| IntegrationBridge | class | False | N/A | ✅ |
| decorator | function | False | 1 | ❌ |
| __init__ | function | False | 3 | ❌ |
| BaseLucasModule | class | False | N/A | ❌ |
| LucasPlugin | class | False | N/A | ❌ |
| LucasPluginManifest | class | False | N/A | ❌ |
| PluginLoader | class | False | N/A | ❌ |
| CoreRegistryMock | class | False | N/A | ❌ |
| registration_results | constant | False | N/A | ❌ |
| statuses | constant | False | N/A | ❌ |
| broadcast_results | constant | False | N/A | ❌ |
| plugins_with_capability | constant | False | N/A | ❌ |
| plugin_health | constant | False | N/A | ❌ |
| processing_method_name | constant | False | N/A | ❌ |

### lukhas.bridge.symbolic_reasoning_adapter

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 26.67%
- Used/Total Symbols: 0/13

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| ReasoningMode | class | False | N/A | ✅ |
| ReasoningContext | dataclass | False | N/A | ✅ |
| SymbolicReasoningAdapter | class | False | N/A | ✅ |
| context_id | constant | False | N/A | ❌ |
| mode | constant | False | N/A | ❌ |
| symbolic_input | constant | False | N/A | ❌ |
| logical_output | constant | False | N/A | ❌ |
| adaptation_metadata | constant | False | N/A | ❌ |
| __init__ | function | False | 1 | ❌ |
| adapt_symbolic_reasoning | function | False | 1 | ✅ |
| bridge_reasoning_flow | function | False | 1 | ✅ |
| validate_reasoning_coherence | function | False | 1 | ✅ |
| close_reasoning_context | function | False | 2 | ✅ |

### lukhas.bridge.connectors.blockchain_bridge

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 33.33%
- Used/Total Symbols: 0/1

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| anchor_hash | function | False | 2 | ✅ |

### lukhas.bridge.llm_wrappers.perplexity_wrapper

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 33.33%
- Used/Total Symbols: 0/4

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| PerplexityWrapper | class | False | N/A | ❌ |
| __init__ | function | False | 2 | ✅ |
| generate_response | function | False | 5 | ✅ |
| is_available | function | False | 1 | ✅ |

### lukhas.bridge.explainability_interface_layer

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 93.33%
- Used/Total Symbols: 0/50

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| ExplanationType | class | False | N/A | ✅ |
| ExplanationAudience | class | False | N/A | ✅ |
| ExplanationDepth | class | False | N/A | ✅ |
| ExplanationRequest | dataclass | False | N/A | ✅ |
| ExplanationProof | dataclass | False | N/A | ✅ |
| ExplanationOutput | dataclass | False | N/A | ✅ |
| ExplanationGenerator | class | False | N/A | ✅ |
| NaturalLanguageGenerator | class | False | N/A | ✅ |
| FormalProofGenerator | class | False | N/A | ✅ |
| ExplainabilityInterfaceLayer | class | False | N/A | ✅ |
| request_id | constant | False | N/A | ❌ |
| decision_id | constant | False | N/A | ❌ |
| explanation_type | type_alias | False | N/A | ❌ |
| audience | constant | False | N/A | ❌ |
| depth | constant | False | N/A | ❌ |
| context | constant | False | N/A | ❌ |
| custom_template | constant | False | N/A | ❌ |
| requires_proof | constant | False | N/A | ❌ |
| requires_signing | constant | False | N/A | ❌ |
| timestamp | constant | False | N/A | ❌ |
| proof_id | constant | False | N/A | ❌ |
| premises | constant | False | N/A | ❌ |
| inference_rules | constant | False | N/A | ❌ |
| logical_steps | constant | False | N/A | ❌ |
| conclusion | constant | False | N/A | ❌ |
| proof_system | constant | False | N/A | ❌ |
| validity_score | constant | False | N/A | ❌ |
| explanation_id | constant | False | N/A | ❌ |
| natural_language | constant | False | N/A | ❌ |
| formal_proof | constant | False | N/A | ❌ |
| causal_chain | constant | False | N/A | ❌ |
| confidence_score | constant | False | N/A | ❌ |
| uncertainty_bounds | constant | False | N/A | ❌ |
| evidence_sources | constant | False | N/A | ❌ |
| srd_signature | constant | False | N/A | ❌ |
| quality_metrics | constant | False | N/A | ❌ |
| metadata | constant | False | N/A | ❌ |
| __init__ | function | False | 3 | ✅ |
| _load_templates | function | False | 1 | ✅ |
| _get_audience_style | function | False | 1 | ✅ |
| _get_depth_content | function | False | 4 | ✅ |
| _format_proof | function | False | 2 | ✅ |
| _format_simple_proof | function | False | 2 | ✅ |
| _format_technical_proof | function | False | 4 | ✅ |
| _initialize_lukhas_integration | function | False | 6 | ✅ |
| _calculate_completeness | function | False | 1 | ✅ |
| _calculate_clarity | function | False | 4 | ✅ |
| _update_metrics | function | False | 4 | ✅ |
| get_metrics | function | False | 3 | ✅ |
| _calculate_std | function | False | 2 | ✅ |

### lukhas.bridge.llm_wrappers.gemini_wrapper

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 33.33%
- Used/Total Symbols: 0/4

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| GeminiWrapper | class | False | N/A | ❌ |
| __init__ | function | False | 3 | ✅ |
| generate_response | function | False | 3 | ✅ |
| is_available | function | False | 1 | ✅ |

### lukhas.bridge.llm_wrappers.anthropic_wrapper

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 33.33%
- Used/Total Symbols: 0/4

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| AnthropicWrapper | class | False | N/A | ❌ |
| __init__ | function | False | 3 | ✅ |
| generate_response | function | False | 3 | ✅ |
| is_available | function | False | 1 | ✅ |

### lukhas.bridge.llm_wrappers.azure_openai_wrapper

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 33.33%
- Used/Total Symbols: 0/4

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| AzureOpenaiWrapper | class | False | N/A | ❌ |
| __init__ | function | False | 8 | ✅ |
| generate_response | function | False | 3 | ✅ |
| is_available | function | False | 1 | ✅ |

### lukhas.bridge.llm_wrappers.env_loader

**Metrics:**
- Connectivity Score: 50.00%
- Cohesion Score: 0.00%
- Coupling Score: 20.00%
- Used/Total Symbols: 2/4

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| load_lukhas_env | function | False | 8 | ✅ |
| get_api_key | function | True | 2 | ✅ |
| get_openai_config | function | False | 1 | ✅ |
| get_azure_openai_config | function | True | 1 | ✅ |

### lukhas.bridge.llm_wrappers.unified_openai_client

**Metrics:**
- Connectivity Score: 4.17%
- Cohesion Score: 0.00%
- Coupling Score: 66.67%
- Used/Total Symbols: 1/24

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| ConversationMessage | dataclass | False | N/A | ✅ |
| ConversationState | dataclass | False | N/A | ✅ |
| UnifiedOpenAIClient | class | True | N/A | ✅ |
| role | constant | False | N/A | ❌ |
| content | constant | False | N/A | ❌ |
| timestamp | constant | False | N/A | ❌ |
| message_id | constant | False | N/A | ❌ |
| metadata | constant | False | N/A | ❌ |
| function_call | constant | False | N/A | ❌ |
| conversation_id | constant | False | N/A | ❌ |
| session_id | constant | False | N/A | ❌ |
| user_id | constant | False | N/A | ❌ |
| messages | constant | False | N/A | ❌ |
| context | constant | False | N/A | ❌ |
| created_at | constant | False | N/A | ❌ |
| updated_at | constant | False | N/A | ❌ |
| total_tokens | constant | False | N/A | ❌ |
| max_context_length | constant | False | N/A | ❌ |
| __init__ | function | False | 4 | ✅ |
| create_conversation | function | False | 1 | ✅ |
| add_message | function | False | 2 | ✅ |
| get_conversation_messages | function | False | 5 | ✅ |
| chat_completion_sync | function | False | 3 | ✅ |
| get_model_info | function | False | 1 | ✅ |

### lukhas.bridge.shared_state

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 80.00%
- Used/Total Symbols: 0/47

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| StateAccessLevel | class | False | N/A | ✅ |
| StateOperation | class | False | N/A | ✅ |
| StateValue | dataclass | False | N/A | ✅ |
| StateChange | dataclass | False | N/A | ✅ |
| ConflictResolutionStrategy | class | False | N/A | ✅ |
| SharedStateManager | class | False | N/A | ✅ |
| set_shared_state | function | False | 1 | ✅ |
| get_shared_state | function | False | 1 | ✅ |
| delete_shared_state | function | False | 1 | ✅ |
| subscribe_to_state | function | False | 1 | ✅ |
| value | constant | False | N/A | ❌ |
| owner_module | constant | False | N/A | ❌ |
| access_level | constant | False | N/A | ❌ |
| version | constant | False | N/A | ❌ |
| timestamp | constant | False | N/A | ❌ |
| user_id | constant | False | N/A | ❌ |
| tier | constant | False | N/A | ❌ |
| ttl | constant | False | N/A | ❌ |
| metadata | constant | False | N/A | ❌ |
| __post_init__ | function | False | 2 | ❌ |
| key | constant | False | N/A | ❌ |
| old_value | constant | False | N/A | ❌ |
| new_value | constant | False | N/A | ❌ |
| operation | constant | False | N/A | ❌ |
| module | constant | False | N/A | ❌ |
| __init__ | function | False | 4 | ❌ |
| _get_lock | function | False | 2 | ✅ |
| _check_access | function | False | 13 | ✅ |
| _is_expired | function | False | 3 | ✅ |
| _cleanup_expired | function | False | 5 | ✅ |
| _resolve_conflict | function | False | 7 | ✅ |
| _notify_subscribers | function | False | 5 | ✅ |
| _add_change_to_history | function | False | 2 | ✅ |
| set_state | function | False | 12 | ✅ |
| get_state | function | False | 5 | ✅ |
| delete_state | function | False | 4 | ✅ |
| subscribe | function | False | 4 | ✅ |
| unsubscribe | function | False | 4 | ✅ |
| get_keys_by_prefix | function | False | 2 | ✅ |
| get_state_info | function | False | 3 | ✅ |
| get_change_history | function | False | 2 | ✅ |
| get_stats | function | False | 5 | ✅ |
| rollback_to_version | function | False | 15 | ✅ |
| _DummyIdentityClient | class | False | N/A | ❌ |
| get_user_info | function | False | 1 | ❌ |
| target_change_obj | constant | False | N/A | ❌ |
| old_value_for_history | constant | False | N/A | ❌ |

### lukhas.bridge.model_communication_engine

**Metrics:**
- Connectivity Score: 0.00%
- Cohesion Score: 0.00%
- Coupling Score: 66.67%
- Used/Total Symbols: 0/26

**Symbols:**

| Name | Kind | Used | Complexity | Documented |
| --- | --- | --- | --- | --- |
| ModelCommunicationEngine | class | False | N/A | ❌ |
| sinusoids | function | False | 1 | ✅ |
| disable_sdpa | function | False | 1 | ❌ |
| n_mels | constant | False | N/A | ❌ |
| n_audio_ctx | constant | False | N/A | ❌ |
| n_audio_state | constant | False | N/A | ❌ |
| n_audio_head | constant | False | N/A | ❌ |
| n_audio_layer | constant | False | N/A | ❌ |
| n_vocab | constant | False | N/A | ❌ |
| n_text_ctx | constant | False | N/A | ❌ |
| n_text_state | constant | False | N/A | ❌ |
| n_text_head | constant | False | N/A | ❌ |
| n_text_layer | constant | False | N/A | ❌ |
| forward | function | False | 1 | ❌ |
| _conv_forward | function | False | 1 | ❌ |
| __init__ | function | False | 1 | ❌ |
| qkv_attention | function | False | 5 | ❌ |
| set_alignment_heads | function | False | 1 | ❌ |
| embed_audio | function | False | 1 | ❌ |
| logits | function | False | 1 | ❌ |
| device | function | False | 1 | ❌ |
| is_multilingual | function | False | 1 | ❌ |
| num_languages | function | False | 1 | ❌ |
| install_kv_cache_hooks | function | False | 4 | ✅ |
| save_to_cache | function | False | 3 | ❌ |
| install_hooks | function | False | 2 | ❌ |

