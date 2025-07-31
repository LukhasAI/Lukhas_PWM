"""
══════════════════════════════════════════════════════════════════════════════════
║ 🧠 LUKHAS AI - USAGE LEARNING
║ A system for learning from user interactions with documentation.
║ Copyright (c) 2025 LUKHAS AI. All rights reserved.
╠══════════════════════════════════════════════════════════════════════════════════
║ Module: usage_learning.py
║ Path: lukhas/learning/usage_learning.py
║ Version: 1.1 | Created: N/A | Modified: 2025-07-25
║ Authors: Jules-04
╠══════════════════════════════════════════════════════════════════════════════════
║ DESCRIPTION
╠══════════════════════════════════════════════════════════════════════════════════
║ Implements a system for learning from user interactions with documentation (DocuTutor).
║ It identifies patterns, updates user preferences, and assesses document
║ effectiveness to provide recommendations.
╚══════════════════════════════════════════════════════════════════════════════════
"""

from typing import Dict, List, Optional, Any # Added Any
from datetime import datetime, timezone # Added timezone
import numpy as np
from collections import defaultdict
import structlog # ΛTRACE: Using structlog for structured logging

# ΛTRACE: Initialize logger for usage learning system
logger = structlog.get_logger().bind(tag="usage_learning")

# # UserInteraction class
# ΛEXPOSE: Data class representing a single user interaction event.
class UserInteraction:
    """Represents a single user interaction with a document."""
    # # Initialization
    def __init__(self, user_id: str, doc_id: str, interaction_type: str, metadata: Dict[str, Any]): # Added type hint for metadata
        # ΛNOTE: Captures key details of a user interaction.
        # ΛSEED: Each interaction instance is a seed for pattern identification.
        self.user_id = user_id
        self.doc_id = doc_id
        self.interaction_type = interaction_type # e.g., "view", "search_click", "feedback_positive"
        self.metadata = metadata # e.g., {"time_spent_seconds": 120, "success": True}
        self.timestamp = datetime.now(timezone.utc) # Use timezone-aware datetime
        # ΛTRACE: UserInteraction created
        logger.debug("user_interaction_created", user_id=user_id, doc_id=doc_id, type=interaction_type)

# # InteractionPattern class
# ΛEXPOSE: Data class for an identified sequence of user interactions.
class InteractionPattern:
    """Represents a common sequence of interactions."""
    # # Initialization
    def __init__(self):
        # ΛNOTE: Tracks frequency and success rate of interaction sequences.
        self.sequence: List[str] = []
        self.frequency: int = 0
        self.last_observed: datetime = datetime.now(timezone.utc)
        self.success_rate: float = 1.0 # Initial assumption, updated with data
        # ΛTRACE: InteractionPattern initialized
        # logger.debug("interaction_pattern_initialized_empty") # Can be too verbose if many are created

    # # Update pattern statistics based on a new observation
    def update(self, success: bool):
        """Update pattern statistics."""
        # ΛDREAM_LOOP: Updating pattern success rates based on outcomes is a learning mechanism.
        # ΛTRACE: Updating interaction pattern
        # logger.debug("interaction_pattern_update", sequence_preview=self.sequence[:3] if self.sequence else "N/A", success=success) # Can be verbose
        self.frequency += 1
        self.last_observed = datetime.now(timezone.utc)
        # Rolling average for success rate
        self.success_rate = (self.success_rate * (self.frequency - 1) + (1.0 if success else 0.0)) / self.frequency


# # UsageBasedLearning class
# ΛEXPOSE: Main class for the usage-based learning system.
class UsageBasedLearning:
    """
    System for learning from user interactions with documentation (e.g., DocuTutor).
    Identifies common patterns, updates user preferences, and assesses document effectiveness.
    """
    # # Initialization
    def __init__(self):
        # ΛNOTE: Initializes storage for interactions, patterns, preferences, and statistics.
        self.interactions: List[UserInteraction] = []
        self.patterns: Dict[str, InteractionPattern] = {} # Key: pattern_key string, Value: InteractionPattern
        self.user_preferences: Dict[str, Dict[str, Any]] = defaultdict(dict) # user_id -> preference dict
        self.doc_statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: { # Using lambda for complex default
            'views': 0, 'total_time_spent_seconds': 0.0, 'successful_uses_count': 0, 'failed_uses_count': 0 # Renamed keys
        })
        # ΛTRACE: UsageBasedLearning system initialized
        logger.info("usage_based_learning_system_initialized")

    # # Record a user interaction with documentation
    # ΛEXPOSE: Primary method for logging user interactions.
    def record_interaction(self, user_id: str, doc_id: str, interaction_type: str, metadata: Dict[str, Any]):
        """Record a user interaction with documentation."""
        # ΛTRACE: Recording user interaction
        logger.info("record_interaction_start", user_id=user_id, doc_id=doc_id, type=interaction_type)
        interaction = UserInteraction(user_id, doc_id, interaction_type, metadata)
        self.interactions.append(interaction)

        stats = self.doc_statistics[doc_id]
        stats['views'] += 1
        if 'time_spent_seconds' in metadata and isinstance(metadata['time_spent_seconds'], (int, float)): # Added type check
            stats['total_time_spent_seconds'] += metadata['time_spent_seconds']

        if 'success' in metadata:
            if metadata['success']: stats['successful_uses_count'] += 1
            else: stats['failed_uses_count'] += 1
        logger.debug("interaction_recorded_and_stats_updated", doc_id=doc_id, views=stats['views'])

    # # Identify common interaction patterns from the recorded interactions
    # ΛEXPOSE: Analyzes interaction history to find recurring sequences.
    def identify_patterns(self, window_size: int = 3):
        """Identify common interaction patterns."""
        # ΛDREAM_LOOP: Identifying and reinforcing common successful patterns is a form of learning.
        # ΛTRACE: Identifying interaction patterns
        logger.info("identify_interaction_patterns_start", window_size=window_size, total_interactions=len(self.interactions))
        if len(self.interactions) < window_size:
            logger.debug("not_enough_interactions_to_identify_patterns", count=len(self.interactions), window=window_size)
            return

        for i in range(len(self.interactions) - window_size + 1):
            sequence_interactions = self.interactions[i : i + window_size] # Renamed
            # Create a more robust pattern key, e.g., based on interaction types or key metadata
            # For simplicity, using doc_id:interaction_type
            current_sequence = [f"{interaction.doc_id}:{interaction.interaction_type}" for interaction in sequence_interactions] # Renamed
            pattern_key_str = "->".join(current_sequence) # Renamed

            # ΛNOTE: Assumes success for now. Real implementation would need to evaluate sequence outcome.
            # ΛCAUTION: Defaulting pattern success to True is a simplification.
            interaction_success = sequence_interactions[-1].metadata.get("success", True) # Example: use success of last interaction in sequence

            if pattern_key_str not in self.patterns: self.patterns[pattern_key_str] = InteractionPattern()
            self.patterns[pattern_key_str].sequence = current_sequence
            self.patterns[pattern_key_str].update(interaction_success)
        logger.info("identify_interaction_patterns_end", num_distinct_patterns=len(self.patterns))

    # # Update stored preferences for a user
    # ΛEXPOSE: Allows updating user-specific preferences.
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]): # Type hint
        """Update stored preferences for a user."""
        # ΛNOTE: Stores user preferences, which can be used for personalization.
        # ΛSEED: User-provided preferences are explicit seeds for personalization.
        # ΛTRACE: Updating user preferences
        logger.info("update_user_preferences_start", user_id=user_id, num_preferences=len(preferences))
        self.user_preferences[user_id].update(preferences)
        logger.debug("user_preferences_updated", user_id=user_id, updated_preferences=preferences)

    # # Calculate the effectiveness score for a document
    # ΛEXPOSE: Assesses document effectiveness based on usage statistics.
    def get_document_effectiveness(self, doc_id: str) -> float:
        """Calculate the effectiveness score for a document."""
        # ΛTRACE: Calculating document effectiveness
        logger.debug("get_document_effectiveness_start", doc_id=doc_id)
        stats = self.doc_statistics[doc_id]
        total_uses = stats['successful_uses_count'] + stats['failed_uses_count']
        if total_uses == 0: return 0.0 # Avoid division by zero

        effectiveness = stats['successful_uses_count'] / total_uses
        logger.debug("document_effectiveness_calculated", doc_id=doc_id, score=effectiveness)
        return effectiveness

    # # Get commonly observed interaction sequences
    # ΛEXPOSE: Retrieves popular interaction patterns.
    def get_popular_sequences(self, min_frequency: int = 2) -> List[Dict[str, Any]]: # Type hint
        """Get commonly observed interaction sequences."""
        # ΛTRACE: Getting popular sequences
        logger.info("get_popular_sequences_start", min_frequency=min_frequency)
        popular_seqs = [{'sequence': p.sequence, 'frequency': p.frequency, 'success_rate': p.success_rate} for p in self.patterns.values() if p.frequency >= min_frequency] # Renamed
        logger.info("get_popular_sequences_end", count=len(popular_seqs))
        return popular_seqs

    # # Recommend next documents based on patterns and user preferences
    # ΛEXPOSE: Provides document recommendations.
    def recommend_next_docs(self, current_doc_id: str, user_id: str) -> List[str]: # Renamed current_doc
        """Recommend next documents based on patterns and user preferences."""
        # ΛDREAM_LOOP: Recommendations adapting to user behavior and pattern success is a learning loop.
        # ΛTRACE: Recommending next documents
        logger.info("recommend_next_docs_start", current_doc_id=current_doc_id, user_id=user_id)
        recommendations_list: List[str] = [] # Renamed
        user_prefs_data = self.user_preferences.get(user_id, {}) # Renamed

        # ΛNOTE: Recommendation logic is currently pattern-based. Could be enhanced with collaborative filtering, etc.
        for pattern_obj in self.patterns.values(): # Renamed pattern
            if pattern_obj.sequence and pattern_obj.sequence[0].startswith(current_doc_id) and len(pattern_obj.sequence) > 1: # Check sequence not empty
                # Consider success rate of the pattern for recommendation strength
                if pattern_obj.success_rate > 0.6: # Example threshold
                    next_doc_candidate = pattern_obj.sequence[1].split(':')[0] # Renamed
                    if next_doc_candidate not in recommendations_list:
                        recommendations_list.append(next_doc_candidate)

        # ΛNOTE: Sort recommendations by document effectiveness.
        recommendations_list.sort(key=lambda doc_id_val: self.get_document_effectiveness(doc_id_val), reverse=True) # Renamed doc_id
        logger.info("recommend_next_docs_end", num_recommendations=len(recommendations_list), for_doc=current_doc_id)
        return recommendations_list[:5] # Return top 5

"""
═══════════════════════════════════════════════════════════════════════════════
║ 📋 FOOTER - LUKHAS AI
╠══════════════════════════════════════════════════════════════════════════════
║ VALIDATION:
║   - Tests: lukhas/tests/learning/test_usage_learning.py
║   - Coverage: N/A
║   - Linting: N/A
║
║ MONITORING:
║   - Metrics: N/A
║   - Logs: usage_learning
║   - Alerts: N/A
║
║ COMPLIANCE:
║   - Standards: N/A
║   - Ethics: N/A
║   - Safety: N/A
║
║ REFERENCES:
║   - Docs: N/A
║   - Issues: N/A
║   - Wiki: N/A
║
║ COPYRIGHT & LICENSE:
║   Copyright (c) 2025 LUKHAS AI. All rights reserved.
║   Licensed under the LUKHAS AI Proprietary License.
║   Unauthorized use, reproduction, or distribution is prohibited.
║
║ DISCLAIMER:
║   This module is part of the LUKHAS AGI system. Use only as intended
║   within the system architecture. Modifications may affect system
║   stability and require approval from the LUKHAS Architecture Board.
╚═══════════════════════════════════════════════════════════════════════════
"""
