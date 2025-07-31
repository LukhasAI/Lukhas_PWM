"""
Voice Bio Adapter

This adapter provides voice-specific functionality for the bio orchestrator,
bridging between voice processing modules and the bio-symbolic orchestration layer.

Created: 2025-07-26
"""

from typing import Dict, Any, Optional
import logging


logger = logging.getLogger("LUKHAS.VoiceBioAdapter")


class VoiceBioAdapter:
    """
    Adapter for voice-specific bio orchestration needs.

    This class wraps the main BioOrchestrator to provide voice-specific
    functionality and optimizations.
    """

    def __init__(self, bio_orchestrator: Optional[BioOrchestrator] = None):
        """
        Initialize the voice adapter.

        Args:
            bio_orchestrator: Existing orchestrator instance or None to create new
        """
        self.orchestrator = bio_orchestrator or BioOrchestrator(
            total_energy_capacity=1.5,  # Voice processing needs more resources
            monitoring_interval=2.0,    # Faster monitoring for real-time audio
            auto_repair=True
        )

        # Voice-specific configuration
        self.voice_config = {
            'sample_rate': 16000,
            'chunk_size': 1024,
            'latency_threshold': 0.05,  # 50ms max latency
            'priority': ResourcePriority.HIGH
        }

        # Register voice-specific modules with high priority
        self._register_voice_modules()

    def _register_voice_modules(self):
        """Register standard voice processing modules"""
        # This would be extended with actual voice modules
        logger.info("Voice adapter initialized with high-priority configuration")

    def process_audio_chunk(self, audio_data: bytes,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an audio chunk through the bio orchestrator.

        Args:
            audio_data: Raw audio bytes
            metadata: Optional metadata about the audio

        Returns:
            Dict with processing results
        """
        # Prioritize voice processing
        task_data = {
            'type': 'voice_processing',
            'priority': self.voice_config['priority'].value,
            'data': audio_data,
            'metadata': metadata or {}
        }

        # Use orchestrator's resource allocation
        allocated_tasks = self.orchestrator.allocate_resources([{
            'module_id': 'voice_processor',
            'complexity': 0.3,  # Voice is moderately complex
            'priority': self.voice_config['priority'].value
        }])

        if allocated_tasks:
            # Process through orchestrator
            success, result = self.orchestrator.invoke_module(
                'voice_processor',
                'process',
                audio_data,
                metadata
            )

            return {
                'success': success,
                'result': result,
                'latency': 0.0  # Would be calculated in real implementation
            }
        else:
            logger.warning("Insufficient resources for voice processing")
            return {
                'success': False,
                'error': 'Insufficient resources',
                'result': None
            }

    def optimize_for_realtime(self):
        """Optimize orchestrator settings for real-time voice processing"""
        # Adjust energy allocation for voice priority
        if 'voice_processor' in self.orchestrator.registered_modules:
            self.orchestrator.update_module(
                'voice_processor',
                priority=ResourcePriority.CRITICAL,
                energy_cost=0.05  # Lower cost for faster allocation
            )

        logger.info("Optimized for real-time voice processing")

    def get_voice_metrics(self) -> Dict[str, Any]:
        """Get voice-specific performance metrics"""
        base_metrics = self.orchestrator.get_system_status()

        # Add voice-specific metrics
        voice_metrics = {
            'base_metrics': base_metrics,
            'voice_config': self.voice_config,
            'optimized_for_realtime': True
        }

        return voice_metrics