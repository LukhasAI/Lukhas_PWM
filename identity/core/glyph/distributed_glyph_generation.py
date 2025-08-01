"""
Distributed GLYPH Generation with Colony Coordination

Manages distributed generation of identity GLYPHs using colony-based
parallel processing and tier-aware steganographic embedding.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from PIL import Image
import hashlib
import base64
import io

# Import colony infrastructure
from core.colonies.base_colony import BaseColony, ConsensusResult
from core.swarm import SwarmAgent, AgentState, SwarmTask

# Import identity components
from identity.core.events import (
    IdentityEventPublisher, IdentityEventType,
    get_identity_event_publisher
)
from identity.core.tier import TierLevel
from identity.core.visualization.lukhas_orb import OrbVisualization

logger = logging.getLogger('LUKHAS_DISTRIBUTED_GLYPH')


class GLYPHType(Enum):
    """Types of identity GLYPHs."""
    AUTHENTICATION = "auth"           # Authentication GLYPH
    VERIFICATION = "verify"          # Verification GLYPH
    SIGNATURE = "signature"          # Digital signature GLYPH
    RECOVERY = "recovery"            # Account recovery GLYPH
    QUANTUM = "quantum"              # Quantum-resistant GLYPH
    CONSCIOUSNESS = "consciousness"   # Consciousness-encoded GLYPH
    DREAM = "dream"                  # Dream-state GLYPH


class GLYPHComplexity(Enum):
    """GLYPH generation complexity levels."""
    BASIC = 1      # Simple pattern generation
    STANDARD = 2   # Standard complexity with basic embedding
    ENHANCED = 3   # Enhanced with multi-layer embedding
    ADVANCED = 4   # Advanced with quantum patterns
    TRANSCENDENT = 5  # Full consciousness integration


@dataclass
class GLYPHGenerationTask:
    """Task for distributed GLYPH generation."""
    task_id: str
    lambda_id: str
    glyph_type: GLYPHType
    tier_level: int
    complexity: GLYPHComplexity
    identity_data: Dict[str, Any]
    orb_state: Optional[OrbVisualization] = None
    steganographic_data: Optional[Dict[str, Any]] = None
    quantum_seed: Optional[bytes] = None
    consciousness_pattern: Optional[np.ndarray] = None
    dream_sequence: Optional[List[Dict[str, Any]]] = None
    deadline: Optional[datetime] = None


@dataclass
class GLYPHFragment:
    """Fragment of a GLYPH generated by an agent."""
    fragment_id: str
    agent_id: str
    fragment_type: str  # 'pattern', 'color', 'quantum', 'consciousness'
    data: np.ndarray
    metadata: Dict[str, Any]
    generation_time: float
    quality_score: float


@dataclass
class GeneratedGLYPH:
    """Complete generated GLYPH with metadata."""
    glyph_id: str
    lambda_id: str
    glyph_type: GLYPHType
    tier_level: int
    image_data: np.ndarray
    embedded_data: Optional[Dict[str, Any]]
    generation_metadata: Dict[str, Any]
    fragments_used: List[str]
    consensus_achieved: bool
    quality_metrics: Dict[str, float]
    timestamp: datetime

    def to_pil_image(self) -> Image.Image:
        """Convert to PIL Image."""
        return Image.fromarray(self.image_data.astype(np.uint8))

    def to_base64(self) -> str:
        """Convert to base64 string."""
        img = self.to_pil_image()
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()


class GLYPHGenerationAgent(SwarmAgent):
    """
    Specialized agent for GLYPH fragment generation.
    """

    def __init__(self, agent_id: str, colony: 'DistributedGLYPHColony',
                 specialization: str):
        super().__init__(agent_id, colony, capabilities=[specialization])
        self.specialization = specialization
        self.fragments_generated = 0
        self.quality_scores: List[float] = []

        logger.info(f"GLYPH agent {agent_id} initialized for {specialization}")

    async def generate_fragment(
        self,
        task: GLYPHGenerationTask,
        fragment_params: Dict[str, Any]
    ) -> GLYPHFragment:
        """Generate a GLYPH fragment based on specialization."""

        start_time = datetime.utcnow()

        try:
            if self.specialization == "pattern":
                fragment_data = await self._generate_pattern_fragment(task, fragment_params)
            elif self.specialization == "color":
                fragment_data = await self._generate_color_fragment(task, fragment_params)
            elif self.specialization == "quantum":
                fragment_data = await self._generate_quantum_fragment(task, fragment_params)
            elif self.specialization == "consciousness":
                fragment_data = await self._generate_consciousness_fragment(task, fragment_params)
            elif self.specialization == "embedding":
                fragment_data = await self._generate_embedding_fragment(task, fragment_params)
            else:
                raise ValueError(f"Unknown specialization: {self.specialization}")

            # Calculate quality score
            quality_score = self._evaluate_fragment_quality(fragment_data, task.tier_level)

            # Create fragment
            fragment = GLYPHFragment(
                fragment_id=f"{task.task_id}_{self.agent_id}_{self.fragments_generated}",
                agent_id=self.agent_id,
                fragment_type=self.specialization,
                data=fragment_data,
                metadata={
                    "tier_level": task.tier_level,
                    "complexity": task.complexity.name,
                    "generation_params": fragment_params
                },
                generation_time=(datetime.utcnow() - start_time).total_seconds(),
                quality_score=quality_score
            )

            self.fragments_generated += 1
            self.quality_scores.append(quality_score)

            return fragment

        except Exception as e:
            logger.error(f"Fragment generation error in {self.agent_id}: {e}")
            raise

    async def _generate_pattern_fragment(
        self,
        task: GLYPHGenerationTask,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Generate geometric pattern fragment."""

        size = params.get("size", (256, 256))
        pattern_type = params.get("pattern_type", "sacred_geometry")

        # Create base pattern
        pattern = np.zeros((*size, 4), dtype=np.uint8)  # RGBA

        if pattern_type == "sacred_geometry":
            # Generate sacred geometry patterns based on tier
            if task.tier_level >= 3:
                pattern = self._create_metatron_cube(size)
            else:
                pattern = self._create_flower_of_life(size)

        elif pattern_type == "fibonacci":
            pattern = self._create_fibonacci_spiral(size)

        elif pattern_type == "fractal":
            pattern = self._create_fractal_pattern(size, task.tier_level)

        # Add identity-specific modulation
        identity_hash = hashlib.sha256(task.lambda_id.encode()).digest()
        modulation = np.frombuffer(identity_hash, dtype=np.uint8)[:3]

        for i in range(3):
            pattern[:, :, i] = (pattern[:, :, i] * (0.8 + 0.2 * modulation[i] / 255)).astype(np.uint8)

        return pattern

    async def _generate_color_fragment(
        self,
        task: GLYPHGenerationTask,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Generate color palette fragment."""

        size = params.get("size", (256, 256))

        # Generate tier-specific color palette
        if task.tier_level == 0:
            base_colors = [(100, 100, 100), (150, 150, 150)]  # Gray
        elif task.tier_level == 1:
            base_colors = [(100, 150, 200), (150, 200, 250)]  # Blue
        elif task.tier_level == 2:
            base_colors = [(100, 200, 150), (150, 250, 200)]  # Green
        elif task.tier_level == 3:
            base_colors = [(200, 150, 100), (250, 200, 150)]  # Gold
        elif task.tier_level == 4:
            base_colors = [(200, 100, 200), (250, 150, 250)]  # Purple
        else:  # Tier 5
            base_colors = [(255, 255, 255), (200, 200, 255)]  # Iridescent

        # Create gradient with identity modulation
        gradient = self._create_identity_gradient(size, base_colors, task.lambda_id)

        # Add consciousness colors if available
        if task.consciousness_pattern is not None:
            gradient = self._blend_consciousness_colors(gradient, task.consciousness_pattern)

        return gradient

    async def _generate_quantum_fragment(
        self,
        task: GLYPHGenerationTask,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Generate quantum-entangled pattern fragment."""

        size = params.get("size", (256, 256))

        # Use quantum seed if available
        if task.quantum_seed:
            np.random.seed(int.from_bytes(task.quantum_seed[:4], 'big'))

        # Generate quantum interference pattern
        x, y = np.meshgrid(np.linspace(-5, 5, size[0]), np.linspace(-5, 5, size[1]))

        # Multiple wave functions
        psi1 = np.exp(-((x-1)**2 + y**2) / 2) * np.exp(1j * (x + y))
        psi2 = np.exp(-((x+1)**2 + y**2) / 2) * np.exp(1j * (x - y))

        # Quantum superposition
        if task.tier_level >= 4:
            # Add more complex quantum states
            psi3 = np.exp(-(x**2 + (y-1)**2) / 2) * np.exp(1j * x * y)
            interference = np.abs(psi1 + psi2 + psi3)**2
        else:
            interference = np.abs(psi1 + psi2)**2

        # Normalize and convert to image
        interference = (interference / interference.max() * 255).astype(np.uint8)

        # Create RGBA image with quantum patterns
        quantum_pattern = np.zeros((*size, 4), dtype=np.uint8)
        quantum_pattern[:, :, 0] = interference  # Red channel
        quantum_pattern[:, :, 1] = np.roll(interference, 50, axis=0)  # Green shifted
        quantum_pattern[:, :, 2] = np.roll(interference, -50, axis=1)  # Blue shifted
        quantum_pattern[:, :, 3] = 255  # Full alpha

        return quantum_pattern

    async def _generate_consciousness_fragment(
        self,
        task: GLYPHGenerationTask,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Generate consciousness-encoded fragment."""

        size = params.get("size", (256, 256))

        if task.consciousness_pattern is None:
            # Generate default consciousness pattern
            pattern = self._create_default_consciousness_pattern(size)
        else:
            # Use provided consciousness data
            pattern = self._encode_consciousness_data(task.consciousness_pattern, size)

        # Add dream layer for Tier 5
        if task.tier_level >= 5 and task.dream_sequence:
            pattern = self._overlay_dream_patterns(pattern, task.dream_sequence)

        return pattern

    async def _generate_embedding_fragment(
        self,
        task: GLYPHGenerationTask,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Generate steganographic embedding layer."""

        size = params.get("size", (256, 256))

        # Create embedding carrier
        carrier = np.ones((*size, 4), dtype=np.uint8) * 128  # Mid-gray base

        if task.steganographic_data:
            # Prepare data for embedding
            data_to_embed = {
                "lambda_id": task.lambda_id,
                "tier_level": task.tier_level,
                "glyph_type": task.glyph_type.value,
                "timestamp": datetime.utcnow().isoformat(),
                **task.steganographic_data
            }

            # Simulate embedding (actual implementation would use steganography module)
            embedded = self._simulate_steganographic_embedding(carrier, data_to_embed)
            return embedded

        return carrier

    # Helper methods for pattern generation

    def _create_flower_of_life(self, size: Tuple[int, int]) -> np.ndarray:
        """Create Flower of Life sacred geometry pattern."""
        pattern = np.zeros((*size, 4), dtype=np.uint8)
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = min(size) // 8

        # Create overlapping circles
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]

        for angle in angles:
            cx = int(center_x + radius * np.cos(angle))
            cy = int(center_y + radius * np.sin(angle))

            # Draw circle
            y, x = np.ogrid[:size[1], :size[0]]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            pattern[mask, :3] = 255
            pattern[mask, 3] = 200

        # Center circle
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        pattern[mask, :3] = 255
        pattern[mask, 3] = 255

        return pattern

    def _create_metatron_cube(self, size: Tuple[int, int]) -> np.ndarray:
        """Create Metatron's Cube pattern."""
        pattern = self._create_flower_of_life(size)

        # Add connecting lines between circle centers
        # This would be more complex in full implementation

        return pattern

    def _create_fibonacci_spiral(self, size: Tuple[int, int]) -> np.ndarray:
        """Create Fibonacci spiral pattern."""
        pattern = np.zeros((*size, 4), dtype=np.uint8)

        # Generate spiral points
        theta = np.linspace(0, 8 * np.pi, 1000)
        r = np.exp(0.1 * theta)

        # Convert to coordinates
        x = r * np.cos(theta) + size[0] // 2
        y = r * np.sin(theta) + size[1] // 2

        # Draw spiral
        for i in range(len(x) - 1):
            if 0 <= x[i] < size[0] and 0 <= y[i] < size[1]:
                pattern[int(y[i]), int(x[i]), :3] = 255
                pattern[int(y[i]), int(x[i]), 3] = 255

        return pattern

    def _create_fractal_pattern(self, size: Tuple[int, int], iterations: int) -> np.ndarray:
        """Create fractal pattern with tier-based complexity."""
        pattern = np.zeros((*size, 4), dtype=np.uint8)

        # Simple Mandelbrot set
        x = np.linspace(-2, 1, size[0])
        y = np.linspace(-1.5, 1.5, size[1])
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        Z = np.zeros_like(C)
        M = np.zeros(C.shape)

        for i in range(iterations * 10):
            mask = np.abs(Z) < 2
            Z[mask] = Z[mask]**2 + C[mask]
            M[mask] = i

        # Normalize and colorize
        M = (M / M.max() * 255).astype(np.uint8)
        pattern[:, :, 0] = M
        pattern[:, :, 1] = np.roll(M, 30, axis=0)
        pattern[:, :, 2] = np.roll(M, -30, axis=1)
        pattern[:, :, 3] = 255

        return pattern

    def _create_identity_gradient(
        self,
        size: Tuple[int, int],
        colors: List[Tuple[int, int, int]],
        identity: str
    ) -> np.ndarray:
        """Create identity-specific gradient."""
        gradient = np.zeros((*size, 4), dtype=np.uint8)

        # Create base gradient
        for i in range(size[1]):
            t = i / size[1]
            color = [
                int(colors[0][j] * (1 - t) + colors[1][j] * t)
                for j in range(3)
            ]
            gradient[i, :, :3] = color
            gradient[i, :, 3] = 255

        # Add identity-specific noise
        identity_hash = hashlib.sha256(identity.encode()).digest()
        np.random.seed(int.from_bytes(identity_hash[:4], 'big'))
        noise = np.random.normal(0, 10, (*size, 3))

        gradient[:, :, :3] = np.clip(gradient[:, :, :3] + noise, 0, 255)

        return gradient

    def _blend_consciousness_colors(
        self,
        base: np.ndarray,
        consciousness_pattern: np.ndarray
    ) -> np.ndarray:
        """Blend consciousness data into color pattern."""
        # Normalize consciousness pattern
        if consciousness_pattern.max() > 0:
            normalized = consciousness_pattern / consciousness_pattern.max()
        else:
            normalized = consciousness_pattern

        # Resize if needed
        if normalized.shape[:2] != base.shape[:2]:
            from scipy import ndimage
            normalized = ndimage.zoom(
                normalized,
                (base.shape[0] / normalized.shape[0], base.shape[1] / normalized.shape[1]),
                order=1
            )

        # Blend with base
        blended = base.copy()
        blended[:, :, :3] = (base[:, :, :3] * 0.7 + normalized[:, :, None] * 255 * 0.3).astype(np.uint8)

        return blended

    def _create_default_consciousness_pattern(self, size: Tuple[int, int]) -> np.ndarray:
        """Create default consciousness pattern."""
        pattern = np.zeros((*size, 4), dtype=np.uint8)

        # Create wave interference pattern
        x, y = np.meshgrid(np.linspace(0, 10, size[0]), np.linspace(0, 10, size[1]))
        wave1 = np.sin(x) * np.cos(y)
        wave2 = np.sin(x * 0.7) * np.cos(y * 1.3)

        interference = (wave1 + wave2) / 2
        normalized = ((interference + 1) / 2 * 255).astype(np.uint8)

        pattern[:, :, 0] = normalized
        pattern[:, :, 1] = np.roll(normalized, 20, axis=0)
        pattern[:, :, 2] = np.roll(normalized, -20, axis=1)
        pattern[:, :, 3] = 255

        return pattern

    def _encode_consciousness_data(
        self,
        consciousness_data: np.ndarray,
        size: Tuple[int, int]
    ) -> np.ndarray:
        """Encode consciousness data into visual pattern."""
        # This would implement actual consciousness encoding
        return self._create_default_consciousness_pattern(size)

    def _overlay_dream_patterns(
        self,
        base: np.ndarray,
        dream_sequence: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Overlay dream sequence patterns."""
        # This would implement dream pattern overlay
        return base

    def _simulate_steganographic_embedding(
        self,
        carrier: np.ndarray,
        data: Dict[str, Any]
    ) -> np.ndarray:
        """Simulate steganographic embedding."""
        # Add subtle pattern to indicate embedded data
        embedded = carrier.copy()

        # Create data signature
        import json
        data_str = json.dumps(data, sort_keys=True)
        signature = hashlib.sha256(data_str.encode()).digest()

        # Embed signature pattern (simplified)
        for i, byte_val in enumerate(signature[:16]):
            row = i // embedded.shape[1]
            col = i % embedded.shape[1]
            if row < embedded.shape[0]:
                embedded[row, col, 0] = (embedded[row, col, 0] & 0xFE) | (byte_val & 1)

        return embedded

    def _evaluate_fragment_quality(self, fragment: np.ndarray, tier_level: int) -> float:
        """Evaluate quality of generated fragment."""
        # Calculate various quality metrics

        # Entropy (information content)
        hist, _ = np.histogram(fragment.flatten(), bins=256, range=(0, 255))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        # Contrast
        contrast = fragment.std() / 255.0

        # Tier-specific quality requirements
        tier_weights = {
            0: {"entropy": 0.3, "contrast": 0.7},
            1: {"entropy": 0.4, "contrast": 0.6},
            2: {"entropy": 0.5, "contrast": 0.5},
            3: {"entropy": 0.6, "contrast": 0.4},
            4: {"entropy": 0.7, "contrast": 0.3},
            5: {"entropy": 0.8, "contrast": 0.2}
        }

        weights = tier_weights.get(tier_level, tier_weights[0])

        # Calculate weighted score
        quality_score = (
            weights["entropy"] * (entropy / 8) +  # Normalize entropy to 0-1
            weights["contrast"] * contrast
        )

        return min(1.0, quality_score)


class DistributedGLYPHColony(BaseColony):
    """
    Colony for distributed GLYPH generation with consensus assembly.
    """

    def __init__(self, colony_id: str = "glyph_generation"):
        super().__init__(
            colony_id=colony_id,
            capabilities=["glyph_generation", "distributed_rendering", "consensus_assembly"]
        )

        self.generation_agents: Dict[str, GLYPHGenerationAgent] = {}
        self.active_tasks: Dict[str, GLYPHGenerationTask] = {}
        self.fragment_pool: Dict[str, List[GLYPHFragment]] = {}
        self.generated_glyphs: Dict[str, GeneratedGLYPH] = {}

        # Colony configuration
        self.agents_per_specialization = 3
        self.fragment_consensus_threshold = 0.8
        self.assembly_timeout = 30.0  # seconds

        # Performance metrics
        self.total_glyphs_generated = 0
        self.average_generation_time = 0.0
        self.quality_scores: List[float] = []

        logger.info(f"Distributed GLYPH Colony {colony_id} initialized")

    async def initialize(self):
        """Initialize the colony with specialized agents."""
        await super().initialize()

        # Create specialized agents
        specializations = ["pattern", "color", "quantum", "consciousness", "embedding"]

        for spec in specializations:
            for i in range(self.agents_per_specialization):
                agent_id = f"{self.colony_id}_agent_{spec}_{i}"
                agent = GLYPHGenerationAgent(agent_id, self, spec)

                self.generation_agents[agent_id] = agent
                self.agents[agent_id] = agent

        # Get event publisher
        from identity.core.events import get_identity_event_publisher
        self.event_publisher = await get_identity_event_publisher()

        logger.info(f"Colony initialized with {len(self.generation_agents)} specialized agents")

    async def generate_identity_glyph(
        self,
        lambda_id: str,
        glyph_type: GLYPHType,
        tier_level: int,
        identity_data: Dict[str, Any],
        orb_state: Optional[OrbVisualization] = None,
        steganographic_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> GeneratedGLYPH:
        """
        Generate an identity GLYPH using distributed agents.
        """

        # Create generation task
        task = GLYPHGenerationTask(
            task_id=f"glyph_{lambda_id}_{glyph_type.value}_{int(datetime.utcnow().timestamp())}",
            lambda_id=lambda_id,
            glyph_type=glyph_type,
            tier_level=tier_level,
            complexity=self._determine_complexity(tier_level),
            identity_data=identity_data,
            orb_state=orb_state,
            steganographic_data=steganographic_data,
            deadline=datetime.utcnow() + timedelta(seconds=self.assembly_timeout)
        )

        # Add quantum/consciousness data for higher tiers
        if tier_level >= 4:
            task.quantum_seed = self._generate_quantum_seed(lambda_id, session_id)

        if tier_level >= 3 and orb_state:
            task.consciousness_pattern = self._extract_consciousness_pattern(orb_state)

        self.active_tasks[task.task_id] = task
        self.fragment_pool[task.task_id] = []

        # Publish generation start event
        await self.event_publisher.publish_glyph_event(
            IdentityEventType.GLYPH_GENERATION_START,
            lambda_id=lambda_id,
            tier_level=tier_level,
            glyph_data={
                "glyph_type": glyph_type.value,
                "complexity": task.complexity.name,
                "task_id": task.task_id
            },
            session_id=session_id
        )

        try:
            # Generate fragments in parallel
            fragment_tasks = []

            # Determine which fragments to generate based on tier
            required_fragments = self._determine_required_fragments(tier_level, glyph_type)

            for fragment_type in required_fragments:
                # Get agents for this fragment type
                agents = [
                    agent for agent in self.generation_agents.values()
                    if agent.specialization == fragment_type and agent.state != AgentState.FAILED
                ]

                if not agents:
                    logger.warning(f"No agents available for {fragment_type}")
                    continue

                # Assign fragment generation to multiple agents
                for agent in agents[:2]:  # Use up to 2 agents per fragment type
                    fragment_params = self._create_fragment_params(
                        fragment_type, tier_level, glyph_type
                    )

                    fragment_task = agent.generate_fragment(task, fragment_params)
                    fragment_tasks.append(fragment_task)

            # Wait for fragments with timeout
            fragments = await asyncio.wait_for(
                asyncio.gather(*fragment_tasks, return_exceptions=True),
                timeout=self.assembly_timeout
            )

            # Collect successful fragments
            for fragment in fragments:
                if isinstance(fragment, GLYPHFragment):
                    self.fragment_pool[task.task_id].append(fragment)
                else:
                    logger.error(f"Fragment generation error: {fragment}")

            # Assemble GLYPH from fragments
            glyph = await self._assemble_glyph(task)

            # Store generated GLYPH
            self.generated_glyphs[glyph.glyph_id] = glyph
            self.total_glyphs_generated += 1

            # Update metrics
            generation_time = (datetime.utcnow() - task.deadline + timedelta(seconds=self.assembly_timeout)).total_seconds()
            self.average_generation_time = (
                (self.average_generation_time * (self.total_glyphs_generated - 1) + generation_time) /
                self.total_glyphs_generated
            )
            self.quality_scores.append(glyph.quality_metrics.get("overall_quality", 0))

            # Publish completion event
            await self.event_publisher.publish_glyph_event(
                IdentityEventType.GLYPH_GENERATED,
                lambda_id=lambda_id,
                tier_level=tier_level,
                glyph_data={
                    "glyph_id": glyph.glyph_id,
                    "glyph_type": glyph_type.value,
                    "quality_score": glyph.quality_metrics.get("overall_quality", 0),
                    "fragments_used": len(glyph.fragments_used),
                    "generation_time": generation_time
                },
                session_id=session_id
            )

            return glyph

        except asyncio.TimeoutError:
            logger.error(f"GLYPH generation timeout for task {task.task_id}")
            raise

        except Exception as e:
            logger.error(f"GLYPH generation error: {e}")
            raise

        finally:
            # Cleanup
            self.active_tasks.pop(task.task_id, None)
            self.fragment_pool.pop(task.task_id, None)

    async def _assemble_glyph(self, task: GLYPHGenerationTask) -> GeneratedGLYPH:
        """Assemble GLYPH from fragments with consensus."""

        fragments = self.fragment_pool.get(task.task_id, [])

        if not fragments:
            raise ValueError("No fragments available for assembly")

        # Group fragments by type
        fragments_by_type = {}
        for fragment in fragments:
            if fragment.fragment_type not in fragments_by_type:
                fragments_by_type[fragment.fragment_type] = []
            fragments_by_type[fragment.fragment_type].append(fragment)

        # Select best fragments through consensus
        selected_fragments = []
        for fragment_type, type_fragments in fragments_by_type.items():
            if len(type_fragments) == 1:
                selected_fragments.append(type_fragments[0])
            else:
                # Vote on best fragment
                best_fragment = max(type_fragments, key=lambda f: f.quality_score)

                # Check if consensus threshold met
                votes = sum(1 for f in type_fragments if f.quality_score >= best_fragment.quality_score * 0.9)
                if votes / len(type_fragments) >= self.fragment_consensus_threshold:
                    selected_fragments.append(best_fragment)

        # Composite fragments into final GLYPH
        final_image = await self._composite_fragments(selected_fragments, task)

        # Calculate quality metrics
        quality_metrics = self._calculate_glyph_quality(final_image, selected_fragments)

        # Extract embedded data if steganographic layer was used
        embedded_data = None
        if task.steganographic_data:
            # Would extract using steganography module
            embedded_data = task.steganographic_data

        # Create GLYPH
        glyph = GeneratedGLYPH(
            glyph_id=f"{task.lambda_id}_{task.glyph_type.value}_{int(datetime.utcnow().timestamp())}",
            lambda_id=task.lambda_id,
            glyph_type=task.glyph_type,
            tier_level=task.tier_level,
            image_data=final_image,
            embedded_data=embedded_data,
            generation_metadata={
                "task_id": task.task_id,
                "complexity": task.complexity.name,
                "fragments_generated": len(fragments),
                "fragments_selected": len(selected_fragments),
                "consensus_achieved": len(selected_fragments) >= len(fragments_by_type) * 0.8
            },
            fragments_used=[f.fragment_id for f in selected_fragments],
            consensus_achieved=len(selected_fragments) >= len(fragments_by_type) * 0.8,
            quality_metrics=quality_metrics,
            timestamp=datetime.utcnow()
        )

        return glyph

    async def _composite_fragments(
        self,
        fragments: List[GLYPHFragment],
        task: GLYPHGenerationTask
    ) -> np.ndarray:
        """Composite fragments into final GLYPH image."""

        # Determine final size
        size = (512, 512) if task.tier_level >= 3 else (256, 256)

        # Create base canvas
        composite = np.zeros((*size, 4), dtype=np.float32)

        # Layer fragments based on type
        layer_order = ["pattern", "color", "quantum", "consciousness", "embedding"]

        for layer_type in layer_order:
            layer_fragments = [f for f in fragments if f.fragment_type == layer_type]

            for fragment in layer_fragments:
                # Resize fragment if needed
                if fragment.data.shape[:2] != size:
                    from scipy import ndimage
                    resized = np.zeros((*size, fragment.data.shape[2]))
                    for c in range(fragment.data.shape[2]):
                        resized[:, :, c] = ndimage.zoom(
                            fragment.data[:, :, c],
                            (size[0] / fragment.data.shape[0], size[1] / fragment.data.shape[1]),
                            order=1
                        )
                    fragment_data = resized
                else:
                    fragment_data = fragment.data.astype(np.float32)

                # Blend based on layer type
                if layer_type == "pattern":
                    # Base layer
                    composite = fragment_data
                elif layer_type == "color":
                    # Color overlay
                    alpha = fragment_data[:, :, 3:4] / 255.0
                    composite[:, :, :3] = composite[:, :, :3] * (1 - alpha) + fragment_data[:, :, :3] * alpha
                    composite[:, :, 3] = np.maximum(composite[:, :, 3], fragment_data[:, :, 3])
                elif layer_type in ["quantum", "consciousness"]:
                    # Additive blending for energy patterns
                    blend_factor = 0.3 if task.tier_level < 4 else 0.5
                    composite[:, :, :3] = np.clip(
                        composite[:, :, :3] + fragment_data[:, :, :3] * blend_factor,
                        0, 255
                    )
                elif layer_type == "embedding":
                    # Subtle embedding layer
                    composite = composite * 0.95 + fragment_data * 0.05

        # Final adjustments
        composite = np.clip(composite, 0, 255).astype(np.uint8)

        # Ensure full alpha
        composite[:, :, 3] = 255

        return composite

    def _determine_complexity(self, tier_level: int) -> GLYPHComplexity:
        """Determine GLYPH complexity based on tier."""
        if tier_level == 0:
            return GLYPHComplexity.BASIC
        elif tier_level <= 2:
            return GLYPHComplexity.STANDARD
        elif tier_level == 3:
            return GLYPHComplexity.ENHANCED
        elif tier_level == 4:
            return GLYPHComplexity.ADVANCED
        else:
            return GLYPHComplexity.TRANSCENDENT

    def _determine_required_fragments(
        self,
        tier_level: int,
        glyph_type: GLYPHType
    ) -> List[str]:
        """Determine which fragment types are required."""

        # Base fragments for all tiers
        required = ["pattern", "color"]

        # Add tier-specific fragments
        if tier_level >= 2:
            required.append("embedding")

        if tier_level >= 3:
            required.append("consciousness")

        if tier_level >= 4:
            required.append("quantum")

        # Type-specific requirements
        if glyph_type == GLYPHType.QUANTUM:
            if "quantum" not in required:
                required.append("quantum")
        elif glyph_type == GLYPHType.CONSCIOUSNESS:
            if "consciousness" not in required:
                required.append("consciousness")

        return required

    def _create_fragment_params(
        self,
        fragment_type: str,
        tier_level: int,
        glyph_type: GLYPHType
    ) -> Dict[str, Any]:
        """Create parameters for fragment generation."""

        base_size = (512, 512) if tier_level >= 3 else (256, 256)

        params = {
            "size": base_size,
            "quality": "high" if tier_level >= 3 else "standard"
        }

        # Type-specific parameters
        if fragment_type == "pattern":
            if glyph_type == GLYPHType.AUTHENTICATION:
                params["pattern_type"] = "sacred_geometry"
            elif glyph_type == GLYPHType.SIGNATURE:
                params["pattern_type"] = "fibonacci"
            else:
                params["pattern_type"] = "fractal"

        elif fragment_type == "quantum":
            params["entanglement_level"] = min(tier_level, 5)
            params["superposition_states"] = 2 ** min(tier_level + 1, 8)

        return params

    def _generate_quantum_seed(self, lambda_id: str, session_id: Optional[str]) -> bytes:
        """Generate quantum seed for GLYPH generation."""
        # Combine identity and session for uniqueness
        seed_data = f"{lambda_id}:{session_id or 'default'}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(seed_data.encode()).digest()

    def _extract_consciousness_pattern(self, orb_state: OrbVisualization) -> np.ndarray:
        """Extract consciousness pattern from ORB state."""
        # This would extract actual consciousness data from ORB
        # For now, generate placeholder pattern
        size = (128, 128)
        pattern = np.random.rand(*size) * orb_state.energy_level
        return pattern

    def _calculate_glyph_quality(
        self,
        image: np.ndarray,
        fragments: List[GLYPHFragment]
    ) -> Dict[str, float]:
        """Calculate quality metrics for generated GLYPH."""

        # Fragment quality average
        fragment_quality = sum(f.quality_score for f in fragments) / len(fragments) if fragments else 0

        # Image quality metrics
        # Entropy
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        # Uniqueness (simplified - would use perceptual hashing in production)
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        uniqueness = len(set(image_hash)) / len(image_hash)

        # Overall quality
        overall_quality = (
            fragment_quality * 0.4 +
            (entropy / 8) * 0.3 +
            uniqueness * 0.3
        )

        return {
            "overall_quality": overall_quality,
            "fragment_quality": fragment_quality,
            "entropy": entropy,
            "uniqueness": uniqueness,
            "complexity": len(fragments) / 5.0  # Normalized by max fragments
        }

    def get_colony_statistics(self) -> Dict[str, Any]:
        """Get colony performance statistics."""

        agent_stats = {}
        for agent in self.generation_agents.values():
            agent_stats[agent.specialization] = {
                "fragments_generated": agent.fragments_generated,
                "avg_quality": sum(agent.quality_scores) / len(agent.quality_scores) if agent.quality_scores else 0
            }

        return {
            "total_glyphs_generated": self.total_glyphs_generated,
            "average_generation_time": self.average_generation_time,
            "average_quality_score": sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0,
            "active_tasks": len(self.active_tasks),
            "agent_statistics": agent_stats,
            "cached_glyphs": len(self.generated_glyphs)
        }