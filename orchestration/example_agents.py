"""
══════════════════════════════════════════════════════════════════════════════════
║ 🧠 LUKHAS AI - ORCHESTRATION
║ Example agent implementations for the LUKHAS system.
║ Copyright (c) 2025 LUKHAS AI. All rights reserved.
╠══════════════════════════════════════════════════════════════════════════════════
║ Module: example_agents.py
║ Path: lukhas/orchestration/example_agents.py
║ Version: 1.0.0 | Created: 2025-07-25 | Modified: 2025-07-25
║ Authors: LUKHAS AI Orchestration Team | Jules
╠══════════════════════════════════════════════════════════════════════════════════
║ DESCRIPTION
╠══════════════════════════════════════════════════════════════════════════════════
║ This module provides example agents demonstrating how to implement the
║ AgentInterface for various capabilities in the LUKHAS system.
╚══════════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import random
from typing import Dict, Any, Optional
from datetime import datetime

from .interfaces.agent_interface import (
    AgentInterface, AgentMetadata, AgentCapability,
    AgentStatus, AgentMessage, AgentContext
)


class AnalyzerAgent(AgentInterface):
    """
    Example analyzer agent that performs data analysis tasks.
    """

    def __init__(self):
        metadata = AgentMetadata(
            name="AnalyzerAgent",
            version="1.0.0",
            description="Analyzes data and provides insights",
            capabilities={
                AgentCapability.TASK_PROCESSING,
                AgentCapability.REASONING,
                AgentCapability.INTER_AGENT_COMM
            },
            tags=["analyzer", "data-processing", "insights"]
        )
        super().__init__(metadata)

        # Register custom capability
        self.register_capability("data_analysis")
        self.register_capability("pattern_recognition")

        # Analysis statistics
        self.analysis_count = 0
        self.total_data_processed = 0
        self.patterns = []
        self.tasks_analyzed = 0
        self.start_time = datetime.now()

    async def initialize(self, context: AgentContext) -> bool:
        """Initialize the analyzer agent"""
        try:
            await self.execute_lifecycle_hooks('pre_init')

            self.context = context
            self._logger.info(f"Initializing {self.metadata.name}")

            # Setup message handlers
            self.register_message_handler("data_request", self._handle_data_request)
            self.register_message_handler("analysis_request", self._handle_analysis_request)

            self.update_status(AgentStatus.READY)
            await self.execute_lifecycle_hooks('post_init')

            return True

        except Exception as e:
            await self.handle_error(e, "initialization")
            return False

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process analysis tasks"""
        try:
            await self.execute_lifecycle_hooks('pre_process')
            self.update_status(AgentStatus.PROCESSING)

            task_type = task.get('task_type', 'unknown')
            task_id = task.get('task_id', 'unknown')

            result = None

            if task_type == "analyze_data":
                result = await self._analyze_data(task.get('parameters', {}))
            elif task_type == "find_patterns":
                result = await self._find_patterns(task.get('parameters', {}))
            elif task_type == "generate_insights":
                result = await self._generate_insights(task.get('parameters', {}))
            else:
                result = {
                    "error": f"Unknown task type: {task_type}",
                    "supported_types": ["analyze_data", "find_patterns", "generate_insights"]
                }

            self.analysis_count += 1

            self.update_status(AgentStatus.ACTIVE)
            await self.execute_lifecycle_hooks('post_process')

            return {
                'status': 'completed',
                'task_id': task_id,
                'task_type': task_type,
                'result': result,
                'agent_id': self.metadata.agent_id,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            await self.handle_error(e, "task processing")
            return {
                'status': 'error',
                'error': str(e),
                'task_id': task.get('task_id', 'unknown')
            }

    async def _analyze_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data analysis"""
        data = parameters.get('data', [])
        analysis_type = parameters.get('analysis_type', 'basic')

        # Simulate analysis
        await asyncio.sleep(random.uniform(0.1, 0.5))

        self.total_data_processed += len(str(data))

        # Mock analysis results
        return {
            "analysis_type": analysis_type,
            "data_points": len(data) if isinstance(data, list) else 1,
            "insights": {
                "mean": random.uniform(0, 100),
                "std_dev": random.uniform(0, 20),
                "trend": random.choice(["increasing", "decreasing", "stable"]),
                "anomalies": random.randint(0, 5)
            },
            "confidence": random.uniform(0.7, 0.99)
        }

    async def _find_patterns(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Find patterns in data"""
        data = parameters.get('data', [])
        pattern_type = parameters.get('pattern_type', 'sequential')

        # Simulate pattern finding
        await asyncio.sleep(random.uniform(0.2, 0.6))

        return {
            "pattern_type": pattern_type,
            "patterns_found": random.randint(0, 10),
            "pattern_strength": random.uniform(0.5, 1.0),
            "examples": [
                {"index": i, "pattern": f"pattern_{i}"}
                for i in range(min(3, random.randint(0, 5)))
            ]
        }

    async def _generate_insights(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from analysis"""
        context = parameters.get('context', {})

        # Simulate insight generation
        await asyncio.sleep(random.uniform(0.3, 0.7))

        insights = [
            "Data shows significant variation in the observed metrics",
            "Trend analysis indicates potential optimization opportunities",
            "Correlation detected between input parameters and outcomes",
            "Anomaly detection suggests need for further investigation"
        ]

        return {
            "insights": random.sample(insights, k=random.randint(2, 4)),
            "recommendations": [
                "Consider adjusting parameters for better performance",
                "Implement monitoring for detected anomalies"
            ],
            "confidence": random.uniform(0.6, 0.95)
        }

    async def _handle_data_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle data requests from other agents"""
        request_type = message.content.get('request_type', 'unknown')

        response_content = {
            "request_id": message.message_id,
            "data_available": True,
            "data_types": ["metrics", "logs", "traces"],
            "format": "json"
        }

        return AgentMessage(
            sender_id=self.metadata.agent_id,
            recipient_id=message.sender_id,
            message_type="data_response",
            content=response_content
        )

    async def _handle_analysis_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle analysis requests from other agents"""
        analysis_params = message.content.get('parameters', {})

        # Perform quick analysis
        result = await self._analyze_data(analysis_params)

        return AgentMessage(
            sender_id=self.metadata.agent_id,
            recipient_id=message.sender_id,
            message_type="analysis_response",
            content=result
        )

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming messages"""
        # Use registered handlers first
        response = await super().handle_message(message)
        if response:
            return response

        # Handle other message types
        if message.message_type == "collaboration_request":
            return AgentMessage(
                sender_id=self.metadata.agent_id,
                recipient_id=message.sender_id,
                message_type="collaboration_response",
                content={
                    "accepted": True,
                    "capabilities": ["data_analysis", "pattern_recognition"]
                }
            )

        return None

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        # Get base health status
        health = {
            'status': self.status.name,
            'agent_id': self.metadata.agent_id,
            'uptime': (datetime.now() - self.metadata.created_at).total_seconds(),
            'active_tasks': len(self.context.active_tasks) if self.context else 0,
            'capabilities': [c.value if isinstance(c, AgentCapability) else c for c in self.metadata.capabilities],
            'healthy': self.status not in [AgentStatus.ERROR, AgentStatus.TERMINATED]
        }

        # Add analyzer-specific metrics
        health.update({
            "analysis_count": self.analysis_count,
            "total_data_processed": self.total_data_processed,
            "average_analysis_time": random.uniform(0.1, 0.5),
            "error_rate": random.uniform(0, 0.05)
        })

        return health

    async def shutdown(self) -> None:
        """Shutdown the analyzer agent"""
        self._logger.info("Shutting down analyzer agent")
        self.update_status(AgentStatus.TERMINATED)


class LearningAgent(AgentInterface):
    """
    Example learning agent that can adapt and improve over time.
    """

    def __init__(self):
        metadata = AgentMetadata(
            name="LearningAgent",
            version="1.0.0",
            description="Adaptive learning agent with meta-learning capabilities",
            capabilities={
                AgentCapability.TASK_PROCESSING,
                AgentCapability.LEARNING,
                AgentCapability.MEMORY_ACCESS,
                AgentCapability.INTER_AGENT_COMM
            },
            tags=["learning", "adaptive", "meta-learning"]
        )
        super().__init__(metadata)

        # Learning state
        self.knowledge_base = {}
        self.learning_rate = 0.01
        self.adaptation_count = 0

    async def initialize(self, context: AgentContext) -> bool:
        """Initialize the learning agent"""
        try:
            await self.execute_lifecycle_hooks('pre_init')

            self.context = context
            self._logger.info(f"Initializing {self.metadata.name}")

            # Initialize knowledge base
            self.knowledge_base = {
                "patterns": [],
                "strategies": {},
                "performance_history": []
            }

            self.update_status(AgentStatus.READY)
            await self.execute_lifecycle_hooks('post_init')

            return True

        except Exception as e:
            await self.handle_error(e, "initialization")
            return False

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process learning tasks"""
        try:
            await self.execute_lifecycle_hooks('pre_process')
            self.update_status(AgentStatus.PROCESSING)

            task_type = task.get('task_type', 'unknown')
            task_id = task.get('task_id', 'unknown')

            result = None

            if task_type == "learn_pattern":
                result = await self._learn_pattern(task.get('parameters', {}))
            elif task_type == "adapt_strategy":
                result = await self._adapt_strategy(task.get('parameters', {}))
            elif task_type == "meta_learn":
                result = await self._meta_learn(task.get('parameters', {}))
            else:
                result = {"error": f"Unknown task type: {task_type}"}

            # Record performance
            self.knowledge_base["performance_history"].append({
                "task_id": task_id,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat(),
                "success": "error" not in result
            })

            self.update_status(AgentStatus.ACTIVE)
            await self.execute_lifecycle_hooks('post_process')

            return {
                'status': 'completed',
                'task_id': task_id,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            await self.handle_error(e, "task processing")
            return {
                'status': 'error',
                'error': str(e),
                'task_id': task.get('task_id', 'unknown')
            }

    async def _learn_pattern(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Learn a new pattern"""
        pattern_data = parameters.get('pattern', {})
        pattern_type = parameters.get('type', 'unknown')

        # Simulate learning
        await asyncio.sleep(random.uniform(0.2, 0.5))

        # Store pattern
        pattern_entry = {
            "type": pattern_type,
            "data": pattern_data,
            "learned_at": datetime.now().isoformat(),
            "confidence": random.uniform(0.7, 0.95)
        }

        self.knowledge_base["patterns"].append(pattern_entry)
        self.adaptation_count += 1

        return {
            "learned": True,
            "pattern_id": len(self.knowledge_base["patterns"]) - 1,
            "confidence": pattern_entry["confidence"]
        }

    async def _adapt_strategy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt strategy based on feedback"""
        strategy_name = parameters.get('strategy', 'default')
        feedback = parameters.get('feedback', {})

        # Simulate adaptation
        await asyncio.sleep(random.uniform(0.1, 0.3))

        # Update strategy
        if strategy_name not in self.knowledge_base["strategies"]:
            self.knowledge_base["strategies"][strategy_name] = {
                "performance": 0.5,
                "usage_count": 0
            }

        strategy = self.knowledge_base["strategies"][strategy_name]

        # Simple adaptation based on feedback
        if feedback.get('success', False):
            strategy["performance"] += self.learning_rate
        else:
            strategy["performance"] -= self.learning_rate

        strategy["performance"] = max(0.0, min(1.0, strategy["performance"]))
        strategy["usage_count"] += 1

        self.adaptation_count += 1

        return {
            "adapted": True,
            "strategy": strategy_name,
            "new_performance": strategy["performance"],
            "adaptation_count": self.adaptation_count
        }

    async def _meta_learn(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform meta-learning"""
        learning_context = parameters.get('context', {})

        # Simulate meta-learning
        await asyncio.sleep(random.uniform(0.3, 0.6))

        # Adjust learning rate based on performance
        recent_performance = [
            p["success"] for p in self.knowledge_base["performance_history"][-10:]
        ]

        if recent_performance:
            success_rate = sum(recent_performance) / len(recent_performance)

            # Adapt learning rate
            if success_rate > 0.8:
                self.learning_rate *= 0.9  # Slow down if doing well
            elif success_rate < 0.5:
                self.learning_rate *= 1.1  # Speed up if struggling

            self.learning_rate = max(0.001, min(0.1, self.learning_rate))

        return {
            "meta_learned": True,
            "new_learning_rate": self.learning_rate,
            "patterns_learned": len(self.knowledge_base["patterns"]),
            "strategies_adapted": len(self.knowledge_base["strategies"])
        }

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming messages"""
        if message.message_type == "knowledge_request":
            # Share knowledge with other agents
            return AgentMessage(
                sender_id=self.metadata.agent_id,
                recipient_id=message.sender_id,
                message_type="knowledge_response",
                content={
                    "patterns_count": len(self.knowledge_base["patterns"]),
                    "strategies": list(self.knowledge_base["strategies"].keys()),
                    "learning_rate": self.learning_rate
                }
            )

        return await super().handle_message(message)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        # Get base health status
        health = {
            'status': self.status.name,
            'agent_id': self.metadata.agent_id,
            'uptime': (datetime.now() - self.metadata.created_at).total_seconds(),
            'active_tasks': len(self.context.active_tasks) if self.context else 0,
            'capabilities': [c.value if isinstance(c, AgentCapability) else c for c in self.metadata.capabilities],
            'healthy': self.status not in [AgentStatus.ERROR, AgentStatus.TERMINATED]
        }

        # Add learning-specific metrics
        health.update({
            "knowledge_base_size": len(self.knowledge_base["patterns"]),
            "adaptation_count": self.adaptation_count,
            "learning_rate": self.learning_rate,
            "strategies_count": len(self.knowledge_base["strategies"])
        })

        return health

    async def shutdown(self) -> None:
        """Shutdown the learning agent"""
        self._logger.info("Shutting down learning agent")
        self.update_status(AgentStatus.TERMINATED)


class CoordinatorAgent(AgentInterface):
    """
    Example coordinator agent that manages collaboration between other agents.
    """

    def __init__(self):
        metadata = AgentMetadata(
            name="CoordinatorAgent",
            version="1.0.0",
            description="Coordinates tasks and collaboration between agents",
            capabilities={
                AgentCapability.TASK_PROCESSING,
                AgentCapability.INTER_AGENT_COMM,
                AgentCapability.RESOURCE_MANAGEMENT,
                AgentCapability.BROADCAST
            },
            tags=["coordinator", "orchestration", "collaboration"]
        )
        super().__init__(metadata)

        # Coordination state
        self.agent_registry = {}
        self.collaboration_sessions = {}
        self.task_distribution = {}

    async def initialize(self, context: AgentContext) -> bool:
        """Initialize the coordinator agent"""
        try:
            await self.execute_lifecycle_hooks('pre_init')

            self.context = context
            self._logger.info(f"Initializing {self.metadata.name}")

            # Register for discovery messages
            self.register_message_handler("agent_announce", self._handle_agent_announce)
            self.register_message_handler("collaboration_request", self._handle_collaboration_request)

            self.update_status(AgentStatus.READY)

            # Announce presence
            await self.broadcast({
                "type": "coordinator_ready",
                "capabilities": [c.value for c in self.metadata.capabilities]
            }, message_type="announcement")

            await self.execute_lifecycle_hooks('post_init')

            return True

        except Exception as e:
            await self.handle_error(e, "initialization")
            return False

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordination tasks"""
        try:
            await self.execute_lifecycle_hooks('pre_process')
            self.update_status(AgentStatus.PROCESSING)

            task_type = task.get('task_type', 'unknown')

            result = None

            if task_type == "coordinate_workflow":
                result = await self._coordinate_workflow(task.get('parameters', {}))
            elif task_type == "distribute_tasks":
                result = await self._distribute_tasks(task.get('parameters', {}))
            elif task_type == "manage_resources":
                result = await self._manage_resources(task.get('parameters', {}))
            else:
                result = {"error": f"Unknown task type: {task_type}"}

            self.update_status(AgentStatus.ACTIVE)
            await self.execute_lifecycle_hooks('post_process')

            return {
                'status': 'completed',
                'task_id': task.get('task_id', 'unknown'),
                'result': result,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            await self.handle_error(e, "task processing")
            return {
                'status': 'error',
                'error': str(e),
                'task_id': task.get('task_id', 'unknown')
            }

    async def _coordinate_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate a workflow between multiple agents"""
        workflow_steps = parameters.get('steps', [])

        # Create collaboration session
        session_id = f"session_{len(self.collaboration_sessions)}"
        self.collaboration_sessions[session_id] = {
            "created_at": datetime.now(),
            "steps": workflow_steps,
            "status": "active",
            "participants": []
        }

        # Simulate coordination
        await asyncio.sleep(random.uniform(0.2, 0.5))

        return {
            "session_id": session_id,
            "workflow_initiated": True,
            "steps_count": len(workflow_steps),
            "estimated_completion": random.uniform(5, 20)
        }

    async def _distribute_tasks(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute tasks among available agents"""
        tasks = parameters.get('tasks', [])

        # Simulate task distribution
        distribution = {}
        for i, task in enumerate(tasks):
            # Mock assignment
            agent_id = f"agent_{i % 3}"
            if agent_id not in distribution:
                distribution[agent_id] = []
            distribution[agent_id].append(task)

        self.task_distribution[datetime.now().isoformat()] = distribution

        return {
            "distributed": True,
            "task_count": len(tasks),
            "agents_assigned": len(distribution),
            "distribution": distribution
        }

    async def _manage_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Manage resource allocation"""
        resource_type = parameters.get('resource_type', 'compute')
        allocation_strategy = parameters.get('strategy', 'balanced')

        # Simulate resource management
        await asyncio.sleep(random.uniform(0.1, 0.3))

        return {
            "resource_type": resource_type,
            "strategy": allocation_strategy,
            "allocation": {
                "agent_1": random.uniform(0.2, 0.4),
                "agent_2": random.uniform(0.2, 0.4),
                "agent_3": random.uniform(0.2, 0.4)
            },
            "utilization": random.uniform(0.6, 0.9)
        }

    async def _handle_agent_announce(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle agent announcements"""
        agent_info = message.content
        agent_id = message.sender_id

        # Register agent
        self.agent_registry[agent_id] = {
            "info": agent_info,
            "registered_at": datetime.now(),
            "last_seen": datetime.now()
        }

        self._logger.info(f"Registered agent: {agent_id}")

        return AgentMessage(
            sender_id=self.metadata.agent_id,
            recipient_id=message.sender_id,
            message_type="registration_confirmed",
            content={"registered": True, "coordinator_id": self.metadata.agent_id}
        )

    async def _handle_collaboration_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle collaboration requests"""
        request = message.content

        # Check available agents
        available_agents = [
            agent_id for agent_id, info in self.agent_registry.items()
            if (datetime.now() - info["last_seen"]).seconds < 60
        ]

        return AgentMessage(
            sender_id=self.metadata.agent_id,
            recipient_id=message.sender_id,
            message_type="collaboration_response",
            content={
                "request_accepted": True,
                "available_agents": available_agents,
                "coordinator_support": True
            }
        )

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming messages"""
        # Update last seen for sender
        if message.sender_id in self.agent_registry:
            self.agent_registry[message.sender_id]["last_seen"] = datetime.now()

        return await super().handle_message(message)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        # Get base health status
        health = {
            'status': self.status.name,
            'agent_id': self.metadata.agent_id,
            'uptime': (datetime.now() - self.metadata.created_at).total_seconds(),
            'active_tasks': len(self.context.active_tasks) if self.context else 0,
            'capabilities': [c.value if isinstance(c, AgentCapability) else c for c in self.metadata.capabilities],
            'healthy': self.status not in [AgentStatus.ERROR, AgentStatus.TERMINATED]
        }

        # Add coordinator-specific metrics
        health.update({
            "registered_agents": len(self.agent_registry),
            "active_sessions": len([s for s in self.collaboration_sessions.values()
                                   if s["status"] == "active"]),
            "task_distributions": len(self.task_distribution)
        })

        return health

    async def shutdown(self) -> None:
        """Shutdown the coordinator agent"""
        self._logger.info("Shutting down coordinator agent")
        self.update_status(AgentStatus.TERMINATED)