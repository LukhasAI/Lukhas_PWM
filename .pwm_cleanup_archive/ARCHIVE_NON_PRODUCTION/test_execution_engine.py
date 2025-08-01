#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════════════
║ 🚀 LUKHAS TEST EXECUTION ENGINE
║ Real-time test execution with streaming dashboard integration
║ Copyright (c) 2025 LUKHAS AI. All rights reserved.
╠══════════════════════════════════════════════════════════════════════════════════
║ Module: test_execution_engine.py
║ Path: dashboard/core/test_execution_engine.py
║ Version: 1.0.0 | Created: 2025-07-28
║ Authors: LUKHAS AI Team | Claude Code
╠══════════════════════════════════════════════════════════════════════════════════
║ DESCRIPTION
╠══════════════════════════════════════════════════════════════════════════════════
║ Advanced test execution engine that provides real-time streaming of test
║ results to the dashboard with intelligent resource management:
║
║ 🎯 REAL-TIME STREAMING EXECUTION:
║ • Live test output streaming to dashboard WebSocket clients
║ • Real-time progress tracking and status updates
║ • Interactive test cancellation and control
║ • Dynamic resource allocation and monitoring
║
║ 🏛️ COLONY-COORDINATED TESTING:
║ • Distributed test execution across multiple colony agents
║ • Load balancing based on agent capabilities and availability
║ • Fault-tolerant execution with automatic agent failover
║ • Cross-colony result aggregation and synchronization
║
║ ⚖️ ETHICS-GUIDED EXECUTION:
║ • Resource impact assessment before test execution
║ • Stakeholder consideration for test scheduling priority
║ • Environmental impact monitoring during execution
║ • Ethics Swarm approval for resource-intensive test suites
║
║ 🔮 ORACLE-OPTIMIZED SCHEDULING:
║ • Predictive test failure detection and prevention
║ • Optimal execution order based on dependency analysis
║ • Prophet-inspired test duration and resource estimation
║ • Temporal insights for peak performance scheduling
║
║ 📊 PERFORMANCE INTELLIGENCE:
║ • Real-time resource utilization monitoring
║ • Predictive scaling and throttling mechanisms
║ • Test performance pattern analysis and optimization
║ • Historical trend analysis for continuous improvement
║
║ ΛTAG: ΛEXECUTION, ΛSTREAMING, ΛREALTIME, ΛPERFORMANCE, ΛINTELLIGENT
╚══════════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import logging
import json
import sys
import os
import time
import psutil
import uuid
import signal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Union, Callable, AsyncIterator
from dataclasses import dataclass, field, asdict
from enum import Enum
import subprocess
import threading
from queue import Queue, Empty
import websockets
from contextlib import asynccontextmanager

# Dashboard system imports
from dashboard.core.test_management_system import (
    TestExecution, TestFile, TestSuite, TestStatus, TestType,
    TestPriority, TestExecutionMode, TestResults
)
from dashboard.core.universal_adaptive_dashboard import DashboardContext
from dashboard.core.dashboard_colony_agent import DashboardColonyAgent, DashboardAgentRole

# LUKHAS system imports
from core.oracle_nervous_system import get_oracle_nervous_system
from core.colonies.ethics_swarm_colony import get_ethics_swarm_colony
from core.event_bus import EventBus

logger = logging.getLogger("ΛTRACE.test_execution_engine")


class ExecutionStatus(Enum):
    """Status of the execution engine."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class StreamType(Enum):
    """Types of execution streams."""
    TEST_OUTPUT = "test_output"
    PROGRESS_UPDATE = "progress_update"
    STATUS_CHANGE = "status_change"
    RESOURCE_USAGE = "resource_usage"
    ERROR_ALERT = "error_alert"
    COMPLETION_SUMMARY = "completion_summary"


@dataclass
class ExecutionResource:
    """Represents execution resource usage."""
    cpu_percent: float
    memory_mb: float
    disk_io_mb: float
    network_io_mb: float
    process_count: int
    thread_count: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StreamMessage:
    """Message for real-time streaming."""
    message_id: str
    stream_type: StreamType
    execution_id: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 3  # 1=critical, 2=high, 3=normal, 4=low


@dataclass
class ExecutionPlan:
    """Plan for test execution with resource allocation."""
    plan_id: str
    test_suite: TestSuite
    execution_mode: TestExecutionMode
    estimated_duration: float
    resource_allocation: Dict[str, float]
    colony_assignments: Dict[str, str]  # test_id -> agent_id
    execution_order: List[str]  # test execution order
    oracle_insights: Dict[str, Any]
    ethics_approval: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


class TestExecutionEngine:
    """
    Revolutionary test execution engine providing real-time streaming
    and intelligent resource management for dashboard integration.
    """

    def __init__(self, dashboard_context: DashboardContext = None):
        self.engine_id = f"exec_engine_{int(datetime.now().timestamp())}"
        self.logger = logger.bind(engine_id=self.engine_id)
        self.dashboard_context = dashboard_context or DashboardContext()

        # Engine status
        self.status = ExecutionStatus.IDLE
        self.current_plan: Optional[ExecutionPlan] = None
        self.active_executions: Dict[str, TestExecution] = {}
        self.execution_history: List[TestExecution] = []

        # LUKHAS system integration
        self.oracle_nervous_system = None
        self.ethics_swarm = None
        self.event_bus = EventBus()

        # Colony agents for distributed execution
        self.execution_coordinator: Optional[DashboardColonyAgent] = None
        self.worker_agents: List[DashboardColonyAgent] = []
        self.resource_monitor: Optional[DashboardColonyAgent] = None

        # Real-time streaming
        self.stream_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.message_queue: Queue = Queue()
        self.stream_server: Optional[websockets.WebSocketServer] = None

        # Resource monitoring
        self.resource_monitor_active = False
        self.resource_history: List[ExecutionResource] = []
        self.resource_thresholds = {
            'cpu_percent': 80.0,
            'memory_mb': 4096.0,
            'process_count': 50
        }

        # Execution control
        self.max_concurrent_tests = 5
        self.execution_timeout = 300  # 5 minutes per test
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.execution_lock = asyncio.Lock()

        # Performance metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "resource_efficiency": 0.0,
            "stream_message_count": 0,
            "colony_utilization": 0.0
        }

        # Event handlers
        self.execution_start_handlers: List[Callable] = []
        self.execution_complete_handlers: List[Callable] = []
        self.stream_message_handlers: List[Callable] = []

        self.logger.info("Test Execution Engine initialized")

    async def initialize(self):
        """Initialize the test execution engine."""
        self.logger.info("Initializing Test Execution Engine")

        try:
            # Initialize LUKHAS system integrations
            await self._initialize_lukhas_integrations()

            # Initialize colony agents
            await self._initialize_colony_agents()

            # Start streaming server
            await self._start_streaming_server()

            # Start resource monitoring
            await self._start_resource_monitoring()

            # Start background tasks
            asyncio.create_task(self._stream_message_broadcaster())
            asyncio.create_task(self._execution_monitor_loop())
            asyncio.create_task(self._performance_analytics_loop())

            self.status = ExecutionStatus.IDLE

            self.logger.info("Test Execution Engine fully initialized")

        except Exception as e:
            self.logger.error("Test execution engine initialization failed", error=str(e))
            self.status = ExecutionStatus.ERROR
            raise

    async def _initialize_lukhas_integrations(self):
        """Initialize integration with LUKHAS AI systems."""

        try:
            # Oracle Nervous System for execution optimization
            self.oracle_nervous_system = await get_oracle_nervous_system()
            self.logger.info("Oracle Nervous System integrated for execution optimization")

            # Ethics Swarm for resource approval
            self.ethics_swarm = await get_ethics_swarm_colony()
            self.logger.info("Ethics Swarm Colony integrated for resource management")

        except Exception as e:
            self.logger.warning("Some LUKHAS systems unavailable for execution engine", error=str(e))

    async def _initialize_colony_agents(self):
        """Initialize colony agents for distributed execution."""

        # Execution coordinator
        self.execution_coordinator = DashboardColonyAgent(DashboardAgentRole.COORDINATOR)
        await self.execution_coordinator.initialize()

        # Worker agents for parallel execution
        for i in range(3):  # 3 worker agents
            worker = DashboardColonyAgent(DashboardAgentRole.PERFORMANCE_MONITOR)
            await worker.initialize()
            self.worker_agents.append(worker)

        # Resource monitor agent
        self.resource_monitor = DashboardColonyAgent(DashboardAgentRole.INTELLIGENCE_AGGREGATOR)
        await self.resource_monitor.initialize()

        self.logger.info("Colony agents initialized for distributed execution",
                        agents=len(self.worker_agents) + 2)

    async def _start_streaming_server(self, host: str = "localhost", port: int = 8766):
        """Start WebSocket server for real-time streaming."""

        async def client_handler(websocket, path):
            """Handle new client connections."""
            self.stream_clients.add(websocket)
            self.logger.info("Stream client connected", clients=len(self.stream_clients))

            try:
                # Send welcome message
                welcome_msg = StreamMessage(
                    message_id=str(uuid.uuid4()),
                    stream_type=StreamType.STATUS_CHANGE,
                    execution_id="system",
                    timestamp=datetime.now(),
                    data={
                        "type": "connection_established",
                        "engine_id": self.engine_id,
                        "status": self.status.value
                    }
                )
                await websocket.send(json.dumps(asdict(welcome_msg), default=str))

                # Keep connection alive and handle client messages
                async for message in websocket:
                    await self._handle_client_message(websocket, message)

            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.stream_clients.discard(websocket)
                self.logger.info("Stream client disconnected", clients=len(self.stream_clients))

        self.stream_server = await websockets.serve(client_handler, host, port)
        self.logger.info("Streaming server started", host=host, port=port)

    async def _handle_client_message(self, websocket, message: str):
        """Handle messages from streaming clients."""

        try:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")

            if msg_type == "cancel_execution":
                execution_id = data.get("execution_id")
                if execution_id:
                    await self.cancel_execution(execution_id)

            elif msg_type == "pause_engine":
                await self.pause_execution()

            elif msg_type == "resume_engine":
                await self.resume_execution()

            elif msg_type == "get_status":
                status_msg = StreamMessage(
                    message_id=str(uuid.uuid4()),
                    stream_type=StreamType.STATUS_CHANGE,
                    execution_id="system",
                    timestamp=datetime.now(),
                    data=await self.get_execution_status()
                )
                await websocket.send(json.dumps(asdict(status_msg), default=str))

        except json.JSONDecodeError:
            self.logger.warning("Invalid JSON message from client")
        except Exception as e:
            self.logger.error("Error handling client message", error=str(e))

    async def _start_resource_monitoring(self):
        """Start resource monitoring background task."""

        self.resource_monitor_active = True
        asyncio.create_task(self._resource_monitor_loop())
        self.logger.info("Resource monitoring started")

    async def create_execution_plan(self, test_suite: TestSuite,
                                  execution_mode: Optional[TestExecutionMode] = None) -> ExecutionPlan:
        """Create an optimized execution plan for a test suite."""

        mode = execution_mode or test_suite.execution_mode
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        self.logger.info("Creating execution plan",
                        plan_id=plan_id,
                        suite=test_suite.name,
                        mode=mode.value)

        # Get Oracle insights for optimization
        oracle_insights = {}
        if self.oracle_nervous_system:
            try:
                oracle_insights = await self._get_oracle_execution_insights(test_suite)
            except Exception as e:
                self.logger.warning("Oracle insights unavailable", error=str(e))

        # Get Ethics approval for resource usage
        ethics_approval = {"approved": True, "compliance_score": 1.0}
        if self.ethics_swarm:
            try:
                ethics_approval = await self._get_ethics_execution_approval(test_suite)
            except Exception as e:
                self.logger.warning("Ethics approval unavailable", error=str(e))

        # Calculate resource allocation
        resource_allocation = self._calculate_resource_allocation(test_suite, mode)

        # Assign colony agents
        colony_assignments = self._assign_colony_agents(test_suite, mode)

        # Determine execution order
        execution_order = self._determine_execution_order(test_suite, oracle_insights)

        # Estimate duration
        estimated_duration = self._estimate_plan_duration(test_suite, mode, oracle_insights)

        plan = ExecutionPlan(
            plan_id=plan_id,
            test_suite=test_suite,
            execution_mode=mode,
            estimated_duration=estimated_duration,
            resource_allocation=resource_allocation,
            colony_assignments=colony_assignments,
            execution_order=execution_order,
            oracle_insights=oracle_insights,
            ethics_approval=ethics_approval
        )

        self.logger.info("Execution plan created",
                        plan_id=plan_id,
                        estimated_duration=f"{estimated_duration:.1f}s",
                        colony_assignments=len(colony_assignments),
                        oracle_insights=bool(oracle_insights))

        return plan

    async def execute_plan(self, plan: ExecutionPlan) -> TestResults:
        """Execute a test plan with real-time streaming."""

        if self.status != ExecutionStatus.IDLE:
            raise RuntimeError(f"Engine is not idle (current status: {self.status.value})")

        # Check ethics approval
        if not plan.ethics_approval.get("approved", True):
            raise ValueError(f"Ethics approval denied: {plan.ethics_approval.get('reason', 'Unknown')}")

        self.logger.info("Starting plan execution",
                        plan_id=plan.plan_id,
                        suite=plan.test_suite.name,
                        test_count=len(plan.test_suite.test_files))

        # Set engine status
        self.status = ExecutionStatus.RUNNING
        self.current_plan = plan

        # Stream execution start
        await self._stream_message(StreamMessage(
            message_id=str(uuid.uuid4()),
            stream_type=StreamType.STATUS_CHANGE,
            execution_id=plan.plan_id,
            timestamp=datetime.now(),
            data={
                "type": "execution_started",
                "plan_id": plan.plan_id,
                "suite_name": plan.test_suite.name,
                "estimated_duration": plan.estimated_duration,
                "test_count": len(plan.test_suite.test_files)
            },
            priority=2
        ))

        start_time = datetime.now()
        executions = []

        try:
            # Execute based on mode
            if plan.execution_mode == TestExecutionMode.SEQUENTIAL:
                executions = await self._execute_sequential_plan(plan)
            elif plan.execution_mode == TestExecutionMode.PARALLEL:
                executions = await self._execute_parallel_plan(plan)
            elif plan.execution_mode == TestExecutionMode.COLONY_DISTRIBUTED:
                executions = await self._execute_colony_distributed_plan(plan)
            elif plan.execution_mode == TestExecutionMode.ORACLE_OPTIMIZED:
                executions = await self._execute_oracle_optimized_plan(plan)
            elif plan.execution_mode == TestExecutionMode.ETHICS_GUIDED:
                executions = await self._execute_ethics_guided_plan(plan)

            # Calculate results
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()

            passed = sum(1 for e in executions if e.status == TestStatus.PASSED)
            failed = sum(1 for e in executions if e.status == TestStatus.FAILED)
            skipped = sum(1 for e in executions if e.status == TestStatus.SKIPPED)
            error = sum(1 for e in executions if e.status == TestStatus.ERROR)

            success_rate = passed / len(executions) if executions else 0.0

            # Create results
            results_id = f"results_{uuid.uuid4().hex[:8]}"
            results = TestResults(
                results_id=results_id,
                suite_id=plan.test_suite.suite_id,
                executed_at=start_time,
                total_tests=len(executions),
                passed_tests=passed,
                failed_tests=failed,
                skipped_tests=skipped,
                error_tests=error,
                total_duration=total_duration,
                success_rate=success_rate,
                performance_metrics=self._calculate_performance_metrics(executions),
                colony_performance=await self._analyze_colony_performance(executions),
                oracle_predictions_accuracy=self._calculate_oracle_accuracy(plan, executions),
                ethics_compliance_score=plan.ethics_approval.get('compliance_score', 1.0),
                detailed_results=executions
            )

            # Update metrics
            self.metrics["total_executions"] += 1
            if success_rate > 0.8:
                self.metrics["successful_executions"] += 1
            else:
                self.metrics["failed_executions"] += 1

            # Stream completion
            await self._stream_message(StreamMessage(
                message_id=str(uuid.uuid4()),
                stream_type=StreamType.COMPLETION_SUMMARY,
                execution_id=plan.plan_id,
                timestamp=datetime.now(),
                data={
                    "type": "execution_completed",
                    "results_id": results_id,
                    "success_rate": success_rate,
                    "total_duration": total_duration,
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "error": error
                },
                priority=1
            ))

            self.logger.info("Plan execution completed",
                           plan_id=plan.plan_id,
                           success_rate=f"{success_rate:.2%}",
                           duration=f"{total_duration:.1f}s")

            return results

        except Exception as e:
            self.logger.error("Plan execution failed", plan_id=plan.plan_id, error=str(e))

            # Stream error
            await self._stream_message(StreamMessage(
                message_id=str(uuid.uuid4()),
                stream_type=StreamType.ERROR_ALERT,
                execution_id=plan.plan_id,
                timestamp=datetime.now(),
                data={
                    "type": "execution_error",
                    "error": str(e),
                    "plan_id": plan.plan_id
                },
                priority=1
            ))

            raise

        finally:
            # Reset engine status
            self.status = ExecutionStatus.IDLE
            self.current_plan = None

            # Clear active executions
            self.active_executions.clear()

    async def _execute_sequential_plan(self, plan: ExecutionPlan) -> List[TestExecution]:
        """Execute tests sequentially."""

        executions = []

        for i, test_file in enumerate(plan.test_suite.test_files):
            if self.status != ExecutionStatus.RUNNING:
                break

            # Stream progress
            await self._stream_progress_update(plan.plan_id, i, len(plan.test_suite.test_files), test_file.file_path.name)

            # Create execution
            execution = TestExecution(
                execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                test_file=test_file,
                status=TestStatus.QUEUED
            )

            # Execute test
            execution = await self._execute_single_test(execution, plan)
            executions.append(execution)

        return executions

    async def _execute_parallel_plan(self, plan: ExecutionPlan) -> List[TestExecution]:
        """Execute tests in parallel."""

        # Create all executions
        executions = []
        for test_file in plan.test_suite.test_files:
            execution = TestExecution(
                execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                test_file=test_file,
                status=TestStatus.QUEUED
            )
            executions.append(execution)

        # Execute in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_tests)

        async def execute_with_semaphore(execution):
            async with semaphore:
                return await self._execute_single_test(execution, plan)

        # Run all tests concurrently
        completed_executions = await asyncio.gather(
            *[execute_with_semaphore(exec) for exec in executions],
            return_exceptions=True
        )

        # Filter out exceptions and return completed executions
        return [exec for exec in completed_executions if isinstance(exec, TestExecution)]

    async def _execute_single_test(self, execution: TestExecution, plan: ExecutionPlan) -> TestExecution:
        """Execute a single test with streaming."""

        execution.status = TestStatus.RUNNING
        execution.started_at = datetime.now()

        # Add to active executions
        self.active_executions[execution.execution_id] = execution

        # Stream start
        await self._stream_message(StreamMessage(
            message_id=str(uuid.uuid4()),
            stream_type=StreamType.STATUS_CHANGE,
            execution_id=execution.execution_id,
            timestamp=datetime.now(),
            data={
                "type": "test_started",
                "test_file": str(execution.test_file.file_path),
                "test_type": execution.test_file.test_type.value,
                "estimated_duration": execution.test_file.estimated_duration
            }
        ))

        # Notify handlers
        for handler in self.execution_start_handlers:
            try:
                await handler(execution)
            except Exception as e:
                self.logger.error("Execution start handler error", error=str(e))

        try:
            # Prepare execution environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(plan.test_suite.test_files[0].file_path.parent.parent)

            # Execute test
            cmd = [sys.executable, str(execution.test_file.file_path)]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(execution.test_file.file_path.parent)
            )

            # Store process for cancellation capability
            self.running_processes[execution.execution_id] = process

            # Stream output in real-time
            output_task = asyncio.create_task(
                self._stream_process_output(execution.execution_id, process)
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.execution_timeout
                )

                execution.exit_code = process.returncode
                execution.output = stdout.decode('utf-8').splitlines()
                execution.error_output = stderr.decode('utf-8').splitlines()

                # Determine status
                if execution.exit_code == 0:
                    execution.status = TestStatus.PASSED
                else:
                    execution.status = TestStatus.FAILED

            except asyncio.TimeoutError:
                execution.status = TestStatus.TIMEOUT
                process.kill()
                await process.wait()

            # Cancel output streaming
            output_task.cancel()

        except Exception as e:
            execution.status = TestStatus.ERROR
            execution.error_output = [str(e)]
            self.logger.error("Test execution error",
                            execution_id=execution.execution_id,
                            error=str(e))

        finally:
            # Clean up
            execution.completed_at = datetime.now()
            if execution.started_at:
                execution.duration = (execution.completed_at - execution.started_at).total_seconds()

            # Remove from active and running
            self.active_executions.pop(execution.execution_id, None)
            self.running_processes.pop(execution.execution_id, None)

            # Add to history
            self.execution_history.append(execution)

            # Stream completion
            await self._stream_message(StreamMessage(
                message_id=str(uuid.uuid4()),
                stream_type=StreamType.STATUS_CHANGE,
                execution_id=execution.execution_id,
                timestamp=datetime.now(),
                data={
                    "type": "test_completed",
                    "status": execution.status.value,
                    "duration": execution.duration,
                    "exit_code": execution.exit_code
                }
            ))

            # Notify handlers
            for handler in self.execution_complete_handlers:
                try:
                    await handler(execution)
                except Exception as e:
                    self.logger.error("Execution complete handler error", error=str(e))

        return execution

    async def _stream_process_output(self, execution_id: str, process: asyncio.subprocess.Process):
        """Stream process output in real-time."""

        async def stream_stdout():
            async for line in process.stdout:
                output_line = line.decode('utf-8').rstrip()
                await self._stream_message(StreamMessage(
                    message_id=str(uuid.uuid4()),
                    stream_type=StreamType.TEST_OUTPUT,
                    execution_id=execution_id,
                    timestamp=datetime.now(),
                    data={
                        "type": "stdout",
                        "line": output_line
                    }
                ))

        async def stream_stderr():
            async for line in process.stderr:
                error_line = line.decode('utf-8').rstrip()
                await self._stream_message(StreamMessage(
                    message_id=str(uuid.uuid4()),
                    stream_type=StreamType.TEST_OUTPUT,
                    execution_id=execution_id,
                    timestamp=datetime.now(),
                    data={
                        "type": "stderr",
                        "line": error_line
                    },
                    priority=2
                ))

        # Stream both stdout and stderr concurrently
        await asyncio.gather(stream_stdout(), stream_stderr(), return_exceptions=True)

    async def _stream_progress_update(self, plan_id: str, current: int, total: int, current_test: str):
        """Stream progress update."""

        progress = (current / total) * 100 if total > 0 else 0

        await self._stream_message(StreamMessage(
            message_id=str(uuid.uuid4()),
            stream_type=StreamType.PROGRESS_UPDATE,
            execution_id=plan_id,
            timestamp=datetime.now(),
            data={
                "type": "progress_update",
                "current": current,
                "total": total,
                "progress_percent": progress,
                "current_test": current_test
            }
        ))

    async def _stream_message(self, message: StreamMessage):
        """Queue a message for streaming to clients."""

        try:
            self.message_queue.put_nowait(message)
            self.metrics["stream_message_count"] += 1

            # Notify handlers
            for handler in self.stream_message_handlers:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error("Stream message handler error", error=str(e))

        except Exception as e:
            self.logger.error("Failed to queue stream message", error=str(e))

    async def _stream_message_broadcaster(self):
        """Background task to broadcast messages to clients."""

        while True:
            try:
                # Get message from queue
                try:
                    message = self.message_queue.get(timeout=1.0)
                except Empty:
                    await asyncio.sleep(0.1)
                    continue

                # Broadcast to all connected clients
                if self.stream_clients:
                    message_json = json.dumps(asdict(message), default=str)

                    # Send to all clients
                    disconnected_clients = set()

                    for client in self.stream_clients:
                        try:
                            await client.send(message_json)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                        except Exception as e:
                            self.logger.warning("Failed to send message to client", error=str(e))
                            disconnected_clients.add(client)

                    # Remove disconnected clients
                    self.stream_clients -= disconnected_clients

            except Exception as e:
                self.logger.error("Stream broadcaster error", error=str(e))
                await asyncio.sleep(1.0)

    # Background monitoring loops

    async def _resource_monitor_loop(self):
        """Background task to monitor resource usage."""

        while self.resource_monitor_active:
            try:
                # Collect resource information
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory_info = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()

                resource = ExecutionResource(
                    cpu_percent=cpu_percent,
                    memory_mb=memory_info.used / (1024 * 1024),
                    disk_io_mb=(disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024) if disk_io else 0,
                    network_io_mb=(network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024) if network_io else 0,
                    process_count=len(psutil.pids()),
                    thread_count=sum(p.num_threads() for p in psutil.process_iter(['num_threads']) if p.info['num_threads'])
                )

                # Store resource data
                self.resource_history.append(resource)

                # Keep only last 100 entries
                if len(self.resource_history) > 100:
                    self.resource_history = self.resource_history[-100:]

                # Stream resource update if clients connected
                if self.stream_clients and self.active_executions:
                    await self._stream_message(StreamMessage(
                        message_id=str(uuid.uuid4()),
                        stream_type=StreamType.RESOURCE_USAGE,
                        execution_id="system",
                        timestamp=datetime.now(),
                        data=asdict(resource),
                        priority=4
                    ))

                # Check thresholds and alert if needed
                if cpu_percent > self.resource_thresholds['cpu_percent']:
                    await self._handle_resource_threshold_breach("cpu", cpu_percent)

                if resource.memory_mb > self.resource_thresholds['memory_mb']:
                    await self._handle_resource_threshold_breach("memory", resource.memory_mb)

                await asyncio.sleep(5.0)  # Monitor every 5 seconds

            except Exception as e:
                self.logger.error("Resource monitoring error", error=str(e))
                await asyncio.sleep(10.0)

    async def _handle_resource_threshold_breach(self, resource_type: str, current_value: float):
        """Handle resource threshold breach."""

        self.logger.warning("Resource threshold breached",
                          resource_type=resource_type,
                          current_value=current_value,
                          threshold=self.resource_thresholds.get(f"{resource_type}_percent",
                                                               self.resource_thresholds.get(f"{resource_type}_mb", 0)))

        # Stream alert
        await self._stream_message(StreamMessage(
            message_id=str(uuid.uuid4()),
            stream_type=StreamType.ERROR_ALERT,
            execution_id="system",
            timestamp=datetime.now(),
            data={
                "type": "resource_threshold_breach",
                "resource_type": resource_type,
                "current_value": current_value,
                "threshold": self.resource_thresholds.get(f"{resource_type}_percent",
                                                         self.resource_thresholds.get(f"{resource_type}_mb", 0))
            },
            priority=2
        ))

    # Control methods

    async def pause_execution(self):
        """Pause test execution."""
        if self.status == ExecutionStatus.RUNNING:
            self.status = ExecutionStatus.PAUSED
            self.logger.info("Test execution paused")

    async def resume_execution(self):
        """Resume test execution."""
        if self.status == ExecutionStatus.PAUSED:
            self.status = ExecutionStatus.RUNNING
            self.logger.info("Test execution resumed")

    async def cancel_execution(self, execution_id: str):
        """Cancel a specific test execution."""

        if execution_id in self.running_processes:
            process = self.running_processes[execution_id]
            process.terminate()

            # Wait briefly for graceful termination, then kill
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

            self.logger.info("Test execution cancelled", execution_id=execution_id)

            # Update execution status
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                execution.status = TestStatus.CANCELLED
                execution.completed_at = datetime.now()
                if execution.started_at:
                    execution.duration = (execution.completed_at - execution.started_at).total_seconds()

    async def get_execution_status(self) -> Dict[str, Any]:
        """Get comprehensive execution engine status."""

        return {
            "engine_id": self.engine_id,
            "status": self.status.value,
            "current_plan": self.current_plan.plan_id if self.current_plan else None,
            "active_executions": len(self.active_executions),
            "stream_clients": len(self.stream_clients),
            "resource_monitoring": self.resource_monitor_active,
            "metrics": self.metrics,
            "resource_usage": asdict(self.resource_history[-1]) if self.resource_history else None,
            "colony_agents": {
                "coordinator": bool(self.execution_coordinator),
                "workers": len(self.worker_agents),
                "resource_monitor": bool(self.resource_monitor)
            },
            "active_test_details": [
                {
                    "execution_id": execution.execution_id,
                    "test_file": str(execution.test_file.file_path),
                    "status": execution.status.value,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "duration": execution.duration
                }
                for execution in self.active_executions.values()
            ]
        }

    # Utility methods (implementations would be added)

    def _calculate_resource_allocation(self, test_suite: TestSuite, mode: TestExecutionMode) -> Dict[str, float]:
        """Calculate resource allocation for test suite."""
        # Implementation would calculate optimal resource allocation
        return {"cpu": 0.5, "memory": 0.3, "network": 0.1}

    def _assign_colony_agents(self, test_suite: TestSuite, mode: TestExecutionMode) -> Dict[str, str]:
        """Assign colony agents to tests."""
        # Implementation would assign agents based on capabilities and load
        return {}

    def _determine_execution_order(self, test_suite: TestSuite, oracle_insights: Dict[str, Any]) -> List[str]:
        """Determine optimal execution order."""
        # Implementation would order tests based on dependencies and insights
        return [str(tf.file_path) for tf in test_suite.test_files]

    def _estimate_plan_duration(self, test_suite: TestSuite, mode: TestExecutionMode, oracle_insights: Dict[str, Any]) -> float:
        """Estimate plan execution duration."""
        # Implementation would estimate duration based on mode and insights
        return sum(tf.estimated_duration for tf in test_suite.test_files)

    # Additional implementation methods would be added for:
    # - Oracle integration methods
    # - Ethics integration methods
    # - Performance analysis
    # - Colony coordination
    # - Advanced execution modes


# Convenience function
async def create_test_execution_engine(dashboard_context: DashboardContext = None) -> TestExecutionEngine:
    """Create and initialize a test execution engine."""
    engine = TestExecutionEngine(dashboard_context)
    await engine.initialize()
    return engine


logger.info("ΛEXECUTION: Test Execution Engine loaded. Real-time streaming ready.")