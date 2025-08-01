#!/usr/bin/env python3
"""
Simple test to verify agent coordination works
"""

import asyncio
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.actor_system import ActorSystem
from core.agent_coordination import (
    CoordinationHub, DataProcessorAgent, AnalyticsAgent,
    SkillLevel, MessagePriority
)

logging.basicConfig(level=logging.INFO)


async def test_basic_coordination():
    """Test basic coordination flow"""
    print("\n=== Testing Basic Agent Coordination ===")

    # Create actor system
    system = ActorSystem()
    await system.start()

    try:
        # Create coordination hub
        hub_ref = await system.create_actor(CoordinationHub, "hub")
        hub = system.get_actor("hub")
        print("✓ Created coordination hub")

        # Create agents
        data_agent_ref = await system.create_actor(DataProcessorAgent, "data_agent")
        analytics_ref = await system.create_actor(AnalyticsAgent, "analytics_agent")

        data_agent = system.get_actor("data_agent")
        analytics = system.get_actor("analytics_agent")

        # Set hub references
        data_agent.coord_hub = hub_ref
        analytics.coord_hub = hub_ref
        print("✓ Created specialized agents")

        # Register skills
        for skill in data_agent.skills:
            await hub.skill_registry.register_skill("data_agent", skill)

        for skill in analytics.skills:
            await hub.skill_registry.register_skill("analytics_agent", skill)
        print("✓ Registered agent skills")

        # Test skill lookup
        data_agents = await hub.skill_registry.find_agents_with_skill("data_cleaning")
        print(f"✓ Found {len(data_agents)} agents with data_cleaning skill")

        # Set actor system reference (needed for get_ref)
        data_agent.actor_system = system
        analytics.actor_system = system

        # Announce a task
        task_id = await data_agent.announce_task(
            description="Clean and analyze dataset",
            required_skills=[
                ("data_validation", SkillLevel.INTERMEDIATE),
                ("statistical_analysis", SkillLevel.INTERMEDIATE)
            ],
            priority=MessagePriority.HIGH
        )
        print(f"✓ Announced task: {task_id}")

        # Wait for coordination
        await asyncio.sleep(2)

        # Check results
        print(f"\nCoordination Results:")
        print(f"  Active announcements: {len(hub.active_announcements)}")
        print(f"  Working groups: {len(hub.working_groups)}")

        if hub.working_groups:
            for task_id, group in hub.working_groups.items():
                print(f"\n  Group {group.group_id[:8]}:")
                print(f"    Task: {group.task.description}")
                print(f"    Members: {list(group.members.keys())}")
                print(f"    Skills covered: {list(group.skills_covered.keys())}")

        print("\n✅ Basic coordination test passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await system.stop()


async def test_skill_metrics():
    """Test skill metric updates"""
    print("\n=== Testing Skill Metrics ===")

    from core.agent_coordination import Skill, SkillLevel

    skill = Skill("test_skill", SkillLevel.INTERMEDIATE)
    print(f"Initial: success_rate={skill.success_rate}, avg_time={skill.avg_completion_time}")

    # Simulate successful task
    skill.update_metrics(success=True, completion_time=10.0)
    print(f"After success: success_rate={skill.success_rate}, avg_time={skill.avg_completion_time}")

    # Simulate failed task
    skill.update_metrics(success=False, completion_time=5.0)
    print(f"After failure: success_rate={skill.success_rate}, avg_time={skill.avg_completion_time}")

    print("✅ Skill metrics test passed!")


async def main():
    """Run all tests"""
    print("Agent Coordination System - Simple Tests")
    print("=" * 50)

    await test_skill_metrics()
    await test_basic_coordination()

    print("\n✨ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())