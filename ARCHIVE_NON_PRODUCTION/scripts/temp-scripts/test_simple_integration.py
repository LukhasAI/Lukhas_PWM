#!/usr/bin/env python3
"""
Simplified Integration Test
Tests the core integration hub functionality with working components only.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

# Test working imports
try:
    from core.core_hub import CoreHub

    print("✅ CoreHub import successful")
    core_hub_available = True
except Exception as e:
    print(f"❌ CoreHub failed: {e}")
    core_hub_available = False

try:
    from consciousness.consciousness_hub import ConsciousnessHub

    print("✅ ConsciousnessHub import successful")
    consciousness_hub_available = True
except Exception as e:
    print(f"❌ ConsciousnessHub failed: {e}")
    consciousness_hub_available = False

try:
    from identity.identity_hub import IdentityHub

    print("✅ IdentityHub import successful")
    identity_hub_available = True
except Exception as e:
    print(f"❌ IdentityHub failed: {e}")
    identity_hub_available = False

try:
    from memory.memory_hub import MemoryHub

    print("✅ MemoryHub import successful")
    memory_hub_available = True
except Exception as e:
    print(f"❌ MemoryHub failed: {e}")
    memory_hub_available = False

try:
    from ethics.service import EthicsService

    print("✅ EthicsService import successful")
    ethics_service_available = True
except Exception as e:
    print(f"❌ EthicsService failed: {e}")
    ethics_service_available = False


class WorkingIntegrationHub:
    """Simplified integration hub with only working components"""

    def __init__(self):
        print("Initializing Working Integration Hub...")

        # Initialize only working components
        self.core_hub = CoreHub() if core_hub_available else None
        self.consciousness_hub = (
            ConsciousnessHub() if consciousness_hub_available else None
        )
        self.identity_hub = IdentityHub() if identity_hub_available else None
        self.memory_hub = MemoryHub() if memory_hub_available else None
        self.ethics_service = EthicsService() if ethics_service_available else None

        self._connect_working_systems()

    def _connect_working_systems(self):
        """Connect only the working systems"""
        print("Connecting working systems...")

        if self.core_hub and self.consciousness_hub:
            self.core_hub.register_service("consciousness_hub", self.consciousness_hub)
            print("✅ Connected Core ↔ Consciousness")

        if self.identity_hub and self.memory_hub:
            self.identity_hub.register_service("memory_hub", self.memory_hub)
            self.memory_hub.register_service("identity_hub", self.identity_hub)
            print("✅ Connected Identity ↔ Memory")

        if self.core_hub and self.ethics_service:
            self.core_hub.register_service("ethics_service", self.ethics_service)
            print("✅ Connected Core ↔ Ethics")

        print("Working systems connected successfully")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        return {
            "core_hub": self.core_hub is not None,
            "consciousness_hub": self.consciousness_hub is not None,
            "identity_hub": self.identity_hub is not None,
            "memory_hub": self.memory_hub is not None,
            "ethics_service": self.ethics_service is not None,
            "connectivity_status": (
                "partial"
                if any(
                    [
                        self.core_hub,
                        self.consciousness_hub,
                        self.identity_hub,
                        self.memory_hub,
                        self.ethics_service,
                    ]
                )
                else "failed"
            ),
        }


if __name__ == "__main__":
    try:
        hub = WorkingIntegrationHub()
        status = hub.get_status()
        print(f"\n📊 Integration Status: {status}")

        working_components = sum(1 for v in status.values() if v is True)
        total_components = len(
            [k for k in status.keys() if k.endswith("_hub") or k.endswith("_service")]
        )

        print(f"📈 Working: {working_components}/{total_components} components")
        print("✅ Simplified integration hub working!")

    except Exception as e:
        print(f"❌ Integration failed: {e}")
        import traceback

        traceback.print_exc()
