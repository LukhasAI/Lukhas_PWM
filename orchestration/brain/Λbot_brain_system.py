#!/usr/bin/env python3
"""
ΛBot AGI System - Legacy Integration Bridge
==========================================
Bridge between legacy ΛBot system and new Lukhas AGI Orchestrator

This file maintains compatibility while migrating to the new
comprehensive AGI orchestration system.

Enhanced: 2025-07-02 with Lukhas AGI integration
"""

import asyncio
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ΛBotAGISystem")

# Import the new Lukhas AGI Orchestrator
try:
    from orchestration.brain.lukhas_agi_orchestrator import orchestration.brain.lukhas_agi_orchestrator, LukhasAGIConfig
    AGI_ORCHESTRATOR_AVAILABLE = True
    logger.info("✅ Lukhas AGI Orchestrator available")
except ImportError as e:
    AGI_ORCHESTRATOR_AVAILABLE = False
    logger.warning(f"Lukhas AGI Orchestrator not available: {e}")

class ΛBotAGISystem:
    """
    Legacy ΛBot AGI System - now bridges to Lukhas AGI Orchestrator
    
    This class maintains backward compatibility while leveraging
    the enhanced AGI capabilities of the new orchestrator.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.orchestrator = lukhas_agi_orchestrator if AGI_ORCHESTRATOR_AVAILABLE else None
        self.active = False
        
        logger.info("🤖 ΛBot AGI System initialized (bridging to Lukhas AGI)")
    
    async def initialize(self) -> bool:
        """Initialize the AGI system"""
        if self.orchestrator:
            return await self.orchestrator.initialize_agi_system()
        else:
            logger.warning("No AGI orchestrator available - running in legacy mode")
            self.active = True
            return True
    
    async def process_request(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a request through the AGI system"""
        if self.orchestrator:
            return await self.orchestrator.process_agi_request(user_input, context)
        else:
            # Legacy fallback processing
            return {
                'response': f"Legacy processing: {user_input}",
                'confidence': 0.5,
                'mode': 'legacy_fallback',
                'timestamp': '2025-07-02'
            }
    
    async def start(self):
        """Start the AGI system"""
        if not self.active:
            await self.initialize()
        
        if self.orchestrator:
            # Start the orchestrator in background
            asyncio.create_task(self.orchestrator.start_agi_orchestration())
        
        logger.info("🚀 ΛBot AGI System started")
    
    async def stop(self):
        """Stop the AGI system"""
        if self.orchestrator:
            await self.orchestrator.stop_agi_orchestration()
        
        self.active = False
        logger.info("🛑 ΛBot AGI System stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        if self.orchestrator:
            status = self.orchestrator.get_agi_status()
            status['legacy_bridge_active'] = True
            status['bridge_version'] = '1.0.0'
            return status
        else:
            return {
                'active': self.active,
                'mode': 'legacy_only',
                'orchestrator_available': False,
                'bridge_version': '1.0.0'
            }


# Global instance for backward compatibility
Λbot_agi_system = ΛBotAGISystem()


# Legacy function aliases for backward compatibility
async def initialize_agi_system():
    """Legacy initialization function"""
    return await Λbot_agi_system.initialize()

async def process_agi_request(user_input: str, context: Optional[Dict] = None):
    """Legacy request processing function"""
    return await Λbot_agi_system.process_request(user_input, context)

def get_agi_status():
    """Legacy status function"""
    return Λbot_agi_system.get_status()

async def main():
    """Main entry point for legacy compatibility"""
    print("🤖 ΛBot AGI System - Legacy Bridge")
    print("Bridging to Lukhas AGI Orchestrator...")
    print("=" * 50)
    
    try:
        await Λbot_agi_system.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ΛBot AGI System...")
        await Λbot_agi_system.stop()

if __name__ == "__main__":
    asyncio.run(main())