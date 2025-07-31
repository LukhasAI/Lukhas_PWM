#!/usr/bin/env python3
"""
🧠 LUKHAS Simple Agent Memory
=============================
Date: 2025-06-23

Simple, working shared memory for multi-agent collaboration.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

class AgentMemory:
    """Simple shared memory for agents"""

    def __init__(self, agent_id: str = "default"):
        """Initialize agent memory"""
        self.agent_id = agent_id
        self.memory_dir = Path("/Users/A_G_I/Lukhas/lukhas_shared_memory/data")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / f"{agent_id}_memory.json"
        self._memory = {}
        self._load_memory()

    def _load_memory(self):
        """Load memory from file"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    self._memory = json.load(f)
        except Exception:
            self._memory = {}

    def _save_memory(self):
        """Save memory to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self._memory, f, indent=2)
        except Exception:
            pass

    async def append_memory(self, key: str, data: Any):
        """Append data to memory"""
        if key not in self._memory:
            self._memory[key] = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self._memory[key].append(entry)
        self._save_memory()

    async def read_memory(self, key: str) -> List[Dict]:
        """Read memory entries for a key"""
        return self._memory.get(key, [])

    def read_all_memory(self) -> Dict:
        """Read all memory"""
        return self._memory.copy()

# For backward compatibility
def append_to_shared_memory(agent_id: str, event_type: str, data: Dict[str, Any]) -> bool:
    """Legacy function for backward compatibility"""
    try:
        memory = AgentMemory(agent_id)
        asyncio.create_task(memory.append_memory(event_type, data))
        return True
    except Exception:
        return False

def read_from_shared_memory(agent_id: str, event_type: str = None, limit: int = 100) -> List[Dict]:
    """Legacy function for backward compatibility"""
    try:
        memory = AgentMemory(agent_id)
        if event_type:
            return asyncio.run(memory.read_memory(event_type))[-limit:]
        else:
            all_memory = memory.read_all_memory()
            all_entries = []
            for entries in all_memory.values():
                all_entries.extend(entries)
            return sorted(all_entries, key=lambda x: x.get('timestamp', ''))[-limit:]
    except Exception:
        return []
