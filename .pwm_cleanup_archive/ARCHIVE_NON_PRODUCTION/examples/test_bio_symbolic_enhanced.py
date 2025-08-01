#!/usr/bin/env python3
"""
Test the enhanced bio_symbolic module
"""

import asyncio
import sys
sys.path.append('.')

from bio.symbolic.bio_symbolic import bio_symbolic, integrate_biological_state


async def test_bio_symbolic():
    """Test the enhanced bio-symbolic functionality."""
    print("Testing Enhanced Bio-Symbolic Module")
    print("=" * 50)

    # Test 1: Process rhythm data
    print("\n1. Testing Rhythm Processing:")
    rhythm_data = {
        'type': 'rhythm',
        'period': 24,  # Circadian
        'phase': 'day',
        'amplitude': 0.8
    }
    result = bio_symbolic.process(rhythm_data)
    print(f"   Rhythm -> {result['glyph']}: {result['meaning']}")

    # Test 2: Process energy data
    print("\n2. Testing Energy Processing:")
    energy_data = {
        'type': 'energy',
        'atp_level': 0.85,
        'efficiency': 0.9,
        'stress': 0.1
    }
    result = bio_symbolic.process(energy_data)
    print(f"   Energy -> {result['power_glyph']}: {result['interpretation']}")

    # Test 3: Full integration
    print("\n3. Testing Full Bio-Symbolic Integration:")
    bio_data = {
        'heart_rate': 72,
        'temperature': 36.8,
        'cortisol': 12,
        'energy_level': 0.8,
        'ph': 7.38,
        'glucose': 95
    }

    integrated = await integrate_biological_state(bio_data)
    print(f"   Primary Symbol: {integrated['primary_symbol']}")
    print(f"   All Symbols: {integrated['all_symbols']}")
    print(f"   Coherence: {integrated['coherence']:.2%}")
    print(f"   Quality: {integrated['integration_quality']}")

    # Test 4: Get statistics
    print("\n4. Bio-Symbolic Statistics:")
    stats = bio_symbolic.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_bio_symbolic())