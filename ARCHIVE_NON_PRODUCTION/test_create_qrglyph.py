

"""
┌────────────────────────────────────────────────────────────────────────────┐
│ 📦 TEST MODULE : test_create_qrglyph.py                                    │
│ 🧪 PURPOSE     : Test script for generating a demo Qrglyph                 │
│ 🧩 TYPE        : Lukhas-ID Utility Test  🔧 VERSION: v0.1.0                 │
│ 🖋️ AUTHOR      : Gonzalo Dominguez      📅 UPDATED: 2025-04-29             │
├────────────────────────────────────────────────────────────────────────────┤
│ 📚 DEPENDENCIES:                                                           │
│   - qrglymph_public.py (must be importable)                                │
│   - A small test payload file (auto-created)                               │
└────────────────────────────────────────────────────────────────────────────┘
"""

import os
from pathlib import Path
import importlib
create_qrglyph = importlib.import_module("lukhas.identity.backend.qrglyphs.qrglymph_public").create_qrglyph

# Configuration
TEST_OUTPUT_DIR = "./test_output"
TEST_PAYLOAD_FILE = os.path.join(TEST_OUTPUT_DIR, "test_payload.txt")

def setup_test_environment():
    Path(TEST_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(TEST_PAYLOAD_FILE, "w") as f:
        f.write("This is a test payload for Qrglyph creation.")

def test_generate_qrglyph():
    print("🔧 Setting up test payload...")
    setup_test_environment()

    print("🚀 Creating Qrglyph...")
    create_qrglyph(TEST_PAYLOAD_FILE, TEST_OUTPUT_DIR)

    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "payload.enc")), "❌ Encrypted file missing."
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "qrglyph_qr.png")), "❌ QR code file missing."
    print("✅ Test Qrglyph generated successfully!")

if __name__ == "__main__":
    test_generate_qrglyph()