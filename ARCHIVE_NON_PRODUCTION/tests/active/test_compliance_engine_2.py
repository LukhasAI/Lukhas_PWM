"""
CRITICAL FILE - DO NOT MODIFY WITHOUT APPROVAL
lukhas AI System - Core Governance Component
File: test_compliance_engine.py
Path: core/governance/test_compliance_engine.py
Created: 2025-06-20
Author: lukhas AI Team
Version: 1.0
This file is part of the lukhas (lukhas Universal Knowledge & Holistic AI System)
Advanced Cognitive Architecture for Artificial General Intelligence
Copyright (c) 2025 lukhas AI Research. All rights reserved.
Licensed under the lukhas Core License - see LICENSE.md for details.
TAGS: [CRITICAL, KeyFile, Governance]
DEPENDENCIES:
  - core/memory/memory_manager.py
  - core/identity/identity_manager.py
"""

import pytest
import time
from core.governance.compliance_engine import ComplianceEngine

def test_anonymize_metadata():
    compliance_engine = ComplianceEngine(gdpr_enabled=True)
    metadata = {
        "user_id": "test_user",
        "location": {"city": "Test City", "country": "Test Country"},
        "device_info": {"type": "mobile", "os": "iOS", "battery_level": 80}
    }

    anonymized = compliance_engine.anonymize_metadata(metadata)

    assert anonymized["user_id"] != "test_user"
    assert "city" not in anonymized["location"]
    assert anonymized["device_info"]["type"] == "anonymized"

def test_should_retain_data_within_retention_period():
    compliance_engine = ComplianceEngine(data_retention_days=30)
    timestamp = time.time() - (15 * 24 * 60 * 60)  # 15 days ago

    assert compliance_engine.should_retain_data(timestamp) is True

def test_should_not_retain_data_after_retention_period():
    compliance_engine = ComplianceEngine(data_retention_days=30)
    timestamp = time.time() - (31 * 24 * 60 * 60)  # 31 days ago

    assert compliance_engine.should_retain_data(timestamp) is False

def test_anonymize_metadata_without_gdpr():
    compliance_engine = ComplianceEngine(gdpr_enabled=False)
    metadata = {
        "user_id": "test_user",
        "location": {"city": "Test City", "country": "Test Country"},
        "device_info": {"type": "mobile", "os": "iOS", "battery_level": 80}
    }

    anonymized = compliance_engine.anonymize_metadata(metadata)

    assert anonymized == metadata  # Should remain unchanged
