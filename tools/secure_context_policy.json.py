secure_context_policy.json
{
  "trusted_devices": {
    "gonzo.dominguez": [
      "MACBOOK-M1-ID1234",
      "LUCASPAD-TESTUNIT"
    ],
    "red_team_alpha": [
      "SECURE-LINUX-BOX-9"
    ]
  },
  "working_hours": {
    "timezone": "Europe/Madrid",
    "start_hour": 8,
    "end_hour": 20,
    "allowed_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
  },
  "access_policies": {
    "symbolic_engine": {
      "off_hours_access": false,
      "requires_2fa": true,
      "allowed_roles": ["admin", "developer"]
    },
    "dream_engine": {
      "off_hours_access": false,
      "requires_2fa": true,
      "allowed_roles": ["admin", "researcher"]
    },
    "compliance_dashboard": {
      "off_hours_access": false,
      "requires_2fa": true,
      "allowed_roles": ["admin", "compliance_officer", "red_team"]
    },
    "memory_folds": {
      "off_hours_access": true,
      "requires_2fa": true,
      "allowed_roles": ["admin", "researcher"]
    }
  },
  "geo_lock": {
    "enabled": true,
    "tier_5_access_locations": [
      "LUCAS_HQ_ZONE",
      "RESEARCH_CAMPUS_ALPHA"
    ]
  },
  "anomaly_detection": {
    "multiple_sessions": true,
    "inactivity_timeout_minutes": 30,
    "sensitive_access_alert": true
  }
}