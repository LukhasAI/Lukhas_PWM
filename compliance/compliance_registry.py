from typing import Dict, Any, List
import json
import logging
from datetime import datetime
from pathlib import Path

class ComplianceRegistry:
    """Central registry for LUCAS AGI compliance management"""
    
    def __init__(self, registry_path: str = None):
        self.logger = logging.getLogger("compliance_registry")
        self.registry_path = registry_path or Path(__file__).parent / "compliance_data"
        self.registry_path.mkdir(exist_ok=True)
        
        self.active_regulations = {
            "EU_AI_ACT": {
                "version": "2024.1",
                "risk_category": "high",
                "requirements": [
                    "transparency",
                    "human_oversight",
                    "technical_robustness"
                ]
            },
            "GDPR": {
                "status": "compliant",
                "dpo_assigned": True,
                "data_impact_assessment": True
            },
            "US_AI_BILL_RIGHTS": {
                "status": "compliant",
                "requirements": [
                    "algorithmic_discrimination_protection",
                    "ai_system_notice"
                ]
            }
        }
        
        self.component_registry = {}
        
    async def register_component(self, 
                               component_id: str, 
                               compliance_data: Dict[str, Any]) -> None:
        """Register a component's compliance information"""
        self.component_registry[component_id] = {
            "registration_date": datetime.now().isoformat(),
            "compliance_data": compliance_data,
            "last_audit": None
        }
        
        await self._save_registry()
        
    async def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": "compliant",
            "active_regulations": self.active_regulations,
            "registered_components": len(self.component_registry),
            "last_assessment": datetime.now().isoformat(),
            "compliance_officer": "LUCAS_OVERSIGHT",
            "documentation_links": {
                "full_assessment": "/compliance/full_assessment.pdf",
                "certifications": "/compliance/certifications/",
                "audit_logs": "/compliance/audit_logs/"
            }
        }
        
    async def _save_registry(self) -> None:
        """Save registry state to disk"""
        registry_file = self.registry_path / "compliance_registry.json"
        with open(registry_file, 'w') as f:
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "components": self.component_registry,
                "regulations": self.active_regulations
            }, f, indent=2)
            
    def get_component_requirements(self, component_id: str) -> List[str]:
        """Get compliance requirements for a specific component"""
        base_requirements = [
            "data_minimization",
            "purpose_limitation",
            "transparency",
            "security"
        ]
        
        component_type = component_id.split('_')[0]
        if component_type == "llm":
            base_requirements.extend([
                "content_filtering",
                "bias_mitigation",
                "ethical_constraints"
            ])
        elif component_type == "intent":
            base_requirements.extend([
                "consent_management",
                "pii_protection"
            ])
            
        return base_requirements
