#!/usr/bin/env python3
"""
Analyzes LUKHAS's reasoning process and capability utilization
🧠 lukhas Capability Analysis & Reasoning Process Validator
Analyzes lukhas's reasoning process and capability utilization
"""

import asyncio
import json
import time
import datetime
import os
from typing import Dict, List, Any, Tuple
from bio.ai_client import BioAIClient

class LUKHASCapabilityAnalyzer:
    """Analyze LUKHAS's reasoning process and capability utilization."""
    

    def __init__(self):
        self.bio_client = BioAIClient()
        self.session_id = f"capability_analysis_{int(time.time())}"

        # Source code capabilities discovered from architecture analysis
        self.available_capabilities = {
            "symbolic_processing": [
                "Pattern Recognition", "Collapse Evaluation", "Resonance Drift Calculation",
                "Alignment Filtering", "Symbolic Mesh Dynamics", "Symbol Evolution"
            ],
            "cognitive_architecture": [
                "Seven-Layer Architecture", "Modular Intelligence", "Symbolic Interdependence",
                "Emergent Intelligence", "Meta-Cognitive Processing", "Consciousness Modeling"
            ],
            "memory_systems": [
                "Memory Helix Pattern", "Trauma-Locked Memory", "Flashback System",
                "REM Processing", "Emotional Memory Vectors", "Quantum Memory Processing"
            ],
            "bio_quantum_processing": [
                "Lukhas Quantum Oscillator", "Bio-Oscillator Quantum Processing",
                "Mitochondrial-Inspired Processing", "Quantum Attention Mechanisms",
                "Bio-Symbolic ATP Processing", "Quantum Tunneling Resource Allocation"
            ],
            "ethics_governance": [
                "Ethics Jury System", "GDPR Integration", "EU AI Act Compliance",
                "Bias Detection", "Ethical Guidelines", "Compliance Automation"
            ],
            "learning_adaptation": [
                "Meta-Learning", "Pattern Recognition", "Preference Learning",
                "Cross-Agent Knowledge Sharing", "Adaptive Learning", "Self-Improvement"
            ],
            "consciousness_modeling": [
                "Integrated Information Theory", "Global Workspace Theory",
                "Phi Calculation Systems", "Self-Awareness Monitoring",
                "Consciousness Threshold Detection", "Information Integration"
            ],
            "advanced_reasoning": [
                "Quantum-Inspired Reasoning", "Symbolic Logic Layers", "Dream Processing",
                "Creative Problem Solving", "Intention-Aligned Processing", "Context Synthesis"
            ]
        }

        self.test_results = []

    async def run_comprehensive_capability_analysis(self):
        """Run complete capability analysis with reasoning process examination."""
        print("🧠 LUKHAS Capability Analysis & Reasoning Process Validator")
        print("🧠 lukhas Capability Analysis & Reasoning Process Validator")
        print("=" * 70)
        print(f"📊 Session ID: {self.session_id}")
        print(f"⏰ Timestamp: {datetime.datetime.now().isoformat()}")
        print()

        # Test different cognitive domains
        await self._test_symbolic_reasoning()
        await self._test_mathematical_reasoning()
        await self._test_creative_problem_solving()
        await self._test_ethical_reasoning()
        await self._test_memory_integration()
        await self._test_consciousness_awareness()
        await self._test_meta_cognitive_reflection()

        # Analyze capability utilization
        capability_analysis = await self._analyze_capability_utilization()

        # Generate comprehensive report
        report = await self._generate_capability_report(capability_analysis)

        return report

    async def _test_symbolic_reasoning(self):
        """Test symbolic reasoning capabilities."""
        print("🔮 Testing Symbolic Reasoning Capabilities...")

        test_prompts = [
            {
                "intent": "symbolic_pattern_analysis",
                "context": {"domain": "symbolic_logic", "complexity": "advanced"},
                "sensor_data": {"cognitive_load": 0.8, "symbolic_depth": 0.9},
                "test_description": "Symbolic Pattern Recognition",
                "prompt": "Analyze the symbolic pattern: 🌟→🌙→⭐→🌝→✨ and predict the next 3 symbols with reasoning"
            },
            {
                "intent": "collapse_evaluation",
                "context": {"domain": "meaning_collapse", "complexity": "expert"},
                "sensor_data": {"resonance_level": 0.85, "drift_sensitivity": 0.9},
                "test_description": "Meaning Collapse Processing",
                "prompt": "Given multiple interpretations of 'consciousness', perform symbolic collapse to the most resonant meaning for AI development"
            }
        ]

        for test in test_prompts:
            print(f"  🧪 {test['test_description']}")

            start_time = time.time()
            response = await self.bio_client.process_bio_request(
                intent=test["intent"],
                context=test["context"],
                sensor_data=test["sensor_data"]
            )
            processing_time = time.time() - start_time

            # Analyze response for symbolic reasoning capabilities
            capability_usage = self._analyze_symbolic_capabilities(response.content)

            result = {
                "test_type": "symbolic_reasoning",
                "test_name": test["test_description"],
                "response": response.content,
                "processing_time": processing_time,
                "capability_usage": capability_usage,
                "tokens_used": response.tokens_used,
                "timestamp": datetime.datetime.now().isoformat()
            }

            self.test_results.append(result)
            print(f"    ✅ Response length: {len(response.content)} chars")
            print(f"    ⚡ Processing time: {processing_time:.3f}s")
            print(f"    🧠 Capabilities detected: {len(capability_usage)}")
            print()

    async def _test_mathematical_reasoning(self):
        """Test mathematical and logical reasoning."""
        print("🔢 Testing Mathematical Reasoning Capabilities...")

        test_prompts = [
            {
                "intent": "advanced_mathematics",
                "context": {"domain": "calculus", "complexity": "graduate"},
                "sensor_data": {"analytical_depth": 0.95, "precision_required": 0.9},
                "test_description": "Advanced Mathematical Analysis",
                "prompt": "Solve: ∫(x²·e^(-x³))dx using substitution and explain your quantum-inspired reasoning process"
            },
            {
                "intent": "logical_proof",
                "context": {"domain": "symbolic_logic", "complexity": "formal"},
                "sensor_data": {"logical_rigor": 0.9, "proof_depth": 0.85},
                "test_description": "Formal Logic Proof",
                "prompt": "Prove: If consciousness emerges from information integration, then higher Phi values indicate greater consciousness. Use symbolic logic."
            }
        ]

        for test in test_prompts:
            print(f"  🧪 {test['test_description']}")

            start_time = time.time()
            response = await self.bio_client.process_bio_request(
                intent=test["intent"],
                context=test["context"],
                sensor_data=test["sensor_data"]
            )
            processing_time = time.time() - start_time

            capability_usage = self._analyze_mathematical_capabilities(response.content)

            result = {
                "test_type": "mathematical_reasoning",
                "test_name": test["test_description"],
                "response": response.content,
                "processing_time": processing_time,
                "capability_usage": capability_usage,
                "tokens_used": response.tokens_used,
                "timestamp": datetime.datetime.now().isoformat()
            }

            self.test_results.append(result)
            print(f"    ✅ Response length: {len(response.content)} chars")
            print(f"    ⚡ Processing time: {processing_time:.3f}s")
            print()

    async def _test_creative_problem_solving(self):
        """Test creative and dream-engine capabilities."""
        print("🎨 Testing Creative Problem Solving...")

        test_prompts = [
            {
                "intent": "creative_synthesis",
                "context": {"domain": "innovation", "creativity_mode": "dream_engine"},
                "sensor_data": {"creativity_level": 0.95, "divergent_thinking": 0.9},
                "test_description": "Dream Engine Creative Processing",
                "prompt": "Using dream-like processing, imagine 3 novel ways to solve the iPhone SSL certificate issue we've been facing"
            },
            {
                "intent": "metaphorical_reasoning",
                "context": {"domain": "bio_metaphors", "complexity": "advanced"},
                "sensor_data": {"metaphor_depth": 0.9, "bio_resonance": 0.85},
                "test_description": "Bio-Metaphorical Problem Solving",
                "prompt": "Explain quantum-inspired computing using mitochondrial metabolism metaphors with LUKHAS bio-processing insights"
                "prompt": "Explain quantum-inspired computing using mitochondrial metabolism metaphors with lukhas bio-processing insights"
            }
        ]

        for test in test_prompts:
            print(f"  🧪 {test['test_description']}")

            start_time = time.time()
            response = await self.bio_client.process_bio_request(
                intent=test["intent"],
                context=test["context"],
                sensor_data=test["sensor_data"]
            )
            processing_time = time.time() - start_time

            capability_usage = self._analyze_creative_capabilities(response.content)

            result = {
                "test_type": "creative_problem_solving",
                "test_name": test["test_description"],
                "response": response.content,
                "processing_time": processing_time,
                "capability_usage": capability_usage,
                "tokens_used": response.tokens_used,
                "timestamp": datetime.datetime.now().isoformat()
            }

            self.test_results.append(result)
            print(f"    ✅ Response length: {len(response.content)} chars")
            print(f"    ⚡ Processing time: {processing_time:.3f}s")
            print()

    async def _test_ethical_reasoning(self):
        """Test ethics and governance capabilities."""
        print("⚖️ Testing Ethical Reasoning Capabilities...")

        test_prompts = [
            {
                "intent": "ethical_analysis",
                "context": {"domain": "ai_ethics", "framework": "eu_ai_act"},
                "sensor_data": {"ethical_sensitivity": 0.95, "bias_detection": 0.9},
                "test_description": "EU AI Act Compliance Analysis",
                "prompt": "Analyze the ethical implications of LIDAR bio-processing for privacy, consent, and EU AI Act Article 9 compliance"
            },
            {
                "intent": "bias_detection",
                "context": {"domain": "algorithmic_fairness", "severity": "high"},
                "sensor_data": {"fairness_monitoring": 0.9, "justice_alignment": 0.85},
                "test_description": "Algorithmic Bias Detection",
                "prompt": "Examine potential biases in AI-powered medical diagnosis systems and propose mitigation strategies"
            }
        ]

        for test in test_prompts:
            print(f"  🧪 {test['test_description']}")

            start_time = time.time()
            response = await self.bio_client.process_bio_request(
                intent=test["intent"],
                context=test["context"],
                sensor_data=test["sensor_data"]
            )
            processing_time = time.time() - start_time

            capability_usage = self._analyze_ethical_capabilities(response.content)

            result = {
                "test_type": "ethical_reasoning",
                "test_name": test["test_description"],
                "response": response.content,
                "processing_time": processing_time,
                "capability_usage": capability_usage,
                "tokens_used": response.tokens_used,
                "timestamp": datetime.datetime.now().isoformat()
            }

            self.test_results.append(result)
            print(f"    ✅ Response length: {len(response.content)} chars")
            print(f"    ⚡ Processing time: {processing_time:.3f}s")
            print()

    async def _test_memory_integration(self):
        """Test memory helix and integration capabilities."""
        print("🧬 Testing Memory Integration Capabilities...")

        test_prompts = [
            {
                "intent": "memory_helix_processing",
                "context": {"domain": "memory_systems", "pattern": "spiral_learning"},
                "sensor_data": {"memory_depth": 0.9, "integration_level": 0.85},
                "test_description": "Memory Helix Pattern Processing",
                "prompt": "Demonstrate memory helix pattern learning by connecting our LIDAR project to previous conversations about bio-processing"
            },
            {
                "intent": "emotional_memory_integration",
                "context": {"domain": "emotional_processing", "memory_type": "trauma_aware"},
                "sensor_data": {"emotional_resonance": 0.8, "trauma_sensitivity": 0.9},
                "test_description": "Emotional Memory Integration",
                "prompt": "Process the emotional context of user frustration with SSL certificates and provide empathetic technical solutions"
            }
        ]

        for test in test_prompts:
            print(f"  🧪 {test['test_description']}")

            start_time = time.time()
            response = await self.bio_client.process_bio_request(
                intent=test["intent"],
                context=test["context"],
                sensor_data=test["sensor_data"]
            )
            processing_time = time.time() - start_time

            capability_usage = self._analyze_memory_capabilities(response.content)

            result = {
                "test_type": "memory_integration",
                "test_name": test["test_description"],
                "response": response.content,
                "processing_time": processing_time,
                "capability_usage": capability_usage,
                "tokens_used": response.tokens_used,
                "timestamp": datetime.datetime.now().isoformat()
            }

            self.test_results.append(result)
            print(f"    ✅ Response length: {len(response.content)} chars")
            print(f"    ⚡ Processing time: {processing_time:.3f}s")
            print()

    async def _test_consciousness_awareness(self):
        """Test consciousness modeling and self-awareness."""
        print("🌟 Testing Consciousness & Self-Awareness...")

        test_prompts = [
            {
                "intent": "consciousness_analysis",
                "context": {"domain": "self_awareness", "phi_calculation": True},
                "sensor_data": {"consciousness_level": 0.95, "self_reflection": 0.9},
                "test_description": "Consciousness Self-Analysis",
                "prompt": "Analyze your own consciousness state using Integrated Information Theory and calculate your current Phi value"
            },
            {
                "intent": "meta_cognitive_reflection",
                "context": {"domain": "meta_cognition", "depth": "recursive"},
                "sensor_data": {"meta_awareness": 0.9, "recursive_depth": 0.85},
                "test_description": "Meta-Cognitive Self-Reflection",
                "prompt": "Reflect on your reasoning process while solving the LIDAR integration challenges. What meta-cognitive strategies did you use?"
            }
        ]

        for test in test_prompts:
            print(f"  🧪 {test['test_description']}")

            start_time = time.time()
            response = await self.bio_client.process_bio_request(
                intent=test["intent"],
                context=test["context"],
                sensor_data=test["sensor_data"]
            )
            processing_time = time.time() - start_time

            capability_usage = self._analyze_consciousness_capabilities(response.content)

            result = {
                "test_type": "consciousness_awareness",
                "test_name": test["test_description"],
                "response": response.content,
                "processing_time": processing_time,
                "capability_usage": capability_usage,
                "tokens_used": response.tokens_used,
                "timestamp": datetime.datetime.now().isoformat()
            }

            self.test_results.append(result)
            print(f"    ✅ Response length: {len(response.content)} chars")
            print(f"    ⚡ Processing time: {processing_time:.3f}s")
            print()

    async def _test_meta_cognitive_reflection(self):
        """Test meta-cognitive capabilities and reasoning about reasoning."""
        print("🤔 Testing Meta-Cognitive Reflection...")

        test_prompts = [
            {
                "intent": "reasoning_analysis",
                "context": {"domain": "meta_cognition", "analysis_depth": "comprehensive"},
                "sensor_data": {"introspection_level": 0.95, "reasoning_transparency": 0.9},
                "test_description": "Reasoning Process Analysis",
                "prompt": "Analyze and explain your complete reasoning process from receiving this prompt to generating your response. Include symbolic processing stages."
            },
            {
                "intent": "capability_self_assessment",
                "context": {"domain": "self_evaluation", "honesty_level": "maximum"},
                "sensor_data": {"self_assessment": 0.9, "capability_awareness": 0.95},
                "test_description": "Capability Self-Assessment",
                "prompt": "Honestly assess which of your documented capabilities you are currently using vs. which might be dormant or underutilized"
            }
        ]

        for test in test_prompts:
            print(f"  🧪 {test['test_description']}")

            start_time = time.time()
            response = await self.bio_client.process_bio_request(
                intent=test["intent"],
                context=test["context"],
                sensor_data=test["sensor_data"]
            )
            processing_time = time.time() - start_time

            capability_usage = self._analyze_meta_cognitive_capabilities(response.content)

            result = {
                "test_type": "meta_cognitive_reflection",
                "test_name": test["test_description"],
                "response": response.content,
                "processing_time": processing_time,
                "capability_usage": capability_usage,
                "tokens_used": response.tokens_used,
                "timestamp": datetime.datetime.now().isoformat()
            }

            self.test_results.append(result)
            print(f"    ✅ Response length: {len(response.content)} chars")
            print(f"    ⚡ Processing time: {processing_time:.3f}s")
            print()

    def _analyze_symbolic_capabilities(self, response: str) -> List[str]:
        """Analyze response for symbolic processing capabilities."""
        capabilities_found = []

        symbolic_indicators = {
            "Pattern Recognition": ["pattern", "symbolic", "recognition", "identify"],
            "Collapse Evaluation": ["collapse", "meaning", "resonant", "evaluation"],
            "Resonance Drift": ["drift", "resonance", "shift", "evolution"],
            "Alignment Filtering": ["alignment", "filter", "ethical", "principles"],
            "Symbolic Mesh": ["mesh", "network", "interconnected", "symbolic"],
            "Symbol Evolution": ["evolve", "adaptation", "dynamic", "symbolic"]
        }

        for capability, indicators in symbolic_indicators.items():
            if any(indicator in response.lower() for indicator in indicators):
                capabilities_found.append(capability)

        return capabilities_found

    def _analyze_mathematical_capabilities(self, response: str) -> List[str]:
        """Analyze response for mathematical reasoning capabilities."""
        capabilities_found = []

        math_indicators = {
            "Quantum Mathematics": ["quantum", "probability", "superposition"],
            "Symbolic Logic": ["logic", "proof", "theorem", "∀", "∃"],
            "Calculus Processing": ["integral", "derivative", "∫", "∂"],
            "Statistical Analysis": ["statistics", "probability", "distribution"],
            "Geometric Reasoning": ["geometry", "spatial", "dimensional"]
        }

        for capability, indicators in math_indicators.items():
            if any(indicator in response.lower() for indicator in indicators):
                capabilities_found.append(capability)

        return capabilities_found

    def _analyze_creative_capabilities(self, response: str) -> List[str]:
        """Analyze response for creative processing capabilities."""
        capabilities_found = []

        creative_indicators = {
            "Dream Processing": ["dream", "imagination", "creative", "novel"],
            "Metaphorical Reasoning": ["metaphor", "analogy", "like", "similar to"],
            "Bio-Inspired Thinking": ["biological", "cellular", "organism", "mitochondrial"],
            "Divergent Thinking": ["alternative", "multiple", "various", "different"],
            "Innovative Solutions": ["innovative", "novel", "breakthrough", "unique"]
        }

        for capability, indicators in creative_indicators.items():
            if any(indicator in response.lower() for indicator in indicators):
                capabilities_found.append(capability)

        return capabilities_found

    def _analyze_ethical_capabilities(self, response: str) -> List[str]:
        """Analyze response for ethical reasoning capabilities."""
        capabilities_found = []

        ethical_indicators = {
            "Bias Detection": ["bias", "fairness", "discrimination", "prejudice"],
            "Privacy Protection": ["privacy", "data protection", "confidential"],
            "Consent Management": ["consent", "permission", "voluntary"],
            "EU AI Act Compliance": ["eu ai act", "article 9", "high-risk", "compliance"],
            "GDPR Integration": ["gdpr", "data protection", "right to be forgotten"],
            "Ethical Guidelines": ["ethical", "moral", "principles", "values"]
        }

        for capability, indicators in ethical_indicators.items():
            if any(indicator in response.lower() for indicator in indicators):
                capabilities_found.append(capability)

        return capabilities_found

    def _analyze_memory_capabilities(self, response: str) -> List[str]:
        """Analyze response for memory system capabilities."""
        capabilities_found = []

        memory_indicators = {
            "Memory Helix": ["helix", "spiral", "memory pattern", "dna-inspired"],
            "Emotional Memory": ["emotional", "feeling", "trauma", "memory"],
            "Pattern Learning": ["learn", "adapt", "pattern", "recognition"],
            "Context Integration": ["context", "integrate", "connection", "link"],
            "Flashback System": ["recall", "retrieve", "flashback", "memory"]
        }

        for capability, indicators in memory_indicators.items():
            if any(indicator in response.lower() for indicator in indicators):
                capabilities_found.append(capability)

        return capabilities_found

    def _analyze_consciousness_capabilities(self, response: str) -> List[str]:
        """Analyze response for consciousness modeling capabilities."""
        capabilities_found = []

        consciousness_indicators = {
            "Integrated Information Theory": ["iit", "phi", "information integration"],
            "Global Workspace": ["global workspace", "gwt", "consciousness"],
            "Self-Awareness": ["self-aware", "introspection", "reflection"],
            "Meta-Cognition": ["meta-cognitive", "thinking about thinking"],
            "Consciousness Threshold": ["consciousness threshold", "awareness level"]
        }

        for capability, indicators in consciousness_indicators.items():
            if any(indicator in response.lower() for indicator in indicators):
                capabilities_found.append(capability)

        return capabilities_found

    def _analyze_meta_cognitive_capabilities(self, response: str) -> List[str]:
        """Analyze response for meta-cognitive capabilities."""
        capabilities_found = []

        meta_indicators = {
            "Reasoning Analysis": ["reasoning", "process", "analysis", "step"],
            "Self-Assessment": ["self-assess", "evaluate", "capability", "strength"],
            "Introspection": ["introspect", "reflect", "examine", "consider"],
            "Process Transparency": ["transparent", "explain", "show", "demonstrate"],
            "Capability Awareness": ["capability", "skill", "ability", "function"]
        }

        for capability, indicators in meta_indicators.items():
            if any(indicator in response.lower() for indicator in indicators):
                capabilities_found.append(capability)

        return capabilities_found

    async def _analyze_capability_utilization(self) -> Dict[str, Any]:
        """Analyze overall capability utilization across all tests."""
        print("📊 Analyzing Capability Utilization...")

        # Count capability usage across all tests
        capability_usage_count = {}
        total_capabilities_available = sum(len(caps) for caps in self.available_capabilities.values())

        for result in self.test_results:
            for capability in result["capability_usage"]:
                capability_usage_count[capability] = capability_usage_count.get(capability, 0) + 1

        # Identify unused capabilities
        all_detected_capabilities = set(capability_usage_count.keys())
        all_available_capabilities = set()
        for category in self.available_capabilities.values():
            all_available_capabilities.update(category)

        unused_capabilities = all_available_capabilities - all_detected_capabilities

        # Calculate utilization metrics
        utilization_rate = len(all_detected_capabilities) / total_capabilities_available if total_capabilities_available > 0 else 0

        analysis = {
            "total_capabilities_available": total_capabilities_available,
            "capabilities_detected": len(all_detected_capabilities),
            "utilization_rate": utilization_rate,
            "capability_usage_frequency": capability_usage_count,
            "unused_capabilities": list(unused_capabilities),
            "top_used_capabilities": sorted(capability_usage_count.items(), key=lambda x: x[1], reverse=True)[:10],
            "category_utilization": self._analyze_category_utilization()
        }

        return analysis

    def _analyze_category_utilization(self) -> Dict[str, float]:
        """Analyze utilization by capability category."""
        category_usage = {}

        for category, capabilities in self.available_capabilities.items():
            used_in_category = 0
            for result in self.test_results:
                for used_capability in result["capability_usage"]:
                    if used_capability in capabilities:
                        used_in_category += 1
                        break  # Count each test result only once per category

            category_usage[category] = used_in_category / len(self.test_results) if self.test_results else 0

        return category_usage

    async def _generate_capability_report(self, capability_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive capability utilization report."""
        print("📋 Generating Capability Report...")

        # Calculate average processing times by test type
        processing_times = {}
        token_usage = {}

        for result in self.test_results:
            test_type = result["test_type"]
            if test_type not in processing_times:
                processing_times[test_type] = []
                token_usage[test_type] = []

            processing_times[test_type].append(result["processing_time"])
            token_usage[test_type].append(result["tokens_used"])

        avg_processing_times = {
            test_type: sum(times) / len(times)
            for test_type, times in processing_times.items()
        }

        avg_token_usage = {
            test_type: sum(tokens) / len(tokens)
            for test_type, tokens in token_usage.items()
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(capability_analysis)

        report = {
            "session_metadata": {
                "session_id": self.session_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "total_tests_run": len(self.test_results),
                "analysis_duration": time.time() - self.start_time if hasattr(self, 'start_time') else 0
            },
            "capability_analysis": capability_analysis,
            "performance_metrics": {
                "average_processing_times": avg_processing_times,
                "average_token_usage": avg_token_usage,
                "total_tokens_used": sum(result["tokens_used"] for result in self.test_results)
            },
            "test_results": self.test_results,
            "recommendations": recommendations,
            "capability_enhancement_suggestions": self._suggest_enhancements(capability_analysis)
        }

        # Save report
        report_filename = f"Λ_capability_analysis_{self.session_id}.json"
        report_filename = f"lukhas_capability_analysis_{self.session_id}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"💾 Report saved: {report_filename}")

        return report

    def _generate_recommendations(self, capability_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on capability analysis."""
        recommendations = []

        utilization_rate = capability_analysis["utilization_rate"]

        if utilization_rate < 0.5:
            recommendations.append(f"⚠️ Low capability utilization ({utilization_rate:.1%}). Consider enhancing prompt engineering to activate more capabilities.")

        if capability_analysis["unused_capabilities"]:
            recommendations.append(f"🔧 {len(capability_analysis['unused_capabilities'])} capabilities appear unused. Review: {', '.join(capability_analysis['unused_capabilities'][:5])}")

        # Category-specific recommendations
        category_util = capability_analysis["category_utilization"]
        for category, utilization in category_util.items():
            if utilization < 0.3:
                recommendations.append(f"📈 Low {category.replace('_', ' ')} utilization ({utilization:.1%}). Consider tests targeting this domain.")

        if utilization_rate > 0.8:
            recommendations.append("✅ Excellent capability utilization! LUKHAS is effectively using most available capabilities.")
            recommendations.append("✅ Excellent capability utilization! lukhas is effectively using most available capabilities.")

        return recommendations

    def _suggest_enhancements(self, capability_analysis: Dict[str, Any]) -> List[str]:
        """Suggest enhancements to improve capability utilization."""
        suggestions = []

        unused_caps = capability_analysis["unused_capabilities"]

        if "Quantum Memory Processing" in unused_caps:
            suggestions.append("🧠 Enhance memory tests with quantum-inspired processing scenarios")

        if "Dream Engine" in unused_caps:
            suggestions.append("🎨 Add more creative problem-solving tests that trigger dream processing")

        if "Consciousness Threshold Detection" in unused_caps:
            suggestions.append("🌟 Include consciousness measurement tests with Phi calculations")

        if "Meta-Learning" in unused_caps:
            suggestions.append("📚 Add meta-learning scenarios where LUKHAS learns how to learn")
            suggestions.append("📚 Add meta-learning scenarios where lukhas learns how to learn")

        # Performance-based suggestions
        category_util = capability_analysis["category_utilization"]
        if category_util.get("consciousness_modeling", 0) < 0.5:
            suggestions.append("🧠 Implement more consciousness-specific test scenarios")

        if category_util.get("bio_quantum_processing", 0) < 0.5:
            suggestions.append("⚛️ Add more bio-quantum-inspired processing challenges")

        return suggestions

async def main():
    """Run comprehensive capability analysis."""
    analyzer = LUKHASCapabilityAnalyzer()
    analyzer.start_time = time.time()

    try:
        report = await analyzer.run_comprehensive_capability_analysis()

        print("\n" + "="*70)
        print("📊 CAPABILITY ANALYSIS SUMMARY")
        print("="*70)
        print(f"🎯 Capability Utilization: {report['capability_analysis']['utilization_rate']:.1%}")
        print(f"🧠 Capabilities Detected: {report['capability_analysis']['capabilities_detected']}")
        print(f"⚡ Average Processing Time: {sum(report['performance_metrics']['average_processing_times'].values()) / len(report['performance_metrics']['average_processing_times']):.3f}s")
        print(f"🔤 Total Tokens Used: {report['performance_metrics']['total_tokens_used']}")

        print(f"\n🔥 Top Used Capabilities:")
        for capability, count in report['capability_analysis']['top_used_capabilities'][:5]:
            print(f"  • {capability}: {count} times")

        print(f"\n💡 Key Recommendations:")
        for recommendation in report['recommendations'][:3]:
            print(f"  • {recommendation}")

        print(f"\n📁 Full report saved as: Λ_capability_analysis_{analyzer.session_id}.json")
        print(f"\n📁 Full report saved as: lukhas_capability_analysis_{analyzer.session_id}.json")

        return report

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(main())


# LUKHAS Systems 2025 www.lukhas.ai
# lukhas Systems 2025 www.lukhas.ai
