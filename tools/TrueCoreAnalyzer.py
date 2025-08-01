#!/usr/bin/env python3
"""
<<<<<<< HEAD
True Core Analyzer: Separate actual Λ AI core from external packages
=======
True Core Analyzer: Separate actual lukhas AI core from external packages
>>>>>>> jules/ecosystem-consolidation-2025
Shows what's actually your AI system vs external dependencies
"""

import os
from datetime import datetime
from collections import defaultdict

class TrueCoreAnalyzer:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
<<<<<<< HEAD
        self.Λ_path = os.path.join(workspace_path, 'lukhas')
=======
        self.lukhas_path = os.path.join(workspace_path, 'lukhas')
>>>>>>> jules/ecosystem-consolidation-2025

        # External packages we know don't belong in core
        self.external_indicators = [
            'Open-Sora', 'HunyuanVideo', 'TTS', 'whisper', 'opensora',
            'recipes', 'models', 'datasets', 'checkpoints', 'weights',
            'pretrained', 'gradio', 'streamlit', 'node_modules', 'dist',
            'build', 'static', 'assets', 'images', 'videos', 'audio'
        ]

        self.categories = {
            'true_core': [],      # Your actual AI core
            'external_packages': [], # Video/audio generation libraries
            'interface_bloat': [],   # Large UI/web packages
            'documentation': [],     # Docs and guides
            'configuration': [],     # Config files
            'unknown': []           # Need manual review
        }

        self.stats = {
            'true_core_files': 0,
            'external_files': 0,
            'interface_files': 0,
            'doc_files': 0,
            'config_files': 0
        }

    def analyze_lukhas_structure(self):
        """Analyze what's actually core AI vs external bloat"""
<<<<<<< HEAD
        if not os.path.exists(self.Λ_path):
            print(f"❌ lukhas/ directory not found")
            return

        print(f"🔍 Analyzing true Λ core vs external packages...")

        for root, dirs, files in os.walk(self.Λ_path):
=======
        if not os.path.exists(self.lukhas_path):
            print(f"❌ lukhas/ directory not found")
            return

        print(f"🔍 Analyzing true lukhas core vs external packages...")

        for root, dirs, files in os.walk(self.lukhas_path):
>>>>>>> jules/ecosystem-consolidation-2025
            for file in files:
                if file.startswith('.'):
                    continue

                file_path = os.path.join(root, file)
<<<<<<< HEAD
                rel_path = os.path.relpath(file_path, self.Λ_path)
=======
                rel_path = os.path.relpath(file_path, self.lukhas_path)
>>>>>>> jules/ecosystem-consolidation-2025

                self._categorize_file(file_path, rel_path)

    def _categorize_file(self, file_path: str, rel_path: str):
        """Categorize each file as core AI or external"""
        file_name = os.path.basename(file_path)
        path_lower = rel_path.lower()

        # Check if it's an external package
        if any(indicator in path_lower for indicator in self.external_indicators):
            self.categories['external_packages'].append(rel_path)
            self.stats['external_files'] += 1
            return

        # Documentation
        if (rel_path.startswith('docs/') or
            file_name.endswith('.md') or
            'readme' in file_name.lower() or
            'license' in file_name.lower()):
            self.categories['documentation'].append(rel_path)
            self.stats['doc_files'] += 1
            return

        # Configuration files
        if (file_name.endswith(('.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.cfg')) or
            'config' in path_lower or 'settings' in path_lower):
            self.categories['configuration'].append(rel_path)
            self.stats['config_files'] += 1
            return

        # Large interface packages (likely external)
<<<<<<< HEAD
        if ('Λ_as_agent' in path_lower and len(rel_path.split('/')) > 4):
=======
        if ('lukhas_as_agent' in path_lower and len(rel_path.split('/')) > 4):
>>>>>>> jules/ecosystem-consolidation-2025
            self.categories['interface_bloat'].append(rel_path)
            self.stats['interface_files'] += 1
            return

        # True core AI components
        core_indicators = [
            'brain/', 'neural/', 'symbolic/', 'agent/', 'memory/',
            'learning/', 'intelligence/', 'orchestration/', 'safety/',
            'governance/', 'id/', 'dreams/', 'quantum/', 'bio/'
        ]

        if (any(indicator in rel_path for indicator in core_indicators) or
            rel_path.startswith('core/') and file_name.endswith('.py')):
            self.categories['true_core'].append(rel_path)
            self.stats['true_core_files'] += 1
            return

        # Everything else needs review
        self.categories['unknown'].append(rel_path)

    def generate_analysis_report(self):
        """Generate analysis showing true core vs bloat"""
        report_path = f"{self.workspace_path}/TRUE_CORE_ANALYSIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_path, 'w') as f:
<<<<<<< HEAD
            f.write("# Λ True Core Analysis\n")
=======
            f.write("# lukhas True Core Analysis\n")
>>>>>>> jules/ecosystem-consolidation-2025
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📊 File Distribution Summary\n\n")
            total_files = sum(len(files) for files in self.categories.values())

            f.write(f"- **Total Files in lukhas/:** {total_files}\n")
            f.write(f"- **True AI Core:** {len(self.categories['true_core'])} files ({len(self.categories['true_core'])/total_files*100:.1f}%)\n")
            f.write(f"- **External Packages:** {len(self.categories['external_packages'])} files ({len(self.categories['external_packages'])/total_files*100:.1f}%)\n")
            f.write(f"- **Interface Bloat:** {len(self.categories['interface_bloat'])} files ({len(self.categories['interface_bloat'])/total_files*100:.1f}%)\n")
            f.write(f"- **Documentation:** {len(self.categories['documentation'])} files ({len(self.categories['documentation'])/total_files*100:.1f}%)\n")
            f.write(f"- **Configuration:** {len(self.categories['configuration'])} files ({len(self.categories['configuration'])/total_files*100:.1f}%)\n")
            f.write(f"- **Unknown/Review:** {len(self.categories['unknown'])} files ({len(self.categories['unknown'])/total_files*100:.1f}%)\n\n")

            # True Core Analysis
            f.write("## 🧠 True AI Core Components\n\n")
            f.write(f"**{len(self.categories['true_core'])} files** - Your actual AI system\n\n")

            if self.categories['true_core']:
                core_by_module = defaultdict(list)
                for file in self.categories['true_core']:
                    if '/' in file:
                        module = file.split('/')[1] if file.startswith('core/') else file.split('/')[0]
                        core_by_module[module].append(file)

                for module, files in sorted(core_by_module.items()):
                    f.write(f"### {module}/ ({len(files)} files)\n")
                    for file in sorted(files)[:5]:  # Show first 5
                        f.write(f"- `{file}`\n")
                    if len(files) > 5:
                        f.write(f"- ... and {len(files)-5} more files\n")
                    f.write("\n")

            # External Packages
            f.write("## 📦 External Packages (Should be Dependencies)\n\n")
            f.write(f"**{len(self.categories['external_packages'])} files** - Not part of your AI core\n\n")

            if self.categories['external_packages']:
                external_by_package = defaultdict(list)
                for file in self.categories['external_packages']:
                    if 'Open-Sora' in file:
                        package = 'Open-Sora (Video Generation)'
                    elif 'HunyuanVideo' in file:
                        package = 'HunyuanVideo (Video Generation)'
                    elif 'TTS' in file:
                        package = 'TTS (Text-to-Speech)'
                    elif 'whisper' in file:
                        package = 'Whisper (Speech Recognition)'
                    else:
                        package = 'Other External'
                    external_by_package[package].append(file)

                for package, files in sorted(external_by_package.items()):
                    f.write(f"### {package} ({len(files)} files)\n")
                    f.write("*These should be pip/npm dependencies, not included in core*\n\n")

            # Interface Bloat
            if self.categories['interface_bloat']:
                f.write("## 🖥️ Interface Bloat\n\n")
                f.write(f"**{len(self.categories['interface_bloat'])} files** - Large interface packages\n\n")
                f.write("*Consider if these should be separate repositories or external dependencies*\n\n")

            # Recommendations
            f.write("## 🎯 Recommendations\n\n")

            true_core_count = len(self.categories['true_core'])
            external_count = len(self.categories['external_packages'])

            if external_count > true_core_count:
                f.write("⚠️ **ALERT**: External packages outnumber your core AI files!\n\n")

            f.write("### Immediate Actions:\n")
            f.write("1. **Move External Packages** - Move video/audio generation to `external_dependencies/`\n")
            f.write("2. **Create requirements.txt** - List external packages as dependencies\n")
<<<<<<< HEAD
            f.write("3. **Clean Core Focus** - Keep only true AI logic in `Λ/core/`\n")
=======
            f.write("3. **Clean Core Focus** - Keep only true AI logic in `lukhas/core/`\n")
>>>>>>> jules/ecosystem-consolidation-2025
            f.write("4. **Separate Interfaces** - Consider moving large UI packages to separate repos\n\n")

            f.write("### Target Structure:\n")
            f.write("```\n")
            f.write("lukhas/\n")
            f.write("├── core/           # YOUR AI SYSTEM (~100-300 files)\n")
            f.write("├── interface/      # Lightweight interfaces\n")
            f.write("├── docs/          # Documentation\n")
            f.write("└── tests/         # Test suite\n")
            f.write("\n")
            f.write("external_dependencies/  # SEPARATE FROM CORE\n")
            f.write("├── video_generation/   # Open-Sora, etc.\n")
            f.write("├── audio_processing/   # TTS, Whisper, etc.\n")
            f.write("└── requirements.txt    # pip install these\n")
            f.write("```\n")

        return report_path

def main():
    workspace_path = "/Users/A_G_I/CodexGPT_Lukhas"

<<<<<<< HEAD
    print("🧠 Λ True Core Analysis")
=======
    print("🧠 lukhas True Core Analysis")
>>>>>>> jules/ecosystem-consolidation-2025
    print("=" * 50)

    analyzer = TrueCoreAnalyzer(workspace_path)
    analyzer.analyze_lukhas_structure()
    report_path = analyzer.generate_analysis_report()

    print(f"\n📊 ANALYSIS COMPLETE")
    print(f"📋 Report: {report_path}")

    true_core = len(analyzer.categories['true_core'])
    external = len(analyzer.categories['external_packages'])
    total = sum(len(files) for files in analyzer.categories.values())

    print(f"\n🎯 KEY FINDINGS:")
    print(f"   • Your True AI Core: {true_core} files ({true_core/total*100:.1f}%)")
    print(f"   • External Packages: {external} files ({external/total*100:.1f}%)")
    print(f"   • Total Files: {total}")

    if external > true_core:
        print(f"\n⚠️  ALERT: External packages are {external/true_core:.1f}x larger than your core!")
        print(f"   Most files are NOT your AI code - they're external libraries!")
    else:
        print(f"\n✅ Good: Your core AI is the primary component")

if __name__ == "__main__":
    main()


<<<<<<< HEAD
# Λ Systems 2025 www.lukhas.ai
=======
# lukhas Systems 2025 www.lukhas.ai
>>>>>>> jules/ecosystem-consolidation-2025
