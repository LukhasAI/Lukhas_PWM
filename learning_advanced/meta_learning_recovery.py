#!/usr/bin/env python3
"""
lukhasMetaLearningRecovery.py - Recovery tool for Meta Learning and Adaptive AI components

This tool specifically targets the Meta Learning and Adaptive AI systems
from the LUKHAS backup directory.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple


class MetaLearningRecovery:
    def __init__(self):
        self.workspace_root = Path("/Users/A_G_I/lukhas")
        self.recovery_log = []

        # Meta Learning backup location
        self.meta_learning_source = "/Users/A_G_I/Archive/old_git_repos/Library/Mobile Documents/com~apple~CloudDocs/LUKHAS_BACKUP_20250530/LUKHAS/CORE/Adaptative_AGI/Meta_Learning"

        # Target directories for different types of components
        self.target_mapping = {
            "meta_learning": "core/meta_learning",
            "adaptive_agi": "core/adaptive_agi",
            "learning": "brain/learning",
            "adaptation": "brain/adaptation",
            "intelligence": "core/intelligence",
            "cognitive": "brain/cognitive",
        }

    def explore_meta_learning_directory(self) -> Dict:
        """Explore the Meta Learning directory structure"""
        exploration_result = {
            "source_path": self.meta_learning_source,
            "exists": False,
            "total_files": 0,
            "python_files": 0,
            "directories": [],
            "files": [],
            "meta_learning_components": [],
            "adaptive_components": [],
            "learning_components": [],
        }

        if not os.path.exists(self.meta_learning_source):
            print(f"❌ Meta Learning source not found: {self.meta_learning_source}")
            return exploration_result

        exploration_result["exists"] = True

        print(f"🔍 Exploring Meta Learning directory: {self.meta_learning_source}")

        try:
            for root, dirs, files in os.walk(self.meta_learning_source):
                exploration_result["directories"].extend(
                    [os.path.join(root, d) for d in dirs]
                )

                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.meta_learning_source)

                    exploration_result["total_files"] += 1
                    exploration_result["files"].append(rel_path)

                    if file.endswith(".py"):
                        exploration_result["python_files"] += 1

                        # Categorize by component type
                        file_lower = file.lower()
                        if any(
                            term in file_lower
                            for term in ["meta_learning", "metalearning", "meta"]
                        ):
                            exploration_result["meta_learning_components"].append(
                                rel_path
                            )
                        elif any(
                            term in file_lower
                            for term in ["adaptive", "adaptation", "adapt"]
                        ):
                            exploration_result["adaptive_components"].append(rel_path)
                        elif any(
                            term in file_lower
                            for term in ["learning", "learn", "train"]
                        ):
                            exploration_result["learning_components"].append(rel_path)

                        print(f"   📄 Found: {rel_path}")

        except Exception as e:
            print(f"❌ Error exploring directory: {e}")

        print(f"📊 Exploration complete:")
        print(f"   Total files: {exploration_result['total_files']}")
        print(f"   Python files: {exploration_result['python_files']}")
        print(
            f"   Meta Learning components: {len(exploration_result['meta_learning_components'])}"
        )
        print(
            f"   Adaptive components: {len(exploration_result['adaptive_components'])}"
        )
        print(
            f"   Learning components: {len(exploration_result['learning_components'])}"
        )

        return exploration_result

    def convert_to_lambda_format(
        self, filename: str, content: str = None
    ) -> Tuple[str, str]:
        """Convert filename and content to lukhas format"""
        if filename.endswith(".py"):
            base_name = filename.replace(".py", "")

            # Convert to lukhasPascalCase if not already
            if not base_name.startswith("lukhas"):
                # Handle special cases for meta learning
                if "meta_learning" in base_name:
                    new_filename = f"lukhas{base_name.replace('meta_learning', 'MetaLearning').replace('_', '').title()}.py"
                elif "adaptive" in base_name:
                    new_filename = f"lukhas{base_name.replace('adaptive', 'Adaptive').replace('_', '').title()}.py"
                elif "learning" in base_name:
                    new_filename = f"lukhas{base_name.replace('learning', 'Learning').replace('_', '').title()}.py"
                else:
                    # General conversion
                    pascal_name = "".join(
                        word.capitalize() for word in base_name.split("_")
                    )
                    new_filename = f"lukhas{pascal_name}.py"
            else:
                new_filename = filename

            # Update content if provided
            new_content = content
            if content and "class " in content:
                # Update class names to lukhas format
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if line.strip().startswith("class ") and not "lukhas" in line:
                        class_def = line.strip()
                        if "(" in class_def:
                            class_name = class_def.split("(")[0].replace("class ", "")
                        else:
                            class_name = class_def.replace("class ", "").replace(
                                ":", ""
                            )

                        if not class_name.startswith("lukhas"):
                            # Special handling for meta learning classes
                            if (
                                "MetaLearning" in class_name
                                or "meta_learning" in class_name.lower()
                            ):
                                new_class_name = f"lukhasMetaLearning{class_name.replace('MetaLearning', '').replace('meta_learning', '').replace('_', '').title()}"
                            elif (
                                "Adaptive" in class_name
                                or "adaptive" in class_name.lower()
                            ):
                                new_class_name = f"lukhasAdaptive{class_name.replace('Adaptive', '').replace('adaptive', '').replace('_', '').title()}"
                            else:
                                pascal_name = "".join(
                                    word.capitalize() for word in class_name.split("_")
                                )
                                new_class_name = f"lukhas{pascal_name}"

                            lines[i] = line.replace(
                                f"class {class_name}", f"class {new_class_name}"
                            )

                new_content = "\n".join(lines)

            return new_filename, new_content

        return filename, content

    def determine_target_directory(self, file_path: str) -> str:
        """Determine the appropriate target directory for a component"""
        file_lower = file_path.lower()

        if "meta_learning" in file_lower or "metalearning" in file_lower:
            return self.target_mapping["meta_learning"]
        elif "adaptive" in file_lower or "adaptation" in file_lower:
            return self.target_mapping["adaptive_agi"]
        elif "learning" in file_lower or "learn" in file_lower:
            return self.target_mapping["learning"]
        elif "cognitive" in file_lower or "cognition" in file_lower:
            return self.target_mapping["cognitive"]
        elif "intelligence" in file_lower or "intelligent" in file_lower:
            return self.target_mapping["intelligence"]
        else:
            return self.target_mapping["meta_learning"]  # Default

    def recover_meta_learning_components(self, exploration_result: Dict) -> Dict:
        """Recover all Meta Learning components"""
        recovery_result = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": self.meta_learning_source,
            "components_processed": 0,
            "components_successful": 0,
            "total_files_recovered": 0,
            "files_recovered": [],
            "errors": [],
        }

        if not exploration_result["exists"]:
            recovery_result["errors"].append("Source directory does not exist")
            return recovery_result

        print("\n🚀 Starting Meta Learning Component Recovery...")

        for file_rel_path in exploration_result["files"]:
            if not file_rel_path.endswith(".py"):
                continue

            source_file_path = os.path.join(self.meta_learning_source, file_rel_path)
            recovery_result["components_processed"] += 1

            try:
                # Read content
                with open(
                    source_file_path, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    content = f.read()

                # Convert to lukhas format
                filename = os.path.basename(file_rel_path)
                new_filename, new_content = self.convert_to_lambda_format(
                    filename, content
                )

                # Determine target directory
                target_dir_name = self.determine_target_directory(file_rel_path)
                target_dir = self.workspace_root / target_dir_name

                # Create target directory
                target_dir.mkdir(parents=True, exist_ok=True)

                # Handle subdirectories
                subdir = os.path.dirname(file_rel_path)
                if subdir:
                    final_target_dir = target_dir / subdir
                    final_target_dir.mkdir(parents=True, exist_ok=True)
                else:
                    final_target_dir = target_dir

                # Write to target
                target_path = final_target_dir / new_filename
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

                recovery_result["files_recovered"].append(
                    {
                        "original": file_rel_path,
                        "new_filename": new_filename,
                        "target_path": str(target_path),
                        "target_category": target_dir_name,
                        "size": len(content),
                    }
                )

                recovery_result["components_successful"] += 1
                recovery_result["total_files_recovered"] += 1

                print(f"✅ Recovered: {file_rel_path} → {new_filename}")
                print(f"   Target: {target_path}")

            except Exception as e:
                error_msg = f"Failed to recover {file_rel_path}: {str(e)}"
                recovery_result["errors"].append(error_msg)
                print(f"❌ {error_msg}")

        return recovery_result

    def execute_recovery(self) -> Dict:
        """Execute complete Meta Learning recovery operation"""
        print("🚀 Starting Meta Learning & Adaptive AI Component Recovery...")
        print("=" * 70)

        # Explore the directory
        exploration_result = self.explore_meta_learning_directory()

        if not exploration_result["exists"]:
            print("❌ Cannot proceed - source directory not found")
            return {"error": "Source directory not found"}

        # Recover components
        recovery_result = self.recover_meta_learning_components(exploration_result)

        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = (
            self.workspace_root / f"lukhasMetaLearning_Recovery_Report_{timestamp}.json"
        )

        complete_report = {
            "exploration": exploration_result,
            "recovery": recovery_result,
        }

        with open(report_file, "w") as f:
            json.dump(complete_report, f, indent=2, default=str)

        # Display summary
        print(f"\n📈 META LEARNING RECOVERY SUMMARY:")
        print(f"   📁 Source: {self.meta_learning_source}")
        print(f"   📦 Components processed: {recovery_result['components_processed']}")
        print(
            f"   ✅ Components successful: {recovery_result['components_successful']}"
        )
        print(
            f"   📄 Total files recovered: {recovery_result['total_files_recovered']}"
        )
        print(f"   ❌ Errors: {len(recovery_result['errors'])}")
        print(f"   💾 Report saved: {report_file}")

        # Show target distribution
        if recovery_result["files_recovered"]:
            target_counts = {}
            for file_info in recovery_result["files_recovered"]:
                category = file_info["target_category"]
                target_counts[category] = target_counts.get(category, 0) + 1

            print(f"\n📊 COMPONENTS BY CATEGORY:")
            for category, count in target_counts.items():
                print(f"   • {category}: {count} files")

        return complete_report


def main():
    recovery = lukhasMetaLearningRecovery()
    results = recovery.execute_recovery()

    if "error" not in results and results["recovery"]["components_successful"] > 0:
        print(
            f"\n🎉 Successfully recovered {results['recovery']['total_files_recovered']} Meta Learning components!"
        )
        print(
            "   lukhasI system enhanced with advanced learning and adaptation capabilities!"
        )
    else:
        print(
            "\n⚠️  No Meta Learning components were recovered. Check paths and permissions."
        )


if __name__ == "__main__":
    main()
