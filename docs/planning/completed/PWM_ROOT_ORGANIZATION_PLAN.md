# 🗂️ PWM Root Directory Organization Plan

## 📋 Current Root Directory Analysis

### Files to Keep in Root (Essential)
- `CLAUDE.md` ✅ (Required by user)
- `README.md` ✅ (Primary documentation)
- `LICENSE` ✅ (Legal requirement)
- `requirements.txt` ✅ (Core dependencies)
- `package.json` ✅ (Node.js dependencies)
- `lukhas_pwm_config.yaml` ✅ (Core configuration)
- `.gitignore`, `.env.example`, `.env.template` ✅ (Git/environment)
- `Lukhas_PWM.code-workspace` ✅ (VS Code workspace)
- Core data files: `helix_memory_store.jsonl`, `lukhas_memory_folds.db` ✅

### Files to Organize into Subdirectories

#### 📊 Analysis Tools → `tools/analysis/`
- `PWM_FUNCTIONAL_ANALYSIS.py`
- `PWM_OPERATIONAL_SUMMARY.py`
- `PWM_WORKSPACE_STATUS_ANALYSIS.py`
- `PWM_MISSING_COMPONENTS_ANALYZER.py`
- `PWM_SECURITY_COMPLIANCE_GAP_ANALYSIS.py`
- `pwm_deep_analysis.py`

#### 📈 Analysis Reports → `docs/reports/`
- `PWM_FUNCTIONAL_ANALYSIS_REPORT.json`
- `PWM_CONNECTIVITY_ANALYSIS.json`
- `PWM_WORKSPACE_STATUS_REPORT.json`
- `PWM_SECURITY_COMPLIANCE_GAP_ANALYSIS.json`

#### 📋 Planning Documents → `docs/planning/completed/`
- `PWM_CHERRY_PICK_PLAN.md`
- `PWM_COMPREHENSIVE_MISSING_COMPONENTS_ANALYSIS.md`
- `PWM_PHASE3_ADVANCED_TOOLS_ANALYSIS.md`
- `PWM_SECURITY_COMPLIANCE_EXPANSION_PLAN.md`

#### 📊 Status Reports → `docs/reports/status/`
- `PWM_CURRENT_STATUS_REPORT.md`
- `PWM_OPERATIONAL_STATUS_REPORT.md`

#### 📚 Archive Documentation → `docs/archive/`
- `README_CONSOLIDATED.md`

#### 🧪 Test Files → `tests/`
- `test_governance.py`
- `test_enhanced_governance.py`
- `test_comprehensive_governance.py`

#### 📦 Build/Container → `deployments/`
- `requirements-container.txt`

## 🎯 Organization Strategy

### Phase 1: Create Directory Structure
```bash
mkdir -p docs/reports/status
mkdir -p docs/reports/analysis
mkdir -p docs/planning/completed
mkdir -p docs/archive
mkdir -p tools/analysis
mkdir -p tools/scripts
mkdir -p tests/governance
mkdir -p deployments/containers
```

### Phase 2: Move Analysis Tools
```bash
mv PWM_*.py tools/analysis/
mv pwm_deep_analysis.py tools/analysis/
```

### Phase 3: Move Reports & Documentation
```bash
mv PWM_*_REPORT.json docs/reports/analysis/
mv PWM_*_STATUS_REPORT.md docs/reports/status/
mv PWM_*_ANALYSIS.json docs/reports/analysis/
```

### Phase 4: Archive Completed Planning
```bash
mv PWM_*_PLAN.md docs/planning/completed/
mv PWM_COMPREHENSIVE_MISSING_COMPONENTS_ANALYSIS.md docs/planning/completed/
mv PWM_PHASE3_ADVANCED_TOOLS_ANALYSIS.md docs/planning/completed/
```

### Phase 5: Move Tests
```bash
mv test_*.py tests/governance/
```

### Phase 6: Move Container Files
```bash
mv requirements-container.txt deployments/containers/
```

### Phase 7: Archive Documentation
```bash
mv README_CONSOLIDATED.md docs/archive/
```

## 📁 Final Clean Root Structure

```
🧠 LUKHAS_PWM/
├── 📄 CLAUDE.md                    # User-required root file
├── 📄 README.md                    # Primary documentation
├── 📄 LICENSE                      # Legal
├── 📄 requirements.txt             # Core dependencies
├── 📄 package.json                 # Node.js dependencies
├── 📄 lukhas_pwm_config.yaml       # Core configuration
├── 📄 .gitignore, .env.example     # Environment files
├── 📄 Lukhas_PWM.code-workspace    # VS Code workspace
├── 📄 helix_memory_store.jsonl     # Core data
├── 📄 lukhas_memory_folds.db       # Core database
├── 📁 docs/                        # All documentation
│   ├── 📁 reports/                 # Analysis reports
│   │   ├── 📁 status/              # Status reports
│   │   └── 📁 analysis/            # Analysis results
│   ├── 📁 planning/                # Planning documents
│   │   └── 📁 completed/           # Completed phases
│   └── 📁 archive/                 # Archived documentation
├── 📁 tools/                       # Analysis & utility tools
│   ├── 📁 analysis/                # Analysis scripts
│   └── 📁 scripts/                 # Utility scripts
├── 📁 tests/                       # Test suites
│   └── 📁 governance/              # Governance tests
├── 📁 deployments/                 # Deployment configs
│   └── 📁 containers/              # Container configs
└── [Existing LUKHAS directories...]
```

## ⚠️ Phase Completion Tracking

### ✅ Completed Phases (Ready for Archive)
1. **Phase 1: Security & Compliance** - Cherry-pick completed
2. **Phase 2: Advanced Learning Systems** - Integration completed
3. **Phase 3: AI Compliance Testing** - Tools integrated

### 📋 Phase Completion Markers
Each completed phase will have:
- Phase completion marker in `docs/planning/completed/`
- Analysis results in `docs/reports/analysis/`
- Status report in `docs/reports/status/`

## 🚀 Implementation Ready

Execute this reorganization to create a clean, professional root directory structure while preserving all PWM work and maintaining LUKHAS functionality.
