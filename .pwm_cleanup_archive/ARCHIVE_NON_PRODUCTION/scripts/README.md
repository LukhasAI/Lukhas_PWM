# 🔧 Scripts - Project Maintenance & Development Utilities

This directory contains project-level scripts for maintenance, development, and deployment operations. These scripts operate at the repository level and provide essential functionality for managing the LUKHAS AGI system.

## Directory Structure

```
scripts/
├── dev/                    # Development utilities
│   ├── backup_all_readmes.py      # README backup and regeneration
│   ├── extract_enumerated_readmes.py  # Documentation packaging
│   └── README.md               # Development scripts documentation
├── setup/                  # Installation and setup scripts
├── testing/                # Test execution and validation
├── analysis/               # Code analysis and metrics
├── organization/           # Repository organization tools
├── docker-build.sh         # Docker container build script
├── docker-entrypoint.sh    # Docker container entry point
├── release.py              # Release management automation
├── run_migrated_tests.py   # Test migration and execution
├── test_api.py            # API testing utilities
└── README.md              # This documentation file
```

## Usage Guidelines

All scripts should be run from the project root directory unless otherwise specified.

### Development Scripts (`dev/`)
```bash
# Generate documentation packages
cd scripts/dev && python3 extract_enumerated_readmes.py

# Backup and regenerate READMEs
cd scripts/dev && python3 backup_all_readmes.py
```

### Container Operations
```bash
# Build Docker container
./scripts/docker-build.sh

# Run with Docker entry point
./scripts/docker-entrypoint.sh
```

### Release Management
```bash
# Prepare new release
python3 scripts/release.py --version 0.3.0
```
