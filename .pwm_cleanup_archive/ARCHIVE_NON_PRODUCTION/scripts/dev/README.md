# Development Scripts - Documentation & Maintenance Utilities

This directory contains development-focused utilities for documentation generation, README management, and development workflow support.

## Scripts Overview

### backup_all_readmes.py
**Purpose:** Comprehensive README backup and regeneration system

**Features:**
- Automated discovery of all README files in the project
- Creates timestamped backups before modifications
- Regenerates module documentation with consistent formatting
- Integrates with git version control for tracking changes
- Supports batch processing of multiple README files

**Usage:**
```bash
cd scripts/dev
python3 backup_all_readmes.py
```

**Output:**
- Backup files in `backup_readmes_YYYYMMDD_HHMMSS/`
- Regenerated README files throughout the project
- Summary report of changes made

### extract_enumerated_readmes.py
**Purpose:** Documentation packaging for website deployment and distribution

**Features:**
- Enumerates all README files with sequential numbering
- Creates compressed documentation packages
- Generates navigation index files
- Prepares website-ready documentation structure
- Calculates package size and file counts

**Usage:**
```bash
cd scripts/dev
python3 extract_enumerated_readmes.py
```

**Output:**
- `lukhas_readmes_enumerated/` directory with numbered files
- `000_INDEX.txt` with navigation structure
- Compressed archive for distribution
- Package statistics and metadata

## Integration with LUKHAS

These scripts are designed to work seamlessly with the LUKHAS AGI project structure:

### Module Awareness
- Understands LUKHAS module hierarchy
- Respects module boundaries and organization
- Maintains documentation consistency across modules

### Configuration Integration
- Uses LUKHAS configuration standards
- Follows project naming conventions
- Integrates with existing development workflows

### Version Control
- Git integration for tracking documentation changes
- Automated commit message generation
- Branch-aware operations for safe modifications

## Development Workflow

### Documentation Updates
1. **Backup existing documentation**
   ```bash
   python3 backup_all_readmes.py
   ```

2. **Make your documentation changes**
   - Edit README files as needed
   - Add new module documentation
   - Update architectural descriptions

3. **Package for deployment**
   ```bash
   python3 extract_enumerated_readmes.py
   ```

4. **Validate results**
   - Review generated packages
   - Check navigation index
   - Verify all modules included

### Best Practices

1. **Always backup** before making bulk changes
2. **Review generated content** for accuracy and consistency
3. **Test navigation** in generated packages
4. **Validate links** and references in documentation
5. **Commit changes** with descriptive messages

## Error Handling

Both scripts include comprehensive error handling:

- **File permission checks** before modifications
- **Backup validation** to ensure data safety
- **Progress reporting** for long-running operations
- **Rollback capabilities** for failed operations
- **Detailed logging** for troubleshooting

## Dependencies

These scripts require:
- Python 3.8+
- Git (for version control integration)
- Write permissions in the project directory
- Standard library modules (no external dependencies)

## Maintenance

These development scripts are maintained as part of the LUKHAS project:

- **Regular testing** with each release cycle
- **Compatibility updates** for new module structures
- **Feature enhancements** based on developer feedback
- **Performance optimizations** for large documentation sets

---

*Development utilities for maintaining high-quality documentation in the LUKHAS AGI ecosystem.*
