#!/bin/bash
# Create a backup before major consolidation

BACKUP_DIR="BACKUP_BEFORE_CONSOLIDATION_$(date +%Y%m%d_%H%M%S)"

echo "Creating backup in $BACKUP_DIR..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Copy key directories that will be affected
cp -r bio "$BACKUP_DIR/"
cp -r core "$BACKUP_DIR/"
cp -r lukhas_personality "$BACKUP_DIR/"
cp -r features "$BACKUP_DIR/"
cp -r quantum "$BACKUP_DIR/"

# Copy Python files in root
cp *.py "$BACKUP_DIR/" 2>/dev/null

echo "Backup created in $BACKUP_DIR"
echo "Total size: $(du -sh "$BACKUP_DIR" | cut -f1)"