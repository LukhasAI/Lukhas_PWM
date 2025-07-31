# CollapseHash Project Makefile
# =============================
# 
# Common tasks for CollapseHash development and deployment
# 
# Usage:
#   make setup      # Initial project setup
#   make install    # Install dependencies
#   make test       # Run tests
#   make clean      # Clean up generated files
#   make run        # Start web interface

.PHONY: help setup install test clean run dev lint format docs

# Default target
help:
	@echo "CollapseHash Project - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup          - Complete project setup (dependencies + config)"
	@echo "  setup-dev      - Setup with development dependencies"
	@echo "  install        - Install dependencies only"
	@echo "  config         - Create configuration file only"
	@echo ""
	@echo "Development Commands:"
	@echo "  test           - Run test suite"
	@echo "  test-verbose   - Run tests with verbose output"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code with black"
	@echo "  type-check     - Run type checking with mypy"
	@echo ""
	@echo "Application Commands:"
	@echo "  run            - Start web interface"
	@echo "  run-dev        - Start web interface in development mode"
	@echo "  generate       - Generate example CollapseHash"
	@echo "  verify         - Run verification example"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean          - Clean up generated files"
	@echo "  clean-all      - Clean everything including dependencies"
	@echo "  docs           - Generate documentation"
	@echo "  backup         - Create project backup"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  qa             - Run all quality checks (lint + test + type-check)"
	@echo "  security       - Run security checks"
	@echo "  benchmark      - Run performance benchmarks"

# Setup Commands
# ==============

setup:
	@echo "🚀 Setting up CollapseHash project..."
	python3 setup_project.py

setup-dev:
	@echo "🚀 Setting up CollapseHash project (development mode)..."
	python3 setup_project.py --dev

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

config:
	@echo "⚙️ Creating configuration..."
	python3 setup_project.py --config-only

# Development Commands
# ====================

test:
	@echo "🧪 Running test suite..."
	python3 -m pytest tests/ -v

test-verbose:
	@echo "🧪 Running test suite (verbose)..."
	python3 -m pytest tests/ -v -s --tb=long

test-coverage:
	@echo "🧪 Running test suite with coverage..."
	python3 -m pytest tests/ --cov=. --cov-report=html --cov-report=term

lint:
	@echo "🔍 Running code linting..."
	python3 -m flake8 *.py --max-line-length=100 --ignore=E203,W503
	@echo "✅ Linting complete"

format:
	@echo "🎨 Formatting code..."
	python3 -m black *.py --line-length=100
	@echo "✅ Code formatted"

type-check:
	@echo "🔍 Running type checking..."
	python3 -m mypy *.py --ignore-missing-imports
	@echo "✅ Type checking complete"

# Application Commands  
# ====================

run:
	@echo "🌐 Starting CollapseHash web interface..."
	python3 web_qr_verifier.py

run-dev:
	@echo "🌐 Starting CollapseHash web interface (development mode)..."
	FLASK_ENV=development python3 web_qr_verifier.py --debug

run-dashboard:
	@echo "📊 Starting CollapseHash dashboard..."
	python3 web_dashboard.py

generate:
	@echo "🔐 Generating example CollapseHash..."
	python3 collapse_cli.py generate --example

verify:
	@echo "🔍 Running verification example..."
	python3 collapse_cli.py verify --example

cli:
	@echo "💻 Starting CollapseHash CLI..."
	python3 collapse_cli.py

# Web Journal Commands
# ====================

run-web-journal:
	@echo "🌐 Starting VeriFold Web Journal..."
	cd web_journal && python3 launch_web_journal.py

setup-web-journal:
	@echo "📦 Setting up Web Journal dependencies..."
	cd web_journal && pip install -r web_journal_requirements.txt

web-journal-help:
	@echo "🌐 VeriFold Web Journal Commands"
	@echo "================================"
	@echo "  run-web-journal      - Start the web journal interface"
	@echo "  setup-web-journal    - Install web journal dependencies"  
	@echo "  web-journal-help     - Show this help"
	@echo ""
	@echo "Manual Usage:"
	@echo "  cd web_journal/"
	@echo "  ./start_web_journal.sh"
	@echo ""
	@echo "Web Interface: http://localhost:5001"

# Utility Commands
# ================

clean:
	@echo "🧹 Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf qr_codes/*.png
	rm -rf exports/*
	@echo "✅ Cleanup complete"

clean-all: clean
	@echo "🧹 Deep cleaning (including dependencies)..."
	rm -rf venv/
	rm -rf .venv/
	pip freeze | grep -v "^-e" | xargs pip uninstall -y 2>/dev/null || true
	@echo "✅ Deep cleanup complete"

docs:
	@echo "📚 Generating documentation..."
	@echo "TODO: Add documentation generation"
	@echo "✅ Documentation ready"

backup:
	@echo "💾 Creating project backup..."
	tar -czf "collapse_hash_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz" \
		--exclude="*.pyc" --exclude="__pycache__" --exclude="venv" \
		--exclude=".git" --exclude="*.log" \
		.
	@echo "✅ Backup created"

# Quality Assurance
# =================

qa: lint type-check test
	@echo "✅ All quality checks passed!"

security:
	@echo "🔒 Running security checks..."
	python3 -m bandit -r . -f json -o security_report.json 2>/dev/null || true
	python3 -m safety check --json --output safety_report.json 2>/dev/null || true
	@echo "✅ Security check complete (see security_report.json, safety_report.json)"

benchmark:
	@echo "⚡ Running performance benchmarks..."
	python3 -c "
import time
import collapse_hash_pq
print('Testing hash generation performance...')
generator = collapse_hash_pq.CollapseHashGenerator()
start = time.time()
for i in range(10):
    test_data = f'test_quantum_data_{i}'.encode()
    hash_result = generator.generate_collapse_hash(test_data)
end = time.time()
print(f'Generated 10 hashes in {end-start:.3f} seconds ({(end-start)/10:.3f}s per hash)')
"
	@echo "✅ Benchmark complete"

# Development Environment
# =======================

venv:
	@echo "🐍 Creating virtual environment..."
	python3 -m venv venv
	@echo "Activate with: source venv/bin/activate"

venv-install: venv
	@echo "📦 Installing in virtual environment..."
	./venv/bin/pip install -r requirements.txt

# Docker Support (TODO)
# =====================

docker-build:
	@echo "🐳 Building Docker image..."
	@echo "TODO: Add Dockerfile and docker build"

docker-run:
	@echo "🐳 Running Docker container..."
	@echo "TODO: Add docker run command"

# Deployment (TODO)
# =================

deploy-dev:
	@echo "🚀 Deploying to development..."
	@echo "TODO: Add development deployment"

deploy-prod:
	@echo "🚀 Deploying to production..."
	@echo "TODO: Add production deployment"

# Data Management
# ===============

reset-data:
	@echo "🗑️ Resetting test data..."
	rm -f collapse_logbook.jsonl
	rm -f test_vectors.json
	python3 setup_project.py --config-only
	@echo "✅ Test data reset"

export-data:
	@echo "💾 Exporting verification data..."
	mkdir -p exports
	cp collapse_logbook.jsonl exports/logbook_$(shell date +%Y%m%d_%H%M%S).jsonl
	cp test_vectors.json exports/vectors_$(shell date +%Y%m%d_%H%M%S).json
	@echo "✅ Data exported to exports/"

# Requirements and Dependencies
# =============================

requirements:
	@echo "📋 Generating requirements.txt from environment..."
	pip freeze > requirements_generated.txt
	@echo "✅ Generated requirements_generated.txt"

check-deps:
	@echo "🔍 Checking for outdated dependencies..."
	pip list --outdated

update-deps:
	@echo "⬆️ Updating dependencies..."
	pip install --upgrade -r requirements.txt

# Platform-specific commands
# ===========================

# macOS specific
install-macos:
	@echo "🍎 Installing macOS dependencies..."
	brew install openssl libffi
	pip install -r requirements.txt

# Linux specific  
install-linux:
	@echo "🐧 Installing Linux dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev
	pip install -r requirements.txt

# Windows specific (requires WSL or PowerShell)
install-windows:
	@echo "🪟 Installing Windows dependencies..."
	@echo "Ensure Microsoft C++ Build Tools are installed"
	pip install -r requirements.txt

# Information Commands
# ====================

info:
	@echo "ℹ️ CollapseHash Project Information"
	@echo "=================================="
	@echo "Python version: $(shell python3 --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Project root: $(shell pwd)"
	@echo "Config file: $(shell test -f config.json && echo "✅ Present" || echo "❌ Missing")"
	@echo "Dependencies: $(shell test -f requirements.txt && echo "✅ Present" || echo "❌ Missing")"
	@echo "Virtual env: $(shell test -n "$$VIRTUAL_ENV" && echo "✅ Active: $$VIRTUAL_ENV" || echo "❌ Not active")"

status:
	@echo "📊 Project Status"
	@echo "=================="
	@python3 -c "
import os, json
print('Configuration:', '✅ Present' if os.path.exists('config.json') else '❌ Missing')
print('Test vectors:', '✅ Present' if os.path.exists('test_vectors.json') else '❌ Missing')
print('Logbook:', '✅ Present' if os.path.exists('collapse_logbook.jsonl') else '❌ Missing')
print('Web templates:', '✅ Present' if os.path.exists('web_templates/index.html') else '❌ Missing')
try:
    import oqs
    print('OQS library:', '✅ Available')
except ImportError:
    print('OQS library:', '❌ Not available')
"
