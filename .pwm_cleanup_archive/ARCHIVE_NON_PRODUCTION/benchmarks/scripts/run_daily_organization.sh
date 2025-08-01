#!/bin/bash
# Manual Daily Organization Runner
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BENCHMARKS_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🗂️ Running Daily Benchmark Organization..."
cd "$BENCHMARKS_ROOT"
python3 "$SCRIPT_DIR/daily_benchmark_organizer.py" --organize
