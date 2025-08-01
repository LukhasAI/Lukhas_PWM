#!/bin/bash
"""
Daily Benchmark Automation Setup
Sets up cron job for automatic daily benchmark organization
"""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ORGANIZER_SCRIPT="$SCRIPT_DIR/daily_benchmark_organizer.py"
BENCHMARKS_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧 Setting up Daily Benchmark Automation"
echo "================================================"

# Make sure organizer script is executable
chmod +x "$ORGANIZER_SCRIPT"

# Create cron job entry (runs at 11:59 PM daily)
CRON_ENTRY="59 23 * * * cd $BENCHMARKS_ROOT && python3 $ORGANIZER_SCRIPT --organize >> $BENCHMARKS_ROOT/logs/daily_organization.log 2>&1"

echo "📅 Recommended cron job entry:"
echo "$CRON_ENTRY"
echo ""
echo "To install this cron job:"
echo "1. Run: crontab -e"
echo "2. Add the above line"
echo "3. Save and exit"
echo ""

# Create logs directory
mkdir -p "$BENCHMARKS_ROOT/logs"

# Create a manual runner script
cat > "$SCRIPT_DIR/run_daily_organization.sh" << 'EOF'
#!/bin/bash
# Manual Daily Organization Runner
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BENCHMARKS_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🗂️ Running Daily Benchmark Organization..."
cd "$BENCHMARKS_ROOT"
python3 "$SCRIPT_DIR/daily_benchmark_organizer.py" --organize
EOF

chmod +x "$SCRIPT_DIR/run_daily_organization.sh"

echo "✅ Setup complete!"
echo "   - Daily organizer script: $ORGANIZER_SCRIPT"
echo "   - Manual runner: $SCRIPT_DIR/run_daily_organization.sh"
echo "   - Logs directory: $BENCHMARKS_ROOT/logs"
echo ""
echo "🚀 To run organization manually:"
echo "   $SCRIPT_DIR/run_daily_organization.sh"