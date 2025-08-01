#!/bin/bash
set -e

echo "🚀 Starting LUKHAS AI API Server..."

# Wait for database if DATABASE_URL is set
if [ -n "$DATABASE_URL" ]; then
    echo "⏳ Waiting for database connection..."
    python -c "
import psycopg2
import os
import time
import sys
from urllib.parse import urlparse

db_url = os.environ.get('DATABASE_URL')
if db_url:
    parsed = urlparse(db_url)
    for i in range(30):
        try:
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:]
            )
            conn.close()
            print('✅ Database is ready!')
            break
        except psycopg2.OperationalError:
            print(f'⏳ Database not ready, retrying... ({i+1}/30)')
            time.sleep(2)
    else:
        print('❌ Database connection failed after 30 attempts')
        sys.exit(1)
"
fi

# Initialize LUKHAS if needed
echo "🧠 Initializing LUKHAS AI System..."
python -c "
try:
    from lukhas import __version__
    print(f'✅ LUKHAS AI Version: {__version__}')
except ImportError:
    print('⚠️ LUKHAS version not found, continuing...')
"

# Run the command passed to the container
echo "🔥 Executing: $@"
exec "$@"