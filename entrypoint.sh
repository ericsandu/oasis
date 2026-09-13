#!/bin/bash
set -e

echo "Starting Gorse in the background..."
mkdir -p /app/data

# Boot Gorse in a single process (master, server, worker combined)
gorse-in-one -c /app/gorse_config.toml > /app/data/gorse.log 2>&1 &

echo "Gorse processes started. Proceeding with OASIS execution..."

# Execute the main command (e.g. python script) in the foreground
exec "$@"
