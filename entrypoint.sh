#!/bin/bash
set -e
if [ -n "$API_KEY" ]; then
    cat > api-keys.json <<EOF
[
    {"key": "$API_KEY", "description": "Environment variable"}
]
EOF
else
    echo "API_KEY not set, using existing api-keys.json or server will generate"
fi
exec python server.py "$@"
