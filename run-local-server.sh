
#!/bin/bash
# run-local-server.sh
# Serves drone_swarm_3d.html locally so the browser can reach the EC2 API.
# Run this on your LOCAL machine (Mac/Linux).
#
# Usage:
#   chmod +x run-local-server.sh
#   ./run-local-server.sh
#
# Then open: http://localhost:3000/drone_swarm_3d.html
 
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=3000
 
echo ""
echo "================================================"
echo "  UAV Swarm — Local HTML Server"
echo "  Serving: $SCRIPT_DIR"
echo "  Open:    http://localhost:$PORT/drone_swarm_3d.html"
echo "  Stop:    Ctrl+C"
echo "================================================"
echo ""
 
cd "$SCRIPT_DIR"
python3 -m http.server $PORT