#!/bin/bash
set -e  # exit if any command fails

# Parse command line arguments
MODE=${1:-"debug"}  # Default to debug mode if no argument provided

# Validate mode argument
if [[ "$MODE" != "debug" && "$MODE" != "run" ]]; then
    echo "Error: Invalid mode. Use 'debug' or 'run'"
    echo "Usage: $0 [debug|run]"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment variables from .env file if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading environment variables from $SCRIPT_DIR/.env"
    set -a  # automatically export all variables
    source "$SCRIPT_DIR/.env"
    set +a
else
    echo "Warning: .env file not found in $SCRIPT_DIR"
fi

echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "OBJ_PATH: $OBJ_PATH"
echo "CAD_PATH: $CAD_PATH"
echo "CAM_PATH: $CAM_PATH"

# check if /templates exists under OUTPUT_DIR, if not run the following blenderproc command to generate it
if [ ! -d "$OUTPUT_DIR/templates" ]; then
    echo "Templates folder does not exist. Generating templates..."
    blenderproc run ./Render/render_obj_templates.py --output_dir $OUTPUT_DIR --obj_path $OBJ_PATH --ply_path $CAD_PATH
fi

# Run server based on mode
if [ "$MODE" == "debug" ]; then
    echo "Starting server in DEBUG mode..."
    fastapi dev start_server.py
else
    echo "Starting server in PRODUCTION mode..."
    fastapi run start_server.py
fi