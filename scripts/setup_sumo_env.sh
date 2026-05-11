# Setup SUMO environment variables (source this file)
export SUMO_HOME="/Users/xuzheng/Library/Python/3.9/lib/python/site-packages/sumo"
export PATH="$SUMO_HOME/bin:$HOME/Library/Python/3.9/bin:$PATH"

# Add traci to Python path
if [[ ":$PYTHONPATH:" != *":$SUMO_HOME/tools:"* ]]; then
    export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
fi

echo "SUMO_HOME=$SUMO_HOME"
echo "SUMO version: $(sumo --version 2>&1 | head -1)"
