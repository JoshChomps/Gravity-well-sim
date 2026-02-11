import sys
IS_WEB = sys.platform == "emscripten"

# Constants
G = 6.67430e-2  # Scaled gravitational constant for simulation
GRID_EXTENT = 50
TRAIL_LENGTH = 200
MAX_GRID_RESOLUTION = 300 if IS_WEB else 500
GRID_RESOLUTION = 100 if IS_WEB else 250
