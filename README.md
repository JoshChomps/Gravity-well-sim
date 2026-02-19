# 3D Gravity Simulator

A physics simulation visualizing gravitational forces and spacetime curvature.

## How to Run (Choose Option A or B)

You do **NOT** need to do both. Just pick the one that is easiest for you.

### Option A: VS Code (Recommended)
1.  Download and install [Python 3.12+](https://www.python.org/downloads/).
2.  Open a terminal/command prompt in this folder.
3.  Install dependencies:
    ```bash
    pip install -r requirements-dev.txt
    ```
4.  Run the simulation:
    ```bash
    python main.py
    ```

*This sets VS Code to always use the correct isolated environment, avoiding conflicts with Miniconda.*

### Option B: Double-Click Script
1.  Navigate to this folder in File Explorer.
2.  Double-click **`run.bat`**.

*This script automatically finds and uses the correct environment for you.*

---

## Web Deployment
To build for the web:
1.  Open a terminal in this folder.
2.  Run `pygbag .`
3.  Go to `localhost:8000` in your browser.
