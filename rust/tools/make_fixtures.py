"""Record reference results from the Python simulators for the Rust tests.

Runs the real code in ../../simulator.py (the original sheet version) and,
for the volume grid, volume_reference/simulator.py (the 3D-grid version's
simulator and config; it shares models.py and ui.py with the sheet version),
without opening a window:
the simulator is built with __new__ like tests/test_physics.py does, and the
OpenGL calls inside the drawing methods are swapped for stubs so the numpy
maths runs and its output can be captured.

Writes ../tests/fixtures/python_reference.json. Run from anywhere with the
repo's venv:

    ..\\.venv\\Scripts\\python.exe tools\\make_fixtures.py
"""
import importlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHEET_REPO = HERE.parent.parent
VOLUME_DIR = HERE / "volume_reference"
OUT = HERE.parent / "tests" / "fixtures" / "python_reference.json"

DT = 1 / 60
CHECKPOINTS = [1, 10, 60, 300, 600, 1200]


def load_simulator(*dirs):
    """Import simulator.py fresh from the first of `dirs` that has each module
    (both versions use the same module names)."""
    for name in ("simulator", "config", "models", "ui"):
        sys.modules.pop(name, None)
    paths = [str(d) for d in dirs]
    sys.path[:0] = paths
    try:
        module = importlib.import_module("simulator")
    finally:
        for p in paths:
            sys.path.remove(p)
    return module


class _Editor:
    fields = {}

    def set_planet(self, planet):
        pass


def headless(module):
    sim = module.GravitySimulator.__new__(module.GravitySimulator)
    sim.planets = []
    sim.planet_counter = 0
    sim.paused = False
    sim.time_scale = 1.0
    sim.selected_planet = None
    sim.ui_dirty = False
    sim.planet_editor = _Editor()
    return sim


def stub_gl(module, captured):
    """Replace the GL names simulator.py imported with no-ops; glVertex3f records."""
    def noop(*args, **kwargs):
        return 1

    for name in ("glGenBuffers", "glBindBuffer", "glBufferData", "glBufferSubData", "glDisable",
                 "glEnable", "glLineWidth", "glColor4f", "glEnableClientState", "glVertexPointer",
                 "glDrawElements", "glDisableClientState", "glBegin", "glEnd"):
        setattr(module, name, noop)
    module.glVertex3f = lambda x, y, z: captured.append([float(x), float(y), float(z)])


def body_state(p):
    return {
        "id": p.id, "name": p.name, "mass": float(p.mass), "radius": float(p.radius),
        "position": [float(v) for v in p.position], "velocity": [float(v) for v in p.velocity],
        "acceleration": [float(v) for v in p.acceleration], "color": [float(c) for c in p.color],
    }


def run_scenario(module, preset, seed):
    random.seed(seed)
    sim = headless(module)
    sim.load_preset(preset)
    start = [body_state(p) for p in sim.planets]
    checkpoints = {}
    for step in range(1, CHECKPOINTS[-1] + 1):
        sim.update_physics(DT)
        if step in CHECKPOINTS:
            checkpoints[str(step)] = [body_state(p) for p in sim.planets]
    return {"preset": preset, "seed": seed, "dt": DT, "start": start, "checkpoints": checkpoints}


def field_bodies(module):
    """Preset 1 plus bodies placed to hit the volume grid's clamp and push-out code."""
    sim = headless(module)
    sim.load_preset(1)
    # The volume lattice point at (30, 0, 0) starts inside this body's 1.1r,
    # so its own pull is zero; the Sun then drags the point deeper into it
    # and the push-out pass has to move it back onto the 1.2r shell.
    sim.planets.append(sim.create_planet([26.0, 0.3, 0.2], [0, 0, 0], 20, 5.0, (1, 1, 1), "Near vertex"))
    sim.planets.append(sim.create_planet([-29.0, 31.0, 30.0], [0, 0, 0], 300, 4.0, (1, 1, 1), "Big"))
    return sim


def capture_grid(module, resolution):
    captured = []
    stub_gl(module, captured)
    sim = field_bodies(module)
    sim.show_grid = True
    sim.show_force_vectors = True
    sim.grid_resolution = resolution
    sim.grid_arrays_dirty = True
    sim.cached_resolution = 0
    sim.vbo_id = None
    sim.ibo_id = None
    sim.draw_curved_grid()
    sim.draw_force_field()
    return {
        "bodies": [body_state(p) for p in sim.planets],
        "resolution": resolution,
        "vertices": sim.grid_vertices.astype(float).tolist(),
        "indices": sim.grid_indices.astype(int).tolist(),
        "arrows": captured,  # start, end, start, end, ...
    }


def main():
    sheet = load_simulator(SHEET_REPO)
    physics = [run_scenario(sheet, preset, seed) for preset, seed in
               [(1, 0), (2, 0), (3, 0), (4, 0), (5, 11), (5, 12), (6, 21)]]
    sheet_grid = capture_grid(sheet, 40)
    volume = load_simulator(VOLUME_DIR, SHEET_REPO)
    volume_grid = capture_grid(volume, 10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "simulator.py and rust/tools/volume_reference/simulator.py",
        "physics": physics, "sheet_grid": sheet_grid, "volume_grid": volume_grid,
    }))
    merges = {f"preset {s['preset']} seed {s['seed']}":
              f"{len(s['start'])} -> {len(s['checkpoints'][str(CHECKPOINTS[-1])])} bodies" for s in physics}
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    for k, v in merges.items():
        print(f"  {k}: {v}")
    print(f"  sheet grid: {len(sheet_grid['vertices'])} vertices, {len(sheet_grid['arrows']) // 2} arrows")
    print(f"  volume grid: {len(volume_grid['vertices'])} vertices, {len(volume_grid['arrows']) // 2} arrows")


if __name__ == "__main__":
    main()
