# What came from each variant

The Rust app merges three earlier versions of the simulator. The original
Python is still at the root of this repo; the other two were separate repos:

| Version | Language | What it was |
|---|---|---|
| Original (this repo's root) | Python (pygame + PyOpenGL) | N-body gravity over a bent 2D "rubber sheet" |
| Gravity-sim-3D-grid | Python | The same app with a warped 3D lattice instead of the sheet |
| Gravity-well-sim-web | JavaScript (three.js) | A browser copy of the sheet version |

The original is the reference: where the versions disagreed, the Rust port
follows it unless another version did something better (listed below).

## From the original (kept as is)

- **Physics, unchanged:** gravity, velocity Verlet, merging collisions. The
  Rust code repeats the Python's arithmetic in the same order, and
  `tests/python_parity.rs` shows identical numbers (zero difference) over
  1,200 steps of every preset, merges included.
- **The sheet view** and its force arrows (also matched point for point).
- **Presets:** the full 10-body solar system with Pluto, binary stars, three
  body, figure-8, collision, cluster.
- **Add planet (`N`):** places a body on a rough orbit.
- **Controls and panels:** time scale 0.25-5x, sheet resolution 20-500, the
  Delete key, the EXIT button (desktop only), and the planet editor with its
  mass / radius / density links.
- **Look:** camera controls, lighting, starfield, trails, velocity arrows,
  and the pulsing selection ring.

## From Gravity-sim-3D-grid

- **The 3D lattice view** (pull toward bodies, then push-out from inside
  them) and its full 3D force arrows, now a second view: press `M` or use the
  View buttons. Each view keeps its own resolution.
- **Fixed:** its "+" button allowed resolution 500, which is 126 million
  points and hangs the app. It now stops at 40 (68,921 points).
- Its `simulator.py` and `config.py` are kept in `tools/volume_reference/` so
  the parity fixtures can be recorded again from this repo alone.

## From Gravity-well-sim-web

- A cap on frame time (0.05 s), so a stalled frame (window dragged, tab
  hidden) cannot fling bodies apart.
- Clickable preset rows and toggles (the Python only listed them).
- Surface g and net force shown read-only.
- Running in a browser, now from the same Rust code compiled to WebAssembly
  instead of a separate JavaScript codebase.

## Dropped

- **The web version's own physics and presets:** its forces used a
  half-loop, trails were recorded per frame, the solar system had only 6
  bodies, the cluster was a random box, `N` used a fixed speed, and time scale
  had no upper limit. The Python versions are the more complete ones.
- **three.js extras:** camera damping, fog, shiny highlights, and the HTML
  side panels. The UI is drawn in the canvas instead, identical on desktop
  and in the browser.
- **Python cosmetics:** Unicode symbols in the UI (box lines, dots, rho,
  subscripts) became plain ASCII and drawn dots, because the built-in font is
  ASCII only. Rounded panel corners and 2-pixel-wide lines are gone too
  (WebGL draws 1-pixel lines).
- **pygbag** (the Python-to-web packager), replaced by the WebAssembly build.

## Changed on purpose

Each change is also noted in the code where it happens.

- **Sub-steps at high time scale:** at 1x the app takes exactly one Python
  step per frame. Faster speeds are split into 1/60 s steps instead of one
  long step, which cut energy drift 24 times at 5x
  (`substepping_is_more_accurate_at_high_time_scale`).
- **Trails** are recorded 60 times a real second, so they stay 3.3 s long on
  any monitor. The Python recorded one point per frame at up to 60 fps.
- **`R`** reloads the current preset. The Python always went back to the
  solar system.
- **Right-drag panning** follows the way the camera faces. The Python panned
  along world X/Z, which only matched the mouse from one angle.
- **Editor:**
  - Typed position or velocity values wait for Apply instead of being
    overwritten by the next physics step.
  - A density edit changes the body right away.
  - Adding, applying or deleting recomputes accelerations immediately. The
    Python left them stale for half a step.
- **Clicks:** the left panel only blocks clicks on the panel itself. The
    Python blocked the whole column down to the bottom of the window.
- **Input order:** clicks, keys and typed characters are handled in the
  order they happened, like pygame's event loop.
- **Status readout:** an FPS counter has been added.
