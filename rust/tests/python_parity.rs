//! Checks the Rust port against numbers recorded from the Python originals
//! by `tools/make_fixtures.py` (the sheet and 3D-grid versions).
//!
//! Physics: the same starting state is stepped 1,200 times at dt = 1/60 s
//! (20 simulated seconds, every run includes merges) and compared at six
//! checkpoints. Grids: the bent sheet, the warped lattice and both sets of
//! force arrows are compared vertex by vertex.
//!
//! Run with `cargo test -- --nocapture` to see the measured differences.

use gravity_sim::field::{self, sheet, volume, Arrow};
use gravity_sim::physics::{Body, Simulation};
use macroquad::math::DVec3;
use serde::Deserialize;
use std::collections::BTreeSet;

/// Largest position or velocity difference accepted, in simulation units
/// (orbits are 5-50 units across, speeds 0.1-3). The port repeats the
/// Python's arithmetic in the same order, so the expected difference is
/// zero; the margin only allows for a different libm `pow` (used in merges)
/// on another platform.
const STATE_TOLERANCE: f64 = 1e-9;

#[derive(Deserialize)]
struct Fixture {
    physics: Vec<Scenario>,
    sheet_grid: GridCapture,
    volume_grid: GridCapture,
}

#[derive(Deserialize)]
struct Scenario {
    preset: u8,
    seed: u64,
    dt: f64,
    start: Vec<State>,
    checkpoints: std::collections::BTreeMap<String, Vec<State>>,
}

#[derive(Deserialize, Clone)]
struct State {
    id: u32,
    name: String,
    mass: f64,
    radius: f64,
    position: [f64; 3],
    velocity: [f64; 3],
    acceleration: [f64; 3],
    color: [f64; 3],
}

#[derive(Deserialize)]
struct GridCapture {
    bodies: Vec<State>,
    resolution: usize,
    vertices: Vec<[f64; 3]>,
    indices: Vec<usize>,
    arrows: Vec<[f64; 3]>,
}

fn fixture() -> Fixture {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/python_reference.json"))
        .expect("fixture missing: run tools/make_fixtures.py");
    serde_json::from_str(&text).unwrap()
}

fn v(a: [f64; 3]) -> DVec3 {
    DVec3::from(a)
}

fn simulation_from(states: &[State]) -> Simulation {
    let mut sim = Simulation::new();
    for s in states {
        let id = sim.add(s.name.clone(), v(s.position), v(s.velocity), s.mass, s.radius, s.color);
        assert_eq!(id, s.id, "ids are handed out in the same order");
    }
    sim
}

fn bodies_from(states: &[State]) -> Vec<Body> {
    simulation_from(states).bodies
}

/// Largest difference in position, velocity, mass and radius between the
/// Rust bodies and a Python checkpoint, after checking it is the same set
/// of bodies in the same order.
fn compare(sim: &Simulation, expected: &[State], what: &str) -> f64 {
    let ids: Vec<_> = sim.bodies.iter().map(|b| (b.id, b.name.as_str())).collect();
    let want: Vec<_> = expected.iter().map(|s| (s.id, s.name.as_str())).collect();
    assert_eq!(ids, want, "{what}: different bodies (a merge happened differently)");
    let mut worst: f64 = 0.0;
    for (b, s) in sim.bodies.iter().zip(expected) {
        worst = worst
            .max((b.position - v(s.position)).abs().max_element())
            .max((b.velocity - v(s.velocity)).abs().max_element())
            .max((b.acceleration - v(s.acceleration)).abs().max_element())
            .max((b.mass - s.mass).abs())
            .max((b.radius - s.radius).abs());
        for k in 0..3 {
            worst = worst.max((b.color[k] - s.color[k]).abs());
        }
    }
    worst
}

#[test]
fn starting_accelerations_match_python() {
    for sc in fixture().physics {
        let mut sim = simulation_from(&sc.start);
        sim.init_accelerations();
        let err = compare(&sim, &sc.start, "start");
        assert!(err == 0.0, "preset {} seed {}: start differs by {err:e}", sc.preset, sc.seed);
    }
}

#[test]
fn stepping_matches_python_for_every_preset() {
    for sc in fixture().physics {
        let mut sim = simulation_from(&sc.start);
        sim.init_accelerations();
        let mut step = 0;
        let mut line = format!("preset {} seed {:>2}: {:>2} bodies", sc.preset, sc.seed, sc.start.len());
        let mut checkpoints: Vec<_> = sc.checkpoints.iter().map(|(k, s)| (k.parse::<usize>().unwrap(), s)).collect();
        checkpoints.sort_by_key(|(k, _)| *k);
        for (at, expected) in checkpoints {
            while step < at {
                sim.step(sc.dt);
                step += 1;
            }
            let err = compare(&sim, expected, &format!("preset {} seed {} step {at}", sc.preset, sc.seed));
            line += &format!(" | step {at}: {} bodies, max diff {err:.1e}", expected.len());
            assert!(
                err <= STATE_TOLERANCE,
                "preset {} seed {} step {at}: differs from Python by {err:e} (> {STATE_TOLERANCE:e})",
                sc.preset,
                sc.seed
            );
        }
        println!("{line}");
    }
}

fn segments_from_python(indices: &[usize]) -> BTreeSet<(usize, usize)> {
    indices.chunks(2).map(|p| (p[0].min(p[1]), p[0].max(p[1]))).collect()
}

fn segments_from_rust(grid: &field::Grid) -> BTreeSet<(usize, usize)> {
    grid.lines.iter().flat_map(|l| l.segments()).map(|(a, b)| (a.min(b), a.max(b))).collect()
}

/// Python keeps grid vertices in a float32 array, so compare the Rust f64
/// values rounded the same way. Returns the largest difference.
fn compare_vertices(rust: &[DVec3], python: &[[f64; 3]]) -> f64 {
    assert_eq!(rust.len(), python.len());
    let mut worst: f64 = 0.0;
    for (r, p) in rust.iter().zip(python) {
        for k in 0..3 {
            worst = worst.max((r[k] as f32 as f64 - p[k]).abs());
        }
    }
    worst
}

fn compare_arrows(rust: &[Arrow], python: &[[f64; 3]]) -> f64 {
    assert_eq!(rust.len() * 2, python.len(), "different number of force arrows");
    let mut worst: f64 = 0.0;
    for (a, p) in rust.iter().zip(python.chunks(2)) {
        worst = worst.max((a.start - v(p[0])).abs().max_element());
        worst = worst.max((a.end - v(p[1])).abs().max_element());
    }
    worst
}

#[test]
fn sheet_grid_matches_python() {
    let cap = fixture().sheet_grid;
    let bodies = bodies_from(&cap.bodies);
    let grid = sheet::grid(cap.resolution);
    assert_eq!(segments_from_rust(&grid), segments_from_python(&cap.indices), "same lines");
    let mut out = Vec::new();
    sheet::displace(&grid, &bodies, &mut out);
    let err = compare_vertices(&out, &cap.vertices);
    let arrows = sheet::arrows(&bodies);
    let arrow_err = compare_arrows(&arrows, &cap.arrows);
    println!(
        "sheet res {}: {} vertices max diff {err:.1e} (float32); {} arrows max diff {arrow_err:.1e}",
        cap.resolution,
        out.len(),
        arrows.len()
    );
    assert_eq!(err, 0.0);
    assert_eq!(arrow_err, 0.0);
}

#[test]
fn volume_grid_matches_python() {
    let cap = fixture().volume_grid;
    let bodies = bodies_from(&cap.bodies);
    let grid = volume::grid(cap.resolution);
    assert_eq!(segments_from_rust(&grid), segments_from_python(&cap.indices), "same lines");
    let mut out = Vec::new();
    volume::displace(&grid, &bodies, &mut out);
    let err = compare_vertices(&out, &cap.vertices);

    // The fixture's bodies are placed so that some points get pushed back
    // out of a body; make sure that path was really compared.
    let on_shell = out
        .iter()
        .filter(|p| bodies.iter().any(|b| ((**p - b.position).length() - b.radius * 1.2).abs() < 1e-9))
        .count();
    let arrows = volume::arrows(&bodies);
    let arrow_err = compare_arrows(&arrows, &cap.arrows);
    println!(
        "volume res {}: {} vertices max diff {err:.1e} (float32), {on_shell} pushed onto a body's shell; \
         {} arrows max diff {arrow_err:.1e}",
        cap.resolution,
        out.len(),
        arrows.len()
    );
    assert!(on_shell > 0, "fixture no longer exercises the push-out pass");
    assert_eq!(err, 0.0);
    assert_eq!(arrow_err, 0.0);
}
