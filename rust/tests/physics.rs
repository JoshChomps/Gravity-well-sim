//! Conservation and behaviour checks, ported from `../tests/test_physics.py`
//! plus checks for what the Rust port adds (sub-stepping, the grid modes).

use gravity_sim::field::{sheet, volume};
use gravity_sim::physics::{self, substeps, Simulation, G, TRAIL_LENGTH};
use gravity_sim::presets::{self, Rng};
use macroquad::math::{dvec3, DVec3};

fn sim_with(bodies: &[([f64; 3], [f64; 3], f64, f64)]) -> Simulation {
    let mut sim = Simulation::new();
    for (p, v, m, r) in bodies {
        sim.add("p", DVec3::from(*p), DVec3::from(*v), *m, *r, [1.0; 3]);
    }
    sim.init_accelerations();
    sim
}

/// A 1e-3 satellite on a circular orbit of radius r around 500 at the origin.
fn circular_orbit() -> (Simulation, f64, f64) {
    let (r, central) = (10.0, 500.0);
    let v = (G * central / r).sqrt();
    (sim_with(&[([0.; 3], [0.; 3], central, 1.0), ([r, 0., 0.], [0., 0., v], 1e-3, 0.1)]), r, v)
}

fn assert_close(a: DVec3, b: DVec3, tol: f64) {
    assert!((a - b).abs().max_element() <= tol, "{a:?} vs {b:?}");
}

#[test]
fn two_body_force_matches_newton() {
    let sim = sim_with(&[([0.; 3], [0.; 3], 100.0, 0.5), ([4., 0., 0.], [0.; 3], 10.0, 0.5)]);
    let f = sim.forces();
    let expected = G * 100.0 * 10.0 / 16.0;
    assert_close(f[0], dvec3(expected, 0., 0.), 1e-12);
    assert_close(f[1], dvec3(-expected, 0., 0.), 1e-12);
}

#[test]
fn forces_sum_to_zero() {
    let mut rng = Rng::new(1);
    let bodies: Vec<_> = (0..8)
        .map(|_| ([rng.uniform(-20., 20.), rng.uniform(-20., 20.), rng.uniform(-20., 20.)], [0.; 3], rng.uniform(1., 100.), 0.1))
        .collect();
    let total: DVec3 = sim_with(&bodies).forces().into_iter().sum();
    assert!(total.length() < 1e-9, "{total:?}");
}

#[test]
fn overlapping_bodies_exert_no_force() {
    let sim = sim_with(&[([0.; 3], [0.; 3], 100.0, 1.0), ([1.5, 0., 0.], [0.; 3], 10.0, 1.0)]);
    for f in sim.forces() {
        assert_eq!(f, DVec3::ZERO);
    }
}

#[test]
fn momentum_is_conserved() {
    let mut sim = sim_with(&[
        ([-4., 0., 0.], [0., 0., 1.2], 300., 0.5),
        ([4., 0., 0.], [0., 0.3, -1.0], 200., 0.5),
        ([0., 2., 12.], [1.5, 0., 0.], 10., 0.3),
    ]);
    let before = sim.momentum();
    for _ in 0..2000 {
        sim.step(0.01);
    }
    assert_eq!(sim.bodies.len(), 3, "bodies merged; the test needs a collision-free setup");
    assert_close(sim.momentum(), before, 1e-9);
}

#[test]
fn circular_orbit_keeps_its_radius() {
    let (mut sim, r, v) = circular_orbit();
    let period = 2.0 * std::f64::consts::PI * r / v;
    let mut worst: f64 = 0.0;
    for _ in 0..(period / 0.01) as usize {
        sim.step(0.01);
        let radius = (sim.bodies[1].position - sim.bodies[0].position).length();
        worst = worst.max((radius - r).abs() / r);
    }
    assert!(worst < 0.01, "radius drifted {worst}");
}

#[test]
fn energy_drift_is_small_over_one_orbit() {
    let (mut sim, r, v) = circular_orbit();
    let start = sim.energy();
    for _ in 0..(2.0 * std::f64::consts::PI * r / v / 0.01) as usize {
        sim.step(0.01);
    }
    let drift = ((sim.energy() - start) / start).abs();
    assert!(drift < 1e-4, "energy drift {drift}");
}

#[test]
fn one_long_step_equals_the_python_time_scale() {
    // The Python multiplied dt by time_scale inside update_physics; here the
    // caller passes the scaled dt. advance() with a large max_step is one step.
    let (mut a, _, _) = circular_orbit();
    let (mut b, _, _) = circular_orbit();
    a.step(0.02);
    b.advance(0.01 * 2.0, 1.0);
    assert_eq!(a.bodies[1].position, b.bodies[1].position);
}

#[test]
fn substeps_split_only_when_needed() {
    let frame = 1.0 / 60.0;
    assert_eq!(substeps(frame, frame), 1, "1x at 60 fps is one step, exactly the Python");
    assert_eq!(substeps(frame * 0.25, frame), 1);
    assert_eq!(substeps(frame * 5.0, frame), 5);
    assert_eq!(substeps(frame * 1.01, frame), 2);
}

#[test]
fn substepping_is_more_accurate_at_high_time_scale() {
    // Binary stars at 5x for 10 simulated seconds: one 5/60 s step per frame
    // (what the Python did) against five 1/60 s steps.
    let run = |max_step: f64| {
        let mut sim = Simulation::new();
        presets::load(&mut sim, 2, &mut Rng::new(0));
        let start = sim.energy();
        for _ in 0..120 {
            sim.advance(5.0 / 60.0, max_step);
        }
        assert_eq!(sim.bodies.len(), 3, "no merges in this window");
        ((sim.energy() - start) / start).abs()
    };
    let single = run(1.0);
    let split = run(1.0 / 60.0);
    println!("binary stars, 5x for 10 s: energy drift {single:.2e} one step/frame, {split:.2e} with sub-steps");
    assert!(split < single / 10.0, "sub-steps {split:e} vs single {single:e}");
}

#[test]
fn trail_is_capped() {
    let (mut sim, _, _) = circular_orbit();
    for _ in 0..TRAIL_LENGTH + 25 {
        sim.step(0.01);
        sim.record_trails();
    }
    assert_eq!(sim.bodies[1].trail.len(), TRAIL_LENGTH);
    assert_eq!(*sim.bodies[1].trail.back().unwrap(), sim.bodies[1].position);
}

#[test]
fn merge_conserves_mass_momentum_and_volume() {
    let mut sim = sim_with(&[([0.; 3], [1., 0., 0.], 30.0, 1.0), ([1.5, 0., 0.], [-2., 1., 0.], 10.0, 0.8)]);
    let p_before = sim.momentum();
    let com = (sim.bodies[0].position * 30.0 + sim.bodies[1].position * 10.0) / 40.0;
    let merges = sim.merge_collisions();
    assert_eq!(merges, vec![physics::Merge { absorbed: [1, 2], into: 3 }]);
    assert_eq!(sim.bodies.len(), 1);
    let m = &sim.bodies[0];
    assert!((m.mass - 40.0).abs() < 1e-12);
    assert_close(m.velocity * m.mass, p_before, 1e-12);
    assert_close(m.position, com, 1e-12);
    assert!((m.radius.powi(3) - (1.0 + 0.8f64.powi(3))).abs() < 1e-12);
}

#[test]
fn merged_body_keeps_heavier_name() {
    let mut sim = Simulation::new();
    sim.add("Light", DVec3::ZERO, DVec3::ZERO, 5.0, 1.0, [1.0; 3]);
    sim.add("Heavy", dvec3(0.5, 0., 0.), DVec3::ZERO, 50.0, 1.0, [1.0; 3]);
    sim.merge_collisions();
    assert_eq!(sim.bodies[0].name, "Heavy");
}

#[test]
fn chain_of_overlaps_leaves_no_overlaps() {
    let bodies: Vec<_> = (0..5).map(|i| ([i as f64 * 0.5, 0., 0.], [0.; 3], 1.0, 0.4)).collect();
    let mut sim = sim_with(&bodies);
    sim.merge_collisions();
    assert!(sim.bodies.len() < 5);
    assert!((sim.total_mass() - 5.0).abs() < 1e-12);
    for (i, a) in sim.bodies.iter().enumerate() {
        for b in &sim.bodies[i + 1..] {
            assert!((a.position - b.position).length() >= a.radius + b.radius);
        }
    }
}

#[test]
fn every_preset_runs_without_blowing_up() {
    for preset in 1..=6 {
        let mut sim = Simulation::new();
        presets::load(&mut sim, preset, &mut Rng::new(preset as u64));
        assert!(!sim.bodies.is_empty(), "preset {preset}");
        let mass = sim.total_mass();
        for _ in 0..500 {
            sim.step(1.0 / 60.0);
        }
        for b in &sim.bodies {
            assert!(b.position.is_finite() && b.velocity.is_finite(), "preset {preset}: {}", b.name);
        }
        assert!((sim.total_mass() - mass).abs() < 1e-9, "preset {preset} lost mass");
    }
}

#[test]
fn random_presets_repeat_with_the_same_seed() {
    for preset in [5, 6] {
        let (mut a, mut b) = (Simulation::new(), Simulation::new());
        presets::load(&mut a, preset, &mut Rng::new(7));
        presets::load(&mut b, preset, &mut Rng::new(7));
        let pos = |s: &Simulation| s.bodies.iter().map(|b| b.position).collect::<Vec<_>>();
        assert_eq!(pos(&a), pos(&b));
    }
}

#[test]
fn density_and_surface_gravity() {
    let mut sim = Simulation::new();
    sim.add("p", DVec3::ZERO, DVec3::ZERO, 4.0 / 3.0 * std::f64::consts::PI, 1.0, [1.0; 3]);
    sim.add("q", DVec3::ZERO, DVec3::ZERO, 100.0, 2.0, [1.0; 3]);
    sim.add("r", DVec3::ZERO, DVec3::ZERO, 5.0, 0.0, [1.0; 3]);
    assert!((sim.bodies[0].density() - 1.0).abs() < 1e-12);
    assert!((sim.bodies[1].surface_gravity() - G * 100.0 / 4.0).abs() < 1e-12);
    assert_eq!(sim.bodies[2].density(), 0.0);
    assert_eq!(sim.bodies[2].surface_gravity(), 0.0);
}

#[test]
fn hsv_matches_colorsys() {
    // Values from Python: colorsys.hsv_to_rgb(h, 0.7, 0.9)
    let cases = [(0.0, [0.9, 0.27, 0.27]), (0.5, [0.27, 0.9, 0.9]), (0.8, [0.774, 0.27, 0.9])];
    for (h, want) in cases {
        let got = presets::hsv_to_rgb(h, 0.7, 0.9);
        for k in 0..3 {
            assert!((got[k] - want[k]).abs() < 1e-9, "h={h}: {got:?}");
        }
    }
}

#[test]
fn the_two_grid_modes_are_independent() {
    let mut sim = Simulation::new();
    presets::load(&mut sim, 1, &mut Rng::new(0));
    let s = sheet::grid(sheet::DEFAULT_RESOLUTION);
    let v = volume::grid(volume::DEFAULT_RESOLUTION);
    assert_eq!(s.base.len(), 251 * 251);
    assert_eq!(v.base.len(), 21 * 21 * 21);
    let (mut so, mut vo) = (Vec::new(), Vec::new());
    sheet::displace(&s, &sim.bodies, &mut so);
    volume::displace(&v, &sim.bodies, &mut vo);
    // The sheet only moves points down; X and Z stay put.
    assert!(so.iter().zip(&s.base).all(|(p, b)| p.x == b.x && p.z == b.z && p.y < 0.0));
    // The lattice moves points sideways too, toward the Sun.
    assert!(vo.iter().zip(&v.base).any(|(p, b)| p.x != b.x));
}

#[test]
fn volume_resolution_limit_stays_sane() {
    // The Python's "+" button could ask for 501^3 points and hang.
    let w = volume::MAX_RESOLUTION + 1;
    assert!(w * w * w < 100_000);
}
