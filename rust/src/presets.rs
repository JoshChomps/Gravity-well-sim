//! The six starting scenarios and the "add a random planet" helper, as in
//! `load_preset` / `add_random_planet` in `../simulator.py`.

use crate::physics::{Simulation, G};
use macroquad::math::{dvec3, DVec3};

pub const NAMES: [&str; 6] = [
    "Solar System",
    "Binary Stars",
    "Three Body",
    "Figure-8",
    "Collision",
    "Cluster",
];

/// Small seeded generator (SplitMix64). Presets 5 and 6 are random; a seed
/// makes them repeatable in tests. It is not Python's Mersenne Twister, so
/// the parity tests load the Python's random starting states from the
/// fixture instead of regenerating them.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Python's `random.uniform(a, b)`.
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }
}

/// `colorsys.hsv_to_rgb`.
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f64; 3] {
    if s == 0.0 {
        return [v, v, v];
    }
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match (i as i64).rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// A random bright colour, like `create_planet` with no colour given.
pub fn random_color(rng: &mut Rng) -> [f64; 3] {
    hsv_to_rgb(rng.random(), 0.7, 0.9)
}

/// Replaces the bodies with preset `n` (1-6) and sets the starting
/// accelerations. Unknown numbers leave the simulation empty.
pub fn load(sim: &mut Simulation, n: u8, rng: &mut Rng) {
    sim.clear();
    let mut add = |name: &str, p: [f64; 3], v: [f64; 3], m: f64, r: f64, c: [f64; 3]| {
        sim.add(name, DVec3::from(p), DVec3::from(v), m, r, c);
    };
    match n {
        1 => {
            add("Sun", [0., 0., 0.], [0., 0., 0.], 500., 1.5, [1.0, 0.9, 0.3]);
            add("Mercury", [5., 0., 0.], [0., 0., 3.2], 5., 0.3, [0.7, 0.7, 0.7]);
            add("Venus", [7., 0., 2.], [0., 0.3, 2.5], 12., 0.45, [0.9, 0.6, 0.3]);
            add("Earth", [10., 0., 0.], [0., 0.2, 2.2], 15., 0.5, [0.3, 0.5, 0.9]);
            add("Mars", [14., 0., -2.], [0., -0.1, 1.8], 8., 0.35, [0.9, 0.4, 0.2]);
            add("Jupiter", [20., 0., 0.], [0., 0.1, 1.3], 60., 0.8, [0.8, 0.7, 0.6]);
            add("Saturn", [28., 0., 5.], [-0.1, 0., 0.9], 50., 0.7, [0.9, 0.8, 0.5]);
            add("Uranus", [36., 0., -5.], [0., -0.1, 0.7], 30., 0.6, [0.6, 0.8, 0.9]);
            add("Neptune", [42., 0., 0.], [0., 0.1, 0.6], 30., 0.6, [0.3, 0.3, 0.9]);
            add("Pluto", [48., 2., 0.], [0., -0.2, 0.5], 0.5, 0.15, [0.7, 0.6, 0.5]);
        }
        2 => {
            add("Star A", [-4., 0., 0.], [0., 0., 1.2], 300., 1.2, [1.0, 0.6, 0.2]);
            add("Star B", [4., 0., 0.], [0., 0., -1.2], 300., 1.2, [0.4, 0.7, 1.0]);
            add("Planet", [0., 0., 12.], [1.5, 0., 0.], 10., 0.4, [0.5, 0.9, 0.5]);
        }
        3 => {
            add("Red Giant", [-5., 0., 0.], [0., 0.5, 0.8], 200., 1.0, [0.9, 0.3, 0.4]);
            add("Green Star", [5., 0., 0.], [0., -0.3, -0.8], 200., 1.0, [0.3, 0.9, 0.5]);
            add("Blue Star", [0., 0., 6.], [-0.5, 0.2, 0.], 200., 1.0, [0.4, 0.5, 1.0]);
        }
        4 => {
            let v = 0.8;
            add("Alpha", [-3., 0., 0.], [0., v * 0.5, v], 150., 0.9, [1.0, 0.5, 0.8]);
            add("Beta", [3., 0., 0.], [0., v * 0.5, v], 150., 0.9, [0.5, 1.0, 0.8]);
            add("Gamma", [0., 0., 0.], [0., -v, -2. * v], 150., 0.9, [0.8, 0.5, 1.0]);
        }
        5 => {
            for (x_range, vx_range) in [((-10., -5.), (1., 2.)), ((5., 10.), (-2., -1.))] {
                for _ in 0..10 {
                    let p = dvec3(rng.uniform(x_range.0, x_range.1), rng.uniform(-2., 2.), rng.uniform(-5., 5.));
                    let v = dvec3(rng.uniform(vx_range.0, vx_range.1), 0., 0.);
                    let (m, r) = (rng.uniform(5., 20.), rng.uniform(0.3, 0.6));
                    let name = format!("Planet {}", sim.next_id());
                    let color = random_color(rng);
                    sim.add(name, p, v, m, r, color);
                }
            }
        }
        6 => {
            for _ in 0..30 {
                add_random_planet(sim, rng);
            }
        }
        _ => {}
    }
    sim.init_accelerations();
}

/// A planet 8-14 units from the origin on a rough circular orbit around the
/// total mass so far (speed `sqrt(G M / d)` times 0.8-1.2), radius growing
/// with mass. Returns its id.
pub fn add_random_planet(sim: &mut Simulation, rng: &mut Rng) -> u32 {
    let angle = rng.uniform(0., 2. * std::f64::consts::PI);
    let dist = rng.uniform(8., 14.);
    let pos = dvec3(dist * angle.cos(), rng.uniform(-2., 2.), dist * angle.sin());
    let speed = if sim.bodies.is_empty() {
        1.0
    } else {
        (G * sim.total_mass() / dist).sqrt() * rng.uniform(0.8, 1.2)
    };
    let vel = dvec3(-angle.sin() * speed, rng.uniform(-0.2, 0.2), angle.cos() * speed);
    let mass = rng.uniform(10., 50.);
    let radius = 0.2 + mass * 0.01;
    let color = random_color(rng);
    let name = format!("Planet {}", sim.next_id());
    sim.add(name, pos, vel, mass, radius, color)
}
