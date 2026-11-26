//! N-body gravity: forces, velocity Verlet, and merging collisions.
//!
//! This is a line-for-line port of `GravitySimulator.calculate_forces`,
//! `handle_collisions` and `update_physics` in `../simulator.py`. The
//! arithmetic is written in the same order as the numpy code (numpy's
//! `norm` and `sum` of a 3-vector are plain left-to-right sums), so the same
//! initial state gives the same numbers; `tests/python_parity.rs` checks
//! that against states recorded from the Python.

use macroquad::math::DVec3;
use std::collections::VecDeque;

/// Scaled gravitational constant (config.py).
pub const G: f64 = 6.67430e-2;
/// Points kept per trail (config.py).
pub const TRAIL_LENGTH: usize = 200;

#[derive(Clone, Debug)]
pub struct Body {
    pub id: u32,
    pub name: String,
    pub position: DVec3,
    pub velocity: DVec3,
    pub acceleration: DVec3,
    pub mass: f64,
    pub radius: f64,
    pub color: [f64; 3],
    pub trail: VecDeque<DVec3>,
}

impl Body {
    /// rho = m / (4/3 pi r^3)
    pub fn density(&self) -> f64 {
        density(self.mass, self.radius)
    }

    /// g = G m / r^2
    pub fn surface_gravity(&self) -> f64 {
        if self.radius > 0.0 {
            G * self.mass / self.radius.powi(2)
        } else {
            0.0
        }
    }
}

pub fn sphere_volume(radius: f64) -> f64 {
    (4.0 / 3.0) * std::f64::consts::PI * radius.powf(3.0)
}

pub fn density(mass: f64, radius: f64) -> f64 {
    let volume = sphere_volume(radius);
    if volume > 0.0 {
        mass / volume
    } else {
        0.0
    }
}

/// Two bodies that merged into a new one (the new body has a fresh id).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Merge {
    pub absorbed: [u32; 2],
    pub into: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Simulation {
    pub bodies: Vec<Body>,
    /// Last id handed out; the Python calls it `planet_counter`.
    counter: u32,
}

impl Simulation {
    pub fn new() -> Self {
        Self::default()
    }

    /// Removes every body and restarts ids at 1, like `load_preset`.
    pub fn clear(&mut self) {
        self.bodies.clear();
        self.counter = 0;
    }

    /// Appends a body and returns its id (`create_planet` + append).
    pub fn add(
        &mut self,
        name: impl Into<String>,
        position: DVec3,
        velocity: DVec3,
        mass: f64,
        radius: f64,
        color: [f64; 3],
    ) -> u32 {
        self.counter += 1;
        self.bodies.push(Body {
            id: self.counter,
            name: name.into(),
            position,
            velocity,
            acceleration: DVec3::ZERO,
            mass,
            radius,
            color,
            trail: VecDeque::new(),
        });
        self.counter
    }

    /// The id the next added body will get.
    pub fn next_id(&self) -> u32 {
        self.counter + 1
    }

    pub fn get(&self, id: u32) -> Option<&Body> {
        self.bodies.iter().find(|b| b.id == id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut Body> {
        self.bodies.iter_mut().find(|b| b.id == id)
    }

    pub fn remove(&mut self, id: u32) -> bool {
        let before = self.bodies.len();
        self.bodies.retain(|b| b.id != id);
        self.bodies.len() != before
    }

    /// Sets every acceleration from the current forces. `load_preset` does
    /// this once after placing bodies, so the first Verlet step has a start.
    pub fn init_accelerations(&mut self) {
        let forces = self.forces();
        for (body, force) in self.bodies.iter_mut().zip(forces) {
            if body.mass > 0.0 {
                body.acceleration = force / body.mass;
            }
        }
    }

    /// Net gravitational force on each body. Overlapping pairs (closer than
    /// the sum of radii) are skipped: they are about to merge.
    ///
    /// Same full i,j double loop as the Python, not the Newton's-third-law
    /// half loop: it is what keeps the rounding identical to the reference.
    pub fn forces(&self) -> Vec<DVec3> {
        let mut forces = Vec::with_capacity(self.bodies.len());
        for (i, planet) in self.bodies.iter().enumerate() {
            let mut force = DVec3::ZERO;
            for (j, other) in self.bodies.iter().enumerate() {
                if i == j {
                    continue;
                }
                let r = other.position - planet.position;
                let dist = r.length();
                if dist < planet.radius + other.radius {
                    continue;
                }
                let force_mag = G * planet.mass * other.mass / (dist * dist);
                force += r * force_mag / dist;
            }
            forces.push(force);
        }
        forces
    }

    /// Merges touching bodies until none touch. Each merge conserves mass,
    /// momentum and volume, puts the result at the centre of mass with a
    /// mass-weighted colour and the heavier body's name, and restarts the
    /// scan (the list changed). The merged body goes to the end of the list.
    pub fn merge_collisions(&mut self) -> Vec<Merge> {
        let mut merges = Vec::new();
        'scan: loop {
            for i in 0..self.bodies.len() {
                for j in (i + 1)..self.bodies.len() {
                    let (p1, p2) = (&self.bodies[i], &self.bodies[j]);
                    let dist_sq = (p1.position - p2.position).length_squared();
                    let min_dist = p1.radius + p2.radius;
                    if dist_sq >= min_dist * min_dist {
                        continue;
                    }

                    let new_mass = p1.mass + p2.mass;
                    let new_radius = (p1.radius.powf(3.0) + p2.radius.powf(3.0)).powf(1.0 / 3.0);
                    let (new_vel, new_pos) = if new_mass > 0.0 {
                        (
                            (p1.velocity * p1.mass + p2.velocity * p2.mass) / new_mass,
                            (p1.position * p1.mass + p2.position * p2.mass) / new_mass,
                        )
                    } else {
                        (DVec3::ZERO, (p1.position + p2.position) / 2.0)
                    };
                    let mut new_color = [0.0; 3];
                    for (k, c) in new_color.iter_mut().enumerate() {
                        *c = (p1.color[k] * p1.mass + p2.color[k] * p2.mass) / new_mass;
                    }
                    let name = if p1.mass >= p2.mass { p1.name.clone() } else { p2.name.clone() };
                    let absorbed = [p1.id, p2.id];

                    self.bodies.remove(j); // larger index first
                    self.bodies.remove(i);
                    let into = self.add(name, new_pos, new_vel, new_mass, new_radius, new_color);
                    merges.push(Merge { absorbed, into });
                    continue 'scan;
                }
            }
            return merges;
        }
    }

    /// One velocity Verlet step of `dt` (already multiplied by the time
    /// scale), with merging between the position and velocity halves.
    /// Trails are not touched here; see [`Simulation::record_trails`].
    pub fn step(&mut self, dt: f64) -> Vec<Merge> {
        for b in &mut self.bodies {
            b.position += b.velocity * dt + b.acceleration * 0.5 * dt * dt;
            b.velocity += b.acceleration * 0.5 * dt;
        }

        let merges = self.merge_collisions();

        let forces = self.forces();
        for (b, force) in self.bodies.iter_mut().zip(forces) {
            let new_acceleration = if b.mass > 0.0 { force / b.mass } else { DVec3::ZERO };
            b.velocity += new_acceleration * 0.5 * dt;
            b.acceleration = new_acceleration;
        }
        merges
    }

    /// Advances `dt` of simulated time in equal steps no longer than
    /// `max_step`. With `dt <= max_step` this is exactly one [`step`].
    ///
    /// The Python always takes one step per frame, so at 5x time scale each
    /// step is 5x longer and close passes lose accuracy; splitting keeps the
    /// step size the same at every time scale.
    ///
    /// [`step`]: Simulation::step
    pub fn advance(&mut self, dt: f64, max_step: f64) -> Vec<Merge> {
        let n = substeps(dt, max_step);
        let h = dt / n as f64;
        let mut merges = Vec::new();
        for _ in 0..n {
            merges.extend(self.step(h));
        }
        merges
    }

    /// Appends each body's position to its trail, keeping the last
    /// [`TRAIL_LENGTH`]. The app calls this once per frame (the Python did
    /// it once per step, which was the same thing at one step per frame).
    pub fn record_trails(&mut self) {
        for b in &mut self.bodies {
            b.trail.push_back(b.position);
            if b.trail.len() > TRAIL_LENGTH {
                b.trail.pop_front();
            }
        }
    }

    pub fn total_mass(&self) -> f64 {
        self.bodies.iter().map(|b| b.mass).sum()
    }

    pub fn momentum(&self) -> DVec3 {
        self.bodies.iter().map(|b| b.velocity * b.mass).sum()
    }

    /// Kinetic plus pairwise potential energy, `-G m_i m_j / d` per pair.
    pub fn energy(&self) -> f64 {
        let kinetic: f64 = self.bodies.iter().map(|b| 0.5 * b.mass * b.velocity.length_squared()).sum();
        let mut potential = 0.0;
        for (i, a) in self.bodies.iter().enumerate() {
            for b in &self.bodies[i + 1..] {
                potential -= G * a.mass * b.mass / (a.position - b.position).length();
            }
        }
        kinetic + potential
    }
}

/// Number of equal sub-steps needed so none is longer than `max_step`.
pub fn substeps(dt: f64, max_step: f64) -> usize {
    // The small slack keeps dt == max_step (e.g. 1/60 at 1x) at one step
    // despite rounding in the division.
    ((dt / max_step) - 1e-9).ceil().max(1.0) as usize
}
