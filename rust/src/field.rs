//! The two spacetime-curvature views and their force-field arrows. Pure
//! maths, no drawing: the app turns the vertices and polylines into lines.
//!
//! * [`sheet`]: the flat "rubber sheet" of `simulator.py` at the repo root
//!   (`draw_curved_grid`, `draw_force_field`).
//! * [`volume`]: the 3D lattice of the 3D-grid version, kept as
//!   `tools/volume_reference/simulator.py`.
//!
//! The two modes share only [`Grid`] and [`Arrow`]; each module owns its own
//! constants and formulas, so changing one cannot change the other.

use crate::physics::{Body, G};
use macroquad::math::DVec3;

/// A run of vertices `start, start + stride, ...` (`len` of them) drawn as
/// one connected line. Grid lines are evenly strided, so this is enough.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Polyline {
    pub start: u32,
    pub stride: u32,
    pub len: u32,
}

impl Polyline {
    pub fn indices(self) -> impl Iterator<Item = usize> {
        (0..self.len).map(move |t| (self.start + t * self.stride) as usize)
    }

    /// Each line segment as a pair of vertex indices.
    pub fn segments(self) -> impl Iterator<Item = (usize, usize)> {
        let first = self.start as usize;
        let stride = self.stride as usize;
        (0..self.len.saturating_sub(1) as usize).map(move |t| (first + t * stride, first + (t + 1) * stride))
    }
}

/// Undisplaced grid points plus the lines between them.
#[derive(Clone, Debug)]
pub struct Grid {
    pub resolution: usize,
    pub base: Vec<DVec3>,
    pub lines: Vec<Polyline>,
}

/// A force arrow from `start` to `end`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arrow {
    pub start: DVec3,
    pub end: DVec3,
}

/// numpy's `linspace(start, stop, num)`: `i * step + start`, last = stop.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let div = (num - 1) as f64;
    let step = (stop - start) / div;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    if num > 1 {
        v[num - 1] = stop;
    }
    v
}

/// numpy's `arange(start, stop, step)` for the integer-valued steps used here.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// Turns arrow field components into arrows the way both Pythons do:
/// skip weak points, direction `F/|F|`, length `min(1.5, 2|F|)`.
fn arrow(start: DVec3, f: DVec3) -> Option<Arrow> {
    let mag = f.length();
    if mag <= 0.01 {
        return None;
    }
    let scale = (mag * 2.0).min(1.5);
    Some(Arrow { start, end: start + f / mag * scale })
}

pub mod sheet {
    //! The flat grid, bent downward into a well under every body.
    use super::*;

    pub const EXTENT: f64 = 50.0;
    pub const DEFAULT_RESOLUTION: usize = 250;
    pub const MIN_RESOLUTION: usize = 20;
    pub const MAX_RESOLUTION: usize = 500;
    pub const RESOLUTION_STEP: usize = 10;
    /// Plummer softening, so a well is rounded instead of a spike.
    pub const SOFTENING: f64 = 1.2;
    const ARROW_SPACING: f64 = 4.0;

    /// A (resolution+1)^2 sheet on the XZ plane. Vertex `i*w + j` sits at
    /// (x_j, 0, z_i), as `np.meshgrid(x, z)` lays it out.
    pub fn grid(resolution: usize) -> Grid {
        let w = resolution + 1;
        let x = linspace(-EXTENT, EXTENT, w);
        let mut base = Vec::with_capacity(w * w);
        for zi in &x {
            for xj in &x {
                base.push(DVec3::new(*xj, 0.0, *zi));
            }
        }
        let mut lines = Vec::with_capacity(2 * w);
        for i in 0..w {
            lines.push(Polyline { start: (i * w) as u32, stride: 1, len: w as u32 }); // along X
        }
        for j in 0..w {
            lines.push(Polyline { start: j as u32, stride: w as u32, len: w as u32 }); // along Z
        }
        Grid { resolution, base, lines }
    }

    /// Depth of the sheet at (x, z): `-sum 0.15 m / (5 sqrt(d^2 + s^2))`,
    /// with d the horizontal distance to each body.
    pub fn height(x: f64, z: f64, bodies: &[Body]) -> f64 {
        let mut y = 0.0;
        for b in bodies {
            let dx = x - b.position.x;
            let dz = z - b.position.z;
            let dist = (dx * dx + dz * dz).sqrt();
            let dist_eff = (dist * dist + SOFTENING * SOFTENING).sqrt();
            let displacement = b.mass / (dist_eff * 5.0);
            y -= displacement * 0.15;
        }
        y
    }

    /// Writes the bent sheet into `out` (one point per grid vertex).
    pub fn displace(grid: &Grid, bodies: &[Body], out: &mut Vec<DVec3>) {
        out.clear();
        out.extend(grid.base.iter().map(|p| DVec3::new(p.x, height(p.x, p.z, bodies), p.z)));
    }

    /// Horizontal field arrows every 4 units, sitting just above the sheet.
    /// Points within 2 radii of a body are left out of that body's pull.
    pub fn arrows(bodies: &[Body]) -> Vec<Arrow> {
        let extent = EXTENT - 2.0;
        let axis = arange(-extent, extent + ARROW_SPACING, ARROW_SPACING);
        let mut out = Vec::new();
        for z in &axis {
            for x in &axis {
                let (mut fx, mut fz) = (0.0, 0.0);
                for b in bodies {
                    let dx = x - b.position.x;
                    let dz = z - b.position.z;
                    let dist = (dx * dx + dz * dz).sqrt();
                    if dist > b.radius * 2.0 {
                        let dist_safe = dist.max(1.0);
                        let force_mag = G * b.mass / (dist_safe * dist_safe);
                        fx += force_mag * (-dx / dist_safe);
                        fz += force_mag * (-dz / dist_safe);
                    }
                }
                let start = DVec3::new(*x, height(*x, *z, bodies) + 0.1, *z);
                if let Some(a) = arrow(start, DVec3::new(fx, 0.0, fz)) {
                    // The arrow stays level: the Python adds no Y to the tip.
                    out.push(Arrow { start: a.start, end: DVec3::new(a.end.x, start.y, a.end.z) });
                }
            }
        }
        out
    }
}

pub mod volume {
    //! The 3D lattice, with every point pulled toward nearby bodies.
    use super::*;

    pub const EXTENT: f64 = 150.0;
    pub const DEFAULT_RESOLUTION: usize = 20;
    pub const MIN_RESOLUTION: usize = 10;
    /// The Python allowed 500 here (501^3 = 126 million points, which hangs
    /// it). 40 is 68,921 points.
    pub const MAX_RESOLUTION: usize = 40;
    pub const RESOLUTION_STEP: usize = 10;
    const ARROW_SPACING: f64 = 8.0;
    const PUSH_OUT_PASSES: usize = 3;

    /// A (resolution+1)^3 lattice. Vertex `i*w^2 + j*w + k` sits at
    /// (x_i, y_j, z_k), as `np.meshgrid(x, y, z, indexing='ij')` lays it out.
    pub fn grid(resolution: usize) -> Grid {
        let w = resolution + 1;
        let a = linspace(-EXTENT, EXTENT, w);
        let mut base = Vec::with_capacity(w * w * w);
        for xi in &a {
            for yj in &a {
                for zk in &a {
                    base.push(DVec3::new(*xi, *yj, *zk));
                }
            }
        }
        let (w, w2) = (w as u32, (w * w) as u32);
        let mut lines = Vec::with_capacity(3 * (w2 as usize));
        for j in 0..w {
            for k in 0..w {
                lines.push(Polyline { start: j * w + k, stride: w2, len: w }); // along X
            }
        }
        for i in 0..w {
            for k in 0..w {
                lines.push(Polyline { start: i * w2 + k, stride: w, len: w }); // along Y
            }
        }
        for i in 0..w {
            for j in 0..w {
                lines.push(Polyline { start: i * w2 + j * w, stride: 1, len: w }); // along Z
            }
        }
        Grid { resolution, base, lines }
    }

    /// How far a lattice point at `p` is pulled toward one body: falls off
    /// as (1 - d/R)^2 out to R = 10 sqrt(m), and never past 1.1 radii from
    /// the body's centre.
    fn pull(p: DVec3, b: &Body) -> DVec3 {
        let d = p - b.position;
        let dist = d.length();
        let dist_safe = dist.max(0.1);
        let influence_radius = b.mass.sqrt() * 10.0;
        if dist >= influence_radius {
            return DVec3::ZERO;
        }
        let normalized = dist_safe / influence_radius;
        let falloff = (1.0 - normalized) * (1.0 - normalized);
        let max_pull = (dist - b.radius * 1.1).max(0.0);
        let pull = (10.0 * falloff).min(max_pull);
        (-d / dist_safe) * pull
    }

    /// Writes the warped lattice into `out`: every body's pull added up,
    /// then three passes pushing points that ended inside 1.2 radii of a
    /// body back out onto that sphere.
    pub fn displace(grid: &Grid, bodies: &[Body], out: &mut Vec<DVec3>) {
        out.clear();
        out.extend(grid.base.iter().map(|&p| {
            let mut offset = DVec3::ZERO;
            for b in bodies {
                offset += pull(p, b);
            }
            p + offset
        }));
        for _ in 0..PUSH_OUT_PASSES {
            for b in bodies {
                let target = b.radius * 1.2;
                let target_sq = target * target;
                for p in out.iter_mut() {
                    let d = *p - b.position;
                    let dist_sq = d.length_squared();
                    if dist_sq < target_sq {
                        let dist = dist_sq.sqrt();
                        let dir = d / dist.max(0.0001);
                        *p += dir * (target - dist);
                    }
                }
            }
        }
    }

    /// Full 3D field arrows every 8 units through the volume. Visits points
    /// in the order of `np.meshgrid(x, y, z)` (y outer, then x, then z).
    pub fn arrows(bodies: &[Body]) -> Vec<Arrow> {
        let extent = EXTENT - 2.0;
        let axis = arange(-extent, extent + ARROW_SPACING, ARROW_SPACING);
        let mut out = Vec::new();
        for y in &axis {
            for x in &axis {
                for z in &axis {
                    let p = DVec3::new(*x, *y, *z);
                    let mut f = DVec3::ZERO;
                    for b in bodies {
                        let d = p - b.position;
                        let dist = d.length();
                        if dist > b.radius * 2.0 {
                            let dist_safe = dist.max(1.0);
                            let force_mag = G * b.mass / (dist_safe * dist_safe);
                            f += (-d / dist_safe) * force_mag;
                        }
                    }
                    out.extend(arrow(p, f));
                }
            }
        }
        out
    }
}
