//! 3D drawing: camera, lit spheres, line batches, stars.
//!
//! The Python used OpenGL fixed-function lighting, which shades per vertex.
//! macroquad has no lighting, so [`Spheres`] does the same per-vertex sum on
//! the CPU (ambient + Lambert diffuse from one light) and hands the GPU
//! coloured vertices. That looks the same and needs no shader, so native
//! and WebGL render identically.

use gravity_sim::physics::Body;
use macroquad::models::{Mesh, Vertex};
use macroquad::prelude::*;

/// Largest batch handed to macroquad at once. The window config raises
/// macroquad's per-draw-call capacity above these.
pub const BATCH_VERTICES: usize = 8_192;
pub const BATCH_INDICES: usize = 16_384;

pub fn rgba(c: [f64; 3], a: f32) -> Color {
    Color::new(c[0] as f32, c[1] as f32, c[2] as f32, a)
}

fn vertex(p: Vec3, color: Color) -> Vertex {
    Vertex::new2(p, Vec2::ZERO, color)
}

/// Orbit camera: yaw/pitch in degrees around `pan`, like `setup_camera`.
pub struct OrbitCamera {
    pub distance: f64,
    pub rot_x: f64,
    pub rot_y: f64,
    pub pan: DVec3,
}

impl OrbitCamera {
    pub fn new() -> Self {
        Self { distance: 40.0, rot_x: 30.0, rot_y: 45.0, pan: DVec3::ZERO }
    }

    pub fn eye(&self) -> DVec3 {
        let (rx, ry) = (self.rot_x.to_radians(), self.rot_y.to_radians());
        self.pan + self.distance * dvec3(rx.cos() * ry.sin(), rx.sin(), rx.cos() * ry.cos())
    }

    pub fn camera(&self) -> Camera3D {
        Camera3D {
            position: self.eye().as_vec3(),
            target: self.pan.as_vec3(),
            up: Vec3::Y,
            fovy: 45f32.to_radians(),
            z_near: 0.1,
            z_far: 500.0,
            ..Default::default()
        }
    }

    /// The Python set its light at (10, 20, 10) while the modelview was
    /// identity, so the light sits in eye space: up and to the right of the
    /// viewer, slightly behind. Returns that point in world space.
    pub fn light_position(&self) -> Vec3 {
        let eye = self.eye().as_vec3();
        let forward = (self.pan.as_vec3() - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        eye + right * 10.0 + up * 20.0 - forward * 10.0
    }

    /// Ray from the near plane through a point on a `size` screen
    /// (`gluUnProject`). Done in f64: in f32 the inverse projection is too
    /// coarse at the far plane to aim the ray.
    pub fn ray(&self, screen: Vec2, size: Vec2) -> (DVec3, DVec3) {
        let proj = DMat4::perspective_rh_gl(45f64.to_radians(), (size.x / size.y) as f64, 0.1, 500.0);
        let view = DMat4::look_at_rh(self.eye(), self.pan, DVec3::Y);
        let inv = (proj * view).inverse();
        let ndc = dvec2(2.0 * screen.x as f64 / size.x as f64 - 1.0, 1.0 - 2.0 * screen.y as f64 / size.y as f64);
        let near = inv.project_point3(ndc.extend(-1.0));
        let far = inv.project_point3(ndc.extend(1.0));
        (near, (far - near).normalize())
    }

    /// Nearest body the ray through `screen` hits (`select_planet_at_screen_pos`).
    pub fn pick(&self, screen: Vec2, bodies: &[Body]) -> Option<u32> {
        let (origin, dir) = self.ray(screen, vec2(screen_width(), screen_height()));
        let mut best = (f64::INFINITY, None);
        for b in bodies {
            let oc = origin - b.position;
            let half_b = oc.dot(dir);
            let c = oc.dot(oc) - b.radius * b.radius;
            let disc = half_b * half_b - c;
            if disc >= 0.0 {
                let t = -half_b - disc.sqrt();
                if t > 0.0 && t < best.0 {
                    best = (t, Some(b.id));
                }
            }
        }
        best.1
    }
}

/// Collects line segments or triangles and hands them to macroquad in
/// large batches (one call per thousands of segments instead of one each).
pub struct Batch {
    mode: DrawMode,
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
}

impl Batch {
    pub fn lines() -> Self {
        Self { mode: DrawMode::Lines, vertices: Vec::new(), indices: Vec::new() }
    }

    pub fn triangles() -> Self {
        Self { mode: DrawMode::Triangles, vertices: Vec::new(), indices: Vec::new() }
    }

    fn make_room(&mut self, vertices: usize, indices: usize) {
        assert!(vertices <= BATCH_VERTICES && indices <= BATCH_INDICES, "primitive too large for one batch");
        if self.vertices.len() + vertices > BATCH_VERTICES || self.indices.len() + indices > BATCH_INDICES {
            self.flush();
        }
    }

    /// Draws what has been collected so far.
    pub fn flush(&mut self) {
        if !self.indices.is_empty() {
            let gl = unsafe { get_internal_gl() }.quad_gl;
            gl.texture(None);
            gl.draw_mode(self.mode);
            gl.geometry(&self.vertices, &self.indices);
        }
        self.vertices.clear();
        self.indices.clear();
    }

    /// A connected line through the points (Lines batches only).
    pub fn polyline(&mut self, points: &[(Vec3, Color)], closed: bool) {
        let n = points.len();
        if n < 2 {
            return;
        }
        let segments = if closed { n } else { n - 1 };
        self.make_room(n, segments * 2);
        let base = self.vertices.len() as u16;
        self.vertices.extend(points.iter().map(|&(p, c)| vertex(p, c)));
        for s in 0..segments {
            self.indices.push(base + s as u16);
            self.indices.push(base + ((s + 1) % n) as u16);
        }
    }

    pub fn segment(&mut self, a: Vec3, b: Vec3, color: Color) {
        self.polyline(&[(a, color), (b, color)], false);
    }

    /// One triangle, or a quad as two (Triangles batches only).
    pub fn polygon(&mut self, corners: &[Vec3], color: Color) {
        let n = corners.len();
        self.make_room(n, (n - 2) * 3);
        let base = self.vertices.len() as u16;
        self.vertices.extend(corners.iter().map(|&p| vertex(p, color)));
        for k in 1..n - 1 {
            self.indices.extend([base, base + k as u16, base + k as u16 + 1]);
        }
    }
}

/// Unit UV sphere (32 slices x 32 stacks, like `gluSphere`) that is scaled,
/// moved and lit per body each frame.
pub struct Spheres {
    unit: Vec<Vec3>,
    mesh: Mesh,
}

impl Spheres {
    pub fn new() -> Self {
        const SLICES: usize = 32;
        const STACKS: usize = 32;
        let mut unit = Vec::with_capacity((SLICES + 1) * (STACKS + 1));
        for i in 0..=STACKS {
            let phi = std::f32::consts::PI * i as f32 / STACKS as f32;
            for j in 0..=SLICES {
                let theta = std::f32::consts::TAU * j as f32 / SLICES as f32;
                unit.push(vec3(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()));
            }
        }
        let mut indices = Vec::with_capacity(SLICES * STACKS * 6);
        let w = (SLICES + 1) as u16;
        for i in 0..STACKS as u16 {
            for j in 0..SLICES as u16 {
                let (a, b) = (i * w + j, (i + 1) * w + j);
                indices.extend([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        let vertices = unit.iter().map(|&p| vertex(p, WHITE)).collect();
        Self { unit, mesh: Mesh { vertices, indices, texture: None } }
    }

    /// Fixed-function GL with `GL_COLOR_MATERIAL`: colour times
    /// (global ambient 0.2 + light ambient + light diffuse * max(0, n.l)).
    /// Light values are the ones `init_gl` set.
    pub fn draw(&mut self, center: Vec3, radius: f32, color: [f32; 3], light: Vec3) {
        const AMBIENT: [f32; 3] = [0.2 + 0.3, 0.2 + 0.3, 0.2 + 0.35];
        const DIFFUSE: [f32; 3] = [0.9, 0.9, 0.85];
        for (v, n) in self.mesh.vertices.iter_mut().zip(&self.unit) {
            let p = center + *n * radius;
            let lambert = n.dot((light - p).normalize()).max(0.0);
            let lit = |k: usize| (color[k] * (AMBIENT[k] + DIFFUSE[k] * lambert)).min(1.0);
            v.position = p;
            v.color = Color::new(lit(0), lit(1), lit(2), 1.0).into();
        }
        draw_mesh(&self.mesh);
    }
}

/// 1,500 stars on a shell 240-360 units out (`Starfield`).
pub struct Stars(Vec<(Vec3, f32)>);

impl Stars {
    pub fn new(rng: &mut gravity_sim::presets::Rng) -> Self {
        let stars = (0..1500)
            .map(|_| {
                let theta = std::f64::consts::TAU * rng.random();
                let phi = (2.0 * rng.random() - 1.0).acos();
                let r = 300.0 * (0.8 + 0.4 * rng.random());
                let p = dvec3(r * phi.sin() * theta.cos(), r * phi.sin() * theta.sin(), r * phi.cos());
                (p.as_vec3(), (0.5 + 0.5 * rng.random()) as f32)
            })
            .collect();
        Self(stars)
    }

    /// Each star is a small camera-facing square about 1.5 px across, the
    /// size of the Python's `glPointSize(1.5)` points.
    pub fn draw(&self, cam: &OrbitCamera, batch: &mut Batch) {
        let eye = cam.eye().as_vec3();
        let forward = (cam.pan.as_vec3() - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        let px_per_unit_at_1 = screen_height() / (2.0 * (22.5f32).to_radians().tan());
        for &(p, b) in &self.0 {
            let half = 0.75 * (p - eye).length() / px_per_unit_at_1;
            let (r, u) = (right * half, up * half);
            batch.polygon(&[p - r - u, p + r - u, p + r + u, p - r + u], Color::new(b, b, b, 0.8));
        }
        batch.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_through_screen_centre_hits_the_target() {
        let cam = OrbitCamera::new();
        let (origin, dir) = cam.ray(vec2(600.0, 400.0), vec2(1200.0, 800.0));
        let to_target = (cam.pan - origin).normalize();
        assert!(dir.dot(to_target) > 0.999_999, "dir {dir:?} vs {to_target:?}");
        assert!((origin - cam.eye()).length() < 0.2, "starts at the near plane");
    }

    #[test]
    fn ray_through_screen_edges_goes_the_right_way() {
        let cam = OrbitCamera { distance: 40.0, rot_x: 0.0, rot_y: 0.0, pan: DVec3::ZERO };
        // Camera on +Z looking at the origin: right of screen is +X, top is +Y.
        let (_, right) = cam.ray(vec2(1200.0, 400.0), vec2(1200.0, 800.0));
        let (_, top) = cam.ray(vec2(600.0, 0.0), vec2(1200.0, 800.0));
        assert!(right.x > 0.3 && top.y > 0.3, "{right:?} {top:?}");
        // Half the vertical field of view (22.5 degrees) at the top edge.
        assert!((top.y.asin().to_degrees() - 22.5).abs() < 1e-6);
    }
}
