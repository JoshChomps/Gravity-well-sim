//! The app: window, input, frame loop and panels. Runs natively and, built
//! for `wasm32-unknown-unknown`, in a browser (see README).

// Release builds are a Windows GUI app, so starting the .exe opens only the
// simulator window, no console. Debug builds keep the console for panics.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use gravity_sim::field::{sheet, volume, Arrow, Grid};
use gravity_sim::physics::{Body, Merge, Simulation};
use gravity_sim::presets::{self, Rng};
use macroquad::input::utils::{register_input_subscriber, repeat_all_miniquad_input};
use macroquad::prelude::*;

mod render;
mod ui;

use render::{Batch, OrbitCamera, Spheres, Stars};
use ui::{Button, Editor, EditorAction, FONT, TITLE_FONT};

/// Longest single physics step. At 1x and 60 fps a frame is exactly one
/// step, as in the Python; faster time scales are split into steps this
/// long instead of taking one long, less accurate step.
const MAX_STEP: f64 = 1.0 / 60.0;
/// A stalled frame (window dragged, tab hidden) counts as at most this
/// long, so it cannot fling bodies apart (the web version's guard).
const MAX_FRAME_DT: f64 = 0.05;
/// Trail points are recorded 60 times a real second, so a trail is the
/// Python's 200 points = 3.3 s long whatever the monitor's refresh rate.
const TRAIL_INTERVAL: f64 = 1.0 / 60.0;

fn window_conf() -> macroquad::conf::Conf {
    macroquad::conf::Conf {
        miniquad_conf: miniquad::conf::Conf {
            window_title: "3D Gravity Simulator - Spacetime Curvature Visualization".into(),
            window_width: 1200,
            window_height: 800,
            window_resizable: true,
            sample_count: 4,
            // Off everywhere. Natively, miniquad would read 1200x800 as
            // physical pixels (960x640 at 125% scaling, too small for the
            // panels). In the browser it reads the pixel ratio once at start
            // while the JS side rereads it on every resize, so a ratio change
            // (zoom, another monitor) pushed the right and bottom panels off
            // screen. Off, every coordinate stays in CSS pixels.
            high_dpi: false,
            ..Default::default()
        },
        // Room for a whole sphere (6,144 indices) or a full line batch.
        draw_call_vertex_capacity: 16_384,
        draw_call_index_capacity: 32_768,
        ..Default::default()
    }
}

/// One way of drawing spacetime curvature. The two views share nothing
/// but this shape; their maths lives in `field::sheet` and `field::volume`.
struct View {
    name: &'static str,
    build: fn(usize) -> Grid,
    displace: fn(&Grid, &[Body], &mut Vec<DVec3>),
    arrows: fn(&[Body]) -> Vec<Arrow>,
    min: usize,
    max: usize,
    step: usize,
    alpha: f32,
    grid: Grid,
    points: Vec<DVec3>,
}

impl View {
    fn sheet() -> Self {
        Self {
            name: "Sheet",
            build: sheet::grid,
            displace: sheet::displace,
            arrows: sheet::arrows,
            min: sheet::MIN_RESOLUTION,
            max: sheet::MAX_RESOLUTION,
            step: sheet::RESOLUTION_STEP,
            alpha: 0.4,
            grid: sheet::grid(sheet::DEFAULT_RESOLUTION),
            points: Vec::new(),
        }
    }

    fn volume() -> Self {
        Self {
            name: "Volume",
            build: volume::grid,
            displace: volume::displace,
            arrows: volume::arrows,
            min: volume::MIN_RESOLUTION,
            max: volume::MAX_RESOLUTION,
            step: volume::RESOLUTION_STEP,
            alpha: 0.25,
            grid: volume::grid(volume::DEFAULT_RESOLUTION),
            points: Vec::new(),
        }
    }

    fn resolution(&self) -> usize {
        self.grid.resolution
    }

    fn change_resolution(&mut self, up: bool) {
        let r = self.resolution();
        let r = if up { (r + self.step).min(self.max) } else { r.saturating_sub(self.step).max(self.min) };
        if r != self.resolution() {
            self.grid = (self.build)(r);
        }
    }

    fn draw(&mut self, bodies: &[Body], batch: &mut Batch) {
        (self.displace)(&self.grid, bodies, &mut self.points);
        let color = Color::new(0.2, 0.4, 0.8, self.alpha);
        let mut line = Vec::new();
        for l in &self.grid.lines {
            line.clear();
            line.extend(l.indices().map(|i| (self.points[i].as_vec3(), color)));
            batch.polyline(&line, false);
        }
        batch.flush();
    }
}

/// Hit areas of the left panel, shared by drawing and clicking.
struct LeftPanel {
    rect: Rect,
    toggle: Rect,
    views: [Rect; 2],
    toggles: [Rect; 5],
    res_minus: Button,
    res_plus: Button,
}

struct PresetsPanel {
    rect: Rect,
    toggle: Rect,
    rows: [Rect; 6],
}

const CONTROLS: [(&str, &str); 14] = [
    ("SPACE", "Pause/Resume"),
    ("R", "Reset preset"),
    ("1-6", "Presets"),
    ("N", "Add planet"),
    ("M", "Sheet / Volume"),
    ("G", "Grid"),
    ("T", "Trails"),
    ("V", "Velocity"),
    ("F", "Forces"),
    ("C", "Track planet"),
    ("+/-", "Time scale"),
    ("Del", "Delete planet"),
    ("Click", "Select"),
    ("Drag", "Rotate L / Pan R"),
];

/// One input event, kept in arrival order.
enum Event {
    LeftDown(Vec2),
    Key(KeyCode),
    Char(char),
}

/// Collects this frame's clicks, key presses and typed characters from
/// macroquad's ordered event replay.
#[derive(Default)]
struct Events(Vec<Event>);

impl miniquad::EventHandler for Events {
    fn update(&mut self) {}
    fn draw(&mut self) {}
    fn mouse_button_down_event(&mut self, button: MouseButton, x: f32, y: f32) {
        if button == MouseButton::Left {
            // Raw events are in physical pixels; the layout is logical.
            self.0.push(Event::LeftDown(vec2(x, y) / screen_dpi_scale()));
        }
    }
    fn key_down_event(&mut self, keycode: KeyCode, _mods: miniquad::KeyMods, repeat: bool) {
        if !repeat {
            self.0.push(Event::Key(keycode));
        }
    }
    fn char_event(&mut self, character: char, _mods: miniquad::KeyMods, _repeat: bool) {
        self.0.push(Event::Char(character));
    }
}

struct App {
    sim: Simulation,
    rng: Rng,
    preset: u8,
    paused: bool,
    time_scale: f64,
    selected: Option<u32>,
    tracking: bool,
    show_grid: bool,
    show_trails: bool,
    show_velocity: bool,
    show_forces: bool,
    views: [View; 2],
    view: usize,
    cam: OrbitCamera,
    rotating: bool,
    panning: bool,
    last_mouse: Vec2,
    left_collapsed: bool,
    presets_collapsed: bool,
    editor: Editor,
    spheres: Spheres,
    stars: Stars,
    lines: Batch,
    tris: Batch,
    trail_clock: f64,
    input_subscriber: usize,
    /// Smoothed frames per second for the status readout.
    fps: f32,
    quit: bool,
}

impl App {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let stars = Stars::new(&mut rng);
        let mut app = Self {
            sim: Simulation::new(),
            rng,
            preset: 1,
            paused: false,
            time_scale: 1.0,
            selected: None,
            tracking: false,
            show_grid: true,
            show_trails: true,
            show_velocity: false,
            show_forces: false,
            views: [View::sheet(), View::volume()],
            view: 0,
            cam: OrbitCamera::new(),
            rotating: false,
            panning: false,
            last_mouse: Vec2::from(mouse_position()),
            left_collapsed: false,
            presets_collapsed: false,
            editor: Editor::new(),
            spheres: Spheres::new(),
            stars,
            lines: Batch::lines(),
            tris: Batch::triangles(),
            trail_clock: 0.0,
            input_subscriber: register_input_subscriber(),
            fps: 60.0,
            quit: false,
        };
        app.load_preset(1);
        app
    }

    fn load_preset(&mut self, n: u8) {
        presets::load(&mut self.sim, n, &mut self.rng);
        self.preset = n;
        self.select(None);
    }

    fn select(&mut self, id: Option<u32>) {
        self.selected = id;
        self.editor.select(id.and_then(|id| self.sim.get(id)));
    }

    fn add_random_planet(&mut self) {
        presets::add_random_planet(&mut self.sim, &mut self.rng);
        // The Python left the new body's acceleration at zero for its first
        // half-step; start it (and its pull on the others) correctly.
        self.sim.init_accelerations();
    }

    fn delete_selected(&mut self) {
        if let Some(id) = self.selected {
            self.sim.remove(id);
            self.sim.init_accelerations();
            self.select(None);
        }
    }

    /// A merge that swallowed the selected body selects the result.
    fn on_merges(&mut self, merges: Vec<Merge>) {
        for m in merges {
            if self.selected.is_some_and(|s| m.absorbed.contains(&s)) {
                self.select(Some(m.into));
            }
        }
    }

    fn left_panel(&self) -> LeftPanel {
        let h = if self.left_collapsed { 40.0 } else { 565f32.min(screen_height() - 20.0).max(40.0) };
        let row = |y: f32| Rect::new(20.0, y - 1.0, 180.0, 16.0);
        LeftPanel {
            rect: Rect::new(10.0, 10.0, 300.0, h),
            toggle: Rect::new(275.0, 10.0, 35.0, 35.0),
            views: [Rect::new(80.0, 150.0, 70.0, 20.0), Rect::new(155.0, 150.0, 70.0, 20.0)],
            toggles: [row(451.0), row(467.0), row(483.0), row(499.0), row(515.0)],
            res_minus: Button::new(160.0, 539.0, 30.0, 20.0, "-", ui::rgb(60, 100, 150)),
            res_plus: Button::new(200.0, 539.0, 30.0, 20.0, "+", ui::rgb(60, 100, 150)),
        }
    }

    fn presets_panel(&self) -> PresetsPanel {
        let h = if self.presets_collapsed { 30.0 } else { 170.0 };
        let y = screen_height() - h - 10.0;
        // Below the left panel when there is room, otherwise beside it.
        let left = self.left_panel().rect;
        let x = if y >= left.y + left.h + 10.0 { 10.0 } else { left.x + left.w + 10.0 };
        PresetsPanel {
            rect: Rect::new(x, y, 180.0, h),
            toggle: Rect::new(x + 145.0, y - 2.0, 35.0, 34.0),
            rows: std::array::from_fn(|i| Rect::new(x + 5.0, y + 29.0 + i as f32 * 18.0, 170.0, 18.0)),
        }
    }

    fn exit_button(&self) -> Option<Button> {
        // A browser tab cannot be closed by the page, so only native has it.
        if cfg!(target_arch = "wasm32") {
            return None;
        }
        Some(Button::new(screen_width() - 110.0, screen_height() - 50.0, 100.0, 35.0, "EXIT", ui::rgb(150, 50, 50)))
    }

    fn in_ui(&self, p: Vec2) -> bool {
        self.left_panel().rect.contains(p)
            || self.editor.panel_rect().contains(p)
            || self.presets_panel().rect.contains(p)
            || self.exit_button().is_some_and(|b| b.hit(p))
    }

    /// Clicks on the left and presets panels. Returns true if one was hit.
    fn click_panels(&mut self, p: Vec2) -> bool {
        if self.exit_button().is_some_and(|b| b.hit(p)) {
            self.quit = true;
            return true;
        }
        let left = self.left_panel();
        if left.toggle.contains(p) {
            self.left_collapsed = !self.left_collapsed;
            return true;
        }
        if !self.left_collapsed && left.rect.contains(p) {
            if let Some(v) = left.views.iter().position(|r| r.contains(p)) {
                self.view = v;
            }
            match left.toggles.iter().position(|r| r.contains(p)) {
                Some(0) => self.show_grid = !self.show_grid,
                Some(1) => self.show_trails = !self.show_trails,
                Some(2) => self.show_velocity = !self.show_velocity,
                Some(3) => self.show_forces = !self.show_forces,
                Some(4) => self.tracking = !self.tracking,
                _ => {}
            }
            if left.res_minus.hit(p) {
                self.views[self.view].change_resolution(false);
            }
            if left.res_plus.hit(p) {
                self.views[self.view].change_resolution(true);
            }
            return true;
        }
        let presets = self.presets_panel();
        if presets.toggle.contains(p) {
            self.presets_collapsed = !self.presets_collapsed;
            return true;
        }
        if presets.rect.contains(p) {
            if !self.presets_collapsed {
                if let Some(i) = presets.rows.iter().position(|r| r.contains(p)) {
                    self.load_preset(i as u8 + 1);
                }
            }
            return true;
        }
        left.rect.contains(p)
    }

    fn shortcut(&mut self, k: KeyCode) {
        match k {
            KeyCode::Key1 | KeyCode::Key2 | KeyCode::Key3 | KeyCode::Key4 | KeyCode::Key5 | KeyCode::Key6 => {
                self.load_preset(k as u8 - KeyCode::Key0 as u8)
            }
            KeyCode::Space => self.paused = !self.paused,
            KeyCode::R => self.load_preset(self.preset),
            KeyCode::N => self.add_random_planet(),
            KeyCode::M => self.view = 1 - self.view,
            KeyCode::G => self.show_grid = !self.show_grid,
            KeyCode::T => self.show_trails = !self.show_trails,
            KeyCode::V => self.show_velocity = !self.show_velocity,
            KeyCode::F => self.show_forces = !self.show_forces,
            KeyCode::C => self.tracking = !self.tracking,
            KeyCode::Equal | KeyCode::KpAdd => self.time_scale = (self.time_scale + 0.25).min(5.0),
            KeyCode::Minus | KeyCode::KpSubtract => self.time_scale = (self.time_scale - 0.25).max(0.25),
            KeyCode::Delete | KeyCode::Backspace => self.delete_selected(),
            _ => {}
        }
    }

    fn left_click(&mut self, p: Vec2) {
        let body = self.selected.and_then(|id| self.sim.get_mut(id));
        match self.editor.click(p, body) {
            EditorAction::Apply => {
                if let Some(b) = self.selected.and_then(|id| self.sim.get_mut(id)) {
                    self.editor.apply_to(b);
                }
                self.sim.init_accelerations();
            }
            EditorAction::Delete => self.delete_selected(),
            EditorAction::Consumed => {}
            EditorAction::None => {
                if !self.click_panels(p) && !self.in_ui(p) {
                    self.rotating = true;
                    let hit = self.cam.pick(p, &self.sim.bodies);
                    self.select(hit);
                }
            }
        }
    }

    fn input(&mut self) {
        let mouse = Vec2::from(mouse_position());

        // Clicks, keys and characters are handled in the order they
        // happened, like the Python's event loop. Several can arrive within
        // one frame; polling them per frame lost that order (a click into a
        // box, the digits typed next and Enter could be applied out of turn).
        let mut events = Events::default();
        repeat_all_miniquad_input(&mut events, self.input_subscriber);
        for e in events.0 {
            match e {
                Event::LeftDown(p) => self.left_click(p),
                Event::Char(c) => {
                    let body = self.selected.and_then(|id| self.sim.get_mut(id));
                    self.editor.type_char(c, body);
                }
                Event::Key(k) if self.editor.typing() => {
                    let body = self.selected.and_then(|id| self.sim.get_mut(id));
                    self.editor.key(k, body);
                }
                Event::Key(k) => self.shortcut(k),
            }
        }

        if is_mouse_button_pressed(MouseButton::Right) && !self.in_ui(mouse) {
            self.panning = true;
        }
        if !is_mouse_button_down(MouseButton::Left) {
            self.rotating = false;
        }
        if !is_mouse_button_down(MouseButton::Right) {
            self.panning = false;
        }

        // A drag starts from where the button went down (the Python reset
        // its last position on every press), so a cursor that jumped while
        // outside the window does not spin the camera.
        if is_mouse_button_pressed(MouseButton::Left) || is_mouse_button_pressed(MouseButton::Right) {
            self.last_mouse = mouse;
        }
        let d = (mouse - self.last_mouse).as_dvec2();
        if !self.in_ui(mouse) {
            if self.rotating {
                self.cam.rot_y += d.x * 0.5;
                self.cam.rot_x = (self.cam.rot_x + d.y * 0.5).clamp(-89.0, 89.0);
            }
            if self.panning {
                // Pan along the ground, relative to where the camera faces
                // (the Python panned along world X/Z, which only matched the
                // mouse when looking straight down the Z axis).
                let ry = self.cam.rot_y.to_radians();
                let right = dvec3(ry.cos(), 0.0, -ry.sin());
                let forward = dvec3(-ry.sin(), 0.0, -ry.cos());
                self.cam.pan += -right * d.x * 0.05 + forward * d.y * 0.05;
            }
        }
        self.last_mouse = mouse;

        let wheel = mouse_wheel().1;
        if wheel > 0.0 {
            self.cam.distance = (self.cam.distance - 2.0).max(10.0);
        } else if wheel < 0.0 {
            self.cam.distance = (self.cam.distance + 2.0).min(100.0);
        }
    }

    fn update(&mut self) {
        if !self.paused {
            let frame = (get_frame_time() as f64).min(MAX_FRAME_DT);
            let merges = self.sim.advance(frame * self.time_scale, MAX_STEP);
            self.on_merges(merges);
            self.trail_clock += frame;
            if self.trail_clock >= TRAIL_INTERVAL {
                self.sim.record_trails();
                self.trail_clock = (self.trail_clock - TRAIL_INTERVAL).min(TRAIL_INTERVAL);
            }
        }
        match self.selected.map(|id| self.sim.get(id)) {
            Some(Some(b)) => {
                self.editor.refresh(b);
                if self.tracking {
                    self.cam.pan = b.position;
                }
            }
            Some(None) => {
                self.tracking = false;
                self.select(None);
            }
            None => {}
        }
        self.editor.layout(screen_width());
        let ft = get_frame_time().max(1e-4);
        self.fps += (1.0 / ft - self.fps) * 0.05;
    }

    fn draw_scene(&mut self) {
        clear_background(Color::new(0.02, 0.02, 0.05, 1.0));
        set_camera(&self.cam.camera());

        self.stars.draw(&self.cam, &mut self.tris);
        if self.show_grid {
            self.views[self.view].draw(&self.sim.bodies, &mut self.lines);
        }
        if self.show_forces {
            let red = Color::new(1.0, 0.2, 0.2, 0.8);
            for a in (self.views[self.view].arrows)(&self.sim.bodies) {
                self.lines.segment(a.start.as_vec3(), a.end.as_vec3(), red);
            }
            self.lines.flush();
        }
        if self.show_trails {
            let mut pts = Vec::new();
            for b in &self.sim.bodies {
                let n = b.trail.len();
                pts.clear();
                pts.extend(b.trail.iter().enumerate().map(|(i, p)| (p.as_vec3(), render::rgba(b.color, i as f32 / n as f32 * 0.8))));
                self.lines.polyline(&pts, false);
            }
            self.lines.flush();
        }

        let light = self.cam.light_position();
        let pulse = 0.5 + 0.5 * (get_time() as f32 * 5.0).sin();
        for b in &self.sim.bodies {
            let mut c = b.color.map(|v| v as f32);
            if Some(b.id) == self.selected {
                c = c.map(|v| (v + 0.3 * pulse).min(1.0));
            }
            self.spheres.draw(b.position.as_vec3(), b.radius as f32, c, light);
        }
        for b in &self.sim.bodies {
            let p = b.position.as_vec3();
            if Some(b.id) == self.selected {
                let r = b.radius as f32 * 1.5;
                let ring: Vec<_> = (0..64)
                    .map(|i| {
                        let a = std::f32::consts::TAU * i as f32 / 64.0;
                        (p + vec3(r * a.cos(), 0.0, r * a.sin()), Color::new(1.0, 1.0, 1.0, 0.8))
                    })
                    .collect();
                self.lines.polyline(&ring, true);
            }
            if self.show_velocity && b.velocity.length() > 0.01 {
                let green = Color::new(0.2, 1.0, 0.3, 0.9);
                let tip = (b.position + b.velocity * 2.0).as_vec3();
                self.lines.segment(p, tip, green);
                let dir = b.velocity.normalize().as_vec3();
                let side = if dir.y.abs() < 0.9 { dir.cross(Vec3::Y) } else { dir.cross(Vec3::X) };
                let side = side.normalize() * 0.15;
                let base = tip - dir * 0.3;
                self.tris.polygon(&[tip, base + side, base - side], green);
            }
        }
        self.lines.flush();
        self.tris.flush();
    }

    fn draw_ui(&mut self) {
        set_default_camera();
        let mouse = Vec2::from(mouse_position());
        let panel_fill = ui::rgba(10, 15, 25, 220);
        let panel_border = ui::rgba(60, 100, 150, 150);

        let left = self.left_panel();
        ui::panel(left.rect, panel_fill, panel_border);
        ui::collapse_box(Rect::new(280.0, 18.0, 24.0, 24.0), self.left_collapsed);
        ui::text("3D GRAVITY SIMULATOR", 20.0, 20.0, TITLE_FONT, ui::TITLE);
        if !self.left_collapsed {
            ui::text("Spacetime Curvature Visualization", 20.0, 45.0, FONT, ui::MUTED);
            let (status, color) = if self.paused { ("PAUSED", ui::rgb(255, 150, 100)) } else { ("RUNNING", ui::rgb(100, 255, 150)) };
            ui::text(&format!("Status: {status}"), 20.0, 75.0, FONT, color);
            ui::text(&format!("Time Scale: {:.2}x", self.time_scale), 20.0, 93.0, FONT, ui::TEXT);
            ui::text(&format!("Planets: {}", self.sim.bodies.len()), 20.0, 111.0, FONT, ui::TEXT);
            ui::text(&format!("FPS: {:.0}", self.fps), 20.0, 129.0, FONT, ui::TEXT);

            ui::text("View:", 25.0, 152.0, FONT, ui::TEXT);
            for (i, r) in left.views.iter().enumerate() {
                let on = i == self.view;
                draw_rectangle(r.x, r.y, r.w, r.h, if on { ui::rgb(60, 100, 150) } else { ui::rgb(25, 35, 50) });
                draw_rectangle_lines(r.x, r.y, r.w, r.h, 1.0, ui::rgb(100, 140, 180));
                ui::text_centered(self.views[i].name, r.x + r.w / 2.0, r.y + r.h / 2.0, FONT, if on { WHITE } else { ui::DIM });
            }

            ui::section("CONTROLS", 20.0, 182.0, 280.0, ui::TITLE);
            for (i, (key, action)) in CONTROLS.iter().enumerate() {
                let y = 202.0 + i as f32 * 16.0;
                ui::text(key, 25.0, y, FONT, ui::rgb(255, 200, 100));
                ui::text(action, 100.0, y, FONT, ui::DIM);
            }

            ui::section("TOGGLES", 20.0, 431.0, 280.0, ui::TITLE);
            let states = [
                ("Grid", self.show_grid),
                ("Trails", self.show_trails),
                ("Velocity", self.show_velocity),
                ("Forces", self.show_forces),
                ("Tracking", self.tracking),
            ];
            for ((name, on), r) in states.iter().zip(&left.toggles) {
                let color = if *on { ui::rgb(100, 255, 150) } else { ui::rgb(255, 100, 100) };
                ui::dot(r.x + 9.0, r.y + 8.0, *on, color);
                ui::text(name, r.x + 20.0, r.y + 1.0, FONT, color);
            }

            let view = &self.views[self.view];
            ui::text(&format!("Grid Res: {}", view.resolution()), 25.0, 541.0, FONT, ui::TEXT);
            left.res_minus.draw(mouse);
            left.res_plus.draw(mouse);
        }

        let selected = self.selected.and_then(|id| self.sim.get(id));
        self.editor.draw(selected, mouse, get_frame_time());

        let presets = self.presets_panel();
        let r = presets.rect;
        ui::panel(r, panel_fill, panel_border);
        ui::collapse_box(Rect::new(r.x + 150.0, r.y + 3.0, 24.0, 24.0), self.presets_collapsed);
        ui::section("PRESETS", r.x + 12.0, r.y + 7.0, 130.0, ui::TITLE);
        if !self.presets_collapsed {
            for (i, row) in presets.rows.iter().enumerate() {
                let current = i as u8 + 1 == self.preset;
                if row.contains(mouse) {
                    draw_rectangle(row.x, row.y, row.w, row.h, ui::rgba(60, 100, 150, 90));
                }
                let color = if current { ui::TITLE } else { ui::DIM };
                ui::text(&format!("{}: {}", i + 1, presets::NAMES[i]), r.x + 10.0, row.y + 1.0, FONT, color);
            }
            let sy = r.y + 30.0 + 6.0 * 18.0 + 5.0;
            draw_line(r.x + 10.0, sy, r.x + 170.0, sy, 1.0, ui::rgb(60, 80, 100));
            ui::text("Scale: 1.0 = Earth", r.x + 10.0, sy + 6.0, FONT, ui::rgb(120, 140, 160));
        }

        if let Some(b) = self.exit_button() {
            b.draw(mouse);
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let seed = (macroquad::miniquad::date::now() * 1000.0) as u64;
    let mut app = App::new(seed);
    loop {
        app.input();
        if app.quit {
            break;
        }
        app.update();
        app.draw_scene();
        app.draw_ui();
        next_frame().await;
    }
}

