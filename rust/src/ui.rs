//! 2D overlay widgets, ported from `../ui.py`: buttons, numeric input
//! fields and the planet editor panel. Drawn every frame (macroquad is
//! immediate mode, so the Python's "dirty texture" cache is not needed).

use gravity_sim::physics::{self, Body, G};
use macroquad::prelude::*;

pub const FONT: f32 = 16.0;
pub const TITLE_FONT: f32 = 22.0;

pub fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::from_rgba(r, g, b, 255)
}

pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
    Color::from_rgba(r, g, b, a)
}

pub const TITLE: Color = Color::new(100. / 255., 180. / 255., 1.0, 1.0);
pub const SECTION: Color = Color::new(80. / 255., 140. / 255., 200. / 255., 1.0);
pub const TEXT: Color = Color::new(200. / 255., 200. / 255., 200. / 255., 1.0);
pub const DIM: Color = Color::new(180. / 255., 180. / 255., 180. / 255., 1.0);
pub const MUTED: Color = Color::new(150. / 255., 150. / 255., 170. / 255., 1.0);

/// Text with its top-left corner at (x, y), like a pygame blit.
pub fn text(s: &str, x: f32, y: f32, size: f32, color: Color) {
    let ascent = measure_text("Ag", None, size as u16, 1.0).offset_y;
    draw_text(s, x.round(), (y + ascent).round(), size, color);
}

pub fn text_width(s: &str, size: f32) -> f32 {
    measure_text(s, None, size as u16, 1.0).width
}

/// Text centred on (cx, cy).
pub fn text_centered(s: &str, cx: f32, cy: f32, size: f32, color: Color) {
    let d = measure_text("Ag", None, size as u16, 1.0);
    text(s, cx - text_width(s, size) / 2.0, cy - d.height / 2.0, size, color);
}

/// "--- LABEL ---" style section header, drawn as lines either side.
pub fn section(label: &str, x: f32, y: f32, width: f32, color: Color) {
    let w = text_width(label, FONT);
    let cy = y + 8.0;
    let gap = 8.0;
    let side = ((width - w) / 2.0 - gap).clamp(6.0, 30.0);
    draw_line(x, cy, x + side, cy, 1.0, color);
    text(label, x + side + gap, y, FONT, color);
    let x2 = x + side + gap + w + gap;
    draw_line(x2, cy, x2 + side, cy, 1.0, color);
}

pub fn panel(r: Rect, fill: Color, border: Color) {
    draw_rectangle(r.x, r.y, r.w, r.h, fill);
    draw_rectangle_lines(r.x, r.y, r.w, r.h, 2.0, border);
}

/// The small square with "-" (expanded) or "+" (collapsed).
pub fn collapse_box(r: Rect, collapsed: bool) {
    draw_rectangle(r.x, r.y, r.w, r.h, rgb(40, 60, 80));
    let (cx, cy) = (r.x + r.w / 2.0, r.y + r.h / 2.0);
    draw_line(cx - 5.0, cy, cx + 5.0, cy, 2.0, WHITE);
    if collapsed {
        draw_line(cx, cy - 5.0, cx, cy + 5.0, 2.0, WHITE);
    }
}

/// Filled (on) or hollow (off) dot, for the toggles list.
pub fn dot(x: f32, y: f32, on: bool, color: Color) {
    if on {
        draw_circle(x, y, 4.5, color);
    } else {
        draw_circle_lines(x, y, 4.5, 1.5, color);
    }
}

pub struct Button {
    pub rect: Rect,
    pub label: &'static str,
    pub color: Color,
}

impl Button {
    pub fn new(x: f32, y: f32, w: f32, h: f32, label: &'static str, color: Color) -> Self {
        Self { rect: Rect::new(x, y, w, h), label, color }
    }

    pub fn hit(&self, p: Vec2) -> bool {
        self.rect.contains(p)
    }

    pub fn draw(&self, mouse: Vec2) {
        let c = if self.hit(mouse) {
            Color::new((self.color.r + 0.12).min(1.0), (self.color.g + 0.12).min(1.0), (self.color.b + 0.12).min(1.0), 1.0)
        } else {
            self.color
        };
        let r = self.rect;
        draw_rectangle(r.x, r.y, r.w, r.h, c);
        draw_rectangle_lines(r.x, r.y, r.w, r.h, 2.0, rgb(100, 140, 180));
        text_centered(self.label, r.x + r.w / 2.0, r.y + r.h / 2.0, FONT, WHITE);
    }
}

/// A numeric text box: click to type (the old text is cleared), Enter or
/// Tab to confirm, Esc to cancel. Values are clamped to [min, max].
pub struct InputField {
    pub rect: Rect,
    pub label: &'static str,
    pub value: f64,
    pub text: String,
    pub active: bool,
    /// Typed and confirmed but not yet applied: live updates leave it alone.
    pub edited: bool,
    pub read_only: bool,
    min: f64,
    max: f64,
    blink: f32,
}

impl InputField {
    fn new(label: &'static str, min: f64, max: f64) -> Self {
        Self {
            rect: Rect::new(0., 0., 0., 0.),
            label,
            value: 0.0,
            text: "0.00".into(),
            active: false,
            edited: false,
            read_only: false,
            min,
            max,
            blink: 0.0,
        }
    }

    /// Shows `v` unless the user is typing in, or has typed into, this box.
    pub fn show(&mut self, v: f64) {
        if !self.active && !self.edited {
            self.value = v;
            self.text = format!("{v:.2}");
        }
    }

    fn force(&mut self, v: f64) {
        self.value = v;
        if !self.active {
            self.text = format!("{v:.2}");
        }
    }

    fn activate(&mut self) {
        self.active = true;
        self.text.clear();
        self.blink = 0.0;
    }

    /// Ends typing; keeps the last number that parsed.
    fn confirm(&mut self) {
        if let Ok(v) = self.text.parse::<f64>() {
            self.value = v.clamp(self.min, self.max);
            self.edited = true;
        }
        self.text = format!("{:.2}", self.value);
        self.active = false;
    }

    fn cancel(&mut self) {
        self.text = format!("{:.2}", self.value);
        self.active = false;
    }

    /// Adds a typed character. Returns true when the value changed (it
    /// updates as you type, like the Python).
    fn type_char(&mut self, c: char) -> bool {
        if !"0123456789.-".contains(c) {
            return false;
        }
        self.text.push(c);
        match self.text.parse::<f64>() {
            Ok(v) => {
                self.value = v.clamp(self.min, self.max);
                true
            }
            Err(_) => false,
        }
    }

    fn draw(&mut self, dt: f32) {
        let r = self.rect;
        let (bg, border) = if self.active { (rgb(40, 50, 70), TITLE) } else { (rgb(25, 35, 50), rgb(60, 80, 100)) };
        draw_rectangle(r.x, r.y, r.w, r.h, bg);
        draw_rectangle_lines(r.x, r.y, r.w, r.h, 2.0, border);
        text(self.label, r.x, r.y - 18.0, FONT, rgb(150, 170, 200));
        self.blink = (self.blink + dt) % 1.0;
        let caret = if self.active && self.blink < 0.5 { "|" } else { "" };
        let color = if self.read_only { rgb(170, 190, 210) } else { WHITE };
        text(&format!("{}{caret}", self.text), r.x + 8.0, r.y + (r.h - 16.0) / 2.0, FONT, color);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum F {
    Mass,
    Radius,
    Density,
    SurfaceG,
    NetForce,
    PosX,
    PosY,
    PosZ,
    VelX,
    VelY,
    VelZ,
}

const FIELDS: [F; 11] = [F::Mass, F::Radius, F::Density, F::SurfaceG, F::NetForce, F::PosX, F::PosY, F::PosZ, F::VelX, F::VelY, F::VelZ];

/// What a click on the editor did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EditorAction {
    None,
    Consumed,
    Apply,
    Delete,
}

/// Right-hand planet editor (`PlanetEditor`). Mass, radius and density are
/// linked (rho = m / V) and change the body as you type; position and
/// velocity follow the body until you type into them, then wait for Apply.
pub struct Editor {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub collapsed: bool,
    fields: Vec<InputField>,
    pub apply: Button,
    pub delete: Button,
}

impl Editor {
    pub fn new() -> Self {
        let mut fields = vec![
            InputField::new("Mass (m)", 1.0, 5000.0),
            InputField::new("Radius (r)", 0.1, 10.0),
            InputField::new("Density (rho)", 0.1, 1000.0),
            InputField::new("Surface g", 0.0, 99999.0),
            InputField::new("Net Force (F = ma)", 0.0, 0.0),
            InputField::new("Pos X", -50.0, 50.0),
            InputField::new("Y", -50.0, 50.0),
            InputField::new("Z", -50.0, 50.0),
            InputField::new("Vel X", -20.0, 20.0),
            InputField::new("Y", -20.0, 20.0),
            InputField::new("Z", -20.0, 20.0),
        ];
        fields[F::SurfaceG as usize].read_only = true;
        fields[F::NetForce as usize].read_only = true;
        let mut e = Self {
            x: 0.0,
            y: 20.0,
            width: 260.0,
            height: 620.0,
            collapsed: false,
            fields,
            apply: Button::new(0., 0., 100., 32., "Apply", rgb(40, 120, 80)),
            delete: Button::new(0., 0., 100., 32., "Delete", rgb(150, 50, 50)),
        };
        e.layout(screen_width());
        e
    }

    pub fn field(&self, f: F) -> &InputField {
        &self.fields[f as usize]
    }

    fn field_mut(&mut self, f: F) -> &mut InputField {
        &mut self.fields[f as usize]
    }

    /// Positions everything for a window `screen_w` wide (same offsets as
    /// `setup_fields`).
    pub fn layout(&mut self, screen_w: f32) {
        self.x = screen_w - 280.0;
        let (x, y) = (self.x, self.y);
        let (c1, c2) = (x + 15.0, x + 120.0);
        let (fw, fh) = (90.0, 26.0);
        let place = |f: &mut InputField, fx: f32, fy: f32, w: f32| f.rect = Rect::new(fx, fy, w, fh);
        place(&mut self.fields[F::Mass as usize], c1, y + 85.0, fw);
        place(&mut self.fields[F::Radius as usize], c2, y + 85.0, fw);
        place(&mut self.fields[F::Density as usize], c1, y + 145.0, fw);
        place(&mut self.fields[F::SurfaceG as usize], c2, y + 145.0, fw);
        place(&mut self.fields[F::NetForce as usize], c1, y + 205.0, fw * 2.0 + 15.0);
        for (k, f) in [F::PosX, F::PosY, F::PosZ].into_iter().enumerate() {
            place(&mut self.fields[f as usize], c1 + 80.0 * k as f32, y + 310.0, 70.0);
        }
        for (k, f) in [F::VelX, F::VelY, F::VelZ].into_iter().enumerate() {
            place(&mut self.fields[f as usize], c1 + 80.0 * k as f32, y + 415.0, 70.0);
        }
        self.apply.rect = Rect::new(c1, y + 480.0, 100.0, 32.0);
        self.delete.rect = Rect::new(c2 + 15.0, y + 480.0, 100.0, 32.0);
    }

    pub fn panel_rect(&self) -> Rect {
        Rect::new(self.x - 10.0, self.y - 10.0, self.width + 20.0, if self.collapsed { 40.0 } else { self.height })
    }

    fn toggle_hit_rect(&self) -> Rect {
        Rect::new(self.x + self.width - 29.0, self.y - 7.0, 34.0, 34.0)
    }

    pub fn typing(&self) -> bool {
        self.fields.iter().any(|f| f.active)
    }

    /// Fills every box from a newly selected body, dropping unapplied edits.
    pub fn select(&mut self, body: Option<&Body>) {
        for f in &mut self.fields {
            f.active = false;
            f.edited = false;
        }
        if let Some(b) = body {
            self.refresh(b);
        }
    }

    /// Keeps the boxes in step with the body as it moves.
    pub fn refresh(&mut self, b: &Body) {
        let values = [
            b.mass,
            b.radius,
            b.density(),
            b.surface_gravity(),
            b.acceleration.length() * b.mass,
            b.position.x,
            b.position.y,
            b.position.z,
            b.velocity.x,
            b.velocity.y,
            b.velocity.z,
        ];
        for (f, v) in self.fields.iter_mut().zip(values) {
            f.show(v);
        }
        // Read-only boxes always show the live value.
        self.field_mut(F::SurfaceG).force(b.surface_gravity());
        self.field_mut(F::NetForce).force(b.acceleration.length() * b.mass);
    }

    /// Handles a left click at `p`. Any box being typed in is confirmed
    /// first, as clicking elsewhere does in the Python.
    pub fn click(&mut self, p: Vec2, body: Option<&mut Body>) -> EditorAction {
        let mut body = body;
        for (i, &f) in FIELDS.iter().enumerate() {
            if self.fields[i].active && !self.fields[i].rect.contains(p) {
                self.fields[i].confirm();
                if let Some(b) = body.as_deref_mut() {
                    self.changed(f, b);
                }
            }
        }
        if self.toggle_hit_rect().contains(p) {
            self.collapsed = !self.collapsed;
            return EditorAction::Consumed;
        }
        if self.collapsed || body.is_none() {
            return if self.panel_rect().contains(p) { EditorAction::Consumed } else { EditorAction::None };
        }
        if self.apply.hit(p) {
            return EditorAction::Apply;
        }
        if self.delete.hit(p) {
            return EditorAction::Delete;
        }
        for f in &mut self.fields {
            if f.rect.contains(p) && !f.read_only {
                f.activate();
                return EditorAction::Consumed;
            }
        }
        if self.panel_rect().contains(p) {
            EditorAction::Consumed
        } else {
            EditorAction::None
        }
    }

    /// A typed character for the active box (ignored when none is active).
    pub fn type_char(&mut self, c: char, body: Option<&mut Body>) {
        let Some(i) = self.fields.iter().position(|f| f.active) else { return };
        if self.fields[i].type_char(c) {
            if let Some(b) = body {
                self.changed(FIELDS[i], b);
            }
        }
    }

    /// An editing key for the active box: Backspace, Esc to cancel, Enter
    /// or Tab to confirm. Other keys are ignored.
    pub fn key(&mut self, k: KeyCode, body: Option<&mut Body>) {
        let Some(i) = self.fields.iter().position(|f| f.active) else { return };
        match k {
            KeyCode::Backspace => {
                self.fields[i].text.pop();
            }
            KeyCode::Escape => self.fields[i].cancel(),
            KeyCode::Enter | KeyCode::KpEnter | KeyCode::Tab => {
                self.fields[i].confirm();
                if let Some(b) = body {
                    self.changed(FIELDS[i], b);
                }
            }
            _ => {}
        }
    }

    /// Live links between boxes: density sets mass; mass or radius update
    /// density and surface gravity. All three change the body at once.
    fn changed(&mut self, f: F, b: &mut Body) {
        match f {
            F::Density => {
                let mass = self.field(F::Density).value * physics::sphere_volume(self.field(F::Radius).value);
                self.field_mut(F::Mass).force(mass);
            }
            F::Mass | F::Radius => {
                let (m, r) = (self.field(F::Mass).value, self.field(F::Radius).value);
                self.field_mut(F::Density).force(physics::density(m, r));
            }
            _ => return,
        }
        b.mass = self.field(F::Mass).value;
        b.radius = self.field(F::Radius).value;
        let g = if b.radius > 0.0 { G * b.mass / (b.radius * b.radius) } else { 0.0 };
        self.field_mut(F::SurfaceG).force(g);
    }

    /// Apply: copies every box onto the body and clears its trail.
    pub fn apply_to(&mut self, b: &mut Body) {
        for f in &mut self.fields {
            if f.active {
                f.confirm();
            }
        }
        let v = |s: &Self, f: F| s.field(f).value;
        b.mass = v(self, F::Mass);
        b.radius = v(self, F::Radius);
        b.position = dvec3(v(self, F::PosX), v(self, F::PosY), v(self, F::PosZ));
        b.velocity = dvec3(v(self, F::VelX), v(self, F::VelY), v(self, F::VelZ));
        b.trail.clear();
        for f in &mut self.fields {
            f.edited = false;
        }
    }

    pub fn draw(&mut self, body: Option<&Body>, mouse: Vec2, dt: f32) {
        let pr = self.panel_rect();
        panel(pr, rgba(15, 20, 30, 240), rgba(60, 100, 150, 200));
        collapse_box(Rect::new(self.x + self.width - 24.0, self.y - 2.0, 24.0, 24.0), self.collapsed);
        let (x, y) = (self.x, self.y);

        if self.collapsed {
            match body {
                Some(b) => {
                    draw_circle(x + 15.0, y + 10.0, 6.0, crate::render::rgba(b.color, 1.0));
                    text(&b.name, x + 25.0, y + 1.0, TITLE_FONT, TITLE);
                }
                None => text("Editor", x + 15.0, y + 1.0, TITLE_FONT, TITLE),
            }
            return;
        }
        let Some(b) = body else {
            text("No Planet Selected", x + 20.0, y + 30.0, TITLE_FONT, MUTED);
            text("Click on a planet to select", x + 20.0, y + 60.0, FONT, rgb(100, 100, 120));
            return;
        };

        let mut name = b.name.clone();
        if name.chars().count() > 18 {
            name = name.chars().take(16).collect::<String>() + "..";
        }
        draw_circle(x + 15.0, y + 18.0, 8.0, crate::render::rgba(b.color, 1.0));
        text(&name, x + 32.0, y + 8.0, TITLE_FONT, TITLE);

        section("Properties", x + 15.0, y + 40.0, 230.0, SECTION);
        section("Position", x + 15.0, y + 265.0, 230.0, SECTION);
        section("Velocity", x + 15.0, y + 370.0, 230.0, SECTION);
        for f in &mut self.fields {
            f.draw(dt);
        }
        self.apply.draw(mouse);
        self.delete.draw(mouse);

        let info_y = y + 530.0;
        section("Formulas", x + 15.0, info_y, 230.0, SECTION);
        let lines = ["Force: F = G*m1*m2/r^2".to_string(), "Surface g: g = G*m/r^2".into(), format!("G = {G:.4} (scaled)")];
        for (i, line) in lines.iter().enumerate() {
            text(line, x + 15.0, info_y + 18.0 + i as f32 * 16.0, FONT, MUTED);
        }
    }
}
