import pygame
from pygame.locals import * # Import constants like K_RETURN, etc.
import numpy as np
import math
from typing import Tuple, Optional
from config import G
from models import Planet

class InputField:
    """A text input field for editing numeric values."""
    def __init__(self, x, y, width, height, label, value, min_val=0.01, max_val=10000):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.value = value
        self.text = f"{value:.2f}"
        self.active = False
        self.min_val = min_val
        self.max_val = max_val
        self.cursor_visible = True
        self.cursor_timer = 0
        
    def handle_event(self, event) -> bool:
        """Handle input events. Returns True if value changed."""
        if event.type == MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
                self.text = ""
                return False
            else:
                if self.active:
                    self.confirm_input()
                self.active = False
                
        if event.type == KEYDOWN and self.active:
            if event.key == K_RETURN or event.key == K_TAB:
                self.confirm_input()
                self.active = False
                return True
            elif event.key == K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == K_ESCAPE:
                self.text = f"{self.value:.2f}"
                self.active = False
            elif event.unicode in '0123456789.-':
                self.text += event.unicode
                # Real-time update
                try:
                    new_val = float(self.text)
                    self.value = max(self.min_val, min(self.max_val, new_val))
                    return True # Signal change immediately
                except ValueError:
                    pass
                
        return False
    
    def confirm_input(self):
        """Confirm and validate the input."""
        try:
            new_val = float(self.text)
            self.value = max(self.min_val, min(self.max_val, new_val))
        except ValueError:
            pass
        self.text = f"{self.value:.2f}"
        
    def update(self, dt):
        """Update cursor blink."""
        self.cursor_timer += dt
        if self.cursor_timer > 500:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0
            return True  # Changed visibility, needs redraw
        return False

            
    def draw(self, surface, font):
        """Draw the input field."""
        bg_color = (40, 50, 70) if self.active else (25, 35, 50)
        border_color = (100, 180, 255) if self.active else (60, 80, 100)
        
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=4)
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=4)
        
        label_surface = font.render(self.label, True, (150, 170, 200))
        surface.blit(label_surface, (self.rect.x, self.rect.y - 18))
        
        display_text = self.text
        if self.active and self.cursor_visible:
            display_text += "|"
        text_surface = font.render(display_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(midleft=(self.rect.x + 8, self.rect.centery))
        surface.blit(text_surface, text_rect)


class Button:
    """A clickable button."""
    def __init__(self, x, y, width, height, text, color=(60, 100, 150)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover = False
        
    def handle_event(self, event) -> bool:
        if event.type == MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False
    
    def draw(self, surface, font):
        color = tuple(min(255, c + 30) for c in self.color) if self.hover else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, (100, 140, 180), self.rect, 2, border_radius=6)
        
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class PlanetEditor:
    """Panel for editing planet properties with input fields."""
    def __init__(self, x, y, width=280, height=620, title_font=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.planet: Optional[Planet] = None
        self.fields = {}
        self.buttons = {}
        self.collapsed = False
        self.setup_fields()
        
    def setup_fields(self):
        """Create input fields for planet properties."""
        field_width = 90
        field_height = 26
        
        # Properties Section
        # Header at y + 40. Fields at y + 85.
        start_y = self.y + 85 
        spacing = 60
        col1_x = self.x + 15
        col2_x = self.x + 120
        
        # Mass and Radius (primary controls for g = Gm/r²)
        self.fields['mass'] = InputField(col1_x, start_y, field_width, field_height, 
                                          "Mass (m)", 100, 1, 5000)
        self.fields['radius'] = InputField(col2_x, start_y, field_width, field_height,
                                            "Radius (r)", 1, 0.1, 10)
        
        # Density (editable - will adjust mass to match)
        self.fields['density'] = InputField(col1_x, start_y + spacing, field_width, field_height,
                                             "Density (ρ)", 23.87, 0.1, 1000)
        
        # Surface gravity (calculated: g = Gm/r²)
        self.fields['surface_g'] = InputField(col2_x, start_y + spacing, field_width, field_height,
                                               "Surface g", 0, 0, 99999)
        
        # Net Force (Real-time metric)
        self.fields['net_force'] = InputField(col1_x, start_y + 120, field_width * 2 + 15, field_height,
                                               "Net Force (F = ma)", 0, 0, 0)
        self.fields['net_force'].active = False # Read-only
        
        # Position Section
        # Header at y + 265. Fields at y + 310.
        pos_y = self.y + 310
        self.fields['pos_x'] = InputField(col1_x, pos_y, 70, field_height,
                                           "Pos X", 0, -50, 50)
        self.fields['pos_y'] = InputField(col1_x + 80, pos_y, 70, field_height,
                                           "Y", 0, -50, 50)
        self.fields['pos_z'] = InputField(col1_x + 160, pos_y, 70, field_height,
                                           "Z", 0, -50, 50)
        
        # Velocity Section
        # Header at y + 370. Fields at y + 415.
        vel_y = self.y + 415
        self.fields['vel_x'] = InputField(col1_x, vel_y, 70, field_height,
                                           "Vel X", 0, -20, 20)
        self.fields['vel_y'] = InputField(col1_x + 80, vel_y, 70, field_height,
                                           "Y", 0, -20, 20)
        self.fields['vel_z'] = InputField(col1_x + 160, vel_y, 70, field_height,
                                           "Z", 0, -20, 20)
        
        # Buttons
        btn_y = self.y + 480
        self.buttons['apply'] = Button(col1_x, btn_y, 100, 32, "Apply", (40, 120, 80))
        self.buttons['delete'] = Button(col2_x + 15, btn_y, 100, 32, "Delete", (150, 50, 50))
        
    def set_planet(self, planet: Optional[Planet]):
        """Set the planet to edit and populate fields."""
        self.planet = planet
        if planet:
            self.fields['mass'].value = planet.mass
            self.fields['mass'].text = f"{planet.mass:.2f}"
            
            self.fields['radius'].value = planet.radius
            self.fields['radius'].text = f"{planet.radius:.2f}"
            
            self.fields['density'].value = planet.density()
            self.fields['density'].text = f"{planet.density():.2f}"
            
            self.fields['surface_g'].value = planet.surface_gravity()
            self.fields['surface_g'].text = f"{planet.surface_gravity():.2f}"
            
            self.fields['pos_x'].value = planet.position[0]
            self.fields['pos_x'].text = f"{planet.position[0]:.2f}"
            self.fields['pos_y'].value = planet.position[1]
            self.fields['pos_y'].text = f"{planet.position[1]:.2f}"
            self.fields['pos_z'].value = planet.position[2]
            self.fields['pos_z'].text = f"{planet.position[2]:.2f}"
            
            self.fields['vel_x'].value = planet.velocity[0]
            self.fields['vel_x'].text = f"{planet.velocity[0]:.2f}"
            self.fields['vel_y'].value = planet.velocity[1]
            self.fields['vel_y'].text = f"{planet.velocity[1]:.2f}"
            self.fields['vel_z'].value = planet.velocity[2]
            self.fields['vel_z'].text = f"{planet.velocity[2]:.2f}"
            
    def handle_event(self, event) -> Tuple[bool, str]:
        """Handle events. Returns (handled, action)."""
            
        # Check toggle button (top right corner of panel)
        # Visual: y - 2. Hitbox: Add padding for easier clicking.
        # Button is 24x24. Let's make hitbox 34x34 centered on it.
        toggle_hit_rect = pygame.Rect(self.x + self.width - 29, self.y - 7, 34, 34)
        if event.type == MOUSEBUTTONDOWN and toggle_hit_rect.collidepoint(event.pos):
            self.collapsed = not self.collapsed
            return True, 'toggle'

        if self.collapsed:
            return False, ''

        if not self.planet:
            return False, ''
            
        # Check buttons first
        for name, button in self.buttons.items():
            if button.handle_event(event):
                return True, name
                
        # Then input fields
        for name, fld in self.fields.items():
            if fld.handle_event(event):
                # When density changes, adjust mass
                if name == 'density':
                    volume = (4/3) * math.pi * self.fields['radius'].value**3
                    self.fields['mass'].value = fld.value * volume
                    self.fields['mass'].text = f"{self.fields['mass'].value:.2f}"
                # When mass or radius changes, update density and surface gravity
                if name in ('mass', 'radius'):
                    self.update_calculated_fields()
                    # Real-time visual update for radius and mass
                    if self.planet:
                        if name == 'radius': self.planet.radius = self.fields['radius'].value
                        if name == 'mass': self.planet.mass = self.fields['mass'].value
                return True, ''
                
        return False, ''
    
    def update_calculated_fields(self):
        """Update density and surface gravity based on mass and radius."""
        mass = self.fields['mass'].value
        radius = self.fields['radius'].value
        
        # Update density: ρ = m / V
        volume = (4/3) * math.pi * radius**3
        density = mass / volume if volume > 0 else 0
        self.fields['density'].value = density
        if not self.fields['density'].active:
            self.fields['density'].text = f"{density:.2f}"
        
        # Update surface gravity: g = Gm/r²
        surface_g = G * mass / (radius ** 2) if radius > 0 else 0
        self.fields['surface_g'].value = surface_g
        self.fields['surface_g'].text = f"{surface_g:.2f}"
        
    def apply_to_planet(self):
        """Apply all field values to the planet."""
        if not self.planet:
            return
            
        self.planet.mass = self.fields['mass'].value
        self.planet.radius = self.fields['radius'].value
        self.planet.position = np.array([
            self.fields['pos_x'].value,
            self.fields['pos_y'].value,
            self.fields['pos_z'].value
        ])
        self.planet.velocity = np.array([
            self.fields['vel_x'].value,
            self.fields['vel_y'].value,
            self.fields['vel_z'].value
        ])
        # Clear trail when properties change significantly
        self.planet.trail.clear()
        
    def update(self, dt) -> bool:
        """Update animations. Returns True if redraw needed."""
        dirty = False
        
        # Update Net Force in real-time if planet is selected
        if self.planet:
            acc_mag = np.linalg.norm(self.planet.acceleration)
            force_mag = acc_mag * self.planet.mass
            self.fields['net_force'].value = force_mag
            self.fields['net_force'].text = f"{force_mag:.2f}"
            
        for fld in self.fields.values():
            if fld.update(dt):
                dirty = True
        return dirty
            
    def draw(self, surface, font, title_font):
        """Draw the editor panel."""
        # Adjust height based on collapsed state
        current_height = 40 if self.collapsed else self.height
        
        panel_rect = pygame.Rect(self.x - 10, self.y - 10, self.width + 20, current_height)
        pygame.draw.rect(surface, (15, 20, 30, 240), panel_rect, border_radius=10)
        pygame.draw.rect(surface, (60, 100, 150, 200), panel_rect, 2, border_radius=10)
        
        # Toggle Button
        toggle_rect = pygame.Rect(self.x + self.width - 24, self.y - 2, 24, 24) # Centered vertically in header
        pygame.draw.rect(surface, (40, 60, 80), toggle_rect, border_radius=4)
        sym = "+" if self.collapsed else "−"
        sym_surf = font.render(sym, True, (255, 255, 255))
        sym_rect = sym_surf.get_rect(center=toggle_rect.center) # Center text
        surface.blit(sym_surf, sym_rect)
        
        if self.collapsed:
            if self.planet:
                # Minimal title when collapsed
                # Header acts as 40px box from y-10 to y+30. Center y is y+10.
                pygame.draw.circle(surface, tuple(int(c * 255) for c in self.planet.color), 
                                  (self.x + 15, self.y + 10), 6)
                
                # Use main title font and color, centered vertically
                title = title_font.render(self.planet.name, True, (100, 180, 255))
                title_rect = title.get_rect(midleft=(self.x + 25, self.y + 10))
                surface.blit(title, title_rect)
            else:
                title = title_font.render("Editor", True, (100, 180, 255))
                title_rect = title.get_rect(midleft=(self.x + 15, self.y + 10))
                surface.blit(title, title_rect)
            return

        if not self.planet:
            msg = title_font.render("No Planet Selected", True, (150, 150, 170))
            surface.blit(msg, (self.x + 20, self.y + 30))
            hint = font.render("Click on a planet to select", True, (100, 100, 120))
            surface.blit(hint, (self.x + 20, self.y + 60))
            return
            
        # Title
        name = self.planet.name
        if len(name) > 18:
            name = name[:16] + ".."
        
        # Use main title color and font
        title = title_font.render(f"  {name}", True, (100, 180, 255))
        surface.blit(title, (self.x + 20, self.y + 8))
        
        # Color indicator circle
        pygame.draw.circle(surface, tuple(int(c * 255) for c in self.planet.color), 
                          (self.x + 15, self.y + 18), 8)
        
        # Section headers
        # start_y = y + 85.
        # Properties at y + 40 (gap 45 to fields)
        # Position at y + 265 (Fields at y + 310. Gap 45).
        # Velocity at y + 370 (Fields at y + 415. Gap 45).
        sections = [
            (self.y + 40, "─── Properties ───"),
            (self.y + 265, "─── Position ───"),
            (self.y + 370, "─── Velocity ───"),
        ]
        for y_pos, text in sections:
            header = font.render(text, True, (80, 140, 200))
            surface.blit(header, (self.x + 15, y_pos))
        
        # Draw all input fields
        for fld in self.fields.values():
            fld.draw(surface, font)
            
        # Draw buttons
        for button in self.buttons.values():
            button.draw(surface, font)
            
        # Formula explanation
        # Position relative to buttons (y + 480)
        # Buttons height 32 -> y + 512. Start formulas at y + 530.
        info_y = self.y + 530
        info_lines = [
            "─── Formulas ───",
            "Force: F = G×m₁×m₂/r²",
            "Surface g: g = G×m/r²", 
            f"G = {G:.4f} (scaled)"
        ]
        for i, line in enumerate(info_lines):
            color = (80, 140, 200) if i == 0 else (150, 150, 170)
            text = font.render(line, True, color)
            surface.blit(text, (self.x + 15, info_y + i * 16))
