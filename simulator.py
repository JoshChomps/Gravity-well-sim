import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import random
import colorsys
import asyncio
from typing import List, Optional

from config import G, GRID_EXTENT, TRAIL_LENGTH, GRID_RESOLUTION, MAX_GRID_RESOLUTION, IS_WEB
from models import Planet
from ui import PlanetEditor, Button

class Starfield:
    def __init__(self, num_stars=1500, radius=300):
        # Reduce star count on web for performance
        if IS_WEB:
            num_stars = 500
        self.stars = []
        for _ in range(num_stars):
            # Random point in sphere
            u = random.random()
            v = random.random()
            theta = 2 * math.pi * u
            phi = math.acos(2 * v - 1)
            r = radius * (0.8 + 0.4 * random.random())  # Varied distance
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            
            # Varying brightness/size
            brightness = 0.5 + 0.5 * random.random()
            self.stars.append(((x, y, z), brightness))
            
    def render(self):
        glDisable(GL_LIGHTING)
        glPointSize(1.5)
        glBegin(GL_POINTS)
        for pos, brightness in self.stars:
            glColor4f(brightness, brightness, brightness, 0.8)
            glVertex3f(*pos)
        glEnd()
        glEnable(GL_LIGHTING)


class GravitySimulator:
    def __init__(self, width=1200, height=800):
        print("Initializing GravitySimulator...")
        pygame.init()
        pygame.display.set_caption("3D Gravity Simulator - Spacetime Curvature Visualization")
        
        self.width = width
        self.height = height
        
        # Web compatibility: Disable RESIZABLE, maybe disable DOUBLEBUF if problematic (usually fine)
        flags = DOUBLEBUF | OPENGL
        if not IS_WEB:
            flags |= RESIZABLE
            
        print(f"Creating display mode: {width}x{height}, flags={flags}")
        self.display = pygame.display.set_mode((width, height), flags)
        
        # Camera
        self.camera_distance = 40
        self.camera_rot_x = 30
        self.camera_rot_y = 45
        self.camera_pan = np.array([0.0, 0.0, 0.0])
        self.camera_tracking = False
        
        # Mouse state
        self.mouse_pressed = [False, False, False]
        self.last_mouse_pos = (0, 0)
        
        # Simulation
        self.planets: List[Planet] = []
        self.running = True
        self.paused = False
        self.time_scale = 1.0
        self.selected_planet: Optional[int] = None
        
        # UI optimization
        self.ui_dirty = True
        self.ui_texture = None
        
        # Visual toggles
        self.show_grid = True
        self.show_trails = True
        self.show_velocity_vectors = False
        self.show_force_vectors = False
        
        # UI State
        self.left_panel_collapsed = False
        self.presets_panel_collapsed = False
        
        self.grid_resolution = GRID_RESOLUTION
        
        # Grid VBO Arrays
        self.grid_arrays_dirty = True
        self.cached_resolution = 0
        self.grid_vertices = None
        self.grid_indices = None
        self.grid_x = None
        self.grid_z = None
        self.vbo_id = None
        self.ibo_id = None
        
        print("Calling init_gl...")
        self.init_gl()
        
        if pygame.font.get_init():
            def get_font(name, size, bold=False):
                try:
                    return pygame.font.SysFont(name, size, bold)
                except:
                    return pygame.font.Font(None, size)
            
            self.font = get_font('freesans,monospace,arial', 14)
            self.title_font = get_font('freesans,monospace,arial', 18, bold=True)
        else:
            self.font = None
            self.title_font = None
        
        self.planet_counter = 0
        # y=20 so that background (drawn at y-10) starts at 10, matching the left panel
        self.planet_editor = PlanetEditor(width - 280, 20, 260)
        
        self.clock = pygame.time.Clock()
        self.starfield = Starfield()
        
        # Grid resolution controls
        self.res_minus_button = Button(110, 0, 30, 20, "-", (60, 100, 150))
        self.res_plus_button = Button(150, 0, 30, 20, "+", (60, 100, 150))
        
        # Quit button in bottom-right area
        self.quit_button = Button(width - 110, height - 50, 100, 35, "EXIT", (150, 50, 50))
        
        # Load default scenario after editor is created
        print("Loading default preset...")
        self.load_preset(1)
        print("Initialization complete.")
        
    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        if not IS_WEB:
            glEnable(GL_LINE_SMOOTH)
        glClearColor(0.02, 0.02, 0.05, 1.0)
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 20.0, 10.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.35, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.85, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        # Initialize UI texture
        print("Generating UI texture...")
        try:
            tex = glGenTextures(1)
            # Handle list/array return
            if hasattr(tex, '__iter__'):
                self.ui_texture = tex[0]
            else:
                self.ui_texture = int(tex)
                
            print(f"UI Texture ID: {self.ui_texture}")
            glBindTexture(GL_TEXTURE_2D, self.ui_texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        except Exception as e:
            print(f"Error creating texture: {e}")
        
    def setup_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        
    def setup_camera(self):
        glLoadIdentity()
        
        # If tracking a planet, update camera pan to follow it
        if self.camera_tracking and self.selected_planet is not None:
            planet = self.get_selected_planet()
            if planet:
                # Smoothly interpolate or just lock? Lock is stricter.
                self.camera_pan = planet.position.copy()
            else:
                self.camera_tracking = False
        
        rad_x = math.radians(self.camera_rot_x)
        rad_y = math.radians(self.camera_rot_y)
        
        cam_x = self.camera_distance * math.cos(rad_x) * math.sin(rad_y)
        cam_y = self.camera_distance * math.sin(rad_x)
        cam_z = self.camera_distance * math.cos(rad_x) * math.cos(rad_y)
        
        gluLookAt(
            cam_x + self.camera_pan[0], cam_y + self.camera_pan[1], cam_z + self.camera_pan[2],
            self.camera_pan[0], self.camera_pan[1], self.camera_pan[2],
            0, 1, 0
        )
        
    def create_planet(self, position, velocity, mass, radius, color=None, name=None) -> Planet:
        if color is None:
            hue = random.random()
            color = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        
        self.planet_counter += 1
        if name is None:
            name = f"Planet {self.planet_counter}"
            
        return Planet(
            position=np.array(position, dtype=float),
            velocity=np.array(velocity, dtype=float),
            mass=mass,
            radius=radius,
            color=color,
            name=name,
            id=self.planet_counter
        )
        
    def load_preset(self, preset_num):
        self.planets.clear()
        self.planet_counter = 0
        
        if preset_num == 1:
            self.planets.append(self.create_planet([0, 0, 0], [0, 0, 0], 500, 1.5, (1.0, 0.9, 0.3), "Sun"))
            self.planets.append(self.create_planet([5, 0, 0], [0, 0, 3.2], 5, 0.3, (0.7, 0.7, 0.7), "Mercury"))
            self.planets.append(self.create_planet([7, 0, 2], [0, 0.3, 2.5], 12, 0.45, (0.9, 0.6, 0.3), "Venus"))
            self.planets.append(self.create_planet([10, 0, 0], [0, 0.2, 2.2], 15, 0.5, (0.3, 0.5, 0.9), "Earth"))
            self.planets.append(self.create_planet([14, 0, -2], [0, -0.1, 1.8], 8, 0.35, (0.9, 0.4, 0.2), "Mars"))
            # Outer giants (scaled distances)
            self.planets.append(self.create_planet([20, 0, 0], [0, 0.1, 1.3], 60, 0.8, (0.8, 0.7, 0.6), "Jupiter"))
            self.planets.append(self.create_planet([28, 0, 5], [-0.1, 0, 0.9], 50, 0.7, (0.9, 0.8, 0.5), "Saturn"))
            self.planets.append(self.create_planet([36, 0, -5], [0, -0.1, 0.7], 30, 0.6, (0.6, 0.8, 0.9), "Uranus"))
            self.planets.append(self.create_planet([42, 0, 0], [0, 0.1, 0.6], 30, 0.6, (0.3, 0.3, 0.9), "Neptune"))
            # Pluto!
            self.planets.append(self.create_planet([48, 2, 0], [0, -0.2, 0.5], 0.5, 0.15, (0.7, 0.6, 0.5), "Pluto"))
            
        elif preset_num == 2:
            self.planets.append(self.create_planet([-4, 0, 0], [0, 0, 1.2], 300, 1.2, (1.0, 0.6, 0.2), "Star A"))
            self.planets.append(self.create_planet([4, 0, 0], [0, 0, -1.2], 300, 1.2, (0.4, 0.7, 1.0), "Star B"))
            self.planets.append(self.create_planet([0, 0, 12], [1.5, 0, 0], 10, 0.4, (0.5, 0.9, 0.5), "Planet"))
        elif preset_num == 3:
            self.planets.append(self.create_planet([-5, 0, 0], [0, 0.5, 0.8], 200, 1.0, (0.9, 0.3, 0.4), "Red Giant"))
            self.planets.append(self.create_planet([5, 0, 0], [0, -0.3, -0.8], 200, 1.0, (0.3, 0.9, 0.5), "Green Star"))
            self.planets.append(self.create_planet([0, 0, 6], [-0.5, 0.2, 0], 200, 1.0, (0.4, 0.5, 1.0), "Blue Star"))
        elif preset_num == 4:
            v = 0.8
            self.planets.append(self.create_planet([-3, 0, 0], [0, v*0.5, v], 150, 0.9, (1.0, 0.5, 0.8), "Alpha"))
            self.planets.append(self.create_planet([3, 0, 0], [0, v*0.5, v], 150, 0.9, (0.5, 1.0, 0.8), "Beta"))
            self.planets.append(self.create_planet([0, 0, 0], [0, -v, -2*v], 150, 0.9, (0.8, 0.5, 1.0), "Gamma"))
        elif preset_num == 5: # Deep Space Collision
            for i in range(10):
                self.planets.append(self.create_planet(
                    [random.uniform(-10, -5), random.uniform(-2, 2), random.uniform(-5, 5)],
                    [random.uniform(1, 2), 0, 0], 
                    random.uniform(5, 20), random.uniform(0.3, 0.6)))
            for i in range(10):
                self.planets.append(self.create_planet(
                    [random.uniform(5, 10), random.uniform(-2, 2), random.uniform(-5, 5)],
                    [random.uniform(-2, -1), 0, 0], 
                    random.uniform(5, 20), random.uniform(0.3, 0.6)))
        elif preset_num == 6: # Random Cluster
            for _ in range(30):
                self.add_random_planet()
        
        self.selected_planet = None
        self.planet_editor.set_planet(None)
        
        # Initialize accelerations for existing planets
        forces = self.calculate_forces()
        for planet, force in zip(self.planets, forces):
            if planet.mass > 0:
                planet.acceleration = force / planet.mass
        
        self.ui_dirty = True
        

    def init_grid_arrays(self):
        """Initialize VBOs and IBOs for rendering."""

        if self.grid_resolution == self.cached_resolution and not self.grid_arrays_dirty:
            return
            
        resolution = self.grid_resolution
        self.cached_resolution = resolution
        extent = GRID_EXTENT
        
        # 1. Create X, Z coordinates (static)
        x = np.linspace(-extent, extent, resolution + 1)
        z = np.linspace(-extent, extent, resolution + 1)
        self.grid_x, self.grid_z = np.meshgrid(x, z)
        
        # 2. Create Indices
        indices = []
        width = resolution + 1
        # Horizontal lines (along X)
        for i in range(resolution + 1):
            for j in range(resolution):
                idx = i * width + j
                indices.extend([idx, idx + 1])
        # Vertical lines (along Z)
        for i in range(resolution):
            for j in range(resolution + 1):
                idx = i * width + j
                indices.extend([idx, idx + width])
                
        # Use uint16 on web if possible for better compatibility (WebGL 1.0)
        index_type = np.uint16 if self.grid_resolution <= 150 else np.uint32
        self.grid_indices = np.array(indices, dtype=index_type)
        
        # 3. Create initial vertex array container (N, 3)
        num_verts = (resolution + 1) * (resolution + 1)
        self.grid_vertices = np.zeros((num_verts, 3), dtype=np.float32)
        self.grid_vertices[:, 0] = self.grid_x.flatten()
        self.grid_vertices[:, 2] = self.grid_z.flatten()
        
        # 4. Generate Buffers if not exists
        if self.vbo_id is None:
            # Handle potential array return from glGenBuffers
            vbos = glGenBuffers(2)
            if hasattr(vbos, '__iter__'):
                self.vbo_id, self.ibo_id = vbos
            else:
                self.vbo_id = vbos
                self.ibo_id = glGenBuffers(1)
            
        # 5. Upload Indices (Static)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo_id)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.grid_indices.nbytes, self.grid_indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        
        # 6. Allocate VBO (Dynamic)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        glBufferData(GL_ARRAY_BUFFER, self.grid_vertices.nbytes, self.grid_vertices, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        self.grid_arrays_dirty = False
        
    def draw_curved_grid(self):
        if not self.show_grid:
            return

        # Ensure arrays are ready
        if self.grid_arrays_dirty or self.cached_resolution != self.grid_resolution:
            self.init_grid_arrays()
            
        # Vectorized displacement calculation
        X = self.grid_x
        Z = self.grid_z
        Y = np.zeros_like(X)
        
        for planet in self.planets:
            dx = X - planet.position[0]
            dz = Z - planet.position[2]
            dist = np.sqrt(dx*dx + dz*dz)
            # Plummer Softening for realistic rounded wells (no plateau)
            # dist_eff = sqrt(d^2 + softening^2)
            softening = 1.2
            dist_eff = np.sqrt(dist*dist + softening*softening)
            displacement = planet.mass / (dist_eff * 5.0)
            Y -= displacement * 0.15
            
        # Update Y coordinate in vertex array
        self.grid_vertices[:, 1] = Y.flatten()
        
        # Render using VBOs
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor4f(0.2, 0.4, 0.8, 0.4) 
        
        glEnableClientState(GL_VERTEX_ARRAY)
        
        # Bind VBO and update data
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.grid_vertices.nbytes, self.grid_vertices)
        glVertexPointer(3, GL_FLOAT, 0, None) # None means use bound buffer
        
        # Bind IBO and Draw
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo_id)
        
        # Logic to choose index type based on IS_WEB or desktop
        if IS_WEB:
             gl_idx_type = GL_UNSIGNED_SHORT
        else:
             gl_idx_type = GL_UNSIGNED_SHORT if self.grid_indices.dtype == np.uint16 else GL_UNSIGNED_INT
             
        glDrawElements(GL_LINES, len(self.grid_indices), gl_idx_type, None) # None means use bound buffer
        
        # Cleanup
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)
        glEnable(GL_LIGHTING)

        
    def draw_force_field(self):
        if not self.show_force_vectors:
            return
            
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        
        spacing = 4 # Increased spacing for better perf
        extent = GRID_EXTENT - 2
        
        # Create grid points for vectors
        x = np.arange(-extent, extent + spacing, spacing, dtype=np.float64)
        z = np.arange(-extent, extent + spacing, spacing, dtype=np.float64)
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X, dtype=np.float64)
        
        # Vectorized Field Calculation
        FX = np.zeros_like(X, dtype=np.float64)
        FZ = np.zeros_like(X, dtype=np.float64)
        
        # Calculate field at all points
        for planet in self.planets:
            dx = X - planet.position[0]
            dz = Z - planet.position[2]
            dist_sq = dx*dx + dz*dz
            dist = np.sqrt(dist_sq)
            
            # Mask out points inside planets
            mask = dist > planet.radius * 2
            
            # Gravity inverse square law
            # Avoid division by zero with maximum
            dist_safe = np.maximum(1.0, dist)
            force_mag = G * planet.mass / (dist_safe**2)
            
            # Accumulate force components
            # Apply mask to accumulation
            FX += (force_mag * (-dx / dist_safe)) * mask
            FZ += (force_mag * (-dz / dist_safe)) * mask
            
            # Also calculate Y displacement for start positions (grid curvature)
            # Also calculate Y displacement for start positions (grid curvature)
            # Plummer Softening matching draw_curved_grid
            softening = 1.2
            dist_eff = np.sqrt(dist*dist + softening*softening)
            disp = planet.mass / (dist_eff * 5.0)
            Y -= disp * 0.15

        # Calculate magnitude and normalize for visualization
        mag = np.sqrt(FX*FX + FZ*FZ)
        
        # Threshold to draw
        mask = mag > 0.01
        
        if not np.any(mask):
            glEnable(GL_LIGHTING)
            return
            
        # Filter points
        X_draw = X[mask]
        Y_draw = Y[mask] + 0.1 # Lift slightly above grid
        Z_draw = Z[mask]
        FX_draw = FX[mask]
        FZ_draw = FZ[mask]
        Mag_draw = mag[mask]
        
        # Normalize direction
        Scale = np.minimum(1.5, Mag_draw * 2.0)
        DirX = FX_draw / Mag_draw * Scale
        DirZ = FZ_draw / Mag_draw * Scale
        
        # Draw Lines (Immediate mode is safe and fast enough for ~600 lines)
        N = len(X_draw)
        glBegin(GL_LINES)
        glColor4f(1.0, 0.2, 0.2, 0.8) # Bright Red for forces
        for i in range(N):
            glVertex3f(X_draw[i], Y_draw[i], Z_draw[i])
            glVertex3f(X_draw[i] + DirX[i], Y_draw[i], Z_draw[i] + DirZ[i])
        glEnd()

        glEnable(GL_LIGHTING)
        
    def draw_arrow_head(self, tip, direction, size):
        if abs(direction[1]) < 0.9:
            perp1 = np.cross(direction, [0, 1, 0])
        else:
            perp1 = np.cross(direction, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1) * size * 0.5
        base = tip - direction * size
        glBegin(GL_TRIANGLES)
        glVertex3f(*tip)
        glVertex3f(*(base + perp1))
        glVertex3f(*(base - perp1))
        glEnd()
        
    def draw_planet(self, planet: Planet, is_selected: bool = False):
        glPushMatrix()
        glTranslatef(*planet.position)
        if is_selected:
            pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.005)
            glColor3f(min(1.0, planet.color[0] + 0.3 * pulse),
                      min(1.0, planet.color[1] + 0.3 * pulse),
                      min(1.0, planet.color[2] + 0.3 * pulse))
        else:
            glColor3f(*planet.color)
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluSphere(quadric, planet.radius, 32, 32)
        gluDeleteQuadric(quadric)
        if is_selected:
            glDisable(GL_LIGHTING)
            glColor4f(1.0, 1.0, 1.0, 0.8)
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            for i in range(64):
                angle = 2 * math.pi * i / 64
                r = planet.radius * 1.5
                glVertex3f(r * math.cos(angle), 0, r * math.sin(angle))
            glEnd()
            glEnable(GL_LIGHTING)
        glPopMatrix()
        
    def draw_trail(self, planet: Planet):
        if not self.show_trails or len(planet.trail) < 2:
            return
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        for i, pos in enumerate(planet.trail):
            alpha = i / len(planet.trail) * 0.8
            glColor4f(planet.color[0], planet.color[1], planet.color[2], alpha)
            glVertex3f(*pos)
        glEnd()
        glEnable(GL_LIGHTING)
        
    def draw_velocity_vector(self, planet: Planet):
        if not self.show_velocity_vectors:
            return
        glDisable(GL_LIGHTING)
        glLineWidth(2.5)
        glColor4f(0.2, 1.0, 0.3, 0.9)
        vel_mag = np.linalg.norm(planet.velocity)
        if vel_mag > 0.01:
            end = planet.position + planet.velocity * 2
            glBegin(GL_LINES)
            glVertex3f(*planet.position)
            glVertex3f(*end)
            glEnd()
            self.draw_arrow_head(end, planet.velocity / vel_mag, 0.3)
        glEnable(GL_LIGHTING)
        


    def calculate_forces(self) -> List[np.ndarray]:
        forces = []
        for i, planet in enumerate(self.planets):
            force = np.array([0.0, 0.0, 0.0])
            for j, other in enumerate(self.planets):
                if i != j:
                    r = other.position - planet.position
                    dist = np.linalg.norm(r)
                    if dist < planet.radius + other.radius:
                        continue
                    force_mag = G * planet.mass * other.mass / (dist * dist)
                    force += force_mag * r / dist
            forces.append(force)
        return forces

    def handle_collisions(self):
        # fast check for collisions
        # We need to restart the check if a collision happens because the list changes
        collision_occurred = True
        while collision_occurred:
            collision_occurred = False
            for i in range(len(self.planets)):
                for j in range(i + 1, len(self.planets)):
                    p1 = self.planets[i]
                    p2 = self.planets[j]
                    
                    dist_sq = np.sum((p1.position - p2.position)**2)
                    min_dist = p1.radius + p2.radius
                    
                    if dist_sq < min_dist * min_dist:
                        # Collision! Merge them
                        new_mass = p1.mass + p2.mass
                        new_radius = (p1.radius**3 + p2.radius**3)**(1/3)
                        
                        # Momentum conservation
                        if new_mass > 0:
                            new_vel = (p1.velocity * p1.mass + p2.velocity * p2.mass) / new_mass
                            new_pos = (p1.position * p1.mass + p2.position * p2.mass) / new_mass
                        else:
                            new_vel = np.zeros(3)
                            new_pos = (p1.position + p2.position) / 2
                            
                        # Weighted color blending
                        c1 = np.array(p1.color)
                        c2 = np.array(p2.color)
                        new_color = tuple((c1 * p1.mass + c2 * p2.mass) / new_mass)
                        
                        # Keep name of the massive planet
                        name = p1.name if p1.mass >= p2.mass else p2.name
                        
                        # Remove old planets
                        self.planets.pop(j) # Pop larger index first
                        self.planets.pop(i)
                        
                        # Add new one
                        new_planet = self.create_planet(new_pos, new_vel, new_mass, new_radius, new_color, name)
                        self.planets.append(new_planet)
                        
                        # If selected planet was destroyed, select the new one
                        if self.selected_planet in (p1.id, p2.id):
                            self.selected_planet = new_planet.id
                            self.planet_editor.set_planet(new_planet)
                        
                        self.ui_dirty = True
                        collision_occurred = True
                        break # Break inner loop
                if collision_occurred:
                    break # Break outer loop

    def update_physics(self, dt):
        if self.paused:
            return
        dt *= self.time_scale
        
        # Velocity Verlet Integration
        # 1. First half-update of velocity and full update of position
        for planet in self.planets:
            planet.position += planet.velocity * dt + 0.5 * planet.acceleration * dt * dt
            planet.velocity += 0.5 * planet.acceleration * dt
            
            # Trail update
            planet.trail.append(planet.position.copy())
            if len(planet.trail) > TRAIL_LENGTH:
                planet.trail.pop(0)

        # 1.5 Handle Collisions
        self.handle_collisions()

        # 2. Calculate new forces and accelerations
        forces = self.calculate_forces()
        
        # 3. Second half-update of velocity
        for planet, force in zip(self.planets, forces):
            if planet.mass > 0:
                new_acceleration = force / planet.mass
            else:
                new_acceleration = np.zeros(3)
                
            planet.velocity += 0.5 * new_acceleration * dt
            planet.acceleration = new_acceleration

        if self.selected_planet is not None:
            self.ui_dirty = True  # Transform values change every frame
            planet = self.get_selected_planet()
            if planet and not any(f.active for f in self.planet_editor.fields.values()):
                self.planet_editor.fields['pos_x'].value = planet.position[0]
                self.planet_editor.fields['pos_x'].text = f"{planet.position[0]:.2f}"
                self.planet_editor.fields['pos_y'].value = planet.position[1]
                self.planet_editor.fields['pos_y'].text = f"{planet.position[1]:.2f}"
                self.planet_editor.fields['pos_z'].value = planet.position[2]
                self.planet_editor.fields['pos_z'].text = f"{planet.position[2]:.2f}"
                self.planet_editor.fields['vel_x'].value = planet.velocity[0]
                self.planet_editor.fields['vel_x'].text = f"{planet.velocity[0]:.2f}"
                self.planet_editor.fields['vel_y'].value = planet.velocity[1]
                self.planet_editor.fields['vel_y'].text = f"{planet.velocity[1]:.2f}"
                self.planet_editor.fields['vel_z'].value = planet.velocity[2]
                self.planet_editor.fields['vel_z'].text = f"{planet.velocity[2]:.2f}"
    
    def get_selected_planet(self) -> Optional[Planet]:
        if self.selected_planet is None:
            return None
        for planet in self.planets:
            if planet.id == self.selected_planet:
                return planet
        return None
    
    def add_random_planet(self):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(8, 14)
        pos = [dist * math.cos(angle), random.uniform(-2, 2), dist * math.sin(angle)]
        if self.planets:
            center_mass = sum(p.mass for p in self.planets)
            orbital_speed = math.sqrt(G * center_mass / dist) * random.uniform(0.8, 1.2)
        else:
            orbital_speed = 1.0
        vel = [-math.sin(angle) * orbital_speed, random.uniform(-0.2, 0.2), math.cos(angle) * orbital_speed]
        mass = random.uniform(10, 50)
        radius = 0.2 + mass * 0.01
        self.planets.append(self.create_planet(pos, vel, mass, radius))
        self.ui_dirty = True
        
    def remove_selected_planet(self):
        if self.selected_planet is not None:
            self.planets = [p for p in self.planets if p.id != self.selected_planet]
            self.selected_planet = None
            self.planet_editor.set_planet(None)
            self.ui_dirty = True
    
    def select_planet_at_screen_pos(self, screen_pos):
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        win_x = screen_pos[0]
        win_y = viewport[3] - screen_pos[1]
        near_point = gluUnProject(win_x, win_y, 0.0, modelview, projection, viewport)
        far_point = gluUnProject(win_x, win_y, 1.0, modelview, projection, viewport)
        ray_origin = np.array(near_point)
        ray_dir = np.array(far_point) - ray_origin
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        closest_planet = None
        closest_dist = float('inf')
        for planet in self.planets:
            oc = ray_origin - planet.position
            a = np.dot(ray_dir, ray_dir)
            b = 2.0 * np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - planet.radius * planet.radius
            discriminant = b * b - 4 * a * c
            if discriminant >= 0:
                t = (-b - math.sqrt(discriminant)) / (2 * a)
                if t > 0 and t < closest_dist:
                    closest_dist = t
                    closest_planet = planet
        if closest_planet:
            self.selected_planet = closest_planet.id
            self.planet_editor.set_planet(closest_planet)
            self.ui_dirty = True
        else:
            self.selected_planet = None
            self.planet_editor.set_planet(None)
            self.ui_dirty = True
            
    def render_ui(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_LIGHTING)
        glLoadIdentity()
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        if self.ui_dirty:
            ui_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            
            # Left panel - dynamic height to fit content
            # Content needs ~540px. Stop before presets (H-180).
            max_h = self.height - 200
            panel_h = min(540, max_h)
            
            # Adjust if collapsed
            if self.left_panel_collapsed:
                panel_h = 40
            
            panel_color = (10, 15, 25, 220)
            pygame.draw.rect(ui_surface, panel_color, (10, 10, 300, panel_h), border_radius=8)
            pygame.draw.rect(ui_surface, (60, 100, 150, 150), (10, 10, 300, panel_h), 2, border_radius=8)
            
            # Left Panel Toggle
            toggle_rect = pygame.Rect(280, 18, 24, 24) # Centered in 40px header (Top 10, Height 40 -> Center 30. 30-12=18)
            pygame.draw.rect(ui_surface, (40, 60, 80), toggle_rect, border_radius=4)
            sym = "+" if self.left_panel_collapsed else "−"
            sym_surf = self.font.render(sym, True, (255, 255, 255))
            sym_rect = sym_surf.get_rect(center=toggle_rect.center)
            ui_surface.blit(sym_surf, sym_rect)

            title = self.title_font.render("3D GRAVITY SIMULATOR", True, (100, 180, 255))
            ui_surface.blit(title, (20, 20))
            
            if not self.left_panel_collapsed:
                subtitle = self.font.render("Spacetime Curvature Visualization", True, (150, 150, 170))
                ui_surface.blit(subtitle, (20, 45))
                
                y = 75
                status = "PAUSED" if self.paused else "RUNNING"
                status_color = (255, 150, 100) if self.paused else (100, 255, 150)
                ui_surface.blit(self.font.render(f"Status: {status}", True, status_color), (20, y))
                y += 20
                ui_surface.blit(self.font.render(f"Time Scale: {self.time_scale:.1f}x", True, (200, 200, 200)), (20, y))
                y += 20
                ui_surface.blit(self.font.render(f"Planets: {len(self.planets)}", True, (200, 200, 200)), (20, y))
                
                # Controls
                y += 30
                ui_surface.blit(self.font.render("─── CONTROLS ───", True, (100, 180, 255)), (20, y))
                controls = [("SPACE", "Pause/Resume"), ("R", "Reset"), ("1-6", "Presets"),
                            ("N", "Add planet"), ("G", "Grid"), ("T", "Trails"),
                            ("V", "Velocity"), ("F", "Forces"), ("C", "Track Planet"),
                            ("+/-", "Time scale"), ("Click", "Select"), ("Mouse", "Camera")]
                y += 20
                for key, action in controls:
                    ui_surface.blit(self.font.render(f"{key:8}", True, (255, 200, 100)), (25, y))
                    ui_surface.blit(self.font.render(action, True, (180, 180, 180)), (100, y))
                    y += 16 # Compact spacing (was 18)
                    
                y += 5
                ui_surface.blit(self.font.render("─── TOGGLES ───", True, (100, 180, 255)), (20, y))
                y += 20
                for name, state in [("Grid", self.show_grid), ("Trails", self.show_trails),
                                ("Velocity", self.show_velocity_vectors), ("Forces", self.show_force_vectors),
                                ("Tracking", self.camera_tracking)]:
                    color = (100, 255, 150) if state else (255, 100, 100)
                    ui_surface.blit(self.font.render(f"{'●' if state else '○'} {name}", True, color), (25, y))
                    y += 16 # Compact spacing (was 18)
                
                if IS_WEB:
                    # Simple text on web to avoid texture/blit issues if any
                    pass
                
                # Grid Resolution Control
                y += 10
                res_text = self.font.render(f"Grid Res: {self.grid_resolution}", True, (200, 200, 200))
                ui_surface.blit(res_text, (25, y))
                
                # Update button positions
                self.res_minus_button.rect.x = 140
                self.res_minus_button.rect.y = y
                self.res_plus_button.rect.x = 180
                self.res_plus_button.rect.y = y
                
                self.res_minus_button.draw(ui_surface, self.font)
                self.res_plus_button.draw(ui_surface, self.font)
            
            # Grid Res Controls drawn above
            
            # Grid Res Controls drawn above
            
            # Planet editor - update position dynamically
            self.planet_editor.x = self.width - 280
            self.planet_editor.draw(ui_surface, self.font, self.title_font)
            
            # Presets panel - dynamic position
            # Height increased for scaling info (was 150)
            presets_h = 170
            if self.presets_panel_collapsed:
                presets_h = 30
            
            # Presets panel - dynamic position
            # Height increased for scaling info (was 150)
            presets_h = 170
            if self.presets_panel_collapsed:
                presets_h = 30
            
            # Revert to left alignment (x=10)
            presets_x = 10
            presets_y = self.height - presets_h - 10
            
            pygame.draw.rect(ui_surface, panel_color, (presets_x, presets_y, 180, presets_h), border_radius=8)
            pygame.draw.rect(ui_surface, (60, 100, 150, 150), (presets_x, presets_y, 180, presets_h), 2, border_radius=8)
            
            # Presets Toggle
            pre_toggle_rect = pygame.Rect(presets_x + 150, presets_y + 3, 24, 24) # Right align in box
            pygame.draw.rect(ui_surface, (40, 60, 80), pre_toggle_rect, border_radius=4)
            sym = "+" if self.presets_panel_collapsed else "−"
            sym_surf = self.font.render(sym, True, (255, 255, 255))
            sym_rect = sym_surf.get_rect(center=pre_toggle_rect.center)
            ui_surface.blit(sym_surf, sym_rect)
            
            # Center title text "--- PRESETS ---"
            # Panel width 180.
            title_text = "─── PRESETS ───"
            title_surf = self.font.render(title_text, True, (100, 180, 255))
            title_rect = title_surf.get_rect(center=(presets_x + 90, presets_y + 15)) # Center of header (30px high)
            ui_surface.blit(title_surf, title_rect)
            
            if not self.presets_panel_collapsed:
                for i, name in enumerate(["1: Solar System", "2: Binary Stars", "3: Three Body", "4: Figure-8", "5: Collision", "6: Cluster"]):
                    ui_surface.blit(self.font.render(name, True, (180, 180, 180)), (presets_x + 10, presets_y + 30 + i * 18))
                
                # Scaling Info
                scale_y = presets_y + 30 + 6 * 18 + 5
                pygame.draw.line(ui_surface, (60, 80, 100), (presets_x + 10, scale_y), (presets_x + 170, scale_y), 1)
                ui_surface.blit(self.font.render("Scale: 1.0 = Earth", True, (120, 140, 160)), (presets_x + 10, scale_y + 5))
            
            # Quit button - dynamic position (bottom right)
            self.quit_button.rect.x = self.width - 110
            self.quit_button.rect.y = self.height - 50
            self.quit_button.draw(ui_surface, self.font)
            
            # Update texture
            texture_data = pygame.image.tostring(ui_surface, "RGBA", True)
            glBindTexture(GL_TEXTURE_2D, self.ui_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            
            self.ui_dirty = False
        
        # Draw texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.ui_texture)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(self.width, 0)
        glTexCoord2f(1, 0); glVertex2f(self.width, self.height)
        glTexCoord2f(0, 0); glVertex2f(0, self.height)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
    def is_point_in_ui(self, pos) -> bool:
        # Left panel (dynamic height)
        # Left panel (dynamic height)
        # Check collapse button first (with generous padding)
        if 275 <= pos[0] <= 310 and 10 <= pos[1] <= 45: return True
        
        panel_h = 40 if self.left_panel_collapsed else self.height - 10
        if 10 <= pos[0] <= 310 and 10 <= pos[1] <= panel_h:
            return True
            
        # Right panel (planet editor) - dynamic position
        # Handled by Editor's own hit test approx
        editor_h = 40 if self.planet_editor.collapsed else 570
        if self.width - 290 <= pos[0] <= self.width - 10 and 10 <= pos[1] <= editor_h + 10:
            return True
            
        # Bottom left presets (x=10 to 190)
        presets_h = 30 if self.presets_panel_collapsed else 170
        presets_y = self.height - presets_h - 10
        if 10 <= pos[0] <= 190 and presets_y <= pos[1] <= self.height - 10:
            return True
        # Quit button - dynamic position (bottom right)
        if self.width - 110 <= pos[0] <= self.width - 10 and self.height - 50 <= pos[1] <= self.height - 15:
            return True
        return False
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                # Only quit if window X button is clicked
                self.running = False
            
            # Handle window resize
            if event.type == VIDEORESIZE:
                self.width = event.w
                self.height = event.h
                self.display = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, self.width, self.height)
                # Update planet editor position
                self.planet_editor.x = self.width - 280
                self.planet_editor.setup_fields()  # Recreate fields at new position
                if self.planet_editor.planet:
                    self.planet_editor.set_planet(self.planet_editor.planet)
                # Update quit button position
                self.quit_button.rect.x = self.width - 110
                self.quit_button.rect.y = self.height - 50
                self.ui_dirty = True
            
            # Check quit button
            if self.quit_button.handle_event(event):
                self.running = False
                continue
            
            # Handle mouse click
            if event.type == MOUSEBUTTONDOWN:
                if self.is_point_in_ui(event.pos):
                    # Check collapse buttons
                    # Left Panel
                    if 275 <= event.pos[0] <= 310 and 10 <= event.pos[1] <= 45:
                        self.left_panel_collapsed = not self.left_panel_collapsed
                        self.ui_dirty = True
                    
                    # Presets Panel Toggle
                    presets_h = 30 if self.presets_panel_collapsed else 170
                    presets_y = self.height - presets_h - 10
                    # Toggle Button at x+150 (10+150=160). y+3.
                    if 155 <= event.pos[0] <= 190 and presets_y - 2 <= event.pos[1] <= presets_y + 32:
                        self.presets_panel_collapsed = not self.presets_panel_collapsed
                        self.ui_dirty = True
                    
                    # Grid Res buttons - handled below to allow hover
                
            # Handle Grid Res buttons (allow hover)
            if not self.left_panel_collapsed:
                if self.res_minus_button.handle_event(event):
                    self.grid_resolution = max(20, self.grid_resolution - 10)
                    self.ui_dirty = True
                if self.res_plus_button.handle_event(event):
                    self.grid_resolution = min(MAX_GRID_RESOLUTION, self.grid_resolution + 10)
                    self.ui_dirty = True
                
            handled, action = self.planet_editor.handle_event(event)
            if handled:
                self.ui_dirty = True
                if action == 'apply':
                    self.planet_editor.apply_to_planet()
                elif action == 'delete':
                    self.remove_selected_planet()
                continue
                
            if event.type == KEYDOWN:
                if any(f.active for f in self.planet_editor.fields.values()):
                    continue
                # Removed ESC key - app only closes via window X button or Quit button
                if event.key == K_SPACE:
                    self.paused = not self.paused
                    self.ui_dirty = True
                elif event.key == K_r:
                    self.load_preset(1)
                elif event.key == K_g:
                    self.show_grid = not self.show_grid
                    self.ui_dirty = True
                elif event.key == K_t:
                    self.show_trails = not self.show_trails
                    self.ui_dirty = True
                elif event.key == K_v:
                    self.show_velocity_vectors = not self.show_velocity_vectors
                    self.ui_dirty = True
                elif event.key == K_f:
                    self.show_force_vectors = not self.show_force_vectors
                    self.ui_dirty = True
                elif event.key == K_c:
                    self.camera_tracking = not self.camera_tracking
                    self.ui_dirty = True
                elif event.key == K_n:
                    self.add_random_planet()
                elif event.key in (K_DELETE, K_BACKSPACE):
                    self.remove_selected_planet()
                elif event.key == K_1:
                    self.load_preset(1)
                elif event.key == K_2:
                    self.load_preset(2)
                elif event.key == K_3:
                    self.load_preset(3)
                elif event.key == K_4:
                    self.load_preset(4)
                elif event.key == K_5:
                    self.load_preset(5)
                elif event.key == K_6:
                    self.load_preset(6)
                elif event.key in (K_PLUS, K_EQUALS, K_KP_PLUS):
                    self.time_scale = min(5.0, self.time_scale + 0.25)
                    self.ui_dirty = True
                elif event.key in (K_MINUS, K_KP_MINUS):
                    self.time_scale = max(0.25, self.time_scale - 0.25)
                    self.ui_dirty = True
                                
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    if not self.is_point_in_ui(event.pos):
                        self.mouse_pressed[0] = True
                        self.select_planet_at_screen_pos(event.pos)
                elif event.button == 3:
                    if not self.is_point_in_ui(event.pos):
                        self.mouse_pressed[2] = True
                elif event.button == 4:
                    self.camera_distance = max(10, self.camera_distance - 2)
                elif event.button == 5:
                    self.camera_distance = min(100, self.camera_distance + 2)
                self.last_mouse_pos = pygame.mouse.get_pos()
                
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_pressed[0] = False
                elif event.button == 3:
                    self.mouse_pressed[2] = False
                    
            elif event.type == MOUSEMOTION:
                dx = event.pos[0] - self.last_mouse_pos[0]
                dy = event.pos[1] - self.last_mouse_pos[1]
                if self.mouse_pressed[0] and not self.is_point_in_ui(event.pos):
                    self.camera_rot_y += dx * 0.5
                    self.camera_rot_x = max(-89, min(89, self.camera_rot_x + dy * 0.5))
                if self.mouse_pressed[2] and not self.is_point_in_ui(event.pos):
                    self.camera_pan[0] -= dx * 0.05
                    self.camera_pan[2] -= dy * 0.05
                self.last_mouse_pos = event.pos
    
    def render(self):
        # debug logs only occasionally to avoid spam
        # print("Render frame...") 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.setup_projection()
        self.setup_camera()
        self.starfield.render()
        self.draw_curved_grid()
        self.draw_force_field()
        for planet in self.planets:
            self.draw_trail(planet)
        for planet in self.planets:
            is_selected = planet.id == self.selected_planet
            self.draw_planet(planet, is_selected)
            self.draw_velocity_vector(planet)
        self.render_ui()
        pygame.display.flip()
        
    async def run(self):
        clock = pygame.time.Clock()
        try:
            while self.running:
                dt = clock.tick(60) / 1000.0
                self.handle_events()
                self.update_physics(dt)
                if self.planet_editor.update(dt * 1000):
                    self.ui_dirty = True
                self.render()
                await asyncio.sleep(0)
        except Exception as e:
            print(f"\nError occurred: {e}")
            import traceback
            traceback.print_exc()
            # On web, input() might freeze/crash, so we avoid it or handle gracefully
            # input("\nPress Enter to exit...") 
        finally:
            pygame.quit()
