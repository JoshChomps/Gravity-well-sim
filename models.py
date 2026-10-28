import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple
from config import G

@dataclass
class Planet:
    """Represents a planetary body in the simulation."""
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    radius: float
    color: Tuple[float, float, float]
    name: str
    trail: List[np.ndarray] = field(default_factory=list)
    id: int = 0
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    
    def density(self) -> float:
        """Calculate density: ρ = m / V = m / (4/3 π r³)"""
        volume = (4/3) * math.pi * self.radius**3
        return self.mass / volume if volume > 0 else 0
    
    def set_density(self, density: float):
        """Set mass based on desired density and current radius."""
        volume = (4/3) * math.pi * self.radius**3
        self.mass = density * volume
        
    def surface_gravity(self) -> float:
        """Calculate surface gravity: g = GM/r²"""
        if self.radius > 0:
            return G * self.mass / (self.radius ** 2)
        return 0
