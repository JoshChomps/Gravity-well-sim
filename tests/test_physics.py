"""Physics checks that run without opening a window.

The simulator class opens a pygame/OpenGL window in __init__, so these tests
build it with __new__ and set only the state the physics methods use.
Run from the repo root: python -m unittest discover -s tests
"""
import math
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import G, TRAIL_LENGTH  # noqa: E402
from models import Planet  # noqa: E402
from simulator import GravitySimulator  # noqa: E402


class _Editor:
    """Stands in for the planet editor panel."""

    fields = {}

    def set_planet(self, planet):
        pass


def make_sim(*bodies):
    """bodies: (position, velocity, mass, radius) tuples."""
    sim = GravitySimulator.__new__(GravitySimulator)
    sim.planets = []
    sim.planet_counter = 0
    sim.paused = False
    sim.time_scale = 1.0
    sim.selected_planet = None
    sim.ui_dirty = False
    sim.planet_editor = _Editor()
    for pos, vel, mass, radius in bodies:
        sim.planets.append(sim.create_planet(pos, vel, mass, radius, (1.0, 1.0, 1.0)))
    for planet, force in zip(sim.planets, sim.calculate_forces(), strict=True):
        planet.acceleration = force / planet.mass
    return sim


def momentum(sim):
    return sum(p.mass * p.velocity for p in sim.planets)


def energy(sim):
    kinetic = sum(0.5 * p.mass * p.velocity @ p.velocity for p in sim.planets)
    potential = 0.0
    for i, a in enumerate(sim.planets):
        for b in sim.planets[i + 1:]:
            potential -= G * a.mass * b.mass / np.linalg.norm(a.position - b.position)
    return kinetic + potential


def circular_orbit(r=10.0, central_mass=500.0, satellite_mass=1e-3):
    v = math.sqrt(G * central_mass / r)
    return make_sim(([0, 0, 0], [0, 0, 0], central_mass, 1.0), ([r, 0, 0], [0, 0, v], satellite_mass, 0.1)), r, v


class ForceTests(unittest.TestCase):
    def test_two_body_force_matches_newton(self):
        sim = make_sim(([0, 0, 0], [0, 0, 0], 100.0, 0.5), ([4, 0, 0], [0, 0, 0], 10.0, 0.5))
        forces = sim.calculate_forces()
        expected = G * 100.0 * 10.0 / 16.0
        np.testing.assert_allclose(forces[0], [expected, 0, 0])
        np.testing.assert_allclose(forces[1], [-expected, 0, 0])

    def test_forces_sum_to_zero(self):
        rng = random.Random(1)
        bodies = [([rng.uniform(-20, 20) for _ in range(3)], [0, 0, 0], rng.uniform(1, 100), 0.1) for _ in range(8)]
        total = sum(make_sim(*bodies).calculate_forces())
        self.assertLess(np.linalg.norm(total), 1e-9)

    def test_overlapping_bodies_exert_no_force(self):
        sim = make_sim(([0, 0, 0], [0, 0, 0], 100.0, 1.0), ([1.5, 0, 0], [0, 0, 0], 10.0, 1.0))
        for force in sim.calculate_forces():
            np.testing.assert_array_equal(force, [0, 0, 0])


class IntegratorTests(unittest.TestCase):
    def test_momentum_is_conserved(self):
        sim = make_sim(([-4, 0, 0], [0, 0, 1.2], 300, 0.5), ([4, 0, 0], [0, 0.3, -1.0], 200, 0.5),
                       ([0, 2, 12], [1.5, 0, 0], 10, 0.3))
        before = momentum(sim)
        for _ in range(2000):
            sim.update_physics(0.01)
        self.assertEqual(len(sim.planets), 3, "bodies merged; the test needs a collision-free setup")
        np.testing.assert_allclose(momentum(sim), before, atol=1e-9)

    def test_circular_orbit_keeps_its_radius(self):
        sim, r, v = circular_orbit()
        period = 2 * math.pi * r / v
        dt = 0.01
        radii = []
        for _ in range(int(period / dt)):
            sim.update_physics(dt)
            radii.append(np.linalg.norm(sim.planets[1].position - sim.planets[0].position))
        self.assertLess(max(abs(x - r) for x in radii) / r, 0.01)

    def test_energy_drift_is_small_over_one_orbit(self):
        sim, r, v = circular_orbit()
        start = energy(sim)
        for _ in range(int(2 * math.pi * r / v / 0.01)):
            sim.update_physics(0.01)
        self.assertLess(abs(energy(sim) - start) / abs(start), 1e-4)

    def test_paused_simulation_does_not_move(self):
        sim, _, _ = circular_orbit()
        sim.paused = True
        before = [p.position.copy() for p in sim.planets]
        sim.update_physics(0.1)
        for p, b in zip(sim.planets, before, strict=True):
            np.testing.assert_array_equal(p.position, b)

    def test_time_scale_matches_a_longer_step(self):
        a, _, _ = circular_orbit()
        b, _, _ = circular_orbit()
        b.time_scale = 2.0
        a.update_physics(0.02)
        b.update_physics(0.01)
        np.testing.assert_allclose(a.planets[1].position, b.planets[1].position)

    def test_trail_is_capped(self):
        sim, _, _ = circular_orbit()
        for _ in range(TRAIL_LENGTH + 25):
            sim.update_physics(0.01)
        self.assertEqual(len(sim.planets[1].trail), TRAIL_LENGTH)


class CollisionTests(unittest.TestCase):
    def test_merge_conserves_mass_momentum_and_volume(self):
        sim = make_sim(([0, 0, 0], [1, 0, 0], 30.0, 1.0), ([1.5, 0, 0], [-2, 1, 0], 10.0, 0.8))
        p_before = momentum(sim)
        com = sum(p.mass * p.position for p in sim.planets) / 40.0
        sim.handle_collisions()
        self.assertEqual(len(sim.planets), 1)
        merged = sim.planets[0]
        self.assertAlmostEqual(merged.mass, 40.0)
        np.testing.assert_allclose(merged.mass * merged.velocity, p_before)
        np.testing.assert_allclose(merged.position, com)
        self.assertAlmostEqual(merged.radius ** 3, 1.0 + 0.8 ** 3)

    def test_merged_body_keeps_heavier_name(self):
        sim = make_sim(([0, 0, 0], [0, 0, 0], 5.0, 1.0), ([0.5, 0, 0], [0, 0, 0], 50.0, 1.0))
        sim.planets[1].name = "Heavy"
        sim.handle_collisions()
        self.assertEqual(sim.planets[0].name, "Heavy")

    def test_chain_of_overlaps_leaves_no_overlaps(self):
        sim = make_sim(*[([i * 0.5, 0, 0], [0, 0, 0], 1.0, 0.4) for i in range(5)])
        sim.handle_collisions()
        self.assertLess(len(sim.planets), 5)
        self.assertAlmostEqual(sum(p.mass for p in sim.planets), 5.0)
        for i, a in enumerate(sim.planets):
            for b in sim.planets[i + 1:]:
                self.assertGreaterEqual(np.linalg.norm(a.position - b.position), a.radius + b.radius)


class PresetTests(unittest.TestCase):
    def test_every_preset_runs_without_blowing_up(self):
        for preset in range(1, 7):
            with self.subTest(preset=preset):
                random.seed(preset)
                sim = make_sim()
                sim.load_preset(preset)
                self.assertGreater(len(sim.planets), 0)
                mass = sum(p.mass for p in sim.planets)
                for _ in range(500):
                    sim.update_physics(1 / 60)
                for p in sim.planets:
                    self.assertTrue(np.all(np.isfinite(p.position)), p.name)
                    self.assertTrue(np.all(np.isfinite(p.velocity)), p.name)
                self.assertAlmostEqual(sum(p.mass for p in sim.planets), mass)


class PlanetTests(unittest.TestCase):
    def planet(self, mass, radius):
        return Planet(np.zeros(3), np.zeros(3), mass, radius, (1, 1, 1), "p")

    def test_density(self):
        self.assertAlmostEqual(self.planet(4 / 3 * math.pi, 1.0).density(), 1.0)
        self.assertEqual(self.planet(5.0, 0.0).density(), 0)

    def test_set_density_round_trips(self):
        p = self.planet(1.0, 2.0)
        p.set_density(3.5)
        self.assertAlmostEqual(p.density(), 3.5)

    def test_surface_gravity(self):
        self.assertAlmostEqual(self.planet(100.0, 2.0).surface_gravity(), G * 100.0 / 4.0)
        self.assertEqual(self.planet(100.0, 0.0).surface_gravity(), 0)


if __name__ == "__main__":
    unittest.main()
