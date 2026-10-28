#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multicolor Diffusion-Binding Simulation

Simulates molecules of different colors diffusing on a 2D surface with
binding/unbinding kinetics. Designed for testing pyS3M analysis pipeline.

Based on:
- Michalet, X.; Berglund, A. J. Phys. Rev. E 2012, 85 (6), 061916.
  (Realistic Brownian motion with camera effects)
- Gillespie, D. T. J. Phys. Chem. 1977, 81 (25), 2340-2361.
  (Stochastic simulation algorithm)

@author: jbeckwith
Created: 2025-11-12
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from scipy.sparse import diags_array
from scipy.spatial.distance import cdist
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from pyS3M.Constants import DriftConstants
import logging
logger = logging.getLogger(__name__)


@dataclass
class Molecule:
    """
    Single molecule with color and diffusion properties.

    Attributes:
        molecule_id: Unique identifier for this molecule
        color: Color label ('R', 'G', or 'B' for red/green/blue)
        position: Current (x, y) coordinates in nm
        D_free: Diffusion coefficient when unbound (nm²/ms)
        D_bound: Diffusion coefficient when bound (nm²/ms)
        is_bound: Current binding state
        bound_partner: Reference to binding partner molecule
        trajectory: History of positions [(x, y), ...]
        times: History of timepoints [t0, t1, ...]
        spectral_profile: Dictionary with A_R, A_G, A_B values
    """
    molecule_id: int
    color: str
    position: np.ndarray
    D_free: float
    D_bound: float = 0.0
    is_bound: bool = False
    bound_partner: Optional['Molecule'] = None
    trajectory: List[np.ndarray] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    spectral_profile: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize spectral profiles based on color."""
        if not self.spectral_profile:
            # Default spectral profiles for R, G, B dyes
            profiles = {
                'R': {'A_R': 0.80, 'A_G': 0.15, 'A_B': 0.05},
                'G': {'A_R': 0.10, 'A_G': 0.80, 'A_B': 0.10},
                'B': {'A_R': 0.05, 'A_G': 0.15, 'A_B': 0.80},
            }
            self.spectral_profile = profiles.get(self.color,
                                                  {'A_R': 0.33, 'A_G': 0.33, 'A_B': 0.34})

        # Initialize trajectory with starting position
        if not self.trajectory:
            self.trajectory = [self.position.copy()]
            self.times = [0.0]

    def get_current_D(self) -> float:
        """Get current diffusion coefficient based on binding state."""
        return self.D_bound if self.is_bound else self.D_free

    def add_position(self, position: np.ndarray, time: float):
        """Add a position to the trajectory."""
        self.trajectory.append(position.copy())
        self.times.append(time)
        self.position = position.copy()


class BindingKinetics:
    """
    Handles binding/unbinding kinetics between molecules using the Gillespie
    algorithm, driven by macroscopic k_on/k_off matrices (concentration-
    dependent), which ensures correct stochastic kinetics.
    """

    def __init__(self, colors: List[str], k_on_matrix: np.ndarray = None,
                 k_off_matrix: np.ndarray = None, binding_radius: float = 100.0):
        """
        Initialize binding kinetics.

        Args:
            colors: List of color labels (e.g., ['R', 'G', 'B'])
            k_on_matrix: 2D array of binding rates (1/(nM·s)) between color pairs
            k_off_matrix: 2D array of unbinding rates (1/s) for each pair
            binding_radius: Distance threshold for binding (nm)
        """
        self.colors = colors
        self.color_to_idx = {c: i for i, c in enumerate(colors)}
        self.binding_radius = binding_radius

        n_colors = len(colors)

        if k_on_matrix is None or k_off_matrix is None:
            raise ValueError("BindingKinetics requires: k_on_matrix, k_off_matrix")

        self.k_on_matrix = np.array(k_on_matrix)
        self.k_off_matrix = np.array(k_off_matrix)

        # Validate
        assert self.k_on_matrix.shape == (n_colors, n_colors), \
            f"k_on_matrix shape {self.k_on_matrix.shape} != ({n_colors}, {n_colors})"
        assert self.k_off_matrix.shape == (n_colors, n_colors), \
            f"k_off_matrix shape {self.k_off_matrix.shape} != ({n_colors}, {n_colors})"

        # Make symmetric
        self.k_on_matrix = (self.k_on_matrix + self.k_on_matrix.T) / 2
        self.k_off_matrix = (self.k_off_matrix + self.k_off_matrix.T) / 2

        # Track binding events for analysis
        self.binding_events: List[Dict] = []
        self.unbinding_events: List[Dict] = []

    def can_bind(self, mol1: Molecule, mol2: Molecule) -> bool:
        """
        Check if two molecules can bind.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            can_bind: True if binding is allowed
        """
        # Can't bind if either is already bound
        if mol1.is_bound or mol2.is_bound:
            return False

        # Check if binding rate is non-zero
        idx1 = self.color_to_idx[mol1.color]
        idx2 = self.color_to_idx[mol2.color]
        return self.k_on_matrix[idx1, idx2] > 0

    def get_binding_rate(self, mol1: Molecule, mol2: Molecule) -> float:
        """
        Get binding rate constant for two molecules.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            k_on: Binding rate (1/(nM·s))
        """
        idx1 = self.color_to_idx[mol1.color]
        idx2 = self.color_to_idx[mol2.color]
        return self.k_on_matrix[idx1, idx2]

    def get_unbinding_rate(self, mol1: Molecule, mol2: Molecule) -> float:
        """
        Get unbinding rate constant for a bound pair.

        Args:
            mol1: First molecule
            mol2: Second molecule

        Returns:
            k_off: Unbinding rate (1/s)
        """
        idx1 = self.color_to_idx[mol1.color]
        idx2 = self.color_to_idx[mol2.color]
        return self.k_off_matrix[idx1, idx2]

    def find_potential_binding_pairs(self, molecules: List[Molecule]) -> List[Tuple[Molecule, Molecule, float]]:
        """
        Find all molecule pairs within binding radius.

        Args:
            molecules: List of all molecules

        Returns:
            pairs: List of (mol1, mol2, distance) tuples within binding radius
        """
        # Get positions of unbound molecules
        unbound_mols = [m for m in molecules if not m.is_bound]

        if len(unbound_mols) < 2:
            return []

        # Compute pairwise distances
        positions = np.array([m.position for m in unbound_mols])
        distances = cdist(positions, positions)

        pairs = []
        for i in range(len(unbound_mols)):
            for j in range(i + 1, len(unbound_mols)):
                mol1 = unbound_mols[i]
                mol2 = unbound_mols[j]
                dist = distances[i, j]

                if dist <= self.binding_radius and self.can_bind(mol1, mol2):
                    pairs.append((mol1, mol2, dist))

        return pairs

    def calculate_propensities(self, molecules: List[Molecule], area: float) -> Tuple[List, List, float]:
        """
        Calculate all binding and unbinding propensities.

        Propensity = k_on × effective_concentration × volume
        (concentration-dependent, volume-based calculation)

        Args:
            molecules: List of all molecules
            area: Simulation area (nm²)

        Returns:
            binding_propensities: List of (mol1, mol2, propensity) for binding events
            unbinding_propensities: List of (mol1, mol2, propensity) for unbinding events
            total_propensity: Sum of all propensities
        """
        binding_propensities = []
        unbinding_propensities = []

        # Find potential binding pairs
        pairs = self.find_potential_binding_pairs(molecules)

        # For surface diffusion: effective concentration ~ 1/area
        # Propensity = k_on × (effective concentration)
        effective_volume = area * 1.0  # nm² × 1 nm height = nm³
        nM_to_nm3 = 1e-9 / 6.022e23 * 1e24  # Convert nM to molecules/nm³

        for mol1, mol2, dist in pairs:
            k_on = self.get_binding_rate(mol1, mol2)
            # Propensity increases as molecules get closer
            # Use simple model: full rate within binding radius
            propensity = k_on * nM_to_nm3 * effective_volume
            binding_propensities.append((mol1, mol2, propensity))

        # Calculate unbinding propensities
        bound_pairs = self._get_bound_pairs(molecules)
        for mol1, mol2 in bound_pairs:
            k_off = self.get_unbinding_rate(mol1, mol2)
            # Convert from 1/s to 1/ms
            unbinding_propensities.append((mol1, mol2, k_off / 1000.0))

        # Total propensity
        total = sum(p[2] for p in binding_propensities) + \
                sum(p[2] for p in unbinding_propensities)

        return binding_propensities, unbinding_propensities, total

    def _get_bound_pairs(self, molecules: List[Molecule]) -> List[Tuple[Molecule, Molecule]]:
        """
        Get all bound molecule pairs (helper to avoid code duplication).

        Returns:
            pairs: List of (mol1, mol2) tuples for bound pairs
        """
        bound_pairs_set = set()
        for mol in molecules:
            if mol.is_bound and mol.bound_partner is not None:
                # Store pairs as sorted tuple to avoid duplicates
                pair = tuple(sorted([mol.molecule_id, mol.bound_partner.molecule_id]))
                bound_pairs_set.add(pair)

        # Convert to list of molecule pairs
        pairs = []
        for pair_ids in bound_pairs_set:
            mol1 = next(m for m in molecules if m.molecule_id == pair_ids[0])
            mol2 = next(m for m in molecules if m.molecule_id == pair_ids[1])
            pairs.append((mol1, mol2))

        return pairs

    def choose_event(self, binding_propensities: List, unbinding_propensities: List,
                    total_propensity: float) -> Optional[Tuple[str, Molecule, Molecule]]:
        """
        Choose next binding/unbinding event stochastically.

        Uses Gillespie algorithm: probability ∝ propensity

        Args:
            binding_propensities: List of binding events
            unbinding_propensities: List of unbinding events
            total_propensity: Sum of all propensities

        Returns:
            event: ('bind', mol1, mol2) or ('unbind', mol1, mol2) or None
        """
        if total_propensity == 0:
            return None

        # Choose event with probability proportional to propensity
        r = np.random.uniform(0, total_propensity)

        cumsum = 0
        # Check binding events
        for mol1, mol2, prop in binding_propensities:
            cumsum += prop
            if r < cumsum:
                return ('bind', mol1, mol2)

        # Check unbinding events
        for mol1, mol2, prop in unbinding_propensities:
            cumsum += prop
            if r < cumsum:
                return ('unbind', mol1, mol2)

        return None

    def execute_binding(self, mol1: Molecule, mol2: Molecule, time: float):
        """
        Execute a binding event.

        Args:
            mol1: First molecule
            mol2: Second molecule
            time: Current simulation time
        """
        mol1.is_bound = True
        mol2.is_bound = True
        mol1.bound_partner = mol2
        mol2.bound_partner = mol1

        # Record event
        self.binding_events.append({
            'time': time,
            'mol1_id': mol1.molecule_id,
            'mol2_id': mol2.molecule_id,
            'mol1_color': mol1.color,
            'mol2_color': mol2.color,
            'position': (mol1.position + mol2.position) / 2
        })

    def execute_unbinding(self, mol1: Molecule, mol2: Molecule, time: float):
        """
        Execute an unbinding event.

        Args:
            mol1: First molecule
            mol2: Second molecule
            time: Current simulation time
        """
        mol1.is_bound = False
        mol2.is_bound = False
        mol1.bound_partner = None
        mol2.bound_partner = None

        # Record event
        self.unbinding_events.append({
            'time': time,
            'mol1_id': mol1.molecule_id,
            'mol2_id': mol2.molecule_id,
            'mol1_color': mol1.color,
            'mol2_color': mol2.color,
            'position': (mol1.position + mol2.position) / 2
        })

    def process_events(self, molecules: List[Molecule], dt: float,
                      area: float, current_time: float) -> int:
        """
        Process binding/unbinding events during a timestep.

        Uses Gillespie algorithm to determine if events occur.

        Args:
            molecules: List of all molecules
            dt: Timestep duration (ms)
            area: Simulation area (nm²)
            current_time: Current simulation time (ms)

        Returns:
            n_events: Number of events that occurred
        """
        n_events = 0

        bind_props, unbind_props, total_prop = self.calculate_propensities(molecules, area)

        if total_prop == 0:
            return 0

        # Time to next event (exponential distribution)
        # Convert rates: k_on is 1/(nM·s), k_off is 1/s
        # Need to convert to 1/ms
        total_prop_per_ms = total_prop / 1000.0  # Convert /s to /ms

        tau = np.random.exponential(1.0 / total_prop_per_ms) if total_prop_per_ms > 0 else np.inf

        # Process events that occur within this timestep
        time_elapsed = 0
        while time_elapsed + tau < dt:
            # Choose which event occurs
            event = self.choose_event(bind_props, unbind_props, total_prop)

            if event is None:
                break

            event_type, mol1, mol2 = event
            event_time = current_time + time_elapsed + tau

            if event_type == 'bind':
                self.execute_binding(mol1, mol2, event_time)
            else:  # unbind
                self.execute_unbinding(mol1, mol2, event_time)

            n_events += 1
            time_elapsed += tau

            # Recalculate propensities after event
            bind_props, unbind_props, total_prop = self.calculate_propensities(molecules, area)

            if total_prop == 0:
                break

            total_prop_per_ms = total_prop / 1000.0
            tau = np.random.exponential(1.0 / total_prop_per_ms) if total_prop_per_ms > 0 else np.inf

        return n_events


class LangevinDiffusion2D:
    """
    2D Brownian diffusion with realistic camera imaging effects.

    Implements the model from Michalet & Berglund (2012):
    - Motion blur from finite exposure time
    - Dynamic localization error
    - Covariance structure for realistic trajectories
    """

    def __init__(self, sigma0: float, s0: float, R: float = 1.0/6):
        """
        Initialize diffusion simulator.

        Args:
            sigma0: Static localization error (nm)
            s0: Standard deviation of PSF (nm)
            R: Motion blur coefficient (default 1/6 for typical camera)
        """
        self.sigma0 = sigma0
        self.s0 = s0
        self.R = R

    def compute_dynamic_localization_error(self, D: float, t_exposure: float) -> float:
        """
        Compute dynamic localization error including motion blur.

        From Michalet & Berglund Eq. 3:
        σ = σ₀√(1 + D·tₑ/s₀²)

        Args:
            D: Diffusion coefficient (nm²/ms)
            t_exposure: Camera exposure time (ms)

        Returns:
            sigma: Dynamic localization error (nm)
        """
        return self.sigma0 * np.sqrt(1.0 + (D * t_exposure) / (self.s0**2))

    def generate_displacement_1D(self, D: float, N: int, dt: float,
                                  t_exposure: float) -> np.ndarray:
        """
        Generate realistic 1D Brownian displacements.

        Uses covariance matrix from Michalet & Berglund Eqs. 2-5:
        - Diagonal: 2D·Δt·(1-2R) + 2σ²
        - Off-diagonal: 2R·D·Δt - σ²

        Args:
            D: Diffusion coefficient (nm²/ms)
            N: Number of steps
            dt: Time step between data points (ms)
            t_exposure: Camera exposure duration (ms)

        Returns:
            positions: Array of N positions (starting from 0)
        """
        if N < 2:
            return np.array([0.0])

        # Compute dynamic localization error
        sigma = self.compute_dynamic_localization_error(D, t_exposure)

        # Build covariance matrix (tridiagonal)
        diagonal = 2 * D * dt * (1 - 2 * self.R) + 2 * sigma**2
        off_diagonal = 2 * self.R * D * dt - sigma**2

        # Create sparse tridiagonal matrix
        stack = np.vstack([
            np.full(N - 1, off_diagonal),
            np.full(N - 1, diagonal),
            np.full(N - 1, off_diagonal),
        ])

        cov = diags_array(stack, offsets=[-1, 0, 1],
                          shape=(N - 1, N - 1)).toarray()

        # Generate correlated random displacements
        rng = np.random.default_rng()
        displacements = rng.multivariate_normal(np.zeros(N - 1), cov)

        # Return cumulative positions (starting from 0)
        positions = np.zeros(N)
        positions[1:] = np.cumsum(displacements)

        return positions

    def generate_trajectory_2D(self, D: float, N: int, dt: float,
                               t_exposure: float,
                               starting_position: np.ndarray) -> np.ndarray:
        """
        Generate realistic 2D Brownian trajectory.

        Args:
            D: Diffusion coefficient (nm²/ms)
            N: Number of steps
            dt: Time step (ms)
            t_exposure: Camera exposure time (ms)
            starting_position: Initial (x, y) position (nm)

        Returns:
            trajectory: Array of shape (N, 2) with (x, y) positions
        """
        # Generate independent x and y displacements
        x_disp = self.generate_displacement_1D(D, N, dt, t_exposure)
        y_disp = self.generate_displacement_1D(D, N, dt, t_exposure)

        # Combine and add starting position
        trajectory = np.column_stack([x_disp, y_disp])
        trajectory += starting_position

        return trajectory


class DiffusionSimulator2D:
    """
    Main simulator for 2D diffusion of multiple molecules.

    Features:
    - Multiple molecules with different diffusion coefficients
    - Boundary conditions (periodic, reflective, absorbing)
    - Realistic camera imaging effects
    - Compatible with pyS3M analysis
    """

    def __init__(self, area: Tuple[float, float], dt: float, t_exposure: float,
                 sigma0: float, s0: float, R: float = 1.0/6,
                 boundary: str = 'reflective',
                 binding_kinetics: Optional[BindingKinetics] = None):
        """
        Initialize 2D diffusion simulator.

        Args:
            area: (width, height) of simulation area in nm
            dt: Timestep for simulation (ms)
            t_exposure: Camera exposure time (ms)
            sigma0: Static localization error (nm)
            s0: PSF standard deviation (nm)
            R: Motion blur coefficient (default 1/6)
            boundary: Boundary condition ('periodic', 'reflective', 'absorbing')
            binding_kinetics: Optional BindingKinetics object for binding/unbinding
        """
        self.area = np.array(area)
        self.dt = dt
        self.t_exposure = t_exposure
        self.boundary = boundary
        self.current_time = 0.0

        # Diffusion engine
        self.diffusion_engine = LangevinDiffusion2D(sigma0, s0, R)

        # Binding kinetics
        self.binding_kinetics = binding_kinetics

        # Molecule storage
        self.molecules: List[Molecule] = []
        self.molecule_counter = 0

    def add_molecule(self, color: str, position: np.ndarray, D_free: float,
                     D_bound: float = 0.0,
                     spectral_profile: Optional[Dict[str, float]] = None) -> Molecule:
        """
        Add a molecule to the simulation.

        Args:
            color: Color label ('R', 'G', or 'B')
            position: Initial (x, y) position in nm
            D_free: Diffusion coefficient when unbound (nm²/ms)
            D_bound: Diffusion coefficient when bound (nm²/ms)
            spectral_profile: Optional custom spectral profile

        Returns:
            molecule: The created Molecule object
        """
        # Ensure position is within bounds
        position = self._apply_boundary_condition(position)

        mol = Molecule(
            molecule_id=self.molecule_counter,
            color=color,
            position=position,
            D_free=D_free,
            D_bound=D_bound,
            spectral_profile=spectral_profile or {}
        )

        self.molecules.append(mol)
        self.molecule_counter += 1

        return mol

    def add_molecules_random(self, n_molecules: int, color: str,
                            D_free: float, D_bound: float = 0.0) -> List[Molecule]:
        """
        Add multiple molecules with random starting positions.

        Args:
            n_molecules: Number of molecules to add
            color: Color label
            D_free: Diffusion coefficient when unbound
            D_bound: Diffusion coefficient when bound

        Returns:
            molecules: List of created molecules
        """
        molecules = []
        for _ in range(n_molecules):
            x = np.random.uniform(0, self.area[0])
            y = np.random.uniform(0, self.area[1])
            position = np.array([x, y])

            mol = self.add_molecule(color, position, D_free, D_bound)
            molecules.append(mol)

        return molecules

    def _apply_boundary_condition(self, position: np.ndarray) -> np.ndarray:
        """
        Apply boundary conditions to a position.

        Args:
            position: (x, y) position

        Returns:
            corrected_position: Position after applying boundary
        """
        pos = position.copy()

        if self.boundary == 'periodic':
            # Wrap around boundaries
            pos = np.mod(pos, self.area)

        elif self.boundary == 'reflective':
            # Reflect at boundaries
            for i in range(2):
                if pos[i] < 0:
                    pos[i] = -pos[i]
                elif pos[i] > self.area[i]:
                    pos[i] = 2 * self.area[i] - pos[i]

            # Clamp to ensure within bounds (in case of multiple reflections)
            pos = np.clip(pos, 0, self.area)

        elif self.boundary == 'absorbing':
            # Mark molecules that leave as invalid (handled elsewhere)
            pass

        return pos

    def _update_bound_pair_positions(self, mol1: Molecule, mol2: Molecule, new_position: np.ndarray):
        """
        Update positions of bound pair to move together.

        Bound molecules move to their center of mass.

        Args:
            mol1: First molecule in bound pair
            mol2: Second molecule in bound pair
            new_position: New position for the pair
        """
        mol1.position = new_position.copy()
        mol2.position = new_position.copy()

    def run(self, n_steps: int, enable_binding: bool = True, chunk_size: int = 10):
        """
        Run simulation for n_steps with optional binding kinetics.

        With binding disabled: Generates full trajectories for all molecules at once
        to preserve proper covariance structure from Michalet & Berglund (2012).

        With binding enabled: Uses chunk-based approach:
        - Generates trajectories in chunks
        - Processes binding/unbinding events between chunks
        - Bound pairs move together at D_bound

        Args:
            n_steps: Number of timesteps to simulate
            enable_binding: Whether to process binding events (default True)
            chunk_size: Steps per chunk when binding enabled (default 10)
        """
        if not enable_binding or self.binding_kinetics is None:
            # Fast path: no binding, generate full trajectories at once
            for mol in self.molecules:
                D = mol.get_current_D()
                trajectory = self.diffusion_engine.generate_trajectory_2D(
                    D=D,
                    N=n_steps,
                    dt=self.dt,
                    t_exposure=self.t_exposure,
                    starting_position=mol.position
                )

                for i in range(n_steps):
                    new_position = self._apply_boundary_condition(trajectory[i])
                    time = mol.times[-1] + (i + 1) * self.dt
                    mol.add_position(new_position, time)

            self.current_time += n_steps * self.dt
            return

        # With binding: process in chunks
        steps_remaining = n_steps
        while steps_remaining > 0:
            current_chunk = min(chunk_size, steps_remaining)

            # Process binding/unbinding events
            area_nm2 = self.area[0] * self.area[1]

            self.binding_kinetics.process_events(
                self.molecules, current_chunk * self.dt, area_nm2, self.current_time,
            )

            # Generate trajectories for this chunk
            # Track which molecules are bound to avoid double-processing
            processed_pairs = set()

            for mol in self.molecules:
                # Skip if this molecule was already processed as part of a bound pair
                if mol.molecule_id in processed_pairs:
                    continue

                D = mol.get_current_D()

                if mol.is_bound and mol.bound_partner is not None:
                    # Generate trajectories for both partners
                    partner = mol.bound_partner

                    traj1 = self.diffusion_engine.generate_trajectory_2D(
                        D=D, N=current_chunk, dt=self.dt,
                        t_exposure=self.t_exposure,
                        starting_position=mol.position
                    )

                    traj2 = self.diffusion_engine.generate_trajectory_2D(
                        D=partner.get_current_D(), N=current_chunk, dt=self.dt,
                        t_exposure=self.t_exposure,
                        starting_position=partner.position
                    )

                    # Move bound pair to center of mass at each step
                    for i in range(current_chunk):
                        pos1 = self._apply_boundary_condition(traj1[i])
                        pos2 = self._apply_boundary_condition(traj2[i])
                        com_pos = (pos1 + pos2) / 2

                        time = self.current_time + (i + 1) * self.dt
                        mol.add_position(com_pos, time)
                        partner.add_position(com_pos, time)

                    processed_pairs.add(mol.molecule_id)
                    processed_pairs.add(partner.molecule_id)

                else:
                    # Unbound molecule moves independently
                    trajectory = self.diffusion_engine.generate_trajectory_2D(
                        D=D, N=current_chunk, dt=self.dt,
                        t_exposure=self.t_exposure,
                        starting_position=mol.position
                    )

                    for i in range(current_chunk):
                        new_position = self._apply_boundary_condition(trajectory[i])
                        time = self.current_time + (i + 1) * self.dt
                        mol.add_position(new_position, time)

                    processed_pairs.add(mol.molecule_id)

            self.current_time += current_chunk * self.dt
            steps_remaining -= current_chunk

    def get_trajectory(self, molecule_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trajectory for a specific molecule.

        Args:
            molecule_id: ID of molecule

        Returns:
            positions: Array of shape (N, 2) with (x, y) positions
            times: Array of timepoints
        """
        mol = self.molecules[molecule_id]
        positions = np.array(mol.trajectory)
        times = np.array(mol.times)

        return positions, times

    def get_all_trajectories(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Get all molecule trajectories.

        Returns:
            trajectories: Dictionary mapping molecule_id -> (positions, times)
        """
        trajectories = {}
        for mol in self.molecules:
            positions = np.array(mol.trajectory)
            times = np.array(mol.times)
            trajectories[mol.molecule_id] = (positions, times)

        return trajectories

    def reset(self):
        """Reset simulation to initial state."""
        self.current_time = 0.0

        for mol in self.molecules:
            # Keep only first position
            mol.trajectory = [mol.trajectory[0].copy()]
            mol.times = [0.0]
            mol.position = mol.trajectory[0].copy()
            mol.is_bound = False
            mol.bound_partner = None


def compute_msd_from_trajectory(trajectory: np.ndarray,
                                 max_tau: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean squared displacement from a trajectory.

    MSD(τ) = ⟨[r(t+τ) - r(t)]²⟩

    Args:
        trajectory: Array of shape (N, 2) with (x, y) positions
        max_tau: Maximum lag time to compute (default: N//4)

    Returns:
        tau_array: Array of lag times (in units of timesteps)
        msd_array: Array of MSD values
    """
    N = len(trajectory)

    if max_tau is None:
        max_tau = N // 4  # Use first quarter to ensure good statistics

    max_tau = min(max_tau, N - 1)

    msd = np.zeros(max_tau)

    for tau in range(1, max_tau + 1):
        displacements = trajectory[tau:] - trajectory[:-tau]
        squared_displacements = np.sum(displacements**2, axis=1)
        msd[tau - 1] = np.mean(squared_displacements)

    tau_array = np.arange(1, max_tau + 1)

    return tau_array, msd


def estimate_D_from_msd(tau_array: np.ndarray, msd_array: np.ndarray,
                        dt: float, n_d: int = 2, fit_points: int = 10) -> float:
    """
    Estimate diffusion coefficient from MSD using linear fit.

    For 2D: MSD(τ) = 4D·τ + offset

    Args:
        tau_array: Array of lag times (in timesteps)
        msd_array: Array of MSD values
        dt: Timestep duration (ms)
        n_d: Number of dimensions (default 2)
        fit_points: Number of initial points to use for linear fit

    Returns:
        D: Estimated diffusion coefficient (nm²/ms)
    """
    # Use first few points for linear fit
    fit_points = min(fit_points, len(tau_array))

    # Convert tau to time
    t = tau_array[:fit_points] * dt
    msd = msd_array[:fit_points]

    # Linear fit: MSD = 2·n_d·D·t + offset
    # Slope = 2·n_d·D
    coeffs = np.polyfit(t, msd, deg=1)
    slope = coeffs[0]

    D = slope / (2 * n_d)

    return D


# =============================================================================
# Camera Imaging Adapter - Converts diffusion trajectories to TIFF movies
# =============================================================================

class CameraAdapter:
    """
    Adapter to convert diffusion simulation trajectories to camera images.

    This class bridges DiffusionSimulator2D output to Multicolour_Simulation_Functions
    for generating realistic TIFF movies with Bayer filtering, PSF, and camera noise.

    Features:
    - Poisson brightness sampling per frame
    - Spectral profile handling for multicolor imaging
    - Blinking support (future extension)
    - Compatible with existing pyS3M analysis pipeline
    """

    def __init__(self, simulator: DiffusionSimulator2D):
        """
        Initialize camera adapter with diffusion simulator.

        Args:
            simulator: DiffusionSimulator2D instance with run trajectories
        """
        self.simulator = simulator

    def prepare_localisations_for_imaging(
        self,
        n_photons_per_dye: Dict[str, float],
        frame_indices: Optional[np.ndarray] = None,
        blinking_probability: Optional[Dict[str, float]] = None,
        poisson_brightness: bool = True,
        random_state: Optional[np.random.Generator] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Convert trajectories to localisation format for image generation.

        This prepares molecule positions, photon counts, and spectral profiles
        for each frame in the format expected by gen_camera_image_stack.

        Args:
            n_photons_per_dye: Mean photon count per dye color {'R': 1000, 'G': 800, ...}
            frame_indices: Frame indices to extract (default: all frames)
            blinking_probability: Probability of blinking off per frame (default: None = always on)
            poisson_brightness: Sample photons from Poisson distribution (default: True)
            random_state: Random generator for reproducibility

        Returns:
            x0y0: Dict[dye_name, positions array (n_frames, 2, n_molecules)], ordered
                (x, y): index 0 is column/x, index 1 is row/y, matching
                gen_camera_image_stack's placement convention
            n_photons: Dict[dye_name, photon counts (n_frames, n_molecules)]
            spectral_profiles: Dict[dye_name, (A_R, A_G, A_B) arrays (n_molecules, 3)]
        """
        if random_state is None:
            random_state = np.random.default_rng()

        # Determine which frames to extract
        if frame_indices is None:
            # Use all frames from simulation
            n_frames = len(self.simulator.molecules[0].trajectory)
            frame_indices = np.arange(n_frames)
        else:
            n_frames = len(frame_indices)

        # Group molecules by color
        molecules_by_color: Dict[str, List[Molecule]] = {}
        for mol in self.simulator.molecules:
            color = mol.color
            if color not in molecules_by_color:
                molecules_by_color[color] = []
            molecules_by_color[color].append(mol)

        # Prepare output dictionaries
        x0y0 = {}
        n_photons = {}
        spectral_profiles = {}

        for color, molecules in molecules_by_color.items():
            n_molecules = len(molecules)

            # Initialize position array: (n_frames, 2, n_molecules)
            positions = np.zeros((n_frames, 2, n_molecules))

            # Initialize photon counts: (n_frames, n_molecules)
            photons = np.zeros((n_frames, n_molecules))

            # Get spectral profiles (constant per molecule)
            profiles = np.zeros((n_molecules, 3))  # (A_R, A_G, A_B)

            for mol_idx, mol in enumerate(molecules):
                # Extract positions at requested frames
                for frame_idx, traj_idx in enumerate(frame_indices):
                    if traj_idx < len(mol.trajectory):
                        # gen_camera_image_stack expects (x, y) order in this axis.
                        positions[frame_idx, 0, mol_idx] = mol.trajectory[traj_idx][0]  # x
                        positions[frame_idx, 1, mol_idx] = mol.trajectory[traj_idx][1]  # y

                        # Sample photons for this frame
                        mean_photons = n_photons_per_dye[color]

                        # Check blinking
                        is_visible = True
                        if blinking_probability is not None and color in blinking_probability:
                            if random_state.random() < blinking_probability[color]:
                                is_visible = False

                        if is_visible:
                            if poisson_brightness:
                                # Poisson sampling for realistic shot noise
                                photons[frame_idx, mol_idx] = random_state.poisson(mean_photons)
                            else:
                                # Deterministic brightness
                                photons[frame_idx, mol_idx] = mean_photons
                        else:
                            # Blinked off
                            photons[frame_idx, mol_idx] = 0

                # Get spectral profile
                profiles[mol_idx, 0] = mol.spectral_profile.get('A_R', 0.33)
                profiles[mol_idx, 1] = mol.spectral_profile.get('A_G', 0.33)
                profiles[mol_idx, 2] = mol.spectral_profile.get('A_B', 0.34)

            # Store in dictionaries
            dye_name = f"dye_{color}"
            x0y0[dye_name] = positions
            n_photons[dye_name] = photons
            spectral_profiles[dye_name] = profiles

        return x0y0, n_photons, spectral_profiles

    def generate_ground_truth_rgb_video(
        self,
        output_path: str,
        frame_indices: Optional[np.ndarray] = None,
        image_size_nm: Optional[Tuple[float, float]] = None,
        pixel_size_nm: float = DriftConstants.XIMEA_PIXEL_SIZE_NM,
        gaussian_width_nm: float = 50.0,
        colormap: str = 'spectral',
        save_video: bool = True,
        background_value: int = 10,
        scale_intensity: bool = True,
        max_intensity: int = 255,
        intensity_percentile: float = 99.5,
    ) -> np.ndarray:
        """
        Generate ground truth RGB video showing molecules as colored Gaussians.

        This creates a "perfect" visualization where each molecule is rendered as a
        Gaussian spot with color determined by its spectral profile. Useful for
        comparing Bayer-filtered analysis results to ground truth.

        Args:
            output_path: Path to save video (TIFF stack)
            frame_indices: Frames to render (default: all)
            image_size_nm: (width, height) in nm (default: use simulator area)
            pixel_size_nm: Pixel size for video (default: 69 nm, matching camera)
            gaussian_width_nm: Width (sigma) of Gaussian PSFs (default: 50 nm)
            colormap: How to map spectral profiles to RGB:
                      'spectral' - Blue (B-dominant) → Red (R-dominant)
                      'direct' - Use (A_R, A_G, A_B) directly as RGB
            save_video: Whether to save TIFF file
            background_value: Background pixel value (default: 10)
            scale_intensity: Whether to auto-scale intensity per frame
            max_intensity: Maximum pixel value (default: 255 for uint8)
            intensity_percentile: Percentile for intensity scaling (default: 99.5).
                                 Lower values (e.g., 95) give brighter videos by allowing
                                 bright spots to saturate. 100 = use absolute max (dimmest).

        Returns:
            rgb_video: RGB video stack (n_frames, height, width, 3) uint8
        """
        # Determine image size
        if image_size_nm is None:
            image_size_nm = self.simulator.area

        width_pixels = int(np.ceil(image_size_nm[0] / pixel_size_nm))
        height_pixels = int(np.ceil(image_size_nm[1] / pixel_size_nm))

        # Determine frames
        if frame_indices is None:
            n_frames = len(self.simulator.molecules[0].trajectory)
            frame_indices = np.arange(n_frames)
        else:
            n_frames = len(frame_indices)

        # Initialize RGB video
        rgb_video = np.zeros((n_frames, height_pixels, width_pixels, 3), dtype=np.uint8)

        # Add background
        rgb_video[:, :, :, :] = background_value

        # Create coordinate grids (in pixels)
        x_pixels = np.arange(width_pixels)
        y_pixels = np.arange(height_pixels)
        X, Y = np.meshgrid(x_pixels, y_pixels)

        # Convert Gaussian width to pixels
        sigma_pixels = gaussian_width_nm / pixel_size_nm

        logger.info("Generating ground truth RGB video: %d frames, %d×%d pixels", n_frames, width_pixels, height_pixels)

        # First pass: render all frames to float arrays
        rgb_video_float = np.zeros((n_frames, height_pixels, width_pixels, 3), dtype=np.float32)

        for frame_idx, traj_idx in enumerate(frame_indices):
            # Accumulate RGB channels
            r_channel = np.zeros((height_pixels, width_pixels), dtype=np.float32)
            g_channel = np.zeros((height_pixels, width_pixels), dtype=np.float32)
            b_channel = np.zeros((height_pixels, width_pixels), dtype=np.float32)

            # Render each molecule
            for mol in self.simulator.molecules:
                if traj_idx >= len(mol.trajectory):
                    continue

                # Get position in nm
                pos_nm = mol.trajectory[traj_idx]
                x_nm, y_nm = pos_nm[0], pos_nm[1]

                # Convert to pixels
                x_px = x_nm / pixel_size_nm
                y_px = y_nm / pixel_size_nm

                # Check if molecule is in field of view
                if x_px < -3*sigma_pixels or x_px > width_pixels + 3*sigma_pixels:
                    continue
                if y_px < -3*sigma_pixels or y_px > height_pixels + 3*sigma_pixels:
                    continue

                # Generate 2D Gaussian
                gaussian = np.exp(-((X - x_px)**2 + (Y - y_px)**2) / (2 * sigma_pixels**2))

                # Get color based on spectral profile
                if colormap == 'spectral':
                    # Map spectral profile to blue→red gradient
                    # Calculate "redness" metric: high A_R = red, high A_B = blue
                    A_R = mol.spectral_profile.get('A_R', 0.33)
                    A_G = mol.spectral_profile.get('A_G', 0.33)
                    A_B = mol.spectral_profile.get('A_B', 0.34)

                    # Compute spectral position (0=blue, 1=red)
                    # Weight: R=1, G=0.5, B=0
                    spectral_pos = (A_R * 1.0 + A_G * 0.5 + A_B * 0.0)

                    # Map to RGB using smooth gradient
                    # Blue (0, 0, 1) → Cyan (0, 1, 1) → Green (0, 1, 0) → Yellow (1, 1, 0) → Red (1, 0, 0)
                    if spectral_pos < 0.25:
                        # Blue → Cyan
                        t = spectral_pos / 0.25
                        rgb = np.array([0.0, t, 1.0])
                    elif spectral_pos < 0.5:
                        # Cyan → Green
                        t = (spectral_pos - 0.25) / 0.25
                        rgb = np.array([0.0, 1.0, 1.0 - t])
                    elif spectral_pos < 0.75:
                        # Green → Yellow
                        t = (spectral_pos - 0.5) / 0.25
                        rgb = np.array([t, 1.0, 0.0])
                    else:
                        # Yellow → Red
                        t = (spectral_pos - 0.75) / 0.25
                        rgb = np.array([1.0, 1.0 - t, 0.0])

                elif colormap == 'direct':
                    # Use spectral profile directly as RGB
                    A_R = mol.spectral_profile.get('A_R', 0.33)
                    A_G = mol.spectral_profile.get('A_G', 0.33)
                    A_B = mol.spectral_profile.get('A_B', 0.34)
                    rgb = np.array([A_R, A_G, A_B])
                    # Normalize to max = 1
                    rgb = rgb / np.max(rgb) if np.max(rgb) > 0 else rgb

                else:
                    raise ValueError(f"Unknown colormap: {colormap}")

                # Add to RGB channels (Gaussian peak = 1.0)
                r_channel += gaussian * rgb[0]
                g_channel += gaussian * rgb[1]
                b_channel += gaussian * rgb[2]

            # Store in float video
            rgb_video_float[frame_idx, :, :, 0] = r_channel
            rgb_video_float[frame_idx, :, :, 1] = g_channel
            rgb_video_float[frame_idx, :, :, 2] = b_channel

            if frame_idx % 100 == 0:
                logger.debug("  Frame %d/%d complete", frame_idx, n_frames)

        logger.info("Rendering complete, applying intensity scaling...")

        # Second pass: apply global intensity scaling
        if scale_intensity:
            # Use percentile-based scaling for better brightness
            # This allows rare bright spots to saturate while boosting typical intensities
            if intensity_percentile >= 100.0:
                # Use absolute max (original behavior)
                scale_target = rgb_video_float.max()
                logger.debug("  Using absolute max intensity: %.6f", scale_target)
            else:
                # Use percentile (makes video brighter by allowing some saturation)
                scale_target = np.percentile(rgb_video_float, intensity_percentile)
                actual_max = rgb_video_float.max()
                logger.debug("  Using %gth percentile: %.6f (max: %.6f)", intensity_percentile, scale_target, actual_max)

            if scale_target > 0:
                # Scale so target intensity → (max_intensity - background_value)
                scale_factor = (max_intensity - background_value) / scale_target
                rgb_video_float = rgb_video_float * scale_factor
                logger.debug("  Applied scale factor: %.2f", scale_factor)
        else:
            # No scaling: use raw Gaussian intensities
            # Multiply by a reasonable factor to make visible
            # (Each Gaussian has peak ~1.0, need to scale to 0-255 range)
            scale_factor = max_intensity - background_value
            rgb_video_float = rgb_video_float * scale_factor
            logger.debug("  No auto-scaling: using fixed scale factor %.2f", scale_factor)

        # Convert to uint8 with background
        rgb_video = np.clip(rgb_video_float + background_value, 0, max_intensity).astype(np.uint8)

        logger.info("Ground truth video generation complete!")

        # Save if requested
        if save_video:
            # Save as proper RGB TIFF using tifffile for ImageJ compatibility
            try:
                import tifffile
                # tifffile expects shape (T, Y, X, C) for RGB time series
                # Our rgb_video is already in this format
                tifffile.imwrite(
                    output_path,
                    rgb_video,
                    photometric='rgb',
                    metadata={'axes': 'TYXC'}
                )
                logger.info("Saved ground truth RGB video to: %s", output_path)
                logger.info("Format: RGB TIFF, %d frames × %d×%d pixels — open in ImageJ/Fiji as RGB composite",
                            n_frames, height_pixels, width_pixels)
            except ImportError:
                # Fallback: save as hyperstack with separate channels
                logger.warning("tifffile not available, using fallback format")
                _dir = str(Path(__file__).parent.parent)
                if _dir not in sys.path:
                    sys.path.insert(0, _dir)

                import pyS3M.IOFunctions as IOFunctions
                io_funcs = IOFunctions.IO_Functions()

                # Save as ImageJ hyperstack: TZCYXS order
                # For RGB composite in ImageJ: (n_frames, 3, height, width)
                tiff_stack = np.zeros((n_frames, 3, height_pixels, width_pixels), dtype=np.uint8)
                for i in range(n_frames):
                    tiff_stack[i, 0] = rgb_video[i, :, :, 0]  # R
                    tiff_stack[i, 1] = rgb_video[i, :, :, 1]  # G
                    tiff_stack[i, 2] = rgb_video[i, :, :, 2]  # B

                # Reshape to (n_frames * 3, height, width) for write_tiff
                tiff_stack = tiff_stack.reshape(n_frames * 3, height_pixels, width_pixels)
                io_funcs.write_tiff(tiff_stack, output_path)
                logger.info("Saved ground truth RGB video to: %s", output_path)
                logger.info("Format: %d frames × 3 channels = %d slices — in ImageJ: Image → Color → Make Composite",
                            n_frames, n_frames * 3)

        return rgb_video
