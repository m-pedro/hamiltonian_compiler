"""
HamiltonianCompiler: Efficient Compilation of Physical Hamiltonians
for Superconducting Quantum Computers

This is the core implementation of the Hamiltonian compiler described in the white paper.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from collections import defaultdict
import itertools
from abc import ABC, abstractmethod

# ============================================================================
# Core Data Structures
# ============================================================================

class PauliString:
    """Represents a Pauli string like 'XYZII'."""
    
    def __init__(self, string: str):
        """Initialize from string representation."""
        if not all(c in 'IXYZ' for c in string):
            raise ValueError("Pauli string must only contain I, X, Y, Z")
        self.string = string
        self.n_qubits = len(string)
    
    def __str__(self):
        return self.string
    
    def __repr__(self):
        return f"PauliString('{self.string}')"
    
    def __eq__(self, other):
        return self.string == other.string
    
    def __hash__(self):
        return hash(self.string)
    
    def commutes_with(self, other: 'PauliString') -> bool:
        """Check if this Pauli string commutes with another."""
        if self.n_qubits != other.n_qubits:
            raise ValueError("Pauli strings must have same length")
        
        # Count number of positions where both are non-identity and different
        anti_commute_count = 0
        for p1, p2 in zip(self.string, other.string):
            if p1 != 'I' and p2 != 'I' and p1 != p2:
                anti_commute_count += 1
        
        # Commute if even number of anti-commuting positions
        return anti_commute_count % 2 == 0
    
    def support(self) -> List[int]:
        """Return list of qubit indices where operator is non-identity."""
        return [i for i, p in enumerate(self.string) if p != 'I']
    
    def weight(self) -> int:
        """Return number of non-identity Paulis."""
        return sum(1 for p in self.string if p != 'I')


@dataclass
class HamiltonianTerm:
    """A single term in a Hamiltonian: coefficient * Pauli_string."""
    coefficient: float
    pauli: PauliString
    
    def __repr__(self):
        return f"{self.coefficient:.4f} * {self.pauli}"


class Hamiltonian:
    """Representation of a quantum Hamiltonian as sum of Pauli strings."""
    
    def __init__(self, n_qubits: int):
        """Initialize Hamiltonian for n_qubits."""
        self.n_qubits = n_qubits
        self.terms: List[HamiltonianTerm] = []
    
    def add_term(self, coefficient: float, pauli_string: str):
        """Add a term to the Hamiltonian.
        
        Args:
            coefficient: Real coefficient
            pauli_string: String of Pauli operators, e.g., "XYZ"
        """
        if len(pauli_string) != self.n_qubits:
            raise ValueError(f"Pauli string length {len(pauli_string)} != n_qubits {self.n_qubits}")
        
        pauli = PauliString(pauli_string)
        term = HamiltonianTerm(coefficient, pauli)
        self.terms.append(term)
    
    @property
    def n_terms(self) -> int:
        """Number of terms in Hamiltonian."""
        return len(self.terms)
    
    @property
    def norm(self) -> float:
        """L1 norm: sum of absolute coefficients."""
        return sum(abs(term.coefficient) for term in self.terms)
    
    def __repr__(self):
        terms_str = "\n  ".join(str(term) for term in self.terms[:5])
        if len(self.terms) > 5:
            terms_str += f"\n  ... ({len(self.terms) - 5} more terms)"
        return f"Hamiltonian({self.n_qubits} qubits, {self.n_terms} terms):\n  {terms_str}"
# ============================================================================
# Quantum Circuit Representation
# ============================================================================

class Gate:
    """Base class for quantum gates."""
    
    def __init__(self, name: str, qubits: List[int], params: List[float] = None):
        self.name = name
        self.qubits = qubits
        self.params = params or []
    
    def __repr__(self):
        params_str = f"({', '.join(f'{p:.4f}' for p in self.params)})" if self.params else ""
        qubits_str = ', '.join(map(str, self.qubits))
        return f"{self.name}{params_str}[{qubits_str}]"


class QuantumCircuit:
    """Simple quantum circuit representation."""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates: List[Gate] = []
    
    def append(self, gate: Gate):
        """Add a gate to the circuit."""
        if any(q >= self.n_qubits for q in gate.qubits):
            raise ValueError(f"Gate acts on qubit >= n_qubits={self.n_qubits}")
        self.gates.append(gate)
    
    def rz(self, theta: float, qubit: int):
        """Add Rz rotation."""
        self.append(Gate('rz', [qubit], [theta]))
    
    def rx(self, theta: float, qubit: int):
        """Add Rx rotation."""
        self.append(Gate('rx', [qubit], [theta]))
    
    def ry(self, theta: float, qubit: int):
        """Add Ry rotation."""
        self.append(Gate('ry', [qubit], [theta]))
    
    def h(self, qubit: int):
        """Add Hadamard gate."""
        self.append(Gate('h', [qubit]))
    
    def x(self, qubit: int):
        """Add X gate."""
        self.append(Gate('x', [qubit]))
    
    def sx(self, qubit: int):
        """Add sqrt(X) gate."""
        self.append(Gate('sx', [qubit]))
    
    def s(self, qubit: int):
        """Add S gate."""
        self.append(Gate('s', [qubit]))
    
    def sdg(self, qubit: int):
        """Add S-dagger gate."""
        self.append(Gate('sdg', [qubit]))
    
    def cx(self, control: int, target: int):
        """Add CNOT gate."""
        self.append(Gate('cx', [control, target]))
    
    def depth(self) -> int:
        """Calculate circuit depth (simple estimate)."""
        if not self.gates:
            return 0
        
        # Track when each qubit is last used
        qubit_time = [0] * self.n_qubits
        
        for gate in self.gates:
            # Gate starts after all involved qubits are free
            start_time = max(qubit_time[q] for q in gate.qubits)
            end_time = start_time + 1
            
            # Update qubit times
            for q in gate.qubits:
                qubit_time[q] = end_time
        
        return max(qubit_time)
    
    def count_ops(self) -> Dict[str, int]:
        """Count operations by gate type."""
        counts = defaultdict(int)
        for gate in self.gates:
            counts[gate.name] += 1
        return dict(counts)
    
    def __repr__(self):
        return f"QuantumCircuit({self.n_qubits} qubits, {len(self.gates)} gates, depth={self.depth()})"
# ============================================================================
# Pauli Exponential Synthesis
# ============================================================================

def synthesize_pauli_exponential(pauli: PauliString, theta: float) -> QuantumCircuit:
    """
    Synthesize circuit for e^{-i*theta*P} where P is a Pauli string.
    
    Algorithm:
    1. Change basis to Z-basis for all non-identity Paulis
    2. Build CNOT ladder to concentrate parity
    3. Apply Rz(2*theta) rotation
    4. Uncompute CNOT ladder
    5. Reverse basis changes
    """
    circuit = QuantumCircuit(pauli.n_qubits)
    
    # Find non-identity positions
    support = pauli.support()
    if not support:
        # Identity string - no gates needed
        return circuit
    
    # Step 1: Change basis to Z
    basis_changes = []
    for qubit in support:
        pauli_type = pauli.string[qubit]
        if pauli_type == 'X':
            circuit.h(qubit)
            basis_changes.append(('h', qubit))
        elif pauli_type == 'Y':
            circuit.sdg(qubit)
            circuit.h(qubit)
            basis_changes.append(('y_basis', qubit))
    
    # Step 2 & 3: CNOT ladder and rotation
    if len(support) == 1:
        # Single Pauli - just rotate
        circuit.rz(2 * theta, support[0])
    else:
        # Multi-qubit Pauli - build CNOT ladder
        target = support[-1]
        
        # Forward CNOT ladder
        for i in range(len(support) - 1):
            circuit.cx(support[i], target)
        
        # Rotation on target
        circuit.rz(2 * theta, target)
        
        # Backward CNOT ladder (uncompute)
        for i in range(len(support) - 2, -1, -1):
            circuit.cx(support[i], target)
    
    # Step 4: Reverse basis changes
    for change_type, qubit in reversed(basis_changes):
        if change_type == 'h':
            circuit.h(qubit)
        elif change_type == 'y_basis':
            circuit.h(qubit)
            circuit.s(qubit)
    
    return circuit
# ============================================================================
# Decomposition Methods
# ============================================================================

class DecompositionMethod(ABC):
    """Abstract base class for Hamiltonian decomposition methods."""
    
    @abstractmethod
    def decompose(self, hamiltonian: Hamiltonian, time: float, 
                  error_budget: float) -> QuantumCircuit:
        """Decompose Hamiltonian evolution into circuit."""
        pass


class FirstOrderTrotter(DecompositionMethod):
    """First-order Trotter-Suzuki decomposition."""
    
    def decompose(self, hamiltonian: Hamiltonian, time: float, 
                  error_budget: float) -> QuantumCircuit:
        """Apply first-order Trotter formula."""
        
        # Estimate required Trotter steps
        # Error bound: ε ≤ t²λ²/(2r) where λ = ||H||_1
        lambda_h = hamiltonian.norm
        r = int(np.ceil((time**2 * lambda_h**2) / (2 * error_budget)))
        r = max(r, 1)  # At least 1 step
        
        dt = time / r
        
        circuit = QuantumCircuit(hamiltonian.n_qubits)
        
        # Apply r Trotter steps
        for step in range(r):
            for term in hamiltonian.terms:
                # Add e^{-i * coefficient * Pauli * dt}
                sub_circuit = synthesize_pauli_exponential(
                    term.pauli, 
                    term.coefficient * dt
                )
                for gate in sub_circuit.gates:
                    circuit.append(gate)
        
        return circuit


class SecondOrderTrotter(DecompositionMethod):
    """Second-order Trotter-Suzuki decomposition."""
    
    def decompose(self, hamiltonian: Hamiltonian, time: float, 
                  error_budget: float) -> QuantumCircuit:
        """Apply second-order Trotter formula with forward-backward sweep."""
        
        # Error bound: ε ≤ C * t³λ³ / r² (approximate)
        lambda_h = hamiltonian.norm
        r = int(np.ceil(np.sqrt((time**3 * lambda_h**3) / error_budget)))
        r = max(r, 1)
        
        dt = time / r
        
        circuit = QuantumCircuit(hamiltonian.n_qubits)
        
        for step in range(r):
            # Forward sweep with dt/2
            for term in hamiltonian.terms:
                sub_circuit = synthesize_pauli_exponential(
                    term.pauli,
                    term.coefficient * dt / 2
                )
                for gate in sub_circuit.gates:
                    circuit.append(gate)
            
            # Backward sweep with dt/2
            for term in reversed(hamiltonian.terms):
                sub_circuit = synthesize_pauli_exponential(
                    term.pauli,
                    term.coefficient * dt / 2
                )
                for gate in sub_circuit.gates:
                    circuit.append(gate)
        
        return circuit


class qDRIFT(DecompositionMethod):
    """Randomized qDRIFT algorithm."""
    
    def decompose(self, hamiltonian: Hamiltonian, time: float, 
                  error_budget: float) -> QuantumCircuit:
        """Apply qDRIFT randomized compilation."""
        
        lambda_h = hamiltonian.norm
        
        # Number of samples: N = λ²t² / ε²
        N = int(np.ceil((lambda_h**2 * time**2) / (error_budget**2)))
        N = max(N, 10)  # Minimum samples
        
        # Compute sampling probabilities
        weights = [abs(term.coefficient) / lambda_h for term in hamiltonian.terms]
        
        circuit = QuantumCircuit(hamiltonian.n_qubits)
        tau = lambda_h * time / N
        
        # Sample N times
        np.random.seed(42)  # For reproducibility in demo
        for k in range(N):
            # Sample term according to weights
            idx = np.random.choice(len(hamiltonian.terms), p=weights)
            term = hamiltonian.terms[idx]
            
            # Apply sampled term
            sign = np.sign(term.coefficient)
            sub_circuit = synthesize_pauli_exponential(term.pauli, sign * tau)
            for gate in sub_circuit.gates:
                circuit.append(gate)
        
        return circuit


class qSWIFT(DecompositionMethod):
    """qSWIFT higher-order randomized algorithm."""
    
    def __init__(self, order: int = 2):
        """Initialize with specified order (1 or 2)."""
        self.order = order
    
    def decompose(self, hamiltonian: Hamiltonian, time: float, 
                  error_budget: float) -> QuantumCircuit:
        """Apply qSWIFT randomized compilation."""
        
        lambda_h = hamiltonian.norm
        
        if self.order == 1:
            # Same as qDRIFT
            N = int(np.ceil((lambda_h**2 * time**2) / (error_budget**2)))
        else:  # order == 2
            # Improved scaling: N ~ (λt)² / ε
            N = int(np.ceil((lambda_h * time)**2 / error_budget))
        
        N = max(N, 10)
        
        weights = [abs(term.coefficient) / lambda_h for term in hamiltonian.terms]
        
        circuit = QuantumCircuit(hamiltonian.n_qubits)
        tau = lambda_h * time / N
        
        np.random.seed(42)
        for k in range(N):
            idx = np.random.choice(len(hamiltonian.terms), p=weights)
            term = hamiltonian.terms[idx]
            
            # For order-2, we would add correction terms here
            # (simplified implementation - full version would compute commutators)
            sign = np.sign(term.coefficient)
            theta = sign * tau
            
            if self.order == 2:
                # Simplified correction (proper implementation needs commutator calculation)
                correction_factor = 1.0 + 0.1 * (tau / time)  # Placeholder
                theta *= correction_factor
            
            sub_circuit = synthesize_pauli_exponential(term.pauli, theta)
            for gate in sub_circuit.gates:
                circuit.append(gate)
        
        return circuit
# ============================================================================
# Circuit Optimization
# ============================================================================

class CircuitOptimizer:
    """Optimize quantum circuits."""
    
    @staticmethod
    def merge_rotations(circuit: QuantumCircuit) -> QuantumCircuit:
        """Merge adjacent rotation gates on same qubit and axis."""
        optimized = QuantumCircuit(circuit.n_qubits)
        
        i = 0
        while i < len(circuit.gates):
            gate = circuit.gates[i]
            
            # Check if this is a rotation gate
            if gate.name in ['rx', 'ry', 'rz'] and i + 1 < len(circuit.gates):
                next_gate = circuit.gates[i + 1]
                
                # Check if next gate is same rotation on same qubit
                if (next_gate.name == gate.name and 
                    next_gate.qubits == gate.qubits):
                    # Merge the rotations
                    merged_angle = gate.params[0] + next_gate.params[0]
                    
                    # Only add if not close to 0 (mod 2π)
                    merged_angle = merged_angle % (2 * np.pi)
                    if abs(merged_angle) > 1e-10 and abs(merged_angle - 2*np.pi) > 1e-10:
                        optimized.append(Gate(gate.name, gate.qubits, [merged_angle]))
                    
                    i += 2  # Skip both gates
                    continue
            
            # No merge - just add the gate
            optimized.append(gate)
            i += 1
        
        return optimized
    
    @staticmethod
    def remove_zero_rotations(circuit: QuantumCircuit, 
                             threshold: float = 1e-10) -> QuantumCircuit:
        """Remove rotation gates with angle ≈ 0."""
        optimized = QuantumCircuit(circuit.n_qubits)
        
        for gate in circuit.gates:
            # Check if rotation with small angle
            if gate.name in ['rx', 'ry', 'rz']:
                angle = gate.params[0] % (2 * np.pi)
                if abs(angle) > threshold and abs(angle - 2*np.pi) > threshold:
                    optimized.append(gate)
            else:
                optimized.append(gate)
        
        return optimized
    
    @staticmethod
    def cancel_inverse_pairs(circuit: QuantumCircuit) -> QuantumCircuit:
        """Cancel adjacent gates that are inverses (H-H, X-X, etc.)."""
        optimized = QuantumCircuit(circuit.n_qubits)
        
        inverse_pairs = {
            ('h', 'h'), ('x', 'x'), ('cx', 'cx'),
            ('s', 'sdg'), ('sdg', 's')
        }
        
        i = 0
        while i < len(circuit.gates):
            if i + 1 < len(circuit.gates):
                gate1 = circuit.gates[i]
                gate2 = circuit.gates[i + 1]
                
                # Check if they form an inverse pair on same qubits
                if ((gate1.name, gate2.name) in inverse_pairs and 
                    gate1.qubits == gate2.qubits):
                    i += 2  # Skip both
                    continue
            
            optimized.append(circuit.gates[i])
            i += 1
        
        return optimized
# ============================================================================
# Main Compiler
# ============================================================================

@dataclass
class CompilerConfig:
    """Configuration for Hamiltonian compiler."""
    decomposition_method: str = "auto"  # "trotter1", "trotter2", "qdrift", "qswift", "auto"
    decomposition_order: int = 1
    optimization_level: int = 2  # 0-3


class HamiltonianCompiler:
    """Main compiler class for Hamiltonian simulation."""
    
    def __init__(self, backend: str = "simulator", 
                 optimization_level: int = 2):
        """Initialize compiler.
        
        Args:
            backend: Target backend (e.g., "ibm_washington", "simulator")
            optimization_level: 0-3, higher = more optimization
        """
        self.backend = backend
        self.optimization_level = optimization_level
        self.optimizer = CircuitOptimizer()
    
    def _select_method(self, hamiltonian: Hamiltonian, 
                      error_budget: float, method: str) -> DecompositionMethod:
        """Select appropriate decomposition method."""
        
        if method == "trotter1":
            return FirstOrderTrotter()
        elif method == "trotter2":
            return SecondOrderTrotter()
        elif method == "qdrift":
            return qDRIFT()
        elif method == "qswift":
            return qSWIFT(order=2)
        elif method == "auto":
            # Automatic selection based on Hamiltonian properties
            if hamiltonian.n_terms <= 5 and error_budget < 1e-3:
                return SecondOrderTrotter()
            elif hamiltonian.n_terms > 20:
                return qSWIFT(order=2)
            else:
                return FirstOrderTrotter()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compile(self, hamiltonian: Hamiltonian, time: float,
                error_budget: float = 1e-3, method: str = "auto",
                config: Optional[CompilerConfig] = None) -> QuantumCircuit:
        """
        Compile Hamiltonian evolution to optimized circuit.
        
        Args:
            hamiltonian: Input Hamiltonian
            time: Evolution time
            error_budget: Maximum acceptable error
            method: Decomposition method to use
            config: Advanced configuration
            
        Returns:
            Optimized quantum circuit implementing e^{-iHt}
        """
        
        # Phase 1: Select decomposition method
        decomposer = self._select_method(hamiltonian, error_budget, method)
        
        # Phase 2: Apply decomposition
        circuit = decomposer.decompose(hamiltonian, time, error_budget)
        
        # Phase 3: Optimize circuit
        if self.optimization_level >= 1:
            circuit = self.optimizer.remove_zero_rotations(circuit)
        
        if self.optimization_level >= 2:
            circuit = self.optimizer.merge_rotations(circuit)
            circuit = self.optimizer.cancel_inverse_pairs(circuit)
        
        if self.optimization_level >= 3:
            # Apply multiple optimization passes
            for _ in range(3):
                circuit = self.optimizer.merge_rotations(circuit)
                circuit = self.optimizer.cancel_inverse_pairs(circuit)
                circuit = self.optimizer.remove_zero_rotations(circuit)
        
        return circuit
    
    def estimate_fidelity(self, circuit: QuantumCircuit,
                         gate_error_1q: float = 0.0004,
                         gate_error_2q: float = 0.008) -> float:
        """Estimate circuit execution fidelity based on gate counts."""
        counts = circuit.count_ops()
        
        n_1q = sum(counts.get(g, 0) for g in ['rx', 'ry', 'rz', 'h', 'x', 'sx', 's', 'sdg'])
        n_2q = counts.get('cx', 0)
        
        # Simple error model: F ≈ (1-ε₁)^n₁ * (1-ε₂)^n₂
        fidelity = (1 - gate_error_1q)**n_1q * (1 - gate_error_2q)**n_2q
        
        return fidelity


# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_molecular_hydrogen():
    """Example: Compile molecular hydrogen Hamiltonian."""
    print("="*70)
    print("Example: Molecular Hydrogen (H₂) Hamiltonian")
    print("="*70)
    
    # H₂ Hamiltonian in STO-3G basis
    H = Hamiltonian(n_qubits=2)
    H.add_term(-1.0523, "II")
    H.add_term(0.3979, "IZ")
    H.add_term(-0.3979, "ZI")
    H.add_term(-0.0112, "ZZ")
    H.add_term(0.1809, "XX")
    
    print(f"\n{H}\n")
    
    # Create compiler
    compiler = HamiltonianCompiler(backend="ibm_washington", optimization_level=3)
    
    # Compile with different methods
    methods = ["trotter1", "trotter2", "qdrift", "qswift"]
    
    for method in methods:
        print(f"\n{method.upper()}:")
        print("-" * 50)
        
        circuit = compiler.compile(H, time=1.0, error_budget=1e-3, method=method)
        
        counts = circuit.count_ops()
        fidelity = compiler.estimate_fidelity(circuit)
        
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  Total gates: {len(circuit.gates)}")
        print(f"  CNOT count: {counts.get('cx', 0)}")
        print(f"  Rotation gates: {sum(counts.get(g, 0) for g in ['rx', 'ry', 'rz'])}")
        print(f"  Estimated fidelity: {fidelity:.4f}")


def example_heisenberg_chain():
    """Example: 1D Heisenberg chain."""
    print("\n" + "="*70)
    print("Example: Heisenberg Chain (4 qubits)")
    print("="*70)
    
    # H = Σᵢ (XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ZᵢZᵢ₊₁)
    n_sites = 4
    H = Hamiltonian(n_qubits=n_sites)
    
    for i in range(n_sites - 1):
        # XX interaction
        pauli = "I" * i + "XX" + "I" * (n_sites - i - 2)
        H.add_term(1.0, pauli)
        
        # YY interaction
        pauli = "I" * i + "YY" + "I" * (n_sites - i - 2)
        H.add_term(1.0, pauli)
        
        # ZZ interaction
        pauli = "I" * i + "ZZ" + "I" * (n_sites - i - 2)
        H.add_term(1.0, pauli)
    
    print(f"\n{H}\n")
    
    compiler = HamiltonianCompiler(optimization_level=3)
    
    circuit = compiler.compile(H, time=2.0, error_budget=1e-2, method="auto")
    
    print(f"\nCompiled Circuit (auto-selected method):")
    print(f"  Circuit depth: {circuit.depth()}")
    print(f"  Total gates: {len(circuit.gates)}")
    print(f"  Gate counts: {circuit.count_ops()}")
    print(f"  Estimated fidelity: {compiler.estimate_fidelity(circuit):.4f}")


def example_ising_model():
    """Example: Transverse field Ising model."""
    print("\n" + "="*70)
    print("Example: Transverse Field Ising Model (6 qubits)")
    print("="*70)
    
    # H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ
    n_sites = 6
    J = 1.0
    h = 0.5
    
    H = Hamiltonian(n_qubits=n_sites)
    
    # ZZ interactions
    for i in range(n_sites - 1):
        pauli = "I" * i + "ZZ" + "I" * (n_sites - i - 2)
        H.add_term(-J, pauli)
    
    # Transverse field
    for i in range(n_sites):
        pauli = "I" * i + "X" + "I" * (n_sites - i - 1)
        H.add_term(-h, pauli)
    
    print(f"\nJ = {J}, h = {h}")
    print(f"Hamiltonian norm: {H.norm:.4f}")
    print(f"Number of terms: {H.n_terms}\n")
    
    compiler = HamiltonianCompiler(optimization_level=3)
    
    # Compare Trotter orders
    for method in ["trotter1", "trotter2"]:
        circuit = compiler.compile(H, time=1.0, error_budget=1e-3, method=method)
        print(f"{method}: depth={circuit.depth()}, gates={len(circuit.gates)}, "
              f"CNOTs={circuit.count_ops().get('cx', 0)}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HAMILTONIAN COMPILER - DEMONSTRATION")
    print("="*70)
    
    # Run examples
    example_molecular_hydrogen()
    example_heisenberg_chain()
    example_ising_model()
    
    print("\n" + "="*70)
    print("Compilation complete!")
    print("="*70)

