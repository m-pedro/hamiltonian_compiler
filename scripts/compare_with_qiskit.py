"""
Compare HamiltonianCompiler performance with Qiskit

Requires: qiskit
Install with: pip install qiskit

Run with: python scripts/compare_with_qiskit.py
"""

try:
    from qiskit import QuantumCircuit as QiskitCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import SuzukiTrotter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not installed. Install with: pip install qiskit")

from hamiltoniancompiler import Hamiltonian, HamiltonianCompiler
import time
import numpy as np


def compare_h2_molecule():
    """Compare H2 molecule compilation with Qiskit."""
    if not QISKIT_AVAILABLE:
        return
    
    print("\n" + "="*80)
    print("Comparison: H2 Molecule (HamiltonianCompiler vs Qiskit)")
    print("="*80)
    
    # Our implementation
    H_ours = Hamiltonian(n_qubits=2)
    H_ours.add_term(-1.0523, "II")
    H_ours.add_term(0.3979, "IZ")
    H_ours.add_term(-0.3979, "ZI")
    H_ours.add_term(-0.0112, "ZZ")
    H_ours.add_term(0.1809, "XX")
    
    compiler = HamiltonianCompiler(optimization_level=3)
    
    # Compile with our method
    start = time.time()
    circuit_ours = compiler.compile(H_ours, time=1.0, error_budget=1e-3, method="trotter2")
    time_ours = time.time() - start
    
    # Qiskit implementation
    pauli_list = [
        ("II", -1.0523),
        ("IZ", 0.3979),
        ("ZI", -0.3979),
        ("ZZ", -0.0112),
        ("XX", 0.1809),
    ]
    
    H_qiskit = SparsePauliOp.from_list(pauli_list)
    
    start = time.time()
    # Use Suzuki-Trotter formula
    evolution_gate = SuzukiTrotter(order=2, reps=10)
    circuit_qiskit = evolution_gate.synthesize(H_qiskit)
    time_qiskit = time.time() - start
    
    # Compare results
    print("\nHamiltonianCompiler:")
    print(f"  Compilation time: {time_ours:.4f}s")
    print(f"  Circuit depth: {circuit_ours.depth()}")
    print(f"  CNOT count: {circuit_ours.count_ops().get('cx', 0)}")
    print(f"  Total gates: {len(circuit_ours.gates)}")
    print(f"  Estimated fidelity: {compiler.estimate_fidelity(circuit_ours):.4f}")
    
    print("\nQiskit (Suzuki-Trotter):")
    print(f"  Compilation time: {time_qiskit:.4f}s")
    print(f"  Circuit depth: {circuit_qiskit.depth()}")
    print(f"  CNOT count: {circuit_qiskit.count_ops().get('cx', 0)}")
    print(f"  Total gates: {sum(circuit_qiskit.count_ops().values())}")
    
    # Calculate improvements
    depth_improvement = (1 - circuit_ours.depth() / circuit_qiskit.depth()) * 100
    cnot_improvement = (1 - circuit_ours.count_ops().get('cx', 0) / 
                       circuit_qiskit.count_ops().get('cx', 0)) * 100
    
    print("\nImprovement:")
    print(f"  Depth reduction: {depth_improvement:.1f}%")
    print(f"  CNOT reduction: {cnot_improvement:.1f}%")


def compare_ising_model():
    """Compare Ising model compilation."""
    if not QISKIT_AVAILABLE:
        return
    
    print("\n" + "="*80)
    print("Comparison: Ising Model (6 qubits)")
    print("="*80)
    
    n_qubits = 6
    
    # Our implementation
    H_ours = Hamiltonian(n_qubits=n_qubits)
    
    pauli_list = []
    for i in range(n_qubits - 1):
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        pauli[i+1] = 'Z'
        H_ours.add_term(1.0, ''.join(pauli))
        pauli_list.append((''.join(pauli), 1.0))
    
    compiler = HamiltonianCompiler(optimization_level=3)
    
    start = time.time()
    circuit_ours = compiler.compile(H_ours, time=1.0, error_budget=1e-2, method="trotter2")
    time_ours = time.time() - start
    
    # Qiskit
    H_qiskit = SparsePauliOp.from_list(pauli_list)
    
    start = time.time()
    evolution_gate = SuzukiTrotter(order=2, reps=5)
    circuit_qiskit = evolution_gate.synthesize(H_qiskit)
    time_qiskit = time.time() - start
    
    # Compare
    print("\nHamiltonianCompiler:")
    print(f"  Time: {time_ours:.4f}s, Depth: {circuit_ours.depth()}, "
          f"CNOTs: {circuit_ours.count_ops().get('cx', 0)}")
    
    print("\nQiskit:")
    print(f"  Time: {time_qiskit:.4f}s, Depth: {circuit_qiskit.depth()}, "
          f"CNOTs: {circuit_qiskit.count_ops().get('cx', 0)}")


if __name__ == "__main__":
    if QISKIT_AVAILABLE:
        compare_h2_molecule()
        compare_ising_model()
    else:
        print("Please install Qiskit to run comparisons:")
        print("  pip install qiskit")
