"""
Profile HamiltonianCompiler performance

Run with: python scripts/profile_compiler.py
"""

import cProfile
import pstats
import io
from hamiltoniancompiler import Hamiltonian, HamiltonianCompiler


def profile_compilation():
    """Profile a typical compilation workflow."""
    # Create test Hamiltonian
    H = Hamiltonian(n_qubits=6)
    
    for i in range(5):
        pauli = ['I'] * 6
        pauli[i] = 'Z'
        pauli[i+1] = 'Z'
        H.add_term(1.0, ''.join(pauli))
    
    # Compile
    compiler = HamiltonianCompiler(optimization_level=3)
    circuit = compiler.compile(H, time=1.0, error_budget=1e-2, method="trotter2")
    
    return circuit


if __name__ == "__main__":
    # Profile the code
    profiler = cProfile.Profile()
    profiler.enable()
    
    circuit = profile_compilation()
    
    profiler.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print(s.getvalue())
    print(f"\nCompiled circuit: depth={circuit.depth()}, gates={len(circuit.gates)}")
