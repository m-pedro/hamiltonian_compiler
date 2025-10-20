"""
HamiltonianCompiler - Efficient Compilation of Physical Hamiltonians
for Superconducting Quantum Computers

Main package initialization file.
"""

__version__ = "1.0.0"
__author__ = "HamiltonianCompiler Contributors"
__license__ = "CC BY-NC-SA 4.0"

# Import core classes for easy access
from .hamiltoniancompiler import (
    # Core data structures
    PauliString,
    HamiltonianTerm,
    Hamiltonian,
    Gate,
    QuantumCircuit,
    # Decomposition methods
    DecompositionMethod,
    FirstOrderTrotter,
    SecondOrderTrotter,
    qDRIFT,
    qSWIFT,
    # Optimization
    CircuitOptimizer,
    # Main compiler
    CompilerConfig,
    HamiltonianCompiler,
    # Utility functions
    synthesize_pauli_exponential,
)

__all__ = [
    "PauliString",
    "HamiltonianTerm",
    "Hamiltonian",
    "Gate",
    "QuantumCircuit",
    "DecompositionMethod",
    "FirstOrderTrotter",
    "SecondOrderTrotter",
    "qDRIFT",
    "qSWIFT",
    "CircuitOptimizer",
    "CompilerConfig",
    "HamiltonianCompiler",
    "synthesize_pauli_exponential",
    "__version__",
]
try:
    from .visualization import (
        plot_circuit_stats,
        plot_hamiltonian_structure,
        plot_compilation_comparison,
        plot_commutation_graph,
    )

    __all__.extend(
        [
            "plot_circuit_stats",
            "plot_hamiltonian_structure",
            "plot_compilation_comparison",
            "plot_commutation_graph",
        ]
    )
except ImportError:
    # Visualization dependencies not installed
    pass
