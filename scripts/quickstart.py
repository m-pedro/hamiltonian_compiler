"""
Quick start demonstration of HamiltonianCompiler

Run with: python scripts/quickstart.py
"""

from hamiltoniancompiler import Hamiltonian, HamiltonianCompiler


def main():
    print("="*80)
    print("HamiltonianCompiler - Quick Start")
    print("="*80)
    
    print("\n1. Creating a Hamiltonian...")
    H = Hamiltonian(n_qubits=2)
    H.add_term(1.0, "ZZ")
    H.add_term(0.5, "XX")
    print(f"   Created: {H.n_terms} terms, norm={H.norm:.2f}")
    
    print("\n2. Creating a compiler...")
    compiler = HamiltonianCompiler(optimization_level=3)
    print("   Compiler ready with optimization level 3")
    
    print("\n3. Compiling Hamiltonian evolution...")
    circuit = compiler.compile(H, time=1.0, error_budget=1e-3, method="auto")
    print(f"   Compiled circuit: depth={circuit.depth()}, gates={len(circuit.gates)}")
    
    print("\n4. Analyzing results...")
    counts = circuit.count_ops()
    fidelity = compiler.estimate_fidelity(circuit)
    print(f"   CNOT gates: {counts.get('cx', 0)}")
    print(f"   Estimated fidelity: {fidelity:.4f}")
    
    print("\n5. Trying different methods...")
    for method in ["trotter1", "trotter2", "qswift"]:
        circ = compiler.compile(H, time=1.0, error_budget=1e-3, method=method)
        print(f"   {method:10s}: depth={circ.depth():3d}, "
              f"CNOTs={circ.count_ops().get('cx', 0):2d}")
    
    print("\n" + "="*80)
    print("Quick start complete! Try modifying the Hamiltonian or parameters.")
    print("See examples/examples.py for more detailed demonstrations.")
    print("="*80)


if __name__ == "__main__":
    main()
```

---

## Summary of New Files

Here's the complete list of new files to add:
```
hamiltonian_compiler/
├── .github/
│   ├── workflows/
│   │   ├── tests.yml
│   │   └── release.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
├── benchmarks/
│   ├── __init__.py
│   └── benchmark_suite.py
├── scripts/
│   ├── profile_compiler.py
│   ├── compare_with_qiskit.py
│   ├── generate_docs.py
│   └── quickstart.py
├── notebooks/
│   └── tutorial.ipynb
├── hamiltoniancompiler/
│   └── visualization.py
├── docs/
│   └── FAQ.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── CODE_OF_CONDUCT.md
