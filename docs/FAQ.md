# Frequently Asked Questions (FAQ)

## General Questions

### What is HamiltonianCompiler?

HamiltonianCompiler is a quantum compilation framework that translates physical system Hamiltonians into optimized quantum circuits for superconducting qubit architectures.

### How is this different from existing quantum compilers?

Traditional quantum compilers (like those in Qiskit or Cirq) start with pre-existing quantum circuits. HamiltonianCompiler starts at the Hamiltonian level, enabling direct compilation of physical problems with specialized optimizations for Hamiltonian simulation.

### What quantum hardware platforms are supported?

Currently optimized for superconducting qubits (IBM, Rigetti, Google). Support for ion traps and neutral atoms is planned for future releases.

## Installation and Setup

### How do I install HamiltonianCompiler?
```bash
pip install git+https://github.com/m-pedro/hamiltonian_compiler.git
```

### What are the minimum requirements?

- Python 3.8 or higher
- NumPy >= 1.20.0
- SciPy >= 1.7.0

### Do I need Qiskit?

No, Qiskit is optional. Install it only if you want to execute circuits on IBM hardware:
```bash
pip install "hamiltoniancompiler[hardware]"
```

## Usage Questions

### How do I create a Hamiltonian?
```python
from hamiltoniancompiler import Hamiltonian

H = Hamiltonian(n_qubits=4)
H.add_term(1.0, "ZZII")  # Coefficient and Pauli string
H.add_term(0.5, "XXII")
```

### Which decomposition method should I use?

- **trotter1**: Fast, good for prototyping
- **trotter2**: Better accuracy, recommended for most cases
- **qswift**: Best for many terms (>20)
- **auto**: Let the compiler decide (recommended)

### How do I interpret the error budget?

The error budget controls the approximation error in the Trotter decomposition. Smaller values (e.g., 1e-4) produce more accurate but longer circuits. Typical values:
- 1e-2: Fast prototyping
- 1e-3: Production use
- 1e-4: High accuracy needed

### Why is my circuit so long?

Several factors affect circuit length:
1. **Error budget**: Tighter budgets require more Trotter steps
2. **Hamiltonian complexity**: More terms or larger coefficients
3. **Evolution time**: Longer times need more steps
4. **Optimization level**: Set `optimization_level=3` for best results

## Performance Questions

### How much faster is this than naive Trotterization?

Typically 20-40% reduction in circuit depth and 30-50% reduction in CNOT gates compared to unoptimized compilation.

### Can I profile compilation performance?

Yes! Run:
```bash
python scripts/profile_compiler.py
```

### How do I benchmark against Qiskit?
```bash
python scripts/compare_with_qiskit.py
```

## Troubleshooting

### ImportError: No module named 'hamiltoniancompiler'

Make sure you installed the package:
```bash
pip install -e .
```

### My circuit has too many gates for hardware

Try:
1. Increase error_budget (e.g., from 1e-3 to 1e-2)
2. Use optimization_level=3
3. Try method="qswift" for better scaling
4. Consider breaking long time evolution into smaller steps

### Tests are failing

Ensure all dependencies are installed:
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### How do I report a bug?

Open an issue on GitHub: https://github.com/m-pedro/hamiltonian_compiler/issues

Include:
- Python version
- Error message
- Minimal code to reproduce

## Advanced Questions

### Can I compile time-dependent Hamiltonians?

Currently not directly supported. Workaround: discretize time and compile each step separately. Full support planned for v1.1.

### How do I add custom decomposition methods?

Subclass `DecompositionMethod` and implement the `decompose()` method. See existing methods in `hamiltoniancompiler.py` for examples.

### Can I integrate this with my own quantum framework?

Yes! The core compilation logic is framework-agnostic. You'll need to write adapters to convert to/from your circuit representation.

### Is there GPU acceleration?

Not yet. This is planned for v2.0 for large-scale routing and scheduling.

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. We welcome:
- Bug reports and fixes
- New features and algorithms
- Documentation improvements
- Benchmark contributions

### Where should I start?

Check the "good first issue" label on GitHub issues.

## Citation

### How do I cite this work?
```bibtex
@software{hamiltoniancompiler2025,
  title = {HamiltonianCompiler: Efficient Compilation for Quantum Simulation},
  author = {HamiltonianCompiler Contributors},
  year = {2025},
  url = {https://github.com/m-pedro/hamiltonian_compiler}
}
```

---

**Still have questions?** Open a discussion on GitHub: https://github.com/m-pedro/hamiltonian_compiler/discussions
