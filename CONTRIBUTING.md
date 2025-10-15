# Contributing to HamiltonianCompiler

We welcome contributions! Here's how to get started:

## Development Setup
```bash
git clone https://github.com/m-pedro/hamiltonian_compiler.git
cd hamiltonian_compiler
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Running Tests
```bash
pytest tests/ -v
```

## Running Examples
```bash
# Run basic examples
python hamiltoniancompiler/hamiltoniancompiler.py

# Run comprehensive examples
python examples/examples.py
```

## Code Style

We use Black for formatting:
```bash
black hamiltoniancompiler/
flake8 hamiltoniancompiler/
```

## Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Code of Conduct

Please be respectful and constructive in all interactions.

## Questions?

Open an issue or contact the maintainers.
```

---
