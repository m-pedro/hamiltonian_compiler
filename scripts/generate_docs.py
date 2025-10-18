"""
Generate API documentation from docstrings

Run with: python scripts/generate_docs.py
"""

import inspect
import hamiltoniancompiler as hc
from pathlib import Path


def generate_api_docs():
    """Generate markdown documentation from docstrings."""
    
    output = []
    output.append("# API Reference\n")
    output.append("Auto-generated API documentation for HamiltonianCompiler\n")
    output.append(f"Version: {hc.__version__}\n\n")
    
    # Core classes
    classes = [
        ('PauliString', hc.PauliString),
        ('Hamiltonian', hc.Hamiltonian),
        ('QuantumCircuit', hc.QuantumCircuit),
        ('HamiltonianCompiler', hc.HamiltonianCompiler),
        ('FirstOrderTrotter', hc.FirstOrderTrotter),
        ('SecondOrderTrotter', hc.SecondOrderTrotter),
        ('qDRIFT', hc.qDRIFT),
        ('qSWIFT', hc.qSWIFT),
        ('CircuitOptimizer', hc.CircuitOptimizer),
    ]
    
    for name, cls in classes:
        output.append(f"## {name}\n\n")
        output.append(f"```python\n{cls.__module__}.{name}\n```\n\n")
        
        # Class docstring
        if cls.__doc__:
            output.append(f"{cls.__doc__}\n\n")
        
        # Methods
        output.append("### Methods\n\n")
        for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not method_name.startswith('_'):
                output.append(f"#### `{method_name}`\n\n")
                if method.__doc__:
                    output.append(f"{method.__doc__}\n\n")
        
        output.append("---\n\n")
    
    # Write to file
    docs_path = Path('docs') / 'api_reference.md'
    docs_path.parent.mkdir(exist_ok=True)
    
    with open(docs_path, 'w') as f:
        f.write(''.join(output))
    
    print(f"API documentation generated at {docs_path}")


if __name__ == "__main__":
    generate_api_docs()
