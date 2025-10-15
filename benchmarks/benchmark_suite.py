"""
Comprehensive benchmark suite for HamiltonianCompiler

Run with: python benchmarks/benchmark_suite.py
"""

import time
import numpy as np
from hamiltoniancompiler import Hamiltonian, HamiltonianCompiler
import json
from datetime import datetime


class BenchmarkResults:
    """Store and format benchmark results."""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, name, **kwargs):
        """Add a benchmark result."""
        self.results.append({
            'name': name,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
    
    def save_json(self, filename='benchmark_results.json'):
        """Save results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        for result in self.results:
            print(f"\n{result['name']}:")
            for key, value in result.items():
                if key not in ['name', 'timestamp']:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")


def benchmark_molecular_systems():
    """Benchmark molecular Hamiltonian compilation."""
    print("\n" + "="*80)
    print("BENCHMARK 1: Molecular Systems")
    print("="*80)
    
    results = BenchmarkResults()
    
    # H2 molecule
    H_h2 = Hamiltonian(n_qubits=2)
    H_h2.add_term(-1.0523, "II")
    H_h2.add_term(0.3979, "IZ")
    H_h2.add_term(-0.3979, "ZI")
    H_h2.add_term(-0.0112, "ZZ")
    H_h2.add_term(0.1809, "XX")
    
    compiler = HamiltonianCompiler(optimization_level=3)
    
    methods = ["trotter1", "trotter2", "qdrift", "qswift"]
    
    for method in methods:
        start_time = time.time()
        circuit = compiler.compile(H_h2, time=1.0, error_budget=1e-3, method=method)
        compile_time = time.time() - start_time
        
        fidelity = compiler.estimate_fidelity(circuit)
        counts = circuit.count_ops()
        
        results.add_result(
            name=f"H2_{method}",
            method=method,
            compile_time=compile_time,
            depth=circuit.depth(),
            cnot_count=counts.get('cx', 0),
            total_gates=len(circuit.gates),
            fidelity=fidelity
        )
        
        print(f"{method:12s}: depth={circuit.depth():4d}, "
              f"CNOTs={counts.get('cx', 0):3d}, "
              f"time={compile_time:.4f}s, "
              f"fidelity={fidelity:.4f}")
    
    return results


def benchmark_scaling():
    """Benchmark compilation scaling with system size."""
    print("\n" + "="*80)
    print("BENCHMARK 2: System Size Scaling")
    print("="*80)
    
    results = BenchmarkResults()
    compiler = HamiltonianCompiler(optimization_level=3)
    
    print(f"{'Qubits':>7} {'Terms':>6} {'Depth':>6} {'CNOTs':>6} {'Time(s)':>8}")
    print("-" * 45)
    
    for n_qubits in [2, 4, 6, 8, 10]:
        # Create nearest-neighbor Ising model
        H = Hamiltonian(n_qubits=n_qubits)
        
        for i in range(n_qubits - 1):
            pauli = ['I'] * n_qubits
            pauli[i] = 'Z'
            pauli[i+1] = 'Z'
            H.add_term(1.0, ''.join(pauli))
        
        start_time = time.time()
        circuit = compiler.compile(H, time=1.0, error_budget=1e-2, method="trotter2")
        compile_time = time.time() - start_time
        
        counts = circuit.count_ops()
        
        results.add_result(
            name=f"Ising_{n_qubits}qubits",
            n_qubits=n_qubits,
            n_terms=H.n_terms,
            depth=circuit.depth(),
            cnot_count=counts.get('cx', 0),
            compile_time=compile_time
        )
        
        print(f"{n_qubits:7d} {H.n_terms:6d} {circuit.depth():6d} "
              f"{counts.get('cx', 0):6d} {compile_time:8.4f}")
    
    return results


def benchmark_optimization_levels():
    """Benchmark different optimization levels."""
    print("\n" + "="*80)
    print("BENCHMARK 3: Optimization Levels")
    print("="*80)
    
    results = BenchmarkResults()
    
    # Test Hamiltonian
    H = Hamiltonian(n_qubits=4)
    H.add_term(1.0, "ZZII")
    H.add_term(0.8, "IZZI")
    H.add_term(0.6, "IIZZ")
    H.add_term(0.5, "XXII")
    H.add_term(0.4, "IXXI")
    
    print(f"{'Level':>6} {'Depth':>6} {'CNOTs':>6} {'Gates':>6} {'Time(s)':>8} {'Fidelity':>10}")
    print("-" * 58)
    
    for opt_level in [0, 1, 2, 3]:
        compiler = HamiltonianCompiler(optimization_level=opt_level)
        
        start_time = time.time()
        circuit = compiler.compile(H, time=1.0, error_budget=1e-2, method="trotter1")
        compile_time = time.time() - start_time
        
        fidelity = compiler.estimate_fidelity(circuit)
        counts = circuit.count_ops()
        
        results.add_result(
            name=f"opt_level_{opt_level}",
            optimization_level=opt_level,
            depth=circuit.depth(),
            cnot_count=counts.get('cx', 0),
            total_gates=len(circuit.gates),
            compile_time=compile_time,
            fidelity=fidelity
        )
        
        print(f"{opt_level:6d} {circuit.depth():6d} {counts.get('cx', 0):6d} "
              f"{len(circuit.gates):6d} {compile_time:8.4f} {fidelity:10.4f}")
    
    return results


def benchmark_error_budgets():
    """Benchmark compilation with different error budgets."""
    print("\n" + "="*80)
    print("BENCHMARK 4: Error Budget Analysis")
    print("="*80)
    
    results = BenchmarkResults()
    
    H = Hamiltonian(n_qubits=4)
    for i in range(3):
        pauli = ['I'] * 4
        pauli[i] = 'Z'
        pauli[i+1] = 'Z'
        H.add_term(1.0, ''.join(pauli))
    
    compiler = HamiltonianCompiler(optimization_level=3)
    
    print(f"{'Error Budget':>13} {'Depth':>6} {'CNOTs':>6} {'Gates':>6}")
    print("-" * 41)
    
    for error_budget in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        circuit = compiler.compile(H, time=1.0, error_budget=error_budget, method="trotter1")
        counts = circuit.count_ops()
        
        results.add_result(
            name=f"error_{error_budget:.0e}",
            error_budget=error_budget,
            depth=circuit.depth(),
            cnot_count=counts.get('cx', 0),
            total_gates=len(circuit.gates)
        )
        
        print(f"{error_budget:13.0e} {circuit.depth():6d} "
              f"{counts.get('cx', 0):6d} {len(circuit.gates):6d}")
    
    return results


def benchmark_heisenberg_models():
    """Benchmark Heisenberg model compilation."""
    print("\n" + "="*80)
    print("BENCHMARK 5: Heisenberg Chain Models")
    print("="*80)
    
    results = BenchmarkResults()
    compiler = HamiltonianCompiler(optimization_level=3)
    
    print(f"{'Sites':>6} {'Terms':>6} {'Depth':>6} {'CNOTs':>6} {'Time(s)':>8}")
    print("-" * 44)
    
    for n_sites in [4, 6, 8, 10]:
        H = Hamiltonian(n_qubits=n_sites)
        
        # Add XX, YY, ZZ interactions
        for i in range(n_sites - 1):
            for pauli_type in ['X', 'Y', 'Z']:
                pauli = ['I'] * n_sites
                pauli[i] = pauli_type
                pauli[i+1] = pauli_type
                H.add_term(1.0, ''.join(pauli))
        
        start_time = time.time()
        circuit = compiler.compile(H, time=1.0, error_budget=1e-2, method="trotter2")
        compile_time = time.time() - start_time
        
        counts = circuit.count_ops()
        
        results.add_result(
            name=f"Heisenberg_{n_sites}sites",
            n_sites=n_sites,
            n_terms=H.n_terms,
            depth=circuit.depth(),
            cnot_count=counts.get('cx', 0),
            compile_time=compile_time
        )
        
        print(f"{n_sites:6d} {H.n_terms:6d} {circuit.depth():6d} "
              f"{counts.get('cx', 0):6d} {compile_time:8.4f}")
    
    return results


def main():
    """Run all benchmarks."""
    print("\n" + "="*80)
    print("HAMILTONIANCOMPILER BENCHMARK SUITE")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"NumPy version: {np.__version__}")
    
    all_results = BenchmarkResults()
    
    # Run all benchmarks
    r1 = benchmark_molecular_systems()
    all_results.results.extend(r1.results)
    
    r2 = benchmark_scaling()
    all_results.results.extend(r2.results)
    
    r3 = benchmark_optimization_levels()
    all_results.results.extend(r3.results)
    
    r4 = benchmark_error_budgets()
    all_results.results.extend(r4.results)
    
    r5 = benchmark_heisenberg_models()
    all_results.results.extend(r5.results)
    
    # Save results
    all_results.save_json('benchmarks/benchmark_results.json')
    print("\n" + "="*80)
    print("Benchmarks complete! Results saved to benchmark_results.json")
    print("="*80)


if __name__ == "__main__":
    main()
