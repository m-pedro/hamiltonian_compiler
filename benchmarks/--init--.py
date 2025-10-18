"""
Benchmarks module for HamiltonianCompiler

This module contains benchmark suites for evaluating compiler performance
across different Hamiltonians, system sizes, and compilation strategies.
"""

__version__ = "1.0.0"

from .benchmark_suite import (
    BenchmarkResults,
    benchmark_molecular_systems,
    benchmark_scaling,
    benchmark_optimization_levels,
    benchmark_error_budgets,
    benchmark_heisenberg_models,
)

__all__ = [
    'BenchmarkResults',
    'benchmark_molecular_systems',
    'benchmark_scaling',
    'benchmark_optimization_levels',
    'benchmark_error_budgets',
    'benchmark_heisenberg_models',
]
