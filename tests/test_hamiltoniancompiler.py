"""
Basic tests for HamiltonianCompiler 
"""

import pytest
import numpy as np
from hamiltoniancompiler import (
    Hamiltonian, 
    HamiltonianCompiler, 
    PauliString,
    QuantumCircuit
)


class TestPauliString:
    """Test PauliString class."""
    
    def test_initialization(self):
        """Test PauliString creation."""
        p = PauliString("XYZ")
        assert p.string == "XYZ"
        assert p.n_qubits == 3
    
    def test_support(self):
        """Test support calculation."""
        p = PauliString("IXYI")
        assert p.support() == [1, 2]
    
    def test_weight(self):
        """Test weight calculation."""
        p = PauliString("IXYZ")
        assert p.weight() == 3


class TestHamiltonian:
    """Test Hamiltonian class."""
    
    def test_initialization(self):
        """Test Hamiltonian creation."""
        H = Hamiltonian(n_qubits=2)
        assert H.n_qubits == 2
        assert H.n_terms == 0
    
    def test_add_term(self):
        """Test adding terms."""
        H = Hamiltonian(n_qubits=2)
        H.add_term(1.0, "XX")
        H.add_term(0.5, "ZZ")
        assert H.n_terms == 2
    
    def test_norm(self):
        """Test L1 norm calculation."""
        H = Hamiltonian(n_qubits=2)
        H.add_term(1.0, "XX")
        H.add_term(-0.5, "ZZ")
        assert abs(H.norm - 1.5) < 1e-10


class TestQuantumCircuit:
    """Test QuantumCircuit class."""
    
    def test_initialization(self):
        """Test circuit creation."""
        qc = QuantumCircuit(n_qubits=3)
        assert qc.n_qubits == 3
        assert len(qc.gates) == 0
    
    def test_add_gates(self):
        """Test adding gates."""
        qc = QuantumCircuit(n_qubits=2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(np.pi/4, 1)
        assert len(qc.gates) == 3


class TestHamiltonianCompiler:
    """Test HamiltonianCompiler class."""
    
    def test_compilation_trotter1(self):
        """Test compilation with first-order Trotter."""
        H = Hamiltonian(n_qubits=2)
        H.add_term(1.0, "ZZ")
        H.add_term(0.5, "XX")
        
        compiler = HamiltonianCompiler(optimization_level=2)
        circuit = compiler.compile(H, time=1.0, error_budget=1e-2, method="trotter1")
        
        assert len(circuit.gates) > 0
        assert circuit.depth() > 0
    
    def test_compilation_trotter2(self):
        """Test compilation with second-order Trotter."""
        H = Hamiltonian(n_qubits=2)
        H.add_term(1.0, "ZZ")
        H.add_term(0.5, "XX")
        
        compiler = HamiltonianCompiler(optimization_level=2)
        circuit = compiler.compile(H, time=1.0, error_budget=1e-2, method="trotter2")
        
        assert len(circuit.gates) > 0
    
    def test_fidelity_estimation(self):
        """Test fidelity estimation."""
        H = Hamiltonian(n_qubits=2)
        H.add_term(1.0, "ZZ")
        
        compiler = HamiltonianCompiler()
        circuit = compiler.compile(H, time=1.0, error_budget=1e-2)
        
        fidelity = compiler.estimate_fidelity(circuit)
        assert 0 <= fidelity <= 1
        assert fidelity > 0.5
    
    def test_h2_molecule(self):
        """Test H2 molecule compilation."""
        H = Hamiltonian(n_qubits=2)
        H.add_term(-1.0523, "II")
        H.add_term(0.3979, "IZ")
        H.add_term(-0.3979, "ZI")
        H.add_term(-0.0112, "ZZ")
        H.add_term(0.1809, "XX")
        
        compiler = HamiltonianCompiler(optimization_level=3)
        circuit = compiler.compile(H, time=1.0, error_budget=1e-3, method="auto")
        
        assert circuit.depth() > 0
        assert len(circuit.gates) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
