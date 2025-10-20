"""
Visualization utilities for HamiltonianCompiler

Requires: matplotlib, networkx
Install with: pip install matplotlib networkx
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from typing import List, Dict
import numpy as np


def plot_circuit_stats(circuit, title="Circuit Statistics"):
    """
    Plot circuit statistics.

    Args:
        circuit: QuantumCircuit object
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed. Install with: pip install matplotlib")
        return

    counts = circuit.count_ops()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Gate counts
    gate_types = list(counts.keys())
    gate_counts = list(counts.values())

    ax1.bar(gate_types, gate_counts, color="skyblue")
    ax1.set_xlabel("Gate Type")
    ax1.set_ylabel("Count")
    ax1.set_title("Gate Distribution")
    ax1.grid(axis="y", alpha=0.3)

    # Summary stats
    stats = {
        "Circuit Depth": circuit.depth(),
        "Total Gates": len(circuit.gates),
        "CNOT Count": counts.get("cx", 0),
        "Single-Qubit Gates": sum(counts.get(g, 0) for g in ["rx", "ry", "rz", "h", "x", "sx"]),
    }

    y_pos = np.arange(len(stats))
    ax2.barh(y_pos, list(stats.values()), color="lightcoral")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(list(stats.keys()))
    ax2.set_xlabel("Count")
    ax2.set_title("Circuit Summary")
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_hamiltonian_structure(hamiltonian, title="Hamiltonian Structure"):
    """
    Visualize Hamiltonian term structure.

    Args:
        hamiltonian: Hamiltonian object
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed")
        return

    # Analyze terms
    weights = [abs(term.coefficient) for term in hamiltonian.terms]
    pauli_weights = [term.pauli.weight() for term in hamiltonian.terms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Coefficient distribution
    ax1.hist(weights, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.set_xlabel("|Coefficient|")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Coefficient Distribution")
    ax1.grid(axis="y", alpha=0.3)

    # Pauli weight distribution
    weight_counts = {}
    for w in pauli_weights:
        weight_counts[w] = weight_counts.get(w, 0) + 1

    ax2.bar(weight_counts.keys(), weight_counts.values(), color="coral", edgecolor="black")
    ax2.set_xlabel("Pauli Weight")
    ax2.set_ylabel("Count")
    ax2.set_title("Pauli String Weight Distribution")
    ax2.set_xticks(range(1, max(pauli_weights) + 1))
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"{title}\n{hamiltonian.n_terms} terms, norm={hamiltonian.norm:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_compilation_comparison(results: Dict[str, Dict], title="Compilation Comparison"):
    """
    Compare compilation results across different methods.

    Args:
        results: Dictionary mapping method names to result dictionaries
                 Each result dict should have keys: 'depth', 'cnot_count', 'fidelity'
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed")
        return

    methods = list(results.keys())
    depths = [results[m]["depth"] for m in methods]
    cnots = [results[m]["cnot_count"] for m in methods]
    fidelities = [results[m].get("fidelity", 0) for m in methods]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Circuit depth
    ax1.bar(methods, depths, color="skyblue", edgecolor="black")
    ax1.set_ylabel("Depth")
    ax1.set_title("Circuit Depth")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # CNOT count
    ax2.bar(methods, cnots, color="lightcoral", edgecolor="black")
    ax2.set_ylabel("CNOT Count")
    ax2.set_title("Two-Qubit Gates")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    # Fidelity
    ax3.bar(methods, fidelities, color="lightgreen", edgecolor="black")
    ax3.set_ylabel("Estimated Fidelity")
    ax3.set_title("Circuit Fidelity")
    ax3.set_ylim([0, 1])
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_commutation_graph(hamiltonian, title="Hamiltonian Commutation Graph"):
    """
    Plot commutation graph of Hamiltonian terms.

    Args:
        hamiltonian: Hamiltonian object
        title: Plot title
    """
    if not NETWORKX_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        print("NetworkX and Matplotlib required")
        return

    # Build commutation graph
    G = nx.Graph()

    for i, term in enumerate(hamiltonian.terms):
        G.add_node(i, label=str(term.pauli)[:6])

    # Add edges for non-commuting terms
    for i, term_i in enumerate(hamiltonian.terms):
        for j, term_j in enumerate(hamiltonian.terms):
            if i < j and not term_i.pauli.commutes_with(term_j.pauli):
                G.add_edge(i, j)

    # Plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"), font_size=8)

    plt.title(
        f"{title}\n{len(G.edges())} non-commuting pairs out of "
        f"{len(hamiltonian.terms)*(len(hamiltonian.terms)-1)//2} possible"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()
