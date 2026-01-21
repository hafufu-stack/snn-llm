"""
11D Hypercube Topology for SNN
==============================

Creates the brain-like 11-dimensional hypercube connectivity pattern
based on Blue Brain Project's discovery of 11D clique structures.

Key Properties:
- 2^11 = 2048 neurons
- Each neuron has exactly 11 connections
- Maximum distance between any two neurons: 11 steps
- 186x parameter reduction vs full connection

Author: Hiroto Funasaki (roll)
Date: 2026-01-21
"""

import numpy as np


def create_hypercube_mask(dim: int) -> np.ndarray:
    """
    Create adjacency matrix for n-dimensional hypercube.
    
    In an n-dimensional hypercube:
    - Number of nodes = 2^n
    - Each node is connected to exactly n neighbors
    - Neighbors differ by exactly 1 bit in their binary address
    
    Args:
        dim: Dimension of hypercube (e.g., 11 for 2048 neurons)
    
    Returns:
        Binary adjacency matrix of shape (2^dim, 2^dim)
    
    Example:
        >>> mask = create_hypercube_mask(3)  # 3D cube: 8 nodes, 3 connections each
        >>> mask.shape
        (8, 8)
        >>> mask.sum()  # Total edges
        24.0
    """
    n = 2 ** dim
    mask = np.zeros((n, n), dtype=np.float32)
    
    for node in range(n):
        for d in range(dim):
            # Flip bit d to get neighbor
            neighbor = node ^ (1 << d)
            mask[node, neighbor] = 1.0
    
    return mask


def create_hybrid_hypercube_mask(dim: int, shortcut_prob: float = 0.05) -> np.ndarray:
    """
    Create hybrid hypercube with random long-range shortcuts.
    
    Combines the structured hypercube with Small-World properties
    by adding random long-range connections (Watts-Strogatz style).
    
    Args:
        dim: Dimension of hypercube
        shortcut_prob: Probability of adding random shortcuts
    
    Returns:
        Hybrid adjacency matrix
    """
    mask = create_hypercube_mask(dim)
    n = mask.shape[0]
    
    # Add random shortcuts
    n_shortcuts = int(n * dim * shortcut_prob)
    
    for _ in range(n_shortcuts):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            mask[i, j] = 1.0
            mask[j, i] = 1.0  # Symmetric
    
    return mask


def hypercube_distance(node1: int, node2: int) -> int:
    """
    Calculate Hamming distance between two nodes in hypercube.
    
    This is the number of bit positions that differ,
    which equals the shortest path length in the hypercube.
    
    Args:
        node1: First node index
        node2: Second node index
    
    Returns:
        Hamming distance (shortest path length)
    """
    xor = node1 ^ node2
    distance = 0
    while xor:
        distance += xor & 1
        xor >>= 1
    return distance


def get_hypercube_stats(dim: int) -> dict:
    """
    Get statistics for n-dimensional hypercube.
    
    Args:
        dim: Dimension of hypercube
    
    Returns:
        Dictionary with statistics
    """
    n = 2 ** dim
    edges = n * dim // 2  # Each edge counted twice
    
    return {
        'dimension': dim,
        'nodes': n,
        'edges': edges,
        'connections_per_node': dim,
        'diameter': dim,  # Max distance between any two nodes
        'density': dim / (n - 1),  # Edge density
        'full_connection_ratio': edges / (n * (n - 1) // 2)
    }


if __name__ == "__main__":
    # Demo
    print("11D Hypercube Statistics:")
    print("-" * 40)
    
    for dim in [5, 7, 9, 11]:
        stats = get_hypercube_stats(dim)
        print(f"\n{dim}D Hypercube:")
        print(f"  Nodes: {stats['nodes']:,}")
        print(f"  Edges: {stats['edges']:,}")
        print(f"  Connections/node: {stats['connections_per_node']}")
        print(f"  Diameter: {stats['diameter']}")
        print(f"  Full connection ratio: {stats['full_connection_ratio']:.4%}")
    
    # Create 11D mask
    print("\n\nCreating 11D hypercube mask...")
    mask = create_hypercube_mask(11)
    print(f"Shape: {mask.shape}")
    print(f"Non-zero elements: {int(mask.sum()):,}")
    print(f"Memory: {mask.nbytes / 1024:.1f} KB")
