#!/usr/bin/env python3
"""
Hyper2Vec: A hypergraph embedding method that preserves higher-order relationships.

This implementation follows the approach from:
"Hyper2Vec: Biased Random Walk for Hypergraph Embedding"

The key idea is to perform biased random walks directly on the hypergraph structure
rather than converting to a 2-section graph, thus preserving the multi-way relationships.
"""

import os
import pickle
import numpy as np
import random
from collections import defaultdict, Counter
from gensim.models import Word2Vec
from tqdm import tqdm
import datetime
from typing import List, Dict, Set, Tuple

# Input and output paths
INPUT_HYPERGRAPH_PATH = "data/hypergraphs/hypergraph.pickle"
OUTPUT_DIR = "data/vectors/hyper2vec/"

class Hyper2Vec:
    def __init__(self, hypergraph_data, dimensions=128, walk_length=30, num_walks=200,
                 p=1.0, q=1.0, alpha=0.5, beta=0.5, workers=4):
        """
        Initialize Hyper2Vec model.

        Args:
            hypergraph_data: Dictionary mapping hyperedge IDs to sets of nodes
            dimensions: Embedding dimension
            walk_length: Length of each random walk
            num_walks: Number of random walks per node
            p: Return parameter (controls probability of returning to previous node)
            q: In-out parameter (controls search to explore vs exploit)
            alpha: Balance between node-to-hyperedge and hyperedge-to-node transitions
            beta: Balance between uniform and degree-biased sampling
            workers: Number of parallel workers for Word2Vec
        """
        self.hypergraph_data = hypergraph_data
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.workers = workers

        # Build auxiliary data structures
        self._build_auxiliary_structures()

    def _build_auxiliary_structures(self):
        """Build auxiliary data structures for efficient sampling."""
        # Node to hyperedges mapping
        self.node_to_hyperedges = defaultdict(set)
        # Hyperedge to nodes mapping (already given but convert to consistent format)
        self.hyperedge_to_nodes = {}
        # All nodes in the hypergraph
        self.nodes = set()
        # All hyperedges in the hypergraph
        self.hyperedges = set()

        for hyperedge_id, nodes in self.hypergraph_data.items():
            # Convert nodes to strings for consistency
            nodes = set(str(node) for node in nodes)
            self.hyperedge_to_nodes[str(hyperedge_id)] = nodes
            self.hyperedges.add(str(hyperedge_id))
            self.nodes.update(nodes)

            for node in nodes:
                self.node_to_hyperedges[str(node)].add(str(hyperedge_id))

        # Compute node degrees (number of hyperedges a node belongs to)
        self.node_degrees = {node: len(hyperedges) for node, hyperedges in self.node_to_hyperedges.items()}

        # Compute hyperedge sizes
        self.hyperedge_sizes = {he: len(nodes) for he, nodes in self.hyperedge_to_nodes.items()}

        print(f"Hypergraph structure:")
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Hyperedges: {len(self.hyperedges)}")
        print(f"  Avg nodes per hyperedge: {np.mean(list(self.hyperedge_sizes.values())):.2f}")
        print(f"  Avg hyperedges per node: {np.mean(list(self.node_degrees.values())):.2f}")

    def _sample_next_node_from_hyperedge(self, current_hyperedge: str, previous_node: str = None) -> str:
        """
        Sample next node from a hyperedge using biased sampling.

        Args:
            current_hyperedge: Current hyperedge ID
            previous_node: Previous node (for biased sampling)

        Returns:
            Next node ID
        """
        nodes_in_hyperedge = list(self.hyperedge_to_nodes[current_hyperedge])

        if previous_node is None:
            # Random sampling for the first step
            return random.choice(nodes_in_hyperedge)

        # Biased sampling based on node degrees and return probability
        weights = []
        for node in nodes_in_hyperedge:
            if node == previous_node:
                # Return probability (controlled by p parameter)
                weight = 1.0 / self.p
            else:
                # Forward probability (controlled by q parameter and node degree)
                weight = (1.0 / self.q) * (self.node_degrees[node] ** self.beta)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Sample based on weights
        return np.random.choice(nodes_in_hyperedge, p=weights)

    def _sample_next_hyperedge_from_node(self, current_node: str, previous_hyperedge: str = None) -> str:
        """
        Sample next hyperedge from a node using biased sampling.

        Args:
            current_node: Current node ID
            previous_hyperedge: Previous hyperedge (for biased sampling)

        Returns:
            Next hyperedge ID
        """
        hyperedges_of_node = list(self.node_to_hyperedges[current_node])

        if previous_hyperedge is None or len(hyperedges_of_node) == 1:
            # Random sampling or only one choice
            return random.choice(hyperedges_of_node)

        # Biased sampling based on hyperedge sizes
        weights = []
        for hyperedge in hyperedges_of_node:
            if hyperedge == previous_hyperedge:
                # Return probability
                weight = 1.0 / self.p
            else:
                # Forward probability (prefer larger hyperedges)
                weight = (1.0 / self.q) * (self.hyperedge_sizes[hyperedge] ** self.beta)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Sample based on weights
        return np.random.choice(hyperedges_of_node, p=weights)

    def _generate_random_walk(self, start_node: str) -> List[str]:
        """
        Generate a single random walk starting from a node.

        Args:
            start_node: Starting node ID

        Returns:
            List of node IDs representing the walk
        """
        walk = [start_node]
        current_node = start_node
        previous_hyperedge = None

        for _ in range(self.walk_length - 1):
            # Step 1: From current node, choose a hyperedge
            try:
                current_hyperedge = self._sample_next_hyperedge_from_node(current_node, previous_hyperedge)
            except:
                # If no hyperedges available, stop the walk
                break

            # Step 2: From the hyperedge, choose the next node
            try:
                next_node = self._sample_next_node_from_hyperedge(current_hyperedge, current_node)
            except:
                # If no nodes available, stop the walk
                break

            walk.append(next_node)

            # Update for next iteration
            previous_hyperedge = current_hyperedge
            current_node = next_node

        return walk

    def generate_walks(self) -> List[List[str]]:
        """
        Generate all random walks for the hypergraph.

        Returns:
            List of walks, where each walk is a list of node IDs
        """
        walks = []
        nodes_list = list(self.nodes)

        print(f"Generating {self.num_walks} walks of length {self.walk_length} for each of {len(nodes_list)} nodes...")

        for _ in tqdm(range(self.num_walks), desc="Walk iterations"):
            # Shuffle nodes for each iteration to ensure randomness
            random.shuffle(nodes_list)
            for node in nodes_list:
                walk = self._generate_random_walk(node)
                if len(walk) > 1:  # Only add walks with more than one node
                    walks.append(walk)

        print(f"Generated {len(walks)} walks")
        return walks

    def fit(self, window=10, min_count=1, workers=None):
        """
        Learn node embeddings using Word2Vec on the generated walks.

        Args:
            window: Context window size for Word2Vec
            min_count: Minimum count for Word2Vec
            workers: Number of workers (defaults to self.workers)

        Returns:
            Trained Word2Vec model
        """
        if workers is None:
            workers = self.workers

        # Generate walks
        walks = self.generate_walks()

        # Train Word2Vec model
        print("Training Word2Vec model on walks...")
        model = Word2Vec(
            walks,
            vector_size=self.dimensions,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1,  # Skip-gram
            hs=0,  # Negative sampling
            negative=5,  # Number of negative samples
            alpha=0.025,  # Learning rate
            epochs=5
        )

        return model


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load the hypergraph
    print(f"Loading hypergraph from {INPUT_HYPERGRAPH_PATH}")
    try:
        with open(INPUT_HYPERGRAPH_PATH, 'rb') as f:
            hypergraph_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Hypergraph file {INPUT_HYPERGRAPH_PATH} not found.")
        print("Please run the hypergraph generation script first.")
        return

    # Initialize Hyper2Vec
    print("Initializing Hyper2Vec...")
    hyper2vec = Hyper2Vec(
        hypergraph_data,
        dimensions=128,      # Embedding dimension
        walk_length=30,      # Length of each random walk
        num_walks=200,       # Number of random walks per node
        p=1.0,              # Return parameter
        q=1.0,              # In-out parameter
        alpha=0.5,          # Node-hyperedge transition balance
        beta=0.5,           # Degree bias parameter
        workers=4           # Number of parallel workers
    )

    # Train the model
    print("Training Hyper2Vec model...")
    model = hyper2vec.fit(
        window=10,          # Context window
        min_count=1,        # Minimum word frequency
        workers=4           # Number of workers
    )

    # Save the model and embeddings
    model_output_path = os.path.join(OUTPUT_DIR, f"hyper2vec_model.model")
    embeddings_output_path = os.path.join(OUTPUT_DIR, f"hyper2vec_embeddings.pickle")

    print(f"Saving Hyper2Vec model to {model_output_path}")
    model.save(model_output_path)

    # Extract and save node embeddings
    print(f"Saving node embeddings to {embeddings_output_path}")
    embeddings = {}
    for node in hyper2vec.nodes:
        if node in model.wv:
            embeddings[node] = model.wv[node]
        else:
            print(f"Warning: Node {node} not found in embeddings")

    with open(embeddings_output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print("Hyper2Vec embedding process completed successfully!")
    print(f"Generated embeddings for {len(embeddings)} nodes")

if __name__ == "__main__":
    main()
