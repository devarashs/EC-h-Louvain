#!/usr/bin/env python3
# filepath: /home/south/EC-h-Louvain/generate_2_section_graph_from_hyperedges-senate-committees.py

"""
Script to generate a 2-section graph from a hypergraph.
The script loads the hypergraph from a pickle file, converts it to a 2-section graph,
and saves the result for further analysis.
"""

import os
import pickle
import networkx as nx
from datetime import datetime
import time
import sys

# Input and output paths
input_file = "data/hypergraphs/hyperedges-senate-committees_20250510_203112_70803e19.pickle"
output_dir = "data/2-section/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate timestamp for the output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
hash_value = hex(int(time.time() * 1000000))[2:10]  # Simple unique identifier

# Load the hypergraph
print(f"Loading hypergraph from {input_file}...")
try:
    with open(input_file, 'rb') as f:
        hypergraph_data = pickle.load(f)
except FileNotFoundError:
    print(f"Error: File {input_file} not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading hypergraph: {e}")
    sys.exit(1)

# Determine the base name of the hypergraph file (without directory and extension)
base_name = os.path.basename(input_file)
base_name = os.path.splitext(base_name)[0]
# Remove the timestamp and hash part from the base name if it exists
base_parts = base_name.split('_')
if len(base_parts) >= 3:
    base_name = '_'.join(base_parts[:-2])

# Create a new empty graph
two_section_graph = nx.Graph()

print("Generating 2-section graph...")

# Counters for statistics
num_hyperedges = 0
unique_nodes = set()

# Try to determine the format of the hypergraph data
if isinstance(hypergraph_data, list):
    # Assume it's a list of hyperedges
    num_hyperedges = len(hypergraph_data)
    for hyperedge in hypergraph_data:
        # Add edges between all pairs of nodes in this hyperedge
        nodes = list(hyperedge)  # Convert to list if it's a set
        unique_nodes.update(nodes)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                two_section_graph.add_edge(nodes[i], nodes[j])

elif isinstance(hypergraph_data, dict):
    # Assume it's a dictionary mapping hyperedge IDs to sets of nodes
    num_hyperedges = len(hypergraph_data)
    for hyperedge_id, nodes in hypergraph_data.items():
        nodes = list(nodes)  # Convert to list if it's a set
        unique_nodes.update(nodes)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                two_section_graph.add_edge(nodes[i], nodes[j])

else:
    # Try to handle it as a custom hypergraph object with methods to access hyperedges
    try:
        if hasattr(hypergraph_data, 'get_hyperedges'):
            hyperedges = hypergraph_data.get_hyperedges()
        elif hasattr(hypergraph_data, 'hyperedges'):
            hyperedges = hypergraph_data.hyperedges()
        else:
            raise AttributeError("Couldn't find hyperedges in the data")

        num_hyperedges = len(hyperedges)
        for hyperedge in hyperedges:
            nodes = list(hyperedge)
            unique_nodes.update(nodes)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    two_section_graph.add_edge(nodes[i], nodes[j])
    except Exception as e:
        print(f"Unable to determine the format of the hypergraph data: {e}")
        print("Please modify the script to handle your specific hypergraph format.")
        sys.exit(1)

# Print statistics
print(f"Hypergraph statistics:")
print(f"  Number of hyperedges: {num_hyperedges}")
print(f"  Number of unique nodes: {len(unique_nodes)}")
print(f"2-section graph statistics:")
print(f"  Number of nodes: {two_section_graph.number_of_nodes()}")
print(f"  Number of edges: {two_section_graph.number_of_edges()}")

# Generate output filename
output_file = f"{output_dir}/{base_name}-2section_{timestamp}_{hash_value}.pickle"

# Save the 2-section graph
print(f"Saving 2-section graph to {output_file}...")
try:
    with open(output_file, 'wb') as f:
        pickle.dump(two_section_graph, f)
    print("Done!")
except Exception as e:
    print(f"Error saving 2-section graph: {e}")
    sys.exit(1)