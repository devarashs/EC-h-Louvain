import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import hashlib
import pickle

try:
    import hypernetx as hnx
except ImportError:
    print("Installing hypernetx...")
    os.system("pip install hypernetx")
    import hypernetx as hnx

INPUT_HYPERGRAPH_PATH = "hyperedges-senate-committees.txt"
OUTPUT_DIR = "data/partitions/senate-committees/"


def load_hyperedges(file_path):
    """
    Load hyperedges from a file.
    Each line represents a hyperedge with comma-separated node IDs.
    """
    hyperedges = []
    with open(file_path, 'r') as f:
        for line in f:
            # Skip comments or empty lines
            if line.strip().startswith('//') or not line.strip():
                continue

            # Parse the line to get node IDs
            nodes = [int(node) for node in line.strip().split(',')]
            hyperedges.append(nodes)

    print(f"Loaded {len(hyperedges)} hyperedges")
    return hyperedges


def create_hypergraph(hyperedges):
    """
    Create a hypergraph using HyperNetX.
    """
    # Create a dictionary mapping edge IDs to sets of node IDs
    edges_dict = {f"e{i}": set(edge) for i, edge in enumerate(hyperedges)}

    # Create a HyperNetX Hypergraph
    H = hnx.Hypergraph(edges_dict)

    print(f"Created hypergraph with {len(H.nodes)} nodes and {len(H.edges)} hyperedges")
    return H


def save_hypergraph(H, file_path):
    """
    Save the hypergraph to a file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the hypergraph
    with open(file_path, 'wb') as f:
        pickle.dump(H.incidence_dict, f)

    print(f"Saved hypergraph to {file_path}")


def visualize_hypergraph(H, file_path):
    """
    Visualize the hypergraph using HyperNetX and save the visualization.
    """
    plt.figure(figsize=(16, 12))

    # For large hypergraphs, we might need to adjust node sizes
    node_size = max(50, 500 - len(H.nodes) // 10)  # Adaptive node size

    try:
        # Draw the hypergraph using HyperNetX
        hnx.draw(H,
                with_node_labels=False,   # Hide node labels for large graphs
                with_edge_labels=False,   # Hide edge labels
                node_size=node_size,      # Adjust node size
                edge_color='lightblue',   # Color for hyperedges
                node_color='orange')      # Color for nodes

        plt.title(f"Hypergraph: {len(H.nodes)} nodes, {len(H.edges)} hyperedges")
    except Exception as e:
        print(f"Warning: Could not draw hypergraph: {e}")
        plt.close()
        return False

    # Save the figure
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


def visualize_hypergraph_as_bipartite(H, file_path):
    """
    Visualize the hypergraph as a bipartite graph.
    This representation shows nodes and hyperedges as separate node types.
    """
    plt.figure(figsize=(16, 12))

    # Convert to a bipartite graph
    B = H.bipartite()

    # Get the sets of nodes
    top_nodes = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 0]
    bottom_nodes = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 1]

    # Create a layout for the bipartite graph
    pos = nx.bipartite_layout(B, top_nodes)

    # Draw nodes - orange for original nodes, blue for hyperedges
    nx.draw_networkx_nodes(B, pos, nodelist=top_nodes, node_color='orange',
                          node_size=100, alpha=0.8, label="Nodes")
    nx.draw_networkx_nodes(B, pos, nodelist=bottom_nodes, node_color='lightblue',
                          node_size=200, alpha=0.8, label="Hyperedges")

    # Draw edges with transparency
    nx.draw_networkx_edges(B, pos, width=0.5, alpha=0.3)

    plt.title(f"Bipartite Representation: {len(top_nodes)} nodes, {len(bottom_nodes)} hyperedges")
    plt.legend()

    # Save the figure
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


def visualize_node_connectivity(H, file_path):
    """
    Create a visualization showing node connectivity in the hypergraph.
    """
    plt.figure(figsize=(12, 8))

    # Count the number of hyperedges each node belongs to
    node_counts = {node: 0 for node in H.nodes}
    for edge_id, nodes in H.incidence_dict.items():
        for node in nodes:
            node_counts[node] += 1

    # Sort nodes by their connectivity
    sorted_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)
    nodes = [n[0] for n in sorted_nodes]
    counts = [n[1] for n in sorted_nodes]

    # Plot the distribution
    plt.bar(range(len(nodes)), counts, color='steelblue')
    plt.xlabel('Node (sorted by connectivity)')
    plt.ylabel('Number of Hyperedges')
    plt.title('Node Connectivity in Hypergraph')

    # Add info about most connected nodes
    top_10 = sorted_nodes[:10]
    plt.text(0.02, 0.95, f"Top 10 most connected nodes:\n" +
             "\n".join([f"Node {n}: {c} hyperedges" for n, c in top_10]),
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    return True


def generate_hypergraph_name(file_path):
    """
    Generate a meaningful name for the hypergraph based on the input file.
    """
    # # Get the base name of the file without extension
    # base_name = os.path.splitext(os.path.basename(file_path))[0]

    # # Add a timestamp for uniqueness
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # # Add a hash of the first part of the file content
    # with open(file_path, 'r') as f:
    #     content = f.read(1024)
    # content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

    return f"hypergraph"


def main():
    parser = argparse.ArgumentParser(description='Generate and visualize hypergraphs.')
    parser.add_argument('--input', default='data/'+INPUT_HYPERGRAPH_PATH,
                        help='Path to the input file containing hyperedges.')
    parser.add_argument('--output-dir', default='data/hypergraphs',
                        help='Directory to save the generated hypergraph.')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist.")
        # Try to find the file in possible locations
        possible_locations = [
            'data/'+INPUT_HYPERGRAPH_PATH,
            '../data/'+INPUT_HYPERGRAPH_PATH,
            '../../data/'+INPUT_HYPERGRAPH_PATH,
        ]
        for loc in possible_locations:
            if os.path.exists(loc):
                args.input = loc
                print(f"Found input file at {loc}")
                break
        else:
            raise FileNotFoundError(f"Could not find input file {args.input}")

    # Load hyperedges
    hyperedges = load_hyperedges(args.input)

    # Create hypergraph
    H = create_hypergraph(hyperedges)

    # Generate a name for the hypergraph
    hypergraph_name = generate_hypergraph_name(args.input)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create paths for saving
    hypergraph_file = os.path.join(args.output_dir, f"{hypergraph_name}.pickle")
    visualization_file = os.path.join(args.output_dir, f"{hypergraph_name}.png")
    bipartite_file = os.path.join(args.output_dir, f"{hypergraph_name}_bipartite.png")
    connectivity_file = os.path.join(args.output_dir, f"{hypergraph_name}_connectivity.png")

    # Save hypergraph
    save_hypergraph(H, hypergraph_file)

    # Visualize hypergraph in multiple ways
    print("Generating visualizations...")

    # Direct hypergraph visualization
    if visualize_hypergraph(H, visualization_file):
        print(f"Saved hypergraph visualization to {visualization_file}")

    # Bipartite representation
    if visualize_hypergraph_as_bipartite(H, bipartite_file):
        print(f"Saved bipartite visualization to {bipartite_file}")

    # Node connectivity analysis
    if visualize_node_connectivity(H, connectivity_file):
        print(f"Saved node connectivity analysis to {connectivity_file}")

    print(f"\nSummary of hypergraph {hypergraph_name}:")
    print(f"- {len(H.nodes)} nodes")
    print(f"- {len(H.edges)} hyperedges")
    print(f"- Saved to {hypergraph_file}")


if __name__ == "__main__":
    main()