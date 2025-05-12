import community as community_louvain
import networkx as nx
import json
import pickle
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time
import csv
from pathlib import Path


def save_metrics(metrics_dict, output_file):
    """Save metrics to a CSV file."""
    print(f"Saving metrics to {output_file}")
    file_exists = os.path.isfile(output_file)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Open file and write metrics
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)


def load_initial_clusters(clusters_dir):
    """
    Load all cluster files from the specified directory.
    Returns a list of (name, partition) tuples.
    """
    cluster_files = []

    # Look for cluster files
    for f in os.listdir(clusters_dir):
        file_path = os.path.join(clusters_dir, f)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Process different file types
        if f.endswith('.npy'):
            try:
                # Load npy files (usually contain cluster labels)
                labels = np.load(file_path)
                partition = {i: int(label) for i, label in enumerate(labels)}
                cluster_files.append((f, partition))
                print(f"Loaded {f}: {len(set(partition.values()))} clusters")
            except Exception as e:
                print(f"Error loading {f}: {e}")

        elif f.endswith('.gpickle'):
            try:
                # Load NetworkX graphs with cluster attributes
                G = nx.read_gpickle(file_path)
                partition = {}
                for node, attrs in G.nodes(data=True):
                    if 'cluster' in attrs:
                        partition[node] = attrs['cluster']
                    else:
                        print(f"Warning: Node {node} in {f} has no cluster attribute")
                        partition[node] = 0

                if partition:
                    cluster_files.append((f, partition))
                    print(f"Loaded {f}: {len(set(partition.values()))} clusters")
            except Exception as e:
                print(f"Error loading {f}: {e}")

        elif f.endswith('.json'):
            try:
                # Load JSON partition files
                with open(file_path, 'r') as json_file:
                    partition = json.load(json_file)

                # Convert string keys to integers if needed
                partition = {int(k) if isinstance(k, str) and k.isdigit() else k: v
                             for k, v in partition.items()}

                cluster_files.append((f, partition))
                print(f"Loaded {f}: {len(set(partition.values()))} clusters")
            except Exception as e:
                print(f"Error loading {f}: {e}")

    return cluster_files


def main():
    """
    Improved EC-Louvain implementation:
    1. Load multiple cluster files as potential initial partitions
    2. Load the appropriate graph structure
    3. Run Louvain with each initial partition
    4. Report the best results
    """
    # Start timing the entire process
    start_time = time.time()

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    clusters_dir = os.path.join(base_dir, "data/partitions/kmeans/")

    # Instead of using embeddings, use the original 2-section graph
    graph_path = os.path.join(base_dir, "data/2-section/2_section_graph.pickle")

    print(f"Base directory: {base_dir}")
    print(f"Clusters directory: {clusters_dir}")
    print(f"Graph path: {graph_path}")

    # Create output directories if they don't exist
    os.makedirs(os.path.join(base_dir, "data/partitions"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/visualizations"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/visualizations/ec_louvain_node2vec"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    # Load the graph
    print(f"Loading graph from {graph_path}...")
    try:
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        # Check if G is a NetworkX graph
        if not isinstance(G, nx.Graph):
            print("Warning: Loaded object is not a NetworkX graph. Attempting to convert...")
            G = nx.Graph(G)

        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    # Load all available cluster files
    cluster_files = load_initial_clusters(clusters_dir)

    if not cluster_files:
        print(f"No valid cluster files found in {clusters_dir}")
        return

    best_modularity = -1
    best_partition = None
    best_file = None

    # Process each cluster file
    for cluster_file, initial_partition in cluster_files:
        print(f"\nProcessing clusters from {cluster_file}...")

        # Ensure all graph nodes exist in the partition
        for node in G.nodes():
            if node not in initial_partition:
                print(f"Warning: Node {node} not in partition, adding default community")
                initial_partition[node] = 0  # Assign to default community

        # Run EC-Louvain with this partition
        print("Running EC-Louvain algorithm...")
        clustering_start_time = time.time()
        partition = community_louvain.best_partition(G, partition=initial_partition)
        clustering_time = time.time() - clustering_start_time

        # Calculate modularity
        modularity = community_louvain.modularity(partition, G)
        print(f"Modularity: {modularity}")

        # Count number of communities
        num_communities = len(set(partition.values()))
        print(f"Number of communities: {num_communities}")

        # Keep track of the best result
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition
            best_file = cluster_file

        # Prepare metrics dictionary
        metrics = {
            'algorithm': f'ec_louvain_{cluster_file}-node2vec',
            'dataset': os.path.basename(graph_path),
            'num_clusters': num_communities,
            'modularity': modularity,
            'clustering_time': clustering_time,
            'total_runtime': time.time() - start_time,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'initial_partition': cluster_file,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        leaderboard_file_path = os.path.join(base_dir, "results/leaderboard.csv")
        file_exists_and_not_empty_leaderboard = os.path.isfile(leaderboard_file_path) and os.path.getsize(leaderboard_file_path) > 0

        with open(leaderboard_file_path, mode='a', newline='') as csvfile:
            fieldnames = ["Algorithm","isPartition","Nodes","Edges", "Modularity"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists_and_not_empty_leaderboard:
                writer.writeheader()

            writer.writerow({
                "Algorithm": "EC-Louvain",
                "isPartition": "Node2Vec+kmeans",
                "Nodes": len(G.nodes()),
                "Edges": len(G.edges()),
                "Modularity": modularity
            })
        print(f"Leaderboard saved to {leaderboard_file_path}")

        # Save metrics
        metrics_path = os.path.join(base_dir, "results/louvain_scores.csv")
        save_metrics(metrics, metrics_path)

        # Save this partition
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(partition, base_dir, f"{cluster_file}_{timestamp}")

    # Process the best result
    if best_partition:
        print(f"\nBest result from {best_file} with modularity {best_modularity}")
        analyze_communities(best_partition)

        # Visualize the best communities
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualize_communities(G, best_partition, base_dir, f"best_{timestamp}")

        # Create node2vec visualization
        node2vec_viz_dir = os.path.join(base_dir, "data/visualizations/ec_louvain_node2vec/")
        visualize_node2vec_communities(G, best_partition, node2vec_viz_dir, f"best_node2vec_{timestamp}")

    # Calculate total runtime
    total_runtime = time.time() - start_time
    print(f"Total runtime: {total_runtime:.2f} seconds")


def analyze_communities(partition):
    """
    Analyze and print information about the detected communities.
    """
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    print(f"Number of communities: {len(communities)}")

    # Community size distribution
    sizes = [len(nodes) for nodes in communities.values()]
    print(f"Average community size: {sum(sizes)/len(sizes):.2f}")
    print(f"Largest community: {max(sizes)} nodes")
    print(f"Smallest community: {min(sizes)} nodes")

    # Print details for the largest communities
    for i, (community_id, nodes) in enumerate(sorted(communities.items(),
                                              key=lambda x: len(x[1]),
                                              reverse=True)):
        if i < 5:  # Just print details for the first few largest communities
            print(f"Community {community_id}: {len(nodes)} nodes")
            if len(nodes) < 10:
                print(f"  Nodes: {nodes}")

    if len(communities) > 5:
        print(f"... and {len(communities) - 5} more communities")


def save_results(partition, base_dir, file_suffix):
    """
    Save the partition results to a JSON file.
    """
    output_path = os.path.join(base_dir, f"data/partitions/ec_louvain_result_{file_suffix}.json")

    # Convert keys to strings for JSON compatibility
    partition_for_json = {str(k): v for k, v in partition.items()}

    with open(output_path, 'w') as f:
        json.dump(partition_for_json, f, indent=2)

    print(f"Results saved to {output_path}")


def visualize_communities(G, partition, base_dir, file_suffix):
    """
    Visualize the communities in the graph.
    """
    # Only visualize if the graph is not too large
    if G.number_of_nodes() > 1000:
        print("Graph too large for visualization, skipping...")
        return

    try:
        plt.figure(figsize=(12, 12))

        # Layout for the graph
        print("Computing layout for visualization...")
        pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility

        # Draw nodes colored by community
        print("Drawing graph...")
        nx.draw_networkx(
            G, pos,
            node_color=list(partition.values()),
            cmap=plt.cm.rainbow,
            node_size=50,
            edge_color='gray',
            alpha=0.8,
            with_labels=False
        )

        plt.title("Communities detected by EC-Louvain - Node2Vec")
        plt.axis('off')

        # Save the visualization
        viz_path = os.path.join(base_dir, f"data/visualizations/ec_louvain_node2vec/ec_louvain_{file_suffix}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {viz_path}")

        plt.close()

    except Exception as e:
        print(f"Error during visualization: {e}")


def visualize_node2vec_communities(G, partition, output_dir, file_suffix):
    """
    Create enhanced visualization of communities with clear color coding.
    """
    # Only visualize if the graph is not too large
    if G.number_of_nodes() > 1000:
        print("Graph too large for visualization, skipping node2vec visualization...")
        return

    try:
        plt.figure(figsize=(16, 16))

        # Create a more well-separated layout
        print("Computing layout for node2vec visualization...")
        pos = nx.spring_layout(G, seed=42, k=0.3)  # Increase k for more spacing

        # Get unique communities
        communities = set(partition.values())
        num_communities = len(communities)
        print(f"Visualizing {num_communities} communities...")

        # Create color map with distinct colors
        cmap = plt.cm.get_cmap('tab20' if num_communities <= 20 else 'rainbow', max(20, num_communities))

        # Draw edges first (in background)
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.2,
            width=0.5,
            edge_color='gray'
        )

        # Draw nodes by community with better colors
        for i, comm in enumerate(sorted(communities)):
            # Get list of nodes in this community
            nodes = [node for node, com in partition.items() if com == comm]

            # Draw nodes for this community
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=[cmap(i % cmap.N)],
                node_size=80,
                alpha=0.8,
                label=f"Community {comm}"
            )

        # Add labels for larger nodes (optional, can be resource intensive)
        if G.number_of_nodes() < 1000:  # Only label if graph is small enough
            # Get the top 10% largest communities
            comm_sizes = {}
            for comm in communities:
                comm_sizes[comm] = len([n for n, c in partition.items() if c == comm])

            largest_comms = sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)[:max(3, int(len(communities)*0.1))]
            largest_comm_ids = [c[0] for c in largest_comms]

            # Get nodes in largest communities
            important_nodes = {n: n for n, c in partition.items() if c in largest_comm_ids}

            nx.draw_networkx_labels(
                G, pos,
                labels=important_nodes,
                font_size=8,
                font_color="black",
                alpha=0.6
            )

        plt.title(f"Communities detected by EC-Louvain with Node2Vec\nModularity: {community_louvain.modularity(partition, G):.4f}")
        plt.axis('off')

        # Add legend (limited to prevent overcrowding)
        if num_communities <= 20:
            plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='lower right')

        # Save the visualization
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, f"{file_suffix}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Node2Vec visualization saved to {viz_path}")

        # Also save a high resolution version
        viz_path_hires = os.path.join(output_dir, f"{file_suffix}_hires.png")
        plt.savefig(viz_path_hires, dpi=600, bbox_inches='tight')

        plt.close()

    except Exception as e:
        print(f"Error during node2vec visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()