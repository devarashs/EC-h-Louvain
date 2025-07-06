#!/usr/bin/env python3
"""
EC-Louvain algorithm using Hyper2Vec embeddings for initial partitioning.

This script loads the Hyper2Vec K-means partitions and applies the EC-Louvain
algorithm to refine the community detection on the hypergraph structure.
"""

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
    Load all cluster files from the Hyper2Vec K-means directory.
    Returns a list of (name, partition) tuples.
    """
    cluster_files = []

    # Look for cluster files
    for f in os.listdir(clusters_dir):
        file_path = os.path.join(clusters_dir, f)

        # Skip directories and non-partition files
        if os.path.isdir(file_path) or 'metadata' in f:
            continue

        # Process JSON partition files
        if f.endswith('.json') and 'partition' in f:
            try:
                with open(file_path, 'r') as json_file:
                    partition = json.load(json_file)

                # Convert string keys to appropriate format (keep as strings for consistency)
                partition = {str(k): int(v) for k, v in partition.items()}

                cluster_files.append((f, partition))
                print(f"Loaded {f}: {len(set(partition.values()))} clusters")
            except Exception as e:
                print(f"Error loading {f}: {e}")

    return cluster_files


def build_2_section_graph_from_hypergraph(hypergraph_data):
    """
    Build a 2-section graph from hypergraph data for Louvain algorithm.
    This is necessary because the standard Louvain algorithm works on regular graphs.
    """
    print("Building 2-section graph from hypergraph...")
    G = nx.Graph()

    # Add edges between all pairs of nodes within each hyperedge
    for hyperedge_id, nodes in hypergraph_data.items():
        nodes = list(nodes)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                # Convert nodes to strings for consistency
                node1, node2 = str(nodes[i]), str(nodes[j])
                if G.has_edge(node1, node2):
                    # Increment edge weight if edge exists
                    G[node1][node2]['weight'] += 1
                else:
                    # Add new edge with weight 1
                    G.add_edge(node1, node2, weight=1)

    print(f"Built 2-section graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def calculate_hypergraph_modularity(partition, hypergraph_data):
    """
    Calculate modularity for hypergraph using the generalized modularity formula.
    This is more appropriate for hypergraphs than the standard graph modularity.
    """
    # Convert partition values to ensure they're hashable
    partition = {str(k): v for k, v in partition.items()}

    # Get all communities
    communities = set(partition.values())

    # Calculate hypergraph modularity
    total_degree = 0
    community_internal_degree = {}
    community_degree = {}

    # Initialize community metrics
    for c in communities:
        community_internal_degree[c] = 0
        community_degree[c] = 0

    # Calculate degrees and internal edges
    for hyperedge_id, nodes in hypergraph_data.items():
        nodes = [str(node) for node in nodes]
        hyperedge_size = len(nodes)

        # Skip empty or single-node hyperedges
        if hyperedge_size <= 1:
            continue

        # Calculate contribution to total degree
        total_degree += hyperedge_size * (hyperedge_size - 1)

        # Group nodes by community
        community_nodes = {}
        for node in nodes:
            if node in partition:
                community = partition[node]
                if community not in community_nodes:
                    community_nodes[community] = []
                community_nodes[community].append(node)

        # Calculate internal and total degree contributions
        for community, comm_nodes in community_nodes.items():
            comm_size = len(comm_nodes)

            # Internal edges within this community in this hyperedge
            if comm_size > 1:
                community_internal_degree[community] += comm_size * (comm_size - 1)

            # Total degree for this community from this hyperedge
            community_degree[community] += comm_size * (hyperedge_size - 1)

    # Calculate modularity
    if total_degree == 0:
        return 0

    modularity = 0
    for c in communities:
        internal = community_internal_degree[c]
        degree = community_degree[c]
        modularity += internal - (degree * degree) / total_degree

    modularity /= total_degree
    return modularity


def visualize_partition(G, partition, output_path, title="Community Detection Results"):
    """Visualize the partition results."""
    try:
        plt.figure(figsize=(16, 12))

        # Create position layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Get unique communities
        communities = set(partition.values())
        colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))

        # Draw nodes colored by community
        for community, color in zip(communities, colors):
            nodes_in_community = [node for node, comm in partition.items() if comm == community]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_community,
                                  node_color=[color], node_size=50, alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)

        plt.title(f"{title}\n{len(communities)} communities, {G.number_of_nodes()} nodes")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization to {output_path}")
        return True
    except Exception as e:
        print(f"Error creating visualization: {e}")
        plt.close()
        return False


def main():
    """
    Hyper2Vec EC-Louvain implementation:
    1. Load Hyper2Vec K-means partitions as initial partitions
    2. Load the hypergraph structure
    3. Build 2-section graph for Louvain algorithm
    4. Run EC-Louvain with each initial partition
    5. Report the best results using both graph and hypergraph modularity
    """
    # Start timing the entire process
    start_time = time.time()

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    clusters_dir = os.path.join(base_dir, "data/partitions/hyper2vec_kmeans/")
    hypergraph_path = os.path.join(base_dir, "data/hypergraphs/hypergraph.pickle")

    print(f"Base directory: {base_dir}")
    print(f"Clusters directory: {clusters_dir}")
    print(f"Hypergraph path: {hypergraph_path}")

    # Create output directories if they don't exist
    os.makedirs(os.path.join(base_dir, "data/partitions"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/visualizations"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/visualizations/ec_louvain_hyper2vec"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    # Load the hypergraph
    print(f"Loading hypergraph from {hypergraph_path}...")
    try:
        with open(hypergraph_path, 'rb') as f:
            hypergraph_data = pickle.load(f)

        print(f"Hypergraph has {len(set().union(*hypergraph_data.values()))} nodes and {len(hypergraph_data)} hyperedges")
    except Exception as e:
        print(f"Error loading hypergraph: {e}")
        return

    # Build 2-section graph for Louvain algorithm
    G = build_2_section_graph_from_hypergraph(hypergraph_data)

    # Load all available cluster files
    cluster_files = load_initial_clusters(clusters_dir)

    if not cluster_files:
        print(f"No valid cluster files found in {clusters_dir}")
        print("Please run the Hyper2Vec K-means partitioning script first.")
        return

    best_graph_modularity = -1
    best_hypergraph_modularity = -1
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

        # Calculate graph modularity (standard)
        graph_modularity = community_louvain.modularity(partition, G)

        # Calculate hypergraph modularity (more appropriate)
        hypergraph_modularity = calculate_hypergraph_modularity(partition, hypergraph_data)

        print(f"Graph Modularity: {graph_modularity:.4f}")
        print(f"Hypergraph Modularity: {hypergraph_modularity:.4f}")

        # Count number of communities
        num_communities = len(set(partition.values()))
        print(f"Number of communities: {num_communities}")

        # Keep track of the best result (using hypergraph modularity)
        if hypergraph_modularity > best_hypergraph_modularity:
            best_hypergraph_modularity = hypergraph_modularity
            best_graph_modularity = graph_modularity
            best_partition = partition
            best_file = cluster_file

        # Generate timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save partition results
        partition_output_path = os.path.join(base_dir, f"data/partitions/ec_louvain_hyper2vec_result_partition_{timestamp}.json")
        with open(partition_output_path, 'w') as f:
            json.dump(partition, f, indent=2)

        # Create visualization
        viz_output_path = os.path.join(base_dir, f"data/visualizations/ec_louvain_hyper2vec/ec_louvain_hyper2vec_{timestamp}.png")
        visualize_partition(G, partition, viz_output_path,
                          f"EC-Louvain with Hyper2Vec ({cluster_file})")

        # Prepare metrics dictionary
        metrics = {
            'algorithm': f'ec_louvain_hyper2vec_{cluster_file}',
            'dataset': os.path.basename(hypergraph_path),
            'num_clusters': num_communities,
            'graph_modularity': graph_modularity,
            'hypergraph_modularity': hypergraph_modularity,
            'clustering_time': clustering_time,
            'total_runtime': time.time() - start_time,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_hyperedges': len(hypergraph_data),
            'initial_partition': cluster_file,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }

        # Save to leaderboard
        leaderboard_file_path = os.path.join(base_dir, "results/leaderboard.csv")
        file_exists_and_not_empty_leaderboard = os.path.isfile(leaderboard_file_path) and os.path.getsize(leaderboard_file_path) > 0

        with open(leaderboard_file_path, mode='a', newline='') as csvfile:
            fieldnames = ["Algorithm", "isPartition", "Nodes", "Edges", "Modularity", "Hypergraph_Modularity"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists_and_not_empty_leaderboard:
                writer.writeheader()

            writer.writerow({
                "Algorithm": metrics['algorithm'],
                "isPartition": "Yes",
                "Nodes": metrics['num_nodes'],
                "Edges": metrics['num_edges'],
                "Modularity": metrics['graph_modularity'],
                "Hypergraph_Modularity": metrics['hypergraph_modularity']
            })

        # Save detailed metrics
        detailed_metrics_file = os.path.join(base_dir, "results/hyper2vec_ec_louvain_detailed.csv")
        save_metrics(metrics, detailed_metrics_file)

    # Report best results
    total_time = time.time() - start_time
    print(f"\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Best result from: {best_file}")
    print(f"Best Graph Modularity: {best_graph_modularity:.4f}")
    print(f"Best Hypergraph Modularity: {best_hypergraph_modularity:.4f}")
    print(f"Number of communities: {len(set(best_partition.values()))}")
    print(f"Total execution time: {total_time:.2f} seconds")

    # Save best partition
    best_partition_path = os.path.join(base_dir, "data/partitions/ec_louvain_hyper2vec_best_partition.json")
    with open(best_partition_path, 'w') as f:
        json.dump(best_partition, f, indent=2)
    print(f"Saved best partition to: {best_partition_path}")

    print("\nHyper2Vec EC-Louvain completed successfully!")


if __name__ == "__main__":
    main()
