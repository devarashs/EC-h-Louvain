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


def main():
    """
    Implement EC-Louvain community detection:
    1. Start Louvain with an initial partition instead of singleton partition
    2. Optimize modularity from this starting point
    """
    # Start timing the entire process
    start_time = time.time()

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    partition_path = os.path.join(base_dir, "data/partitions/senate-committees/partition.json")
    embeddings_path = os.path.join(base_dir, "data/vectors/senate-committees/node2vec_embeddings_20250510_210250.pickle")

    print(f"Base directory: {base_dir}")
    print(f"Partition path: {partition_path}")
    print(f"Embeddings path: {embeddings_path}")

    # Create output directories if they don't exist
    os.makedirs(os.path.join(base_dir, "data/partitions"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/visualizations"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    # Load initial partition
    print(f"Loading initial partition from {partition_path}")
    with open(partition_path, 'r') as f:
        initial_partition = json.load(f)

    # Convert keys to integers if stored as strings in the JSON
    initial_partition = {int(k) if isinstance(k, str) and k.isdigit() else k: v
                         for k, v in initial_partition.items()}

    # Load the embedded graph data
    print(f"Loading embeddings from {embeddings_path}")
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    # Process the embeddings data to create a graph
    G = create_graph_from_embeddings(embeddings)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Ensure all graph nodes exist in the partition
    for node in G.nodes():
        if str(node) not in initial_partition and node not in initial_partition:
            print(f"Warning: Node {node} not in initial partition, adding default community")
            initial_partition[node] = 0  # Assign to default community

    # Run EC-Louvain (Louvain with initial partition)
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

    # Calculate total runtime
    total_runtime = time.time() - start_time
    print(f"Total runtime: {total_runtime:.2f} seconds")

    # Analyze communities
    analyze_communities(partition)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(partition, base_dir, timestamp)

    # Prepare metrics dictionary
    metrics = {
        'algorithm': 'ec_louvain',
        'dataset': os.path.basename(embeddings_path),
        'num_clusters': num_communities,
        'modularity': modularity,
        'clustering_time': clustering_time,
        'total_runtime': total_runtime,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'initial_partition': os.path.basename(partition_path),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }

    # Save metrics
    metrics_path = os.path.join(base_dir, "results/louvain_scores.csv")
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")

    # Visualize communities
    visualize_communities(G, partition, base_dir, timestamp)

def create_graph_from_embeddings(embeddings):
    """
    Create a NetworkX graph from node embeddings.
    Edges are created based on embedding similarity.
    """
    G = nx.Graph()

    # Add nodes with embeddings as attributes
    for node, embedding in embeddings.items():
        G.add_node(node, embedding=embedding)

    # Add edges based on embedding similarity
    print("Creating edges based on cosine similarity...")
    nodes = list(embeddings.keys())
    edge_count = 0

    for i in range(len(nodes)):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(nodes)} nodes, created {edge_count} edges")

        for j in range(i+1, len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            emb1, emb2 = np.array(embeddings[node1]), np.array(embeddings[node2])

            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            # Add edge if similarity is above threshold
            threshold = 0.5  # Adjust this threshold as needed
            if similarity > threshold:
                G.add_edge(node1, node2, weight=similarity)
                edge_count += 1

    return G


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


def save_results(partition, base_dir, timestamp):
    """
    Save the partition results to a JSON file.
    """
    output_path = os.path.join(base_dir, f"data/partitions/ec_louvain_result_{timestamp}.json")

    # Convert keys to strings for JSON compatibility
    partition_for_json = {str(k): v for k, v in partition.items()}

    with open(output_path, 'w') as f:
        json.dump(partition_for_json, f, indent=2)

    print(f"Results saved to {output_path}")


def visualize_communities(G, partition, base_dir, timestamp):
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

        plt.title("Communities detected by EC-Louvain")
        plt.axis('off')

        # Save the visualization
        viz_path = os.path.join(base_dir, f"data/visualizations/ec_louvain_{timestamp}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {viz_path}")

        plt.close()

    except Exception as e:
        print(f"Error during visualization: {e}")


if __name__ == "__main__":
    main()