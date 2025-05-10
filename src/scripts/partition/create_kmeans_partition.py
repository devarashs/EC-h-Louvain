import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
import networkx as nx
from kneed import KneeLocator
import seaborn as sns
from collections import defaultdict
import json
import logging
import time
import csv
import community as community_louvain  # python-louvain package for modularity
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Input and output paths - MODIFY THESE
INPUT_FILE = "data/vectors/senate-committees/node2vec_embeddings_20250510_234928.pickle"
OUTPUT_DIR = "data/partitions/senate-committees/"

def calculate_modularity(G, partition):
    """Calculate the modularity of a partition."""
    logger.info("Calculating modularity")
    # Convert partition from node:cluster to format needed by community_louvain
    return community_louvain.modularity(partition, G)

def save_metrics(metrics_dict, output_file):
    """Save metrics to a CSV file."""
    logger.info(f"Saving metrics to {output_file}")
    file_exists = os.path.isfile(output_file)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Open file and write metrics
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)

def load_graph(input_file):
    """Load the pickled graph and embeddings."""
    logger.info(f"Loading graph from {input_file}")
    try:
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        raise

def find_optimal_k(embeddings, max_k=20):
    """Find the optimal number of clusters using elbow method and silhouette score."""
    logger.info("Finding optimal K for KMeans clustering")

    # Calculate inertia for different K values
    inertias = []
    silhouette_scores = []
    k_range = range(2, min(max_k + 1, len(embeddings) // 2 + 1))

    for k in k_range:
        logger.info(f"Testing k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

        # Calculate silhouette score
        if k > 1:
            labels = kmeans.labels_
            try:
                score = silhouette_score(embeddings, labels)
                silhouette_scores.append(score)
                logger.info(f"Silhouette score for k={k}: {score:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score for k={k}: {e}")
                silhouette_scores.append(-1)

    # Use knee/elbow detection to find optimal k
    try:
        kneedle = KneeLocator(list(k_range), inertias, curve='convex', direction='decreasing')
        elbow_k = kneedle.elbow if kneedle.elbow else k_range[0]
        logger.info(f"Elbow method suggests k={elbow_k}")
    except Exception as e:
        logger.warning(f"Error in knee detection: {e}")
        elbow_k = k_range[0]

    # Find k with highest silhouette score
    if silhouette_scores:
        best_silhouette_k = k_range[np.argmax(silhouette_scores) + 1]
        logger.info(f"Best silhouette score at k={best_silhouette_k}")
    else:
        best_silhouette_k = elbow_k

    # Plot elbow curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(list(k_range), inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    if elbow_k:
        plt.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow at k={elbow_k}')
    plt.legend()

    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(list(k_range), silhouette_scores, 'go-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    if best_silhouette_k:
        plt.axvline(x=best_silhouette_k, color='r', linestyle='--', label=f'Best at k={best_silhouette_k}')
    plt.legend()
    plt.tight_layout()

    # Decide final k based on both methods
    optimal_k = best_silhouette_k if silhouette_scores else elbow_k
    logger.info(f"Selected optimal k={optimal_k}")

    return optimal_k, plt.gcf()

def perform_kmeans_clustering(embeddings, n_clusters):
    """Perform KMeans clustering with the optimal number of clusters."""
    logger.info(f"Performing KMeans clustering with k={n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans.cluster_centers_

def visualize_clusters(G, cluster_labels, node_pos=None, output_file=None):
    """Visualize the graph with nodes colored by cluster."""
    logger.info("Visualizing clusters")
    plt.figure(figsize=(12, 10))

    # Create node color map
    node_list = list(G.nodes())
    color_map = {}
    for i, node in enumerate(node_list):
        color_map[node] = cluster_labels[i]

    # Choose a color palette
    num_clusters = len(np.unique(cluster_labels))
    colors = sns.color_palette("husl", num_clusters)

    # If we don't have positions, use spring layout
    if node_pos is None:
        logger.info("Calculating node positions with spring layout")
        node_pos = nx.spring_layout(G, seed=42)

    # Draw nodes colored by cluster
    for cluster_id in range(num_clusters):
        nodelist = [node for node in G.nodes() if color_map[node] == cluster_id]
        nx.draw_networkx_nodes(G, node_pos,
                              nodelist=nodelist,
                              node_color=[colors[cluster_id]] * len(nodelist),
                              node_size=50,
                              alpha=0.8,
                              label=f"Cluster {cluster_id}")

    # Draw edges with low alpha for clarity
    nx.draw_networkx_edges(G, node_pos, alpha=0.2)

    plt.title(f"Graph Clustering with KMeans (k={num_clusters})")
    plt.legend(scatterpoints=1)
    plt.axis('off')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_file}")

    return plt.gcf()

def save_partition_for_louvain(G, cluster_labels, output_file):
    """Save the partition in a format suitable for Louvain algorithm."""
    logger.info(f"Saving partition for Louvain to {output_file}")
    node_list = list(G.nodes())
    partition = {node: int(cluster_labels[i]) for i, node in enumerate(node_list)}

    with open(output_file, 'w') as f:
        json.dump(partition, f)

    return partition

def save_partitioned_graphs(G, cluster_labels, output_dir):
    """Save individual subgraphs for each cluster."""
    logger.info("Saving partitioned graphs")
    node_list = list(G.nodes())

    # Group nodes by cluster
    clusters = defaultdict(list)
    for i, node in enumerate(node_list):
        clusters[int(cluster_labels[i])].append(node)

    # Create and save subgraphs
    for cluster_id, nodes in clusters.items():
        subgraph = G.subgraph(nodes)
        output_file = os.path.join(output_dir, f"cluster_{cluster_id}.gpickle")
        with open(output_file, 'wb') as f:
            pickle.dump(subgraph, f)
        logger.info(f"Saved cluster {cluster_id} with {len(nodes)} nodes to {output_file}")

    # Save a summary file
    summary = {
        'num_clusters': len(clusters),
        'cluster_sizes': {k: len(v) for k, v in clusters.items()},
        'total_nodes': len(node_list)
    }
    with open(os.path.join(output_dir, 'cluster_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    """Main function to execute the KMeans partitioning workflow."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the data
    start_time = time.time()
    data = load_graph(INPUT_FILE)

    # For your specific data structure: a dictionary with node IDs as keys and embeddings as values
    logger.info(f"Loaded embeddings for {len(data)} nodes")

    # Create a graph from the node IDs
    G = nx.Graph()
    G.add_nodes_from(data.keys())

    # Extract embeddings as a numpy array while maintaining node order
    nodes = list(data.keys())
    embeddings = np.array([data[node] for node in nodes])

    logger.info(f"Prepared graph with {G.number_of_nodes()} nodes")
    logger.info(f"Embedding matrix shape: {embeddings.shape}")

    # Find optimal k
    optimal_k, elbow_plot = find_optimal_k(embeddings)
    elbow_plot.savefig(os.path.join(OUTPUT_DIR, 'elbow_plot.png'), dpi=300, bbox_inches='tight')

    # Perform clustering
    clustering_start_time = time.time()
    cluster_labels, centers = perform_kmeans_clustering(embeddings, optimal_k)
    clustering_time = time.time() - clustering_start_time

    # Calculate node positions for visualization (using embeddings if 2D, otherwise spring layout)
    if embeddings.shape[1] == 2:
        node_pos = {node: embeddings[i] for i, node in enumerate(nodes)}
    else:
        node_pos = nx.spring_layout(G, seed=42)

    # Visualize clusters
    cluster_viz = visualize_clusters(G, cluster_labels, node_pos,
                                   os.path.join(OUTPUT_DIR, 'cluster_visualization.png'))

    # Save partition for Louvain algorithm
    partition = save_partition_for_louvain(G, cluster_labels,
                                          os.path.join(OUTPUT_DIR, 'partition.json'))

    # Save partitioned graphs
    save_partitioned_graphs(G, cluster_labels, OUTPUT_DIR)

    # Save raw cluster assignments
    np.save(os.path.join(OUTPUT_DIR, 'cluster_labels.npy'), cluster_labels)

    # Calculate modularity
    # First, add edges to the graph if not already present
    # For demonstration, we'll add edges between nodes that are close in embedding space
    if G.number_of_edges() == 0:
        logger.info("Graph has no edges. Adding edges based on embedding proximity")
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)
        for i in range(len(nodes)):
            for j in indices[i][1:]:  # Skip the first one (itself)
                G.add_edge(nodes[i], nodes[j])

    modularity = calculate_modularity(G, partition)

    # Calculate total runtime
    total_runtime = time.time() - start_time

    # No ground truth for AMI, so we'll set it to -1
    ami = -1

    # Prepare metrics dictionary
    metrics = {
        'algorithm': 'kmeans',
        'dataset': os.path.basename(INPUT_FILE),
        'num_clusters': optimal_k,
        'modularity': modularity,
        'ami': ami,
        'clustering_time': clustering_time,
        'total_runtime': total_runtime,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'date': time.strftime('%Y-%m-%d'),
        'timestamp': time.strftime('%H:%M:%S')
    }

    # Save metrics
    save_metrics(metrics, 'results/partitions_scores.csv')

    logger.info("KMeans partitioning completed successfully")
    logger.info(f"Metrics: Clusters={optimal_k}, Modularity={modularity:.4f}, Runtime={total_runtime:.2f}s")

if __name__ == "__main__":
    main()