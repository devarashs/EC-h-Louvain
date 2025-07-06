#!/usr/bin/env python3
"""
K-means clustering using Hyper2Vec embeddings.

This script loads the Hyper2Vec embeddings and performs K-means clustering
to create initial partitions for the EC-Louvain algorithm.
"""

import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Input and output paths
INPUT_EMBEDDINGS_PATH = "data/vectors/hyper2vec/hyper2vec_embeddings.pickle"
OUTPUT_DIR = "data/partitions/hyper2vec_kmeans/"

def load_embeddings(file_path):
    """Load embeddings from pickle file."""
    print(f"Loading embeddings from {file_path}")
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)

    print(f"Loaded embeddings for {len(embeddings)} nodes")
    return embeddings

def prepare_data(embeddings):
    """Convert embeddings dict to numpy arrays for sklearn."""
    nodes = list(embeddings.keys())
    vectors = np.array([embeddings[node] for node in nodes])

    print(f"Prepared data matrix: {vectors.shape}")
    return nodes, vectors

def find_optimal_k(vectors, max_k=20, min_k=2):
    """
    Find optimal number of clusters using elbow method and silhouette analysis.

    Args:
        vectors: Node embedding vectors
        max_k: Maximum number of clusters to try
        min_k: Minimum number of clusters to try

    Returns:
        Optimal k value
    """
    print(f"Finding optimal number of clusters (k={min_k} to {max_k})...")

    # Adjust max_k if we have fewer samples
    max_k = min(max_k, len(vectors) - 1)

    k_range = range(min_k, max_k + 1)
    inertias = []
    silhouette_scores = []

    for k in k_range:
        print(f"  Testing k={k}...")

        # Fit K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors)

        # Calculate metrics
        inertia = kmeans.inertia_
        if k > 1:  # Silhouette score requires at least 2 clusters
            sil_score = silhouette_score(vectors, cluster_labels)
        else:
            sil_score = 0

        inertias.append(inertia)
        silhouette_scores.append(sil_score)

        print(f"    Inertia: {inertia:.2f}, Silhouette: {sil_score:.3f}")

    # Find optimal k using silhouette score
    best_k_idx = np.argmax(silhouette_scores)
    optimal_k = k_range[best_k_idx]

    print(f"Optimal k based on silhouette score: {optimal_k}")
    print(f"Best silhouette score: {silhouette_scores[best_k_idx]:.3f}")

    return optimal_k, k_range, inertias, silhouette_scores

def plot_clustering_metrics(k_range, inertias, silhouette_scores, optimal_k, output_dir):
    """Plot elbow curve and silhouette scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Elbow curve
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax1.legend()
    ax1.grid(True)

    # Silhouette scores
    ax2.plot(k_range, silhouette_scores, 'go-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'clustering_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved clustering metrics plot to {plot_path}")

def perform_kmeans_clustering(nodes, vectors, k):
    """
    Perform K-means clustering with the specified number of clusters.

    Args:
        nodes: List of node IDs
        vectors: Node embedding vectors
        k: Number of clusters

    Returns:
        Dictionary mapping node IDs to cluster labels
    """
    print(f"Performing K-means clustering with k={k}...")

    # Fit K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)

    # Create node to cluster mapping
    node_to_cluster = {nodes[i]: int(cluster_labels[i]) for i in range(len(nodes))}

    # Calculate clustering metrics
    inertia = kmeans.inertia_
    if k > 1:
        sil_score = silhouette_score(vectors, cluster_labels)
    else:
        sil_score = 0

    print(f"Clustering completed:")
    print(f"  Inertia: {inertia:.2f}")
    print(f"  Silhouette Score: {sil_score:.3f}")

    # Print cluster size distribution
    cluster_sizes = {}
    for cluster_id in cluster_labels:
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    print("Cluster size distribution:")
    for cluster_id, size in sorted(cluster_sizes.items()):
        print(f"  Cluster {cluster_id}: {size} nodes")

    return node_to_cluster, {
        'inertia': inertia,
        'silhouette_score': sil_score,
        'cluster_sizes': cluster_sizes,
        'num_clusters': k
    }

def save_partition(node_to_cluster, metadata, output_dir):
    """Save the partition results."""
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save partition mapping
    partition_file = os.path.join(output_dir, f"hyper2vec_kmeans_partition_{timestamp}.json")
    with open(partition_file, 'w') as f:
        json.dump(node_to_cluster, f, indent=2)


    # Convert cluster_sizes keys to int (for JSON compatibility)
    if 'cluster_sizes' in metadata:
        metadata['cluster_sizes'] = {int(k): int(v) for k, v in metadata['cluster_sizes'].items()}

    metadata_file = os.path.join(output_dir, f"hyper2vec_kmeans_metadata_{timestamp}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved partition to {partition_file}")
    print(f"Saved metadata to {metadata_file}")

    return partition_file, metadata_file

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load embeddings
    try:
        embeddings = load_embeddings(INPUT_EMBEDDINGS_PATH)
    except FileNotFoundError:
        print(f"Error: Embeddings file {INPUT_EMBEDDINGS_PATH} not found.")
        print("Please run the Hyper2Vec embedding script first.")
        return

    # Prepare data
    nodes, vectors = prepare_data(embeddings)

    # Find optimal number of clusters
    optimal_k, k_range, inertias, silhouette_scores = find_optimal_k(vectors)

    # Plot clustering metrics
    plot_clustering_metrics(k_range, inertias, silhouette_scores, optimal_k, OUTPUT_DIR)

    # Perform clustering with optimal k
    node_to_cluster, metadata = perform_kmeans_clustering(nodes, vectors, optimal_k)

    # Add additional metadata
    metadata['method'] = 'hyper2vec_kmeans'
    metadata['embedding_method'] = 'hyper2vec'
    metadata['clustering_method'] = 'kmeans'
    metadata['optimal_k'] = optimal_k
    metadata['total_nodes'] = len(nodes)
    metadata['embedding_dimension'] = vectors.shape[1]

    # Save results
    partition_file, metadata_file = save_partition(node_to_cluster, metadata, OUTPUT_DIR)

    print("\nHyper2Vec K-means clustering completed successfully!")
    print(f"Created {optimal_k} clusters from {len(nodes)} nodes")

if __name__ == "__main__":
    main()
