import networkx as nx
import community as community_louvain
import pickle
import csv
import json  # Add missing import
import time
import os
import datetime
from pathlib import Path

# Use absolute paths or make sure paths are relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_GRAPH_PATH = os.path.join(BASE_DIR, "data/2-section/2_section_graph.pickle")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "results/louvain_scores.csv")
ALGORITHM_NAME = "python_louvain-deepwalk"
INITIAL_PARTITION_PATH = None  # Set to a path if you want to use an initial partition


print(f"BASE_DIR: {BASE_DIR}")
print(f"INPUT_GRAPH_PATH: {INPUT_GRAPH_PATH}")
print(f"OUTPUT_CSV_PATH: {OUTPUT_CSV_PATH}")
print(f"ALGORITHM_NAME: {ALGORITHM_NAME}")
print(f"INITIAL_PARTITION_PATH: {INITIAL_PARTITION_PATH}")

# Rest of the code remains the same...

def main():
    # Start timing the total runtime
    total_start_time = time.time()

    # Load the graph from pickle file
    print(f"Loading graph from {INPUT_GRAPH_PATH}...")
    try:
        with open(INPUT_GRAPH_PATH, 'rb') as f:
            G = pickle.load(f)

        # Check if G is a NetworkX graph
        if not isinstance(G, nx.Graph):
            print("Warning: Loaded object is not a NetworkX graph. Attempting to convert...")
            G = nx.Graph(G)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    # Get graph statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Process initial partition if provided
    initial_partition = None
    initial_partition_name = "None"

    if INITIAL_PARTITION_PATH:
        try:
            with open(INITIAL_PARTITION_PATH, 'r') as f:
                initial_partition = json.load(f)
            initial_partition_name = os.path.basename(INITIAL_PARTITION_PATH)
        except Exception as e:
            print(f"Error loading initial partition: {e}")

    # Start timing the clustering process
    clustering_start_time = time.time()

    # Apply Louvain algorithm
    print("Running Louvain community detection...")
    partition = community_louvain.best_partition(G, partition=initial_partition)

    # Calculate metrics
    clustering_time = time.time() - clustering_start_time
    total_runtime = time.time() - total_start_time
    communities = set(partition.values())
    num_clusters = len(communities)
    modularity = community_louvain.modularity(partition, G)

    # Get dataset name (just the filename)
    dataset = os.path.basename(INPUT_GRAPH_PATH)

    # Generate timestamp
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.datetime.now().strftime('%H:%M:%S')

    # Prepare the data to write to CSV
    data = {
        'algorithm': ALGORITHM_NAME,
        'dataset': dataset,
        'num_clusters': num_clusters,
        'modularity': modularity,
        'clustering_time': clustering_time,
        'total_runtime': total_runtime,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'initial_partition': initial_partition_name,
        'date': current_date,
        'timestamp': current_time
    }

    # Check if the output file exists
    file_exists = os.path.isfile(OUTPUT_CSV_PATH)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    # Write data to CSV
    with open(OUTPUT_CSV_PATH, 'a', newline='') as csvfile:
        fieldnames = ['algorithm', 'dataset', 'num_clusters', 'modularity',
                      'clustering_time', 'total_runtime', 'num_nodes',
                      'num_edges', 'initial_partition', 'date', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


    # Add this after writing to the CSV file
    print(f"File exists after writing: {os.path.exists(OUTPUT_CSV_PATH)}")
    print(f"File size: {os.path.getsize(OUTPUT_CSV_PATH) if os.path.exists(OUTPUT_CSV_PATH) else 0} bytes")
    print(f"Results saved to {OUTPUT_CSV_PATH}")
    print(f"Found {num_clusters} communities with modularity {modularity:.4f}")
    print(f"Clustering time: {clustering_time:.4f} seconds")
    print(f"Total runtime: {total_runtime:.4f} seconds")

if __name__ == "__main__":
    main()