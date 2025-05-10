import os
import pickle
import networkx as nx
from node2vec import Node2Vec
import datetime

# Input and output paths as variables for easy modification
INPUT_GRAPH_PATH = "data/2-section/2_section_graph.pickle"
OUTPUT_DIR = "data/vectors/node2vec/"

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load the 2-section graph
    print(f"Loading 2-section graph from {INPUT_GRAPH_PATH}")
    with open(INPUT_GRAPH_PATH, 'rb') as f:
        graph = pickle.load(f)

    # Check if the graph is already a NetworkX graph, if not convert it
    if not isinstance(graph, nx.Graph):
        print("Converting data to NetworkX graph...")
        if isinstance(graph, dict):
            # If it's a dictionary of edges or adjacency list
            G = nx.Graph()
            for node, neighbors in graph.items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
            graph = G
        else:
            raise TypeError("Unsupported graph format in the pickle file")

    print(f"Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    # Configure and run Node2Vec
    print("Running Node2Vec on the graph...")
    # Parameters for Node2Vec (can be adjusted)
    node2vec = Node2Vec(
        graph,
        dimensions=128,    # Embedding dimension
        walk_length=30,    # Length of each random walk
        num_walks=200,     # Number of random walks per node
        workers=4          # Number of parallel workers
    )

    # Generate embeddings
    model = node2vec.fit(
        window=10,        # Context size for optimization
        min_count=1,      # Ignores all words with total frequency lower than this
        batch_words=4     # Number of words in each batch
    )

    # Save the model and embeddings
    model_output_path = os.path.join(OUTPUT_DIR, f"node2vec_model.model")
    embeddings_output_path = os.path.join(OUTPUT_DIR, f"node2vec_embeddings.pickle")

    print(f"Saving Node2Vec model to {model_output_path}")
    model.save(model_output_path)

    # Extract and save node embeddings
    print(f"Saving node embeddings to {embeddings_output_path}")
    embeddings = {str(node): model.wv[str(node)] for node in graph.nodes() if str(node) in model.wv}
    with open(embeddings_output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print("Node2Vec embedding process completed successfully!")

if __name__ == "__main__":
    main()