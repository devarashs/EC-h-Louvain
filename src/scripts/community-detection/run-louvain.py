import os
import csv
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DATASET_PATH = "data/dataset.txt"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "pure_louvain_scores.csv")
LEADERBOARD_FILE = os.path.join(RESULTS_DIR, "leaderboard.csv")
VIS_DIR = "data/visualizations/pure_louvin"
EDGE_LIST_FILE = os.path.join(VIS_DIR, "projected_graph.edgelist")
GRAPHML_FILE = os.path.join(VIS_DIR, "projected_graph.graphml")

def file_exists_and_not_empty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0

def plot_communities(G, partition, out_path):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(
        G, pos, partition.keys(), node_size=80,
        cmap=cmap, node_color=list(partition.values())
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, format="png")
    plt.close()

def read_hyperedges(path):
    hyperedges = []
    with open(path, "r") as f:
        for line in f:
            nodes = [int(x) for x in line.strip().split(",") if x]
            if nodes:
                hyperedges.append(nodes)
    return hyperedges

def hypergraph_to_graph(hyperedges):
    G = nx.Graph()
    for hedge in hyperedges:
        for i in range(len(hedge)):
            for j in range(i + 1, len(hedge)):
                G.add_edge(hedge[i], hedge[j])
    return G

def save_graph_visualizations(G):
    os.makedirs(VIS_DIR, exist_ok=True)
    nx.write_edgelist(G, EDGE_LIST_FILE, data=False)
    nx.write_graphml(G, GRAPHML_FILE)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    hyperedges = read_hyperedges(DATASET_PATH)
    G = hypergraph_to_graph(hyperedges)
    save_graph_visualizations(G)
    partition = community_louvain.best_partition(G)
    modularity = community_louvain.modularity(partition, G)

    # Save community visualization as image
    os.makedirs(VIS_DIR, exist_ok=True)
    plot_communities(G, partition, os.path.join(VIS_DIR, "communities.png"))

    # Save to pure_louvain_scores.csv
    with open(RESULTS_FILE, mode='a', newline='') as csvfile:
        fieldnames = ["Modularity_qC"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists_and_not_empty(RESULTS_FILE):
            writer.writeheader()
        writer.writerow({"Modularity_qC": modularity})

    # Save to leaderboard.csv
    alphas_out = []
    alphas_config = []
    change_mode_config = "pure_louvain"
    change_frequency_config = "N/A"
    with open(LEADERBOARD_FILE, mode='a', newline='') as csvfile:
        fieldnames = ["Algorithm","isPartition","Nodes","Edges", "Modularity"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists_and_not_empty(LEADERBOARD_FILE):
            writer.writeheader()
        writer.writerow({
            "Algorithm": "pure_louvain",
            "isPartition": "-1",
            "Nodes": len(list(G.nodes())),
            "Edges": len(list(G.edges())),
            "Modularity": modularity
        })
    print(f"Results saved to {RESULTS_FILE} and {LEADERBOARD_FILE}")
    print(f"Graph visualizations saved to {VIS_DIR}")

if __name__ == "__main__":
    main()