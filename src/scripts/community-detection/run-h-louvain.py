import sys
sys.path.append("h-louvain")  # Adjust this if your repo is elsewhere
from h_louvain import load_ABCDH_from_file, hLouvain
import os
import csv
import matplotlib.pyplot as plt
import hypernetx as hnx # For visualization

def main():

    dataset_file_path = "data/dataset.txt"

    # Create directories if they don't exist
    results_dir = "results"
    visualizations_dir = "data/visualizations/h_louvain_pure"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    print(f"Results will be saved in: {os.path.abspath(results_dir)}")
    print(f"Visualizations will be saved in: {os.path.abspath(visualizations_dir)}")

    print(f"Loading hypergraph from: {dataset_file_path}")
    try:
        HG = load_ABCDH_from_file(dataset_file_path)
        print("Hypergraph loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_file_path}")
        print("Please ensure the file path is correct and the script is run from the intended directory.")
        return
    except Exception as e:
        print(f"Error loading hypergraph: {e}")
        return

    print("Initializing hLouvain...")
    # Initialize hLouvain with the loaded hypergraph.
    # Using hmod_tau="infinity" as in your example.
    hL = hLouvain(HG, hmod_tau="infinity")
    print("hLouvain initialized.")

    print("Running h-Louvain community detection...")
    # Parameters for h_louvain_community method, as in your example.
    alphas_config = [0.5, 1, 0.5, 0.7, 0.8, 1]
    change_mode_config = "communities"
    change_frequency_config = 0.5

    try:
        A, q2, alphas_out = hL.h_louvain_community(
            alphas=alphas_config,
            change_mode=change_mode_config,
            change_frequency=change_frequency_config
        )
        print("Community detection finished.")

        print("\nFINAL ANSWER:")
        print("Partition (A):", A)
        print("Modularity (qC):", q2)
        print("Output Alphas:", alphas_out)

        # --- Record measurements to CSV ---
        results_file_path = os.path.join(results_dir, "hLouvain_scores.csv")
        # Check if file exists and is not empty to determine if headers are needed
        file_exists_and_not_empty = os.path.isfile(results_file_path) and os.path.getsize(results_file_path) > 0

        with open(results_file_path, mode='a', newline='') as csvfile:
            fieldnames = [ "Modularity_qC", "Output_Alphas", "Alphas_Config", "Change_Mode", "Change_Frequency"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists_and_not_empty:
                writer.writeheader()

            writer.writerow({
                "Modularity_qC": q2,
                "Output_Alphas": str(alphas_out),
                "Alphas_Config": str(alphas_config),
                "Change_Mode": change_mode_config,
                "Change_Frequency": change_frequency_config
            })
        print(f"Results saved to {results_file_path}")

        # --- Create visualizations ---
        print("Attempting to create visualizations...")
        print("Type of hL.HG:", type(hL.HG))
        if isinstance(hL.HG, dict):
            sample = list(hL.HG.items())[:3]
            print("Sample edges from hL.HG:", sample)
        else:
            print("hL.HG is not a dict. Its value:", hL.HG)
        try:
            # hL.HG should contain the hypergraph data loaded by load_ABCDH_from_file.
            # This typically is a dictionary mapping hyperedge IDs to sets/frozensets of nodes.
            # e.g., {'edge1': {'nodeA', 'nodeB'}, 'edge2': {'nodeB', 'nodeC', 'nodeD'}}
            if not hL.HG:
                 print("Hypergraph data (hL.HG) is empty or not found. Skipping visualization.")
            else:
                # Ensure hL.HG is in a format hypernetx.Hypergraph can consume.
                # If hL.HG is not a dict of iterables, it might need conversion.
                H_viz = hL.HG

                # Generate a layout for the hypergraph visualization
                # Kamada-Kawai layout is often good for network-like structures.
                # This can be computationally intensive for very large hypergraphs.
                print("Generating layout for visualization (this may take a moment for large graphs)...")
                pos = None
                print("Layout generated.")

                # 1. Visualize the original hypergraph structure
                plt.figure(figsize=(12, 10))
                hnx.draw(H_viz, pos=pos, with_node_labels=True, with_edge_labels=False)
                plt.title("Hypergraph Structure")
                hypergraph_plot_path = os.path.join(visualizations_dir, "hypergraph_structure.png")
                plt.savefig(hypergraph_plot_path)
                plt.close()
                print(f"Hypergraph structure visualization saved to {hypergraph_plot_path}")

                # 2. Visualize communities on the hypergraph
                if A: # Check if partition A is not empty
                    plt.figure(figsize=(12, 10))

                    nodes_in_viz = list(H_viz.nodes())
                    node_to_community_id = {node: A.get(node) for node in nodes_in_viz}

                    # Get unique communities to generate a color for each
                    # Filter out None if A.get() returns None for nodes not in partition
                    valid_community_ids = sorted(list(set(cid for cid in node_to_community_id.values() if cid is not None)))

                    if not valid_community_ids:
                        node_colors = ['lightgrey'] * len(nodes_in_viz) # All nodes unassigned or single color
                    else:
                        cmap = plt.get_cmap('viridis', len(valid_community_ids))
                        community_to_color = {cid: cmap(i) for i, cid in enumerate(valid_community_ids)}
                        node_colors = [community_to_color.get(node_to_community_id[node], 'lightgrey') for node in nodes_in_viz]

                    hnx.draw(H_viz,
                             pos=pos, # Use the same layout for consistency
                             with_node_labels=True,
                             with_edge_labels=False,
                             node_size=10, font_size_nodes=8,
                             node_color=node_colors # Pass the list of colors for nodes
                             )
                    plt.title("Hypergraph Communities")
                    communities_plot_path = os.path.join(visualizations_dir, "hypergraph_communities.png")
                    plt.savefig(communities_plot_path)
                    plt.close()
                    print(f"Communities visualization saved to {communities_plot_path}")
                else:
                    print("Partition A is empty. Skipping community visualization.")

        except ImportError:
            print("Visualization libraries not found. Please install hypernetx and matplotlib: pip install hypernetx matplotlib")
        except Exception as ve:
            print(f"Error during visualization: {ve}")
            print("Visualization skipped. Ensure hypergraph data hL.HG is compatible with hypernetx.Hypergraph().")
            print("hL.HG should ideally be a dictionary like: {edge_id: {node1, node2, ...}}")
            import traceback
            traceback.print_exc()


    except Exception as e:
        print(f"Error during h-Louvain community detection or subsequent processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()