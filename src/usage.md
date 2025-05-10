# EC-h-Louvain Usage Guide

This guide explains how to set up your environment and run the EC-h-Louvain pipeline.

## 1. Create and Activate Conda Environment

```bash
conda create -n echlouvain python=3.9
conda activate echlouvain
```

## 2. Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

## 3. Prepare the Dataset

- The file `dataset.txt` is provided.
- You may replace its contents with your own hyperedge dataset as needed.

## 4. Run the Pipeline

Execute the following scripts in order:

```bash
python -m scripts.hypergraph.generate_hypergraph
python -m scripts.hypergraph.generate_2_section_graph
python -m scripts.vector.create_Node2Vec_embeddings
python -m scripts.partition.create_kmeans_partition
python -m scripts.community-detection.run-louvain
python -m scripts.community-detection.run-ec-louvain
```

Each script performs a step in the EC-h-Louvain workflow.

---

Feel free to modify `dataset.txt` and experiment with your own data.
