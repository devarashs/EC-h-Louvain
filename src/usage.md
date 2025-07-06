# EC-h-Louvain Usage Guide

This guide explains how to set up your environment and run the EC-h-Louvain pipeline.

## 1. Create and Activate Conda Environment

```bash
conda create -n cde python=3.12.9
conda activate cde
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

### 4.1 Hyper2Vec Pipeline (Recommended for Hypergraphs)

**Why Hyper2Vec?** Unlike Node2Vec and DeepWalk which work on 2-section graphs and lose hypergraph structure, Hyper2Vec performs biased random walks directly on the hypergraph, preserving higher-order relationships and typically achieving better modularity.

Execute the following scripts in order to run **Hyper2Vec -> kmeans -> EC-Louvain**:

```bash
python -m scripts.hypergraph.generate_hypergraph
python -m scripts.vector.create_hyper2vec_embeddings
python -m scripts.partition.hyper2vec_kmeans_partition
python -m scripts.community-detection.run-ec-louvain_hyper2vec
```

### 4.2 Alternative Embedding Methods

Execute the following scripts in order to run node2vec -> kmeans -> EC-Louvain:

```bash
python -m scripts.hypergraph.generate_hypergraph
python -m scripts.hypergraph.generate_2_section_graph
python -m scripts.vector.create_Node2Vec_embeddings
python -m scripts.partition.node2vec_kmeans_partition
python -m scripts.community-detection.run-ec-louvain_node2vec
```

Execute the following scripts in order to run deepwalk -> kmeans -> EC-Louvain:

```bash
python -m scripts.hypergraph.generate_hypergraph
python -m scripts.hypergraph.generate_2_section_graph
python -m scripts.vector.create_deepwalk_embeddings
python -m scripts.partition.deepwalk_kmeans_partition
python -m scripts.community-detection.run-ec-louvain_deepwalk
```

Execute the following scripts in order to run **Hyper2Vec -> kmeans -> EC-Louvain** (RECOMMENDED for hypergraphs):

```bash
python -m scripts.hypergraph.generate_hypergraph
python -m scripts.vector.create_hyper2vec_embeddings
python -m scripts.partition.hyper2vec_kmeans_partition
python -m scripts.community-detection.run-ec-louvain_hyper2vec
```

### 4.3 Pure Algorithm Comparisons

Execute the following scripts in order to run pure h-louvain on hyperedges:

```bash
python -m scripts.community-detection.run-h-louvain

```

Execute the following scripts in order to run pure louvain on hyperedges:

```bash
python -m scripts.community-detection.run-louvain

```

Each script performs a step in the EC-h-Louvain workflow.

---

Feel free to modify `dataset.txt` and experiment with your own data.
