# EC-h-Louvain Unified Pipeline Runner
# Run this script from the src directory in your activated conda environment

Write-Host "[1/7] Generating hypergraph..."
python -m scripts.hypergraph.generate_hypergraph

Write-Host "[2/7] Generating 2-section graph..."
python -m scripts.hypergraph.generate_2_section_graph

# Node2Vec pipeline
Write-Host "[3/7] Node2Vec pipeline: embeddings, kmeans, EC-Louvain"
python -m scripts.vector.create_Node2Vec_embeddings
python -m scripts.partition.node2vec_kmeans_partition
python -m scripts.community-detection.run-ec-louvain_node2vec

# DeepWalk pipeline
Write-Host "[4/7] DeepWalk pipeline: embeddings, kmeans, EC-Louvain"
python -m scripts.vector.create_deepwalk_embeddings
python -m scripts.partition.deepwalk_kmeans_partition
python -m scripts.community-detection.run-ec-louvain_deepwalk

# Hyper2Vec pipeline
Write-Host "[5/7] Hyper2Vec pipeline: embeddings, kmeans, EC-Louvain"
python -m scripts.vector.create_hyper2vec_embeddings
python -m scripts.partition.hyper2vec_kmeans_partition
python -m scripts.community-detection.run-ec-louvain_hyper2vec

# Pure h-louvain
Write-Host "[6/7] Pure h-louvain on hyperedges"
python -m scripts.community-detection.run-h-louvain

# Pure louvain
Write-Host "[7/7] Pure louvain on hyperedges"
python -m scripts.community-detection.run-louvain

Write-Host "All EC-h-Louvain pipelines completed!"