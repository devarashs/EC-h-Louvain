# Hybrid Hypergraph Modularity Optimization

## Overview

This repository implements a hybrid approach to hypergraph community detection by combining embedding-based clustering with H-Louvain algorithm. The goal is to improve modularity optimization in hypergraphs through a two-stage process.

## Methodology

### The Hybrid Approach

1. **Hypergraph Embedding**: Transform the hypergraph structure into a vector space representation
2. **Initial Clustering**: Apply clustering algorithms to the embeddings to identify initial communities
3. **H-Louvain Refinement**: Use the initial clusters as starting points for H-Louvain algorithm
4. **Performance Evaluation**: Compare results with standalone EC-Louvain and H-Louvain approaches

## Implementation Details

### Hypergraph Embedding

We implement various embedding techniques for hypergraphs, which capture the higher-order relationships between nodes.

### Clustering Methods

After obtaining node embeddings, we apply clustering algorithms to identify initial community structures.

### H-Louvain Initialization

Instead of random initialization, we use the clusters obtained from embedding-based methods as the starting point for H-Louvain.

## Expected Outcomes

- Improved modularity scores compared to standalone methods
- Better community detection in complex hypergraphs
- Potentially faster convergence of the H-Louvain algorithm

## Results and Analysis

This section will be updated with experimental results and performance comparisons.

## References

- EC-Louvain algorithm
- H-Louvain algorithm
- Hypergraph embedding techniques

## Baseline Repositories

The following repositories are used as baselines for our implementation:

- [H-Louvain](https://github.com/pawelwm/h-louvain.git) - Implementation of the Hypergraph Louvain algorithm
- [ECCD](https://github.com/bartoszpankratz/ECCD.git) - Edge-Centric Community Detection (EC-Louvain)
- [OpenNE](https://github.com/thunlp/OpenNE.git) - Open Network Embedding framework for node representation learning

Additional repositories may be incorporated as the project evolves. The implementation also uses standard Python and Julia libraries not listed here.

## Citations

This work is inspired by:

- **EC-Louvain/Leiden**: [Pankratz et al., 2024](https://doi.org/10.1093/comnet/cnae035)
- **h-Louvain (Hypergraph Modularity)**: [Kamiński et al., 2024](https://doi.org/10.1093/comnet/cnae041)
