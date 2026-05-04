# EEL 6878 Final Project: GCN and Graph Transformer Comparison

**University of Central Florida** **Course:** EEL 6878: Modeling and Artificial Intelligence  
**Team Members:** DeAndre Hendrix & David Garzon

## Project Overview

This project compares graph neural network (GCN) and attention-based graph learning methods (Graph Transformer) for node classification on citation networks. We utilize the Cora and CiteSeer benchmark datasets, where papers are represented as nodes, citation links are represented as edges, and each paper is assigned a topic label.

The goal of this project is to evaluate whether the dynamic attention mechanism of a Graph Transformer can outperform the static message-passing scheme of a baseline Graph Convolutional Network on highly homophilous, small-scale datasets.

## Repository Structure

- `data/`: Contains the Cora and CiteSeer dataset files (auto-downloaded by PyG).
- `notebooks/`: Contains `final_results.ipynb`, an interactive Jupyter Notebook that runs the experiments, displays metric tables, and visualizes training curves and confusion matrices.
- `results/`: Contains the saved logs, metrics (CSV/JSON), and generated plots for all model runs.
- `src/`:
    - `gcn_baseline.py`: Contains the data loading utilities, training loops, and the baseline GCN model architecture.
    - `graph_transformer.py`: Contains the implementation of the multi-head self-attention Graph Transformer.

## Setup Instructions

To run this project locally, we recommend creating a Python virtual environment or Conda environment.

1. Clone the repository:
    ```bash
    git clone https://github.com/deandrehendrix/EEL6878_Final_Project
    cd EEL6878_Final_Project
    ```
