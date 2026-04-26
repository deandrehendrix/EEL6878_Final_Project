from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_dataset(name="Cora"):
    dataset = Planetoid(
        root="data",
        name=name,
        transform=NormalizeFeatures(),
    )
    data = dataset[0]

    return dataset, data

def print_dataset_info(dataset, data):
    print("Dataset:", dataset.name)
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Number of node features:", dataset.num_node_features)
    print("Number of classes:", dataset.num_classes)
    print("Training nodes:", int(data.train_mask.sum()))
    print("Validation nodes:", int(data.val_mask.sum()))
    print("Testing nodes:", int(data.test_mask.sum()))

if __name__ == "__main__":
    dataset, data = load_dataset("Cora")
    print_dataset_info(dataset, data)
