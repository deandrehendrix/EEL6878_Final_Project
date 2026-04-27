import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def load_data(name="Cora"):
    dataset = Planetoid(
        root="data",
        name=name,
        transform=NormalizeFeatures(),
    )
    data = dataset[0]
    return dataset, data


def print_data_info(dataset, data):
    print("Dataset:", dataset.name)
    print("Nodes:", data.num_nodes)
    print("Edges:", data.num_edges)
    print("Features per node:", dataset.num_node_features)
    print("Classes:", dataset.num_classes)
    print("Train nodes:", int(data.train_mask.sum()))
    print("Validation nodes:", int(data.val_mask.sum()))
    print("Test nodes:", int(data.test_mask.sum()))


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


def accuracy(model, data, mask):
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())

    return acc


def run_gcn(epochs=200):
    dataset, data = load_data("Cora")
    print_data_info(dataset, data)

    model = GCN(
        num_features=dataset.num_node_features,
        hidden_channels=16,
        num_classes=dataset.num_classes,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer)

        if epoch == 1 or epoch % 20 == 0:
            print("Epoch:", epoch, "Loss:", round(loss, 4))

    print("\nFinal results:")
    print("Train accuracy:", round(accuracy(model, data, data.train_mask), 4))
    print("Validation accuracy:", round(accuracy(model, data, data.val_mask), 4))
    print("Test accuracy:", round(accuracy(model, data, data.test_mask), 4))

    return model, dataset, data


if __name__ == "__main__":
    run_gcn()
