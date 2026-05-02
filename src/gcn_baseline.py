import csv
import json
import torch
import torch.nn.functional as F
import sys
from datetime import datetime
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def get_predictions(model, data, mask):
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    return data.y[mask].cpu(), pred[mask].cpu()


def classification_metrics(y_true, y_pred, num_classes):
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for i in range(len(y_true)):
        confusion[int(y_true[i]), int(y_pred[i])] += 1

    class_rows = []
    f1_scores = []
    supports = []

    for class_id in range(num_classes):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum()) - tp
        fn = int(confusion[class_id, :].sum()) - tp
        support = int(confusion[class_id, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_rows.append(
            {
                "class": class_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        f1_scores.append(f1)
        supports.append(support)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    total_support = sum(supports)
    weighted_f1 = sum(f1_scores[i] * supports[i] for i in range(num_classes)) / total_support

    return confusion, class_rows, macro_f1, weighted_f1


def count_parameters(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    return total


def save_run_files(dataset, data, model, results, epochs, hidden_channels, lr, weight_decay):
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_time + "_" + dataset.name + "_GCN"
    run_dir = PROJECT_ROOT / "results" / "runs" / "gcn" / run_name
    plot_dir = PROJECT_ROOT / "results" / "plots" / "gcn" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # empty folders for the transformer results later
    (PROJECT_ROOT / "results" / "runs" / "graph_transformer").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results" / "plots" / "graph_transformer").mkdir(parents=True, exist_ok=True)

    dataset_table = [
        {
            "dataset": dataset.name,
            "nodes": data.num_nodes,
            "edges": data.num_edges,
            "avg_degree": data.num_edges / data.num_nodes,
            "density": data.num_edges / (data.num_nodes * (data.num_nodes - 1)),
            "features": dataset.num_node_features,
            "classes": dataset.num_classes,
            "train_nodes": int(data.train_mask.sum()),
            "val_nodes": int(data.val_mask.sum()),
            "test_nodes": int(data.test_mask.sum()),
        }
    ]

    main_results_table = [
        {
            "dataset": dataset.name,
            "model": "GCN",
            "train_acc": results["train_acc"],
            "val_acc": results["val_acc"],
            "test_acc": results["test_acc"],
            "macro_f1": results["macro_f1"],
            "weighted_f1": results["weighted_f1"],
        }
    ]

    model_table = [
        {
            "model": "GCN",
            "hidden_channels": hidden_channels,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "parameters": results["parameters"],
        }
    ]

    with open(run_dir / "dataset_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dataset_table[0].keys())
        writer.writeheader()
        writer.writerows(dataset_table)

    with open(run_dir / "main_results_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=main_results_table[0].keys())
        writer.writeheader()
        writer.writerows(main_results_table)

    with open(run_dir / "model_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=model_table[0].keys())
        writer.writeheader()
        writer.writerows(model_table)

    with open(run_dir / "epoch_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_accuracy"])
        writer.writeheader()
        for i in range(len(results["losses"])):
            writer.writerow(
                {
                    "epoch": i + 1,
                    "loss": results["losses"][i],
                    "val_accuracy": results["val_accuracies"][i],
                }
            )

    with open(run_dir / "per_class_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results["per_class_metrics"][0].keys())
        writer.writeheader()
        writer.writerows(results["per_class_metrics"])

    with open(run_dir / "confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results["confusion_matrix"])

    easy_results = {}
    for key, value in results.items():
        if key not in ["losses", "val_accuracies"]:
            easy_results[key] = value
    easy_results["epochs"] = epochs
    easy_results["hidden_channels"] = hidden_channels
    easy_results["learning_rate"] = lr
    easy_results["weight_decay"] = weight_decay

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(easy_results, f, indent=2)

    return str(run_dir), str(plot_dir)


def run_experiment(dataset_name="Cora", epochs=200):
    dataset, data = load_data(dataset_name)
    print_data_info(dataset, data)

    hidden_channels = 16
    lr = 0.01
    weight_decay = 5e-4

    model = GCN(
        num_features=dataset.num_node_features,
        hidden_channels=hidden_channels,
        num_classes=dataset.num_classes,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer)
        val_acc = accuracy(model, data, data.val_mask)

        losses.append(loss)
        val_accuracies.append(val_acc)

        if epoch == 1 or epoch % 20 == 0:
            print("Epoch:", epoch, "Loss:", round(loss, 4), "Val acc:", round(val_acc, 4))

    train_acc = accuracy(model, data, data.train_mask)
    val_acc = accuracy(model, data, data.val_mask)
    test_acc = accuracy(model, data, data.test_mask)
    y_true, y_pred = get_predictions(model, data, data.test_mask)
    confusion, per_class_metrics, macro_f1, weighted_f1 = classification_metrics(
        y_true,
        y_pred,
        dataset.num_classes,
    )

    results = {
        "losses": losses,
        "val_accuracies": val_accuracies,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion.tolist(),
        "parameters": count_parameters(model),
    }

    run_dir, plot_dir = save_run_files(
        dataset,
        data,
        model,
        results,
        epochs,
        hidden_channels,
        lr,
        weight_decay,
    )
    results["run_dir"] = run_dir
    results["plot_dir"] = plot_dir

    print("\nFinal results:")
    print("Train accuracy:", round(train_acc, 4))
    print("Validation accuracy:", round(val_acc, 4))
    print("Test accuracy:", round(test_acc, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("Saved run to:", run_dir)
    print("Plots go to:", plot_dir)

    return model, dataset, data, results


def run_gcn(dataset_name="Cora", epochs=200):
    model, dataset, data, results = run_experiment(dataset_name, epochs)
    return model, dataset, data


if __name__ == "__main__":
    dataset_name = "Cora"

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]

    # PyG wants this exact spelling
    if dataset_name.lower() == "citeseer":
        dataset_name = "CiteSeer"

    run_gcn(dataset_name)
