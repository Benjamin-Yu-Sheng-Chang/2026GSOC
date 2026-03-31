import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, PMLP
import matplotlib.pyplot as plt


def compute_roc_auc(labels, scores):
    """Compute ROC curve and AUC from labels and predicted scores (numpy)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    labels = labels[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    tp = 0
    fp = 0
    tprs = [0.0]
    fprs = [0.0]
    for l in labels:
        if l == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    auc = np.trapz(tprs, fprs)
    return fprs, tprs, auc


print("=" * 60)
print("Loading data")
print("=" * 60)
raw = np.load("../data/QG_jets.npz")
X = raw["X"]
y = raw["y"]


all_particles = X.reshape(-1, 4)
nonzero_mask = all_particles[:, 0] > 0
feat_mean = all_particles[nonzero_mask].mean(axis=0)
feat_std = all_particles[nonzero_mask].std(axis=0)
feat_std[feat_std < 1e-8] = 1.0


def jet_to_graph(particles, label, k=7):
    """Convert one jet (N, 4) into a PyG Data object.

    Nodes  = particles with non-zero pT.
    Edges  = k-nearest neighbours in (rapidity, phi) space.
    """
    mask = particles[:, 0] > 0
    feats = particles[mask]
    if len(feats) == 0:
        feats = particles[:1]

    coords = feats[:, 1:3]

    feats = (feats - feat_mean) / feat_std
    n = len(feats)

    k_actual = min(k, n - 1)
    if k_actual > 0:
        diff = coords[:, None, :] - coords[None, :, :]
        dist = (diff ** 2).sum(axis=-1)
        np.fill_diagonal(dist, np.inf)
        neighbours = np.argsort(dist, axis=1)[:, :k_actual]

        src = np.repeat(np.arange(n), k_actual)
        dst = neighbours.flatten()
        edge_index = np.stack([src, dst], axis=0)
        edge_index = np.concatenate([edge_index, edge_index[[1, 0]]], axis=1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return Data(
        x=torch.tensor(feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor([label], dtype=torch.long),
    )


print("=" * 60)
print("Building graphs")
print("=" * 60)
graphs = [jet_to_graph(X[i], y[i]) for i in range(len(X))]
print(f"Built {len(graphs)} graphs.  Example: {graphs[0]}")

rng = np.random.default_rng(42)
perm = rng.permutation(len(graphs))
n_test = len(graphs) // 5
n_val = len(graphs) // 10
test_graphs = [graphs[i] for i in perm[:n_test]]
val_graphs = [graphs[i] for i in perm[n_test:n_test + n_val]]
train_graphs = [graphs[i] for i in perm[n_test + n_val:]]

train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=256)
test_loader = DataLoader(test_graphs, batch_size=256)


class JetGCN(nn.Module):
    def __init__(self, in_dim=4, hidden=64, out_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.BatchNorm1d(hidden)
        )

        self.conv1 = GCNConv(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden, out_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.encoder(x)
        x = self.conv1(x, edge_index).relu()
        global_mean = global_mean_pool(x, batch)
        return self.classifier(global_mean)


class JetPMLP(nn.Module):
    def __init__(self, in_dim=4, hidden=64, out_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.BatchNorm1d(hidden)
        )
        self.mlp = PMLP(
            in_channels=hidden, hidden_channels=hidden,
            out_channels=hidden, num_layers=2, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden, out_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)
        x = self.mlp(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


def train_and_evaluate(model, name, train_loader, val_loader, test_loader, epochs=15, lr=1e-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        model.eval()
        correct, total = 0, 0
        all_labels, all_scores = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y.view(-1)).sum().item()
                total += batch.num_graphs
                probs = torch.softmax(out, dim=1)[:, 1]
                all_scores.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.view(-1).cpu().numpy())

        _, _, val_auc = compute_roc_auc(all_labels, all_scores)
        print(f"[{name}] Epoch {epoch:2d} | Loss {total_loss / len(train_graphs):.4f} | Val Acc {correct / total:.4f} | Val AUC {val_auc:.4f}")

    model.eval()
    correct, total = 0, 0
    all_labels, all_scores = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y.view(-1)).sum().item()
            total += batch.num_graphs
            probs = torch.softmax(out, dim=1)[:, 1]
            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())

    fprs, tprs, test_auc = compute_roc_auc(all_labels, all_scores)
    print(f"[{name}] Test Accuracy: {correct / total:.4f}")
    print(f"[{name}] Test AUC:      {test_auc:.4f}")
    return fprs, tprs, test_auc


print("\n=== Training GCN ===")
gcn_fprs, gcn_tprs, gcn_auc = train_and_evaluate(
    JetGCN(), "GCN", train_loader, val_loader, test_loader
)

print("\n=== Training pMLP ===")
mlp_fprs, mlp_tprs, mlp_auc = train_and_evaluate(
    JetPMLP(), "pMLP", train_loader, val_loader, test_loader
)

plt.figure()
plt.plot(gcn_fprs, gcn_tprs, label=f"GCN (AUC = {gcn_auc:.4f})")
plt.plot(mlp_fprs, mlp_tprs, label=f"pMLP (AUC = {mlp_auc:.4f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Quark vs Gluon")
plt.legend()
plt.tight_layout()
plt.savefig("../fig/task2_roc.png", dpi=150)
plt.show()

