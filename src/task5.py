import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

n_qubits = 4
n_layers = 2
edges = [(0, 1), (1, 2), (3, 0)]

dev = qml.device("default.qubit", wires=n_qubits)


def encoding_layer(features):
    for i in range(n_qubits):
        qml.RX(features[i, 0], wires=i)
        qml.RZ(features[i, 1], wires=i)
    qml.Barrier()


def qgcn_layer(params, edges):
    for i in range(n_qubits):
        qml.RX(params[i, 0], wires=i)
        qml.RZ(params[i, 1], wires=i)


    for src, dst in edges:
        qml.CNOT(wires=[src, dst])
        qml.RZ(params[src, 3], wires=dst)
        qml.CNOT(wires=[src, dst])

    qml.Barrier()


@qml.qnode(dev)
def qgcn_circuit(features, params):
    encoding_layer(features)

    for l in range(n_layers):
        qgcn_layer(params[l], edges)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


np.random.seed(123)
features = np.random.uniform(0, np.pi, (n_qubits, 2))
params = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 4))

result = qgcn_circuit(features, params)
print("Node-level readout (PauliZ expectations):")
for i, val in enumerate(result):
    print(f"  qubit {i}: {val:.4f}")

graph_pred = np.mean(result)
print(f"\nGraph-level prediction (mean pool): {graph_pred:.4f}")

fig, ax = qml.draw_mpl(qgcn_circuit)(features, params)
plt.title("QGCN Circuit")
plt.savefig("../fig/task5_qgcn_circuit.png", dpi=150, bbox_inches="tight")
plt.show()
