import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev)
def circuit():

    for i in range(5):
        qml.Hadamard(wires=i)
    for i in range(4):
        qml.CNOT(wires=[i, i + 1])
    qml.SWAP(wires=[0, 4])
    qml.RX(np.pi / 2, wires=0)

    return qml.state()

fig, ax = qml.draw_mpl(circuit)()
plt.title("Task 1 Circuit")
plt.savefig("src/task1_circuit.png", dpi=150, bbox_inches="tight")
plt.show()
