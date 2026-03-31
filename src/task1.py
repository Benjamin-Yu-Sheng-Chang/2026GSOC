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
plt.savefig("task1_circuit.png", dpi=150, bbox_inches="tight")
plt.show()


swap_dev = qml.device("default.qubit", wires=5)

@qml.qnode(swap_dev)
def swap_test():
    qml.Hadamard(wires=1)
    qml.RX(np.pi / 3, wires=2)

    qml.Hadamard(wires=3)
    qml.Hadamard(wires=4)

    qml.Hadamard(wires=0)
    qml.CSWAP(wires=[0, 1, 3])
    qml.CSWAP(wires=[0, 2, 4])
    qml.Hadamard(wires=0)

    return qml.probs(wires=0)

probs = swap_test()
print(f"SWAP test probabilities: P(|0>) = {probs[0]:.4f}, P(|1>) = {probs[1]:.4f}")
print(f"Overlap |<q1q2|q3q4>|^2 = {2 * probs[0] - 1:.4f}")

fig, ax = qml.draw_mpl(swap_test)()
plt.title("SWAP Test Circuit")
plt.savefig("../fig/task1_swap_test.png", dpi=150, bbox_inches="tight")
plt.show()
