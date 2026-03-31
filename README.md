# Quantum Graph Neural Networks for High-Energy Physics

This project implements Quantum Machine Learning methods for high-energy physics analysis, specifically Quantum Graph Neural Networks (QGNNs) using PennyLane.

## Project Structure

**Test tasks and discussions are in `paper/`:**

- `task_discussion.typ` — Detailed analysis of each task
- `proposal_2.typ` — GSOC 2026 proposal
- `refs.bib` — Bibliography

**Source code is in `src/`:**

- `task2.py` — GCN and pMLP benchmarking on quark vs. gluon classification
- `task5.py` — Quantum GCN circuit implementation using PennyLane

## Quick Start

Set Up Environment Manager

linux or mac

```
curl -fsSL https://pixi.sh/install.sh | sh
```

windows powershell

```
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

Run Task 2 (classical baseline):

```bash
cd src && python task2.py
```

Run Task 5 (quantum circuit):

```bash
cd src && python task5.py
```

## Key Results

- **Task 2:** GCN achieves ~0.87 AUC on QG jets dataset (competitive with classical methods)
- **Task 5:** Demonstrates quantum graph convolution with PennyLane, including encoding, message passing, and readout

## Author

Benjamin Yu Sheng Chang | University of Toronto
