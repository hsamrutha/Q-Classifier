# Q-Classifier
Quantum Machine Learning Classifier (PennyLane)

An end-to-end implementation of a Variational Quantum Classifier (VQC) using PennyLane with structured benchmarking and noise-aware simulation.

Overview

This project builds a hybrid quantum–classical binary classifier using:

- Angle encoding (RY rotations)

- Parameterized variational layers with CNOT entanglement

- Expectation measurement of Pauli-Z

- Cross-entropy loss with gradient-based optimization

= Random restarts for stability

- Noise Modeling

Training supports:

- Clean simulation (default.qubit)

- Depolarizing noise using default.mixed

- Configurable noise probability

- Benchmarking

- Performance is compared against:

- Support Vector Machine (RBF kernel)

- Multi-Layer Perceptron (16, 8 hidden units)

Each experiment logs:

- Training loss & test accuracy

- Confusion matrices

- Classification reports

- Decision boundary visualizations

- Additional Analysis

- Layer-depth sweep

- Noise sweep

- Circuit expressibility via pairwise fidelity statistics

- Feature sensitivity (interpretability sweeps)

Scope

Implements a reproducible hybrid QML pipeline with noise-aware VQC training and classical baselines for comparison.
