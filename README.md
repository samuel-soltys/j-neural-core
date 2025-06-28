# JNeuralCore – Simple Java MLP Neural Network Framework with Autograd

This project implements a simple **Multilayer Perceptron (MLP)** in Java from scratch, featuring:

- A fully functional **autograd engine** for automatic differentiation
- A custom **neural network builder** (`MLP`, `Neuron`, `Layer`)
- Two example classifiers build on top of the neural network builder:
  - Binary classification
  - Digit recognition (WIP)

No machine learning libraries are used — everything is built from the ground up to demonstrate how neural nets work internally.

## Getting Started

### Prerequisites

- Java 17+
- Maven 3.x
- Git (optional)

### Clone the Repo

```bash
git clone https://github.com/yourusername/j-neural-core.git
cd j-neural-core
```

## Running the project

To run the main training demo:

```bash
mvn compile
mvn exec:java -Dexec.mainClass="app.Main"
```

## How It Works

- The Value class implements a basic computational graph with automatic differentiation (backpropagation).
- MLP, Layer, and Neuron classes build and execute the neural network.
- Binary classifier uses:
  - ReLU activation in hidden layers
  - Sigmoid activation in the output layer.
  - Cross-Entropy loss optimized with stochastic gradient descent and backpropagation.
- Digit classifier is WIP.

---
Project is inspired by [micrograd by Andrej Karpathy](https://github.com/karpathy/micrograd)