# 🧠 j-neural-core

A lightweight Java implementation of a multi-layer perceptron (MLP) neural network — from scratch — with autograd, training, and classification examples.

This project is meant for experimentation with neural networks without using external ML libraries.

## 📦 Features

* Fully custom autograd engine (Value class)
* Multi-Layer Perceptron (MLP) architecture
* Stochastic Gradient Descent (SGD) training loop
* Binary classification using ReLU activations in hidden layers, Sigmoid activation in output layer, and Cross-Entropy loss
* Digit classification (WIP)
* Clean Maven-based project structure

## 📁 Folder Structure

```bash
j-neural-core/
├── src/main/
│   ├── java/
│   │   ├── app/
│   │   │   └── Main.java               # Entry point to run both classifiers
│   │   ├── core/
│   │   │   ├── Layer.java
│   │   │   ├── MLP.java                # Neural network model
│   │   │   ├── Neuron.java
│   │   │   └── Value.java              # Autograd engine
│   │   └── models/
│   │       ├── BinaryClassifier.java   # Trains binary classifier with ReLU+Sigmoid+Cross-Entropy
│   │       └── DigitRecognizer.java    # (WIP) MNIST-style digit recognition
│   └── resources/data/                   # (WIP) Data for digit recognition
│       ├── digits_original.txt
│       ├── digits_test.csv
│       └── digits_train.csv
├── .gitignore
├── README.md
└── pom.xml
```

## 📊 Dataset

The **DigitRecognizer** task uses the *Optical Recognition of Handwritten Digits* dataset collected by **E. Alpaydin & C. Kaynak** at the Department of Computer Engineering, Boğaziçi University, Istanbul (July 1998).

Original images are 32×32 binary bitmaps. For this project, each image was pre‑processed into non‑overlapping 4×4 pixel blocks. For every block we count the number of active (value `1`) pixels, producing an **8×8 matrix whose elements are integers in the range 0‑16**. This reduces dimensionality and provides invariance to small local distortions while retaining the essential character of each digit. The two resulting CSVs are stored in `src/main/resources/data`.

## 🚀 Getting Started

### Prerequisites

- Java 17+
- Maven 3.x
- Git (optional)

### Clone the Repo

```bash
git clone https://github.com/yourusername/j-neural-core.git
cd j-neural-core
```

### Build and run the project (main training demo)

```bash
mvn compile
mvn exec:java -Dexec.mainClass="app.Main"
```

## 📚 How It Works

* The `Value` class implements a basic computational graph with automatic differentiation (backpropagation).
* `MLP`, `Layer`, and `Neuron` classes build and execute the neural network.
* Binary classifier uses:
  * ReLU activation in hidden layers
  * Sigmoid activation in the output layer.
  * Cross-Entropy loss optimized with stochastic gradient descent and backpropagation.
* Digit classifier is WIP.

## 📝 License

This project is open source and available under the MIT License.

---
Project is inspired by micrograd by Andrej Karpathy ([https://github.com/karpathy/micrograd](https://github.com/karpathy/micrograd))