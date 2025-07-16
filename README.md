# 🧠 j-neural-core

A lightweight Java implementation of a multi-layer perceptron (MLP) neural network — from scratch — with autograd, training, and examples of classification models built on top of the MLP class.

This project is meant for experimentation with neural networks without using any external ML libraries.

## 📦 Features

* Fully custom autograd engine (Value class)
* Multi-Layer Perceptron (MLP) architecture
* Stochastic Gradient Descent (SGD) training loop
* **Automatic switching** between:
  - Binary classification: Sigmoid activation + Binary Cross-Entropy loss
  - Multiclass classification: Softmax activation + Categorical Cross-Entropy loss  
* Binary classification example with simple data
* Digit classification using processed handwritten digit dataset
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
│   │   ├── engine/
│   │   │   └── Trainer.java            # Trainer class with activation/loss switching logic
│   │   ├── data/
│   │   │   └── DigitDataLoader.java    # Loads and parses digit dataset
│   │   └── models/
│   │       ├── BinaryClassifier.java   # Binary classification using sigmoid + BCE
│   │       └── DigitRecognizer.java    # Multiclass digit recognition using softmax + CE
│   └── resources/data/
│       ├── digits_original.txt
│       ├── digits_test.csv
│       └── digits_train.csv
├── .gitignore
├── README.md
└── pom.xml
```

## 📊 Dataset

The **DigitRecognizer** model uses the *Optical Recognition of Handwritten Digits* dataset collected by:

**E. Alpaydin, C. Kaynak**  
Department of Computer Engineering  
Boğaziçi University, Istanbul, Turkey  
(July 1998)

Original images are 32×32 bitmaps. For this project, they were **preprocessed into 8×8 matrices** by dividing into 4×4 non-overlapping blocks and **counting the number of active pixels (value 1) in each block**. This produces inputs with integer values between 0–16, reducing dimensionality while preserving useful information and adding robustness to small distortions.

The training and testing CSV files are located in `src/main/resources/data/`.

## 🚀 Getting Started

### Prerequisites

- Java 17+
- Maven 3.x
- Git (optional)

### Clone the Repo

```bash
git clone http://github.com/samuel-soltys/j-neural-core
cd j-neural-core
```

### Build and run the project (main training demo)

```bash
mvn compile
mvn exec:java -Dexec.mainClass="app.Main"
```

Alternatively, you can run the individual models directly by setting their class as the entry point:

```bash
mvn exec:java -Dexec.mainClass="models.BinaryClassifier"
mvn exec:java -Dexec.mainClass="models.DigitRecognizer"
```

## 🧪 Classification Logic

* **Auto-switching logic** is based on the number of output neurons:
  - If output layer has **1 neuron**, use Sigmoid + Binary Cross-Entropy loss.
  - If output layer has **>1 neurons**, use Softmax + Categorical Cross-Entropy loss.
* Models:
  - `BinaryClassifier`: Predicts binary labels using hidden ReLU layers and sigmoid output.
  - `DigitRecognizer`: Predicts digits (0–9) using softmax output with 10 neurons.

## 📝 License

This project is open source and available under the MIT License.

---

Project inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy