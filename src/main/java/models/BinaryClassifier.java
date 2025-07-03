package models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import core.MLP;
import core.Value;

public class BinaryClassifier {
    private final MLP model;
    
    public BinaryClassifier(int[] layers) {
        this.model = new MLP(layers);
    }

    public int getModelParametersCount() {
        return model.parameters().size();
    }
    
    /**
     * Train the binary classifier using Stochastic Gradient Descent.
     * 
     * @param X Input features, a 2D array where each row is a sample.
     * @param y Labels, a 1D array where each element corresponds to the label of the sample in X.
     * @param learningRate Learning rate for the optimizer.
     * @param epochs Number of training epochs.
     */
    public void train(double[][] X, int[] y, double learningRate, int epochs) {
        if (X.length != y.length) {
            throw new IllegalArgumentException("Input and output arrays must have the same length.");
        }
        if (X.length == 0 || X[0].length == 0) {
            throw new IllegalArgumentException("Input array must not be empty.");
        }
        if (y.length == 0) {
            throw new IllegalArgumentException("Output array must not be empty.");
        }        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            for (int i = 0; i < X.length; i++) {
                // Wrap inputs in Value objects
                List<Value> inputs = new ArrayList<>();
                for (double d : X[i]) {
                    inputs.add(new Value(d));
                }

                // Forward pass
                List<Value> outs = model.forward(inputs);
                Value prediction = outs.get(0);
                
                // Calculating cross-entropy loss 
                Value groundTruth = new Value(y[i]);
                Value loss = groundTruth.mul(prediction.log()).add(
                    new Value(1.0).add(groundTruth.mul(new Value(-1.0))).mul(
                        new Value(1.0).add(prediction.mul(new Value(-1.0))).log()
                    )
                ).mul(new Value(-1.0));
                totalLoss += loss.data;

                // Backward pass
                model.zeroGrad();
                loss.backward();

                // Update parameters with Stochastic Gradient Descent
                for (Value param : model.parameters()) {
                    param.data -= learningRate * param.grad;
                }
            }
            if (epoch % 10 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d, Loss: %.8f%n", epoch, totalLoss);
            }
        }
    }

    /**
     * Test the trained model on new data.
     * 
     * @param X Input features, a 2D array where each row is a sample.
     * @param y Labels, a 1D array where each element corresponds to the label of the sample in X.
     */
    public void test(double[][] X, int[] y) {
        for (int i = 0; i < X.length; i++) {
            List<Value> inputs = new ArrayList<>();
            for (double d : X[i]) {
                inputs.add(new Value(d));
            }
            List<Value> out = model.forward(inputs);
            System.out.printf("Input: %s -> Prediction: %.3f, Ground Truth: %d%n",
                Arrays.toString(X[i]), out.get(0).data, y[i]);
        }
    }
}
