package engine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import core.MLP;
import core.Value;

public class Trainer {
    private final MLP model;
    
    public Trainer(MLP model) {
        this.model = model;
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
            // learning rate decay 
            // if (epoch > 40 && epoch % 20 == 0) {
            //     learningRate *= 0.8; // Decay learning rate every 20 epochs
            // }
            double totalLoss = 0.0;
            for (int i = 0; i < X.length; i++) {
                // Wrap inputs in Value objects
                List<Value> inputs = new ArrayList<>();
                for (double d : X[i]) {
                    inputs.add(new Value(d));
                }

                // Forward pass
                List<Value> outs = model.forward(inputs);
                
                Value loss;
                // If binary classification, calculate binary cross-entropy loss
                // Assuming outs is a single output neuron with sigmoid activation
                if (outs.size() == 1) {
                    Value prediction = outs.get(0);

                    // Calculating binary cross-entropy loss
                    Value groundTruth = new Value(y[i]);
                    loss = groundTruth.mul(prediction.log()).add(
                        new Value(1.0).add(groundTruth.mul(new Value(-1.0))).mul(
                            new Value(1.0).add(prediction.mul(new Value(-1.0))).log()
                        )
                    ).mul(new Value(-1.0));
                    totalLoss += loss.data;
                // Else multi-class classification, calculate categorical cross-entropy loss
                // Assuming outs is softmaxed already and contains probabilities for each class
                } else {
                    // Takes the prediction for the class number corresponding to y[i] (the ground truth label)
                    Value prediction = outs.get(y[i]);

                    // Cross-entropy loss: -log(p_correct_class)
                    loss = prediction.log().mul(new Value(-1.0));
                    totalLoss += loss.data;
                }
                
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
    public double test(double[][] X, int[] y) {
        double accuracy = 0.0;
        int correct = 0;
        for (int i = 0; i < X.length; i++) {
            List<Value> inputs = new ArrayList<>();
            for (double d : X[i]) {
                inputs.add(new Value(d));
            }
            List<Value> out = model.forward(inputs);
            System.out.println("Input: " + Arrays.toString(X[i]) + "-> Prediction: " + out + ", Ground Truth: " + y[i]);
            
            int predicted;
            // If binary classification, threshold at 0.5
            if (out.size() == 1) {
                predicted = out.get(0).data >= 0.5 ? 1 : 0;
            // Else Multi-class: pick the class with the highest probability
            } else {
                double max = Double.NEGATIVE_INFINITY;
                int maxIdx = -1;
                for (int j = 0; j < out.size(); j++) {
                    if (out.get(j).data > max) {
                        max = out.get(j).data;
                        maxIdx = j;
                    }
                }
                predicted = maxIdx;
            }
            if (predicted == y[i]) {
                correct++;
            }
        }
        
        accuracy = (double) correct / X.length;
        return accuracy;   
    }
}
