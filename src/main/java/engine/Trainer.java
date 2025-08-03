package engine;

import java.util.ArrayList;
// import java.util.Arrays;
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
     * @param X_val Validation input features (optional, can be null)
     * @param y_val Validation labels (optional, can be null)
     * @param learningRateDecay Whether to apply learning rate decay.
     * @param decayStartEpoch Epoch to start applying learning rate decay.
     * @param decayEvery How often to apply learning rate decay (every n epochs).
     */
    public void train(double[][] X, int[] y, double learningRate, int epochs, double[][] X_val, int[] y_val, boolean learningRateDecay, int decayStartEpoch, int decayEvery) {
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
            double totalValLoss = 0.0;
            // Learning rate decay 
            if (learningRateDecay && epoch > decayStartEpoch && epoch % decayEvery == 0) {
                learningRate *= 0.9; 
            }
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
                    Value groundTruth = new Value(y[i]);
                    loss = binaryCE(groundTruth, prediction);
                    totalLoss += loss.data;
                // Else multi-class classification, calculate categorical cross-entropy loss
                // Assuming outs is softmaxed already and contains probabilities for each class
                } else {
                    // Takes the prediction for the class number corresponding to y[i] (the ground truth label)
                    Value prediction = outs.get(y[i]);
                    loss = prediction.log().mul(new Value(-1.0));
                    totalLoss += loss.data;
                }
                
                loss = l2Regularization(loss);
                
                // Backward pass
                model.zeroGrad();
                loss.backward();

                // Update parameters with Stochastic Gradient Descent
                for (Value param : model.parameters()) {
                    param.data -= learningRate * param.grad;
                }
            }
            // If validation dataset provided
            if (X_val != null) {
                // Calculate validation loss
                for (int i = 0; i < X_val.length; i++) {
                    Value valLoss;
                    List<Value> valInputs = new ArrayList<>();
                    for (double d : X_val[i]) {
                        valInputs.add(new Value(d));
                    }

                    List<Value> valOuts = model.forward(valInputs);

                    if (valOuts.size() == 1) {
                        // Binary classification loss
                        Value valPrediction = valOuts.get(0);
                        Value groundTruth = new Value(y_val[i]);
                        valLoss = binaryCE(groundTruth, valPrediction);
                    } else {
                        // Multi-class loss
                        Value valPrediction = valOuts.get(y_val[i]);
                        valLoss = valPrediction.log().mul(new Value(-1.0));
                    }
                    totalValLoss += valLoss.data;
                }
            }

            if (epoch % 1 == 0 || epoch == epochs - 1) {
                System.out.println("-------------------");
                System.out.println("Epoch " + epoch + ":");
                System.out.printf("Training loss: %.8f%n", totalLoss);
                double avgTrainLoss = totalLoss / X.length;
                System.out.printf("avgTrainLoss: %.8f%n", avgTrainLoss);
                
                if (X_val != null) {
                    System.out.printf("Validation Loss: %.8f%n", totalValLoss);
                    double avgValLoss = totalValLoss / X_val.length;
                    System.out.printf("avgValLoss: %.8f%n", avgValLoss);
                }

            }
            // Early stopping: check if 'Q' is pressed
            try {
                if (System.in.available() > 0) {
                    int ch = System.in.read();
                    if (ch == 'Q' || ch == 'q') {
                        System.out.println("Early stopping triggered by user.");
                        break;
                    }
                }
            } catch (Exception e) {
                // Ignore input errors
            }
        }
    }
    // Binary Cross Entropy helper function
    private Value binaryCE(Value groundTruth, Value prediction) {
        return groundTruth.mul(prediction.log()).add(
            new Value(1.0).add(groundTruth.mul(new Value(-1.0))).mul(
                new Value(1.0).add(prediction.mul(new Value(-1.0))).log()
            )
        ).mul(new Value(-1.0));
    }
    // L2 Regularization helper function
    private Value l2Regularization(Value loss) {
        double lambda = 1e-4;
        Value l2Penalty = new Value(0.0);
        for (Value param : model.weights()) {
            l2Penalty = l2Penalty.add(param.pow(2.0));
        }
        Value l2Loss = new Value(lambda).mul(l2Penalty);
        loss = loss.add(l2Loss);
        return loss;
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
            // System.out.println("Input: " + Arrays.toString(X[i]) + "-> Prediction: " + out + ", Ground Truth: " + y[i]);
            
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
            } else {
                System.out.printf("Misclassified sample %d: Predicted %d, Actual %d%n", i, predicted, y[i]);
                System.out.println("Wrong prediction probability: " + out.get(predicted).data + ", Expected value probability: " + out.get(y[i]).data);
            }
        }
        
        accuracy = (double) correct / X.length;
        return accuracy;   
    }
}
