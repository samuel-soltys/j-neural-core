import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        // Sample 2D input points, labels 0 or 1
        double[][] X = {
            {2.0, 3.0, 1.0},
            {1.0, 0.0, -1.0},
            {3.0, 1.0, 0.5},
            {-1.0, -2.0, 0.0},
            {0.0, 2.0, 3.0},
            {-3.0, -1.0, -2.0}
        };
        int[] y = {1, 0, 1, 0, 1, 0};
        
        // Initializing the model
        int[] layers = new int[]{3, 4, 4, 1};
        MLP model = new MLP(layers);

        // Training the model
        double learningRate = 0.05;
        int epochs = 100;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            // Go through each input
            for (int i = 0; i < X.length; i++) {
                // Wrap inputs in Value objects
                List<Value> inputs = new ArrayList<>();
                for (double d : X[i]) {
                    inputs.add(new Value(d));
                }

                // Forward pass
                List<Value> outs = model.forward(inputs);
                Value prediction = outs.get(0);
                
                // Cross-entropy loss 
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
            if (epoch % 1 == 0) {
                System.out.printf("Epoch %d, Loss: %.8f%n", epoch, totalLoss);
            }
        }
        
        // Test trained model
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