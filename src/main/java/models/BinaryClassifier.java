package models;

import core.MLP;
import engine.Trainer;

public class BinaryClassifier {
    public static void main(String[] args) {
        // Example data for binary classification
        double[][] X = {
            {2.0, 3.0, 1.0},
            {1.0, 0.0, -1.0},
            {3.0, 1.0, 0.5},
            {-1.0, -2.0, 0.0},
            {0.0, 2.0, 3.0},
            {-3.0, -1.0, -2.0}
        };
        int[] y = {1, 0, 1, 0, 1, 0};
        int[] layersBinary = new int[]{3, 4, 4, 1};
        MLP modelBinary = new MLP(layersBinary);

        Trainer trainer = new Trainer(modelBinary);
        System.out.println("Model parameters count: " + trainer.getModelParametersCount());

        trainer.train(X, y, 0.05, 10, null, null, false, 0, 0);
        
        // Test the model with new data
        double[][] testX = {
            {2.5, 2.0, 1.5},
            {0.5, -1.0, 0.0},
            {-2.0, -1.5, -1.0}
        };
        int[] testY = {1, 0, 0};
        double accuracy = trainer.test(testX, testY);
        System.out.println("Test accuracy: " + accuracy);
    }
}
