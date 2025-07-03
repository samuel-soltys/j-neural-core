package app;

import models.BinaryClassifier;

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
        int[] layers = new int[]{3, 4, 4, 1};

        BinaryClassifier classifier = new BinaryClassifier(layers);
        System.out.println("Model parameters count: " + classifier.getModelParametersCount());

        classifier.train(X, y, 0.05, 200);
        
        // Test the model with new data
        double[][] testX = {
            {2.5, 2.0, 1.5},
            {0.5, -1.0, 0.0},
            {-2.0, -1.5, -1.0}
        };
        int[] testY = {1, 0, 0};
        classifier.test(testX, testY);

        // TO DO: Digits recognizer
        // DigitRecognizer digitRecognizer = new DigitRecognizer();
        // digitRecognizer.train("path/to/digits/train.csv");
        // digitRecognizer.test("path/to/digits/test.csv");
    }
}