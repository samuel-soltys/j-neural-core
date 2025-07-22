package models;

import core.MLP;
import engine.Trainer;
import data.DigitDataLoader;
import data.DigitDataLoader.DataSet;

public class DigitRecognizer {
    public static void main(String[] args) {
        double[][] X = null;   // training images
        int[] y = null;     // training labels
        
        // Loading training digit data
        try {
            DataSet train = DigitDataLoader.load("/data/digits_train.csv");
            X = train.images;
            y = train.labels;
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("Training data loaded: " + X.length + " images, " + y.length + " labels.");

        if (X == null || y == null) {
            System.err.println("Null training data.");
            return;
        }

        // Loading validation digit data
        double[][] X_val = null;   // validation images
        int[] y_val = null;     // validation labels
        try {
            DataSet val = DigitDataLoader.load("/data/digits_validation.csv");
            X_val = val.images;
            y_val = val.labels;
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("Validation data loaded: " + X_val.length + " images, " + y_val.length + " labels.");

        int[] layersDigit = new int[]{64, 64, 32, 10};
        MLP modelDigit = new MLP(layersDigit);

        Trainer trainer = new Trainer(modelDigit);
        System.out.println("Model parameters count: " + trainer.getModelParametersCount());

        // String modelPath = "./digits_classifier.txt";
        // modelDigit.loadModel(modelPath);
        trainer.train(X, y, 0.008, 200, X_val, y_val);
        
        // Loading test digit data
        double[][] testX = null;   // test images
        int[] testY = null;     // test labels
        try {
            DataSet test = DigitDataLoader.load("/data/digits_test.csv");
            testX = test.images;
            testY = test.labels;
        } catch (Exception e) {
            e.printStackTrace();
        }

        double accuracy = trainer.test(testX, testY);
        System.out.println("Test accuracy: " + accuracy);
    }
}
