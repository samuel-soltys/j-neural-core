package core;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class MLP {
    List<Layer> layers;

    public MLP(int[] layerSizes) {
        this.layers = new ArrayList<>(layerSizes.length);
        for(int i = 0; i < layerSizes.length - 1; i++) {
            layers.add(new Layer(layerSizes[i], layerSizes[i + 1]));
        }
    }

    public List<Value> forward(List<Value> inputs) {
        List<Value> outputs = inputs;
        for(int i = 0; i < layers.size(); i++) {
            Boolean isOutputLayer = (i == layers.size() - 1);
            outputs = layers.get(i).forward(outputs, isOutputLayer);
        }
        return outputs;
    }

    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        return params;
    }
    public List<Value> weights() {
        List<Value> weights = new ArrayList<>();
        for (Layer layer : layers) {
            weights.addAll(layer.weights());
        }
        return weights;
    }

    public void saveModel(String filePath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            this.parameters().forEach(param -> {
                writer.println(param.data);
            });
            writer.close();
        } catch (IOException e) {
            System.err.println("Error saving model: " + e.getMessage());
        }
    }
    public void loadModel(String filePath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            List<Value> params = this.parameters();
            int i = 0;
            while ((line = reader.readLine()) != null && i < params.size()) {
                params.get(i).data = Double.parseDouble(line);
                i++;
            }
        } catch (IOException e) {
            System.err.println("Error loading model: " + e.getMessage());
        }
    }

    public void zeroGrad() {
        for (Value p : parameters()) {
            p.grad = 0.0;
        }
    }
}