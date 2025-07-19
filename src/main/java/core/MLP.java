package core;

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

    public void zeroGrad() {
        for (Value p : parameters()) {
            p.grad = 0.0;
        }
    }
}