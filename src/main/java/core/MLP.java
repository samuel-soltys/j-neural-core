package core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

class Neuron {
    private List<Value> w;
    private Value b;

    public Neuron(int numInputs) {
        this.w = new Random().doubles(numInputs, -1, 1)
                    .mapToObj(Value::new)
                    .collect(Collectors.toList());
        this.b = new Value(0.0);
    }

    public Value forward(List<Value> inputs, boolean isOutputLayer) {
        Value z = b;
        for (int i = 0; i < w.size(); i++) {
            z = z.add(w.get(i).mul(inputs.get(i)));
        }
        if (isOutputLayer) {
            return z.sigmoid();  // use sigmoid in output layer only
        } else {
            return z.relu();    // use ReLU in hidden layers
        }
    }

    public List<Value> parameters() {
        List<Value> params = new ArrayList<>(w);
        params.add(b);
        return params;
    }
}

class Layer {
    private List<Neuron> neurons;
    
    public Layer(int numInputs, int numNeurons) {
        this.neurons = new ArrayList<>(numNeurons);
        for (int i = 0; i < numNeurons; i++) {
            neurons.add(new Neuron(numInputs));
        }
    }

    public List<Value> forward(List<Value> inputs) {
        List<Value> outputs = new ArrayList<>();
        for (int i = 0; i < neurons.size(); i++) {
            boolean isOutputLayer = (i == neurons.size() - 1);
            outputs.add(neurons.get(i).forward(inputs, isOutputLayer));
        }
        return outputs;
    }

    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Neuron neuron : neurons) {
            params.addAll(neuron.parameters());
        }
        return params;
    }
}

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
        for(Layer layer : layers) {
            outputs = layer.forward(outputs);
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

    public void zeroGrad() {
        for (Value p : parameters()) {
            p.grad = 0.0;
        }
    }
}