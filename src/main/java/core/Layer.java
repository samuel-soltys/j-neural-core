package core;

import java.util.ArrayList;
import java.util.List;

public class Layer {
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
