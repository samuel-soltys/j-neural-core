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

    public List<Value> forward(List<Value> inputs, Boolean isOutputLayer) {
        List<Value> outputs = new ArrayList<>();
        for (int i = 0; i < neurons.size(); i++) {
            outputs.add(neurons.get(i).forward(inputs, isOutputLayer));
        }
        if (isOutputLayer) {
            // If multiple outputs, apply softmax, else apply sigmoid activation function
            if (outputs.size() > 1) {
                Value sum = new Value(0.0);
                for (Value o : outputs) {
                    sum = sum.add(o.exp());
                }
                
                List<Value> outputsSoftmax = new ArrayList<>();
                for (Value o : outputs) {
                    outputsSoftmax.add(o.exp().div(sum));
                }
                return outputsSoftmax;
            } else {
                outputs.set(0, outputs.get(0).sigmoid());
            }
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
    public List<Value> weights() {
        List<Value> weights = new ArrayList<>();
        for (Neuron neuron : neurons) {
            weights.addAll(neuron.weights());
        }
        return weights;
    }
}
