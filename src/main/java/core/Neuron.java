package core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class Neuron {
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
            // Returning raw logits to allow using appropriate activation function for the output layer
            return z;
        } else {
            // using ReLU in hidden layers
            return z.relu();
        }
    }

    public List<Value> parameters() {
        List<Value> params = new ArrayList<>(w);
        params.add(b);
        return params;
    }
    public List<Value> weights() {
        List<Value> weights = new ArrayList<>(w);
        return weights;
    }
}
