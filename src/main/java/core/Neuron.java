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
            return z;  // Returning raw logits for output layer, change to z.sigmoid() if binary classification (for now assuming multi-class)
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
