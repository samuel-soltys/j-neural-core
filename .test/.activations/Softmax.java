package core.activations;

import java.util.Set;

import core.Value;

public class Softmax {
    public Value[] softmax(Value[] inputs) {
        double max = Double.NEGATIVE_INFINITY;
        for (Value v : inputs) {
            if (v.data > max) max = v.data;
        }
        double[] exps = new double[inputs.length];
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            exps[i] = Math.exp(inputs[i].data - max);
            sum += exps[i];
        }
        final double finalSum = sum;
        Value[] outputs = new Value[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            double softmaxVal = exps[i] / finalSum;
            Value out = new Value(softmaxVal, Set.of(inputs[i]), "softmax");
            int idx = i;
            out.backward = () -> {
                for (int j = 0; j < inputs.length; j++) {
                    double grad = out.grad * softmaxVal * ((idx == j ? 1 : 0) - (exps[j] / finalSum));
                    inputs[j].grad += grad;
                }
            };
            outputs[i] = out;
        }
        return outputs;
    }
}
