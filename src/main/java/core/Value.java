package core;

import java.util.List;
import java.util.Set;
import java.util.ArrayList;
import java.util.HashSet;

public class Value {
    public double data;
    public double grad;
    
    private Set<Value> prev;
    private Runnable backward = () -> {};
    @SuppressWarnings("unused") // String op is used for debugging
    private String op;

    // Constructor for leaf node
    public Value(double data) {
        this.data = data;
        this.grad = 0.0;
        this.prev = Set.of();
        this.op = "";
    }
    
    // Constructor for internal nodes
    public Value(double data, Set<Value> prev, String op) {
        this.data = data;
        this.grad = 0.0;
        this.prev = prev;
        this.op = op;
    }

    // Add method: create new Value with backprop rule
    public Value add(Value other) {
        Value result = new Value(this.data + other.data, Set.of(this, other), "+");
        result.backward = () -> {
            this.grad += 1.0 * result.grad;
            other.grad += 1.0 * result.grad;
        };
        return result;
    }
    
    // Pow function only for double powers
    public Value pow(double p) {
        Value result = new Value(Math.pow(this.data, p), Set.of(this), "pow");
        result.backward = () -> {
            this.grad += (p * Math.pow(this.data, p - 1)) * result.grad;
        };
        return result;
    }

    // Multiplying with another Value object
    public Value mul(Value other) {
        Value result = new Value(this.data * other.data, Set.of(this, other), "*");
        result.backward = () -> {
            this.grad += other.data * result.grad;
            other.grad += this.data * result.grad;
        };
        return result;
    }
    
    public Value log() {
        Value result = new Value(Math.log(this.data), Set.of(this), "log");
        result.backward = () -> {
            this.grad += (1.0 / this.data) * result.grad;
        };
        return result;
    }

    public Value exp() {
        // Clamp input to avoid overflow/underflow in Math.exp
        double safeInput = Math.max(-200, Math.min(200, this.data));
        double expValue = Math.exp(safeInput);
        Value result = new Value(expValue, Set.of(this), "exp");
        result.backward = () -> {
            this.grad += result.data * result.grad;
        };
        return result;
    }

    // Division by another Value object
    public Value div(Value other) {
        if (other.data == 0) {
            throw new ArithmeticException("Division by zero");
        }
        Value result = new Value(this.data / other.data, Set.of(this, other), "/");
        result.backward = () -> {
            this.grad += (1.0 / other.data) * result.grad;
            other.grad += (-this.data / (other.data * other.data)) * result.grad;
        };
        return result;
    }

    // Activation functions
    public Value relu() {
        Value result = new Value((this.data < 0) ? 0 : this.data, Set.of(this), "ReLU");
        result.backward = () -> {
            this.grad += ((result.data > 0) ? 1 : 0) * result.grad;
        };
        return result;
    }
    public Value sigmoid() {
        Value result = new Value(1.0 / (1.0 + Math.exp(-this.data)), Set.of(this), "sigmoid");
        result.backward = () -> {
            this.grad += result.data * (1 - result.data) * result.grad;
        };
        return result;
    }

    // Global backward pass for the whole computational graph
    public void backward() {
        // Setting grad of the final node
        this.grad = 1.0;

        // Topological order (depth-first)
        List<Value> topo = new ArrayList<>();
        buildTopo(this, topo, new HashSet<>());

        // Traverse in reverse topo order, calling stored backprop functions
        for (int i = topo.size() - 1; i >= 0; i--) {
            topo.get(i).backward.run();
        }

    }
    private static void buildTopo(Value v, List<Value> topo, Set<Value> seen) {
        if (!seen.contains(v)) {
            seen.add(v);
            for (Value p : v.prev) buildTopo(p, topo, seen);
            topo.add(v);
        }
    }

    // Overriding default print function
    @Override
    public String toString() {
        return "Value(" + data + ")";
    }
}