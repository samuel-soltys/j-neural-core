import java.util.List;
import java.util.Set;
import java.util.ArrayList;
import java.util.HashSet;

class Value {
    public double data;
    public double grad;
    
    private Set<Value> prev;
    private String op;
    private Runnable backward = () -> {};

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
    public Value pow(double other) {
        Value result = new Value(Math.pow(this.data, other), Set.of(this), "pow");
        result.backward = () -> {
            this.grad += (other * Math.pow(this.data, other - 1)) * result.grad;
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
    
    // Activation function
    public Value relu() {
        Value result = new Value((this.data < 0) ? 0 : this.data, Set.of(this), "ReLU");
        result.backward = () -> {
            this.grad += ((result.data > 0) ? 1 : 0) * result.grad;
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

public class Main {
    public static void main(String[] args) {
        Value x1 = new Value(2.0);
        Value x2 = new Value(0.0);

        Value w1 = new Value(-3.0);
        Value w2 = new Value(1.0);

        Value b = new Value(6.8813735870196432);

        Value x1w1 = x1.mul(w1);
        Value x2w2 = x2.mul(w2);
        Value x1w1x2w2 = x1w1.add(x2w2);

        Value n = x1w1x2w2.add(b);
        Value o = n.relu();
        o.backward();

        System.out.println("x1: " + x1.grad + " | w1: " + w1.grad + "\nx2: " + x2.grad + " | w2: " + w2.grad);
    }
}