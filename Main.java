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

    public Value forward(List<Value> inputs) {
        Value z = b;
        for (int i = 0; i < w.size(); i++) {
            z = z.add(w.get(i).mul(inputs.get(i)));
        }
        return z.relu();
    }
}

public class Main {
    public static void main(String[] args) {
        List<Value> a = new ArrayList<>();
        a.add(new Value(2.0));
        a.add(new Value(3.0));
        Neuron n = new Neuron(2);
        System.out.println(n.forward(a));
    }
}