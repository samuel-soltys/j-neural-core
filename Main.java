class Value {
    public double data;
    public double grad;

    public Value(double data) {
        this.data = data;
        this.grad = 0.0;
    }
    
    // Adding with another Value object
    public Value add(Value other) {
        Value out = new Value(this.data + other.data);
        return out;
    }
    // Adding with scalar
    public Value add(double other) {
        Value out = new Value(this.data + other);
        return out;
    }
    
    // Multiplying with another Value object
    public Value mul(Value other) {
        Value out = new Value(this.data * other.data);
        return out;
    }
    // Multiplying with scalar
    public Value mul(double other) {
        Value out = new Value(this.data * other);
        return out;
    }

    // Overriding default print function
    @Override
    public String toString() {
        return "Value(" + data + ")";
    }
}

public class Main {
    public static void main(String[] args) {
        Value a = new Value(2.0); // 2.0
        Value b = a.add(1.0);   // 3.0
        Value c = b.mul(2.0);   // 6.0
        Value d = c.mul(b); // 18.0
        
        System.out.println("a: " + a + "\nb: " + b + "\nc: " + c + "\nd: " + d);
    }
}