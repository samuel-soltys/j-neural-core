class Value {
    public double data;

    public Value(double data) {
        this.data = data;
    }

    @Override
    public String toString() {
        return "Value(" + data + ")";
    }
}

public class Main {
    public static void main(String[] args) {
        Value a = new Value(2.0);
        System.out.println(a);
    }
}