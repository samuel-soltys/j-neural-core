package data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class DigitDataLoader {
    public static class DataSet {
        public final double[][] images;
        public final int[] labels;

        public DataSet(double[][] images, int[] labels) {
            this.images = images;
            this.labels = labels;
        }
    }

    public static DataSet load(String resourcePath) throws Exception {
        List<double[]> imageList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();
        
        InputStream is = DigitDataLoader.class.getResourceAsStream(resourcePath);
        if (is == null) {
            throw new IOException("Resource not found: " + resourcePath);
        }
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));

        // Reading the file line by line
        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.trim().split(",");
            if (parts.length != 65) continue;

            // Converting to double arrays
            // Assuming the first 64 parts are pixel values and the last part is the label
            double[] image = new double[64];
            for (int i = 0; i < 64; i++) {
                image[i] = Double.parseDouble(parts[i]);
            }
            int label = Integer.parseInt(parts[64]);

            imageList.add(image);
            labelList.add(label);
        }
        
        // Converting lists to plain arrays for easier handling
        double[][] images = imageList.toArray(new double[0][64]);
        int[] labels = labelList.stream().mapToInt(Integer::intValue).toArray();

        // Normalizing image values to [0, 1] for better training
        for (int i = 0; i < images.length; i++) {
            for (int j = 0; j < images[i].length; j++) {
                images[i][j] /= 16.0;
            }
        }

        return new DataSet(images, labels);
    }
}
