import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Data
class Neuron {

    private Integer numberOfInputs;
    private List<Double> weights;

    Neuron(int numberOfInputs, double minWeight, double maxWeight) {
        this.numberOfInputs = numberOfInputs;
        Random rand = new Random();
        weights = new ArrayList<>();

        if (minWeight > maxWeight) minWeight = maxWeight;

        for (int i = 0; i < numberOfInputs; i++) {
            weights.add(rand.nextDouble()*(maxWeight - minWeight) + minWeight);
        }
    }

    Neuron(List<Double> weights) {
        this.numberOfInputs = weights.size();
        this.weights = weights;
    }
}
