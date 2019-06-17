import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@Data
public class MultilayerPerceptron {

    private List<Neuron> outputNeurons;
    private List<Neuron> hiddenNeurons;
    private boolean isBIAS;
    public List<Double> lastInputs;

    MultilayerPerceptron(int hiddenLayerSize, int inputOutputLayerSize, double minWeight, double maxWeight,
                         boolean isBIAS) {
        this.isBIAS = isBIAS;
        int increaseSize = isBIAS ? 1 : 0;

        this.hiddenNeurons = new ArrayList<>();
        for (int i = 0; i < hiddenLayerSize; i++) {
            hiddenNeurons.add(new Neuron(inputOutputLayerSize + increaseSize, minWeight, maxWeight));
        }

        this.outputNeurons = new ArrayList<>();
        for (int i = 0; i < inputOutputLayerSize; i++) {
            outputNeurons.add(new Neuron(hiddenLayerSize + increaseSize, minWeight, maxWeight));
        }
    }

    List<Double> calculateOutputs(List<Double> inputs) {
        List<Double> modifiedInputs = new ArrayList<>(inputs);
        if (isBIAS) {
            modifiedInputs.add(1.0);
        }

        List<Double> results = getResults(modifiedInputs, hiddenNeurons);
        this.lastInputs = results;

        List<Double> modifiedHiddenOutputs = new ArrayList<>(results);
        if (isBIAS) {
            modifiedHiddenOutputs.add(1.0);
        }

        return getResults(modifiedHiddenOutputs, outputNeurons);
    }

    private List<Double> getResults(List<Double> inputs, List<Neuron> neurons) {
        List<Double> results = new ArrayList<>();

        neurons.forEach(neuron -> {
            double sum = 0;
            for (int i = 0; i < inputs.size(); i++) {
                sum += inputs.get(i) * neuron.getWeights().get(i);
            }
            results.add(1 / (1 + Math.exp(0 - sum)));
        });

        return results;
    }

    void teach(List<TrainingPattern> patterns, int epochs, double step) {
        for (int i = 0; i < epochs; i++) {
            for (TrainingPattern pattern : patterns) {
                teach(pattern, step);
            }
        }
    }

    private void teach(TrainingPattern pattern, double step) {
        List<Double> inputs = new ArrayList<>(pattern.getInputs());
        if (isBIAS) {
            inputs.add(1.0);
        }

        // Applies inputs from input layer to the hidden layer, prepares inputs for output layer
        List<Double> hiddenResults = getResults(inputs, hiddenNeurons);
        if (isBIAS) {
            hiddenResults.add(1.0);
        }

        // Applies inputs to the output layer
        List<Double> outputResults = getResults(hiddenResults, outputNeurons);

        // Calculates signal error for output layer
        List<Double> signalErrorsOutput = IntStream.range(0, outputResults.size()).mapToObj(i ->
                (pattern.getExpectedOutputs()
                        .get(i) - outputResults.get(i)) * calculateSigmoidalFrom(outputResults.get(i))
        ).collect(Collectors.toList());

        // Calculates signal error for hidden layer
        List<Double> signalErrorsHidden = IntStream.range(0, hiddenResults.size()).mapToObj(i ->
                calculateSigmoidalFrom(hiddenResults.get(i)) * (IntStream.range(0, outputResults.size())
                        .mapToDouble(j -> outputNeurons.get(j).getWeights().get(i) * signalErrorsOutput.get(j)).sum())
        ).collect(Collectors.toList());

        this.outputNeurons = adjustWeightsBy(outputNeurons, signalErrorsOutput, hiddenResults, step);
        this.hiddenNeurons = adjustWeightsBy(hiddenNeurons, signalErrorsHidden, inputs, step);
    }

    private List<Neuron> adjustWeightsBy(List<Neuron> neurons, List<Double> errors, List<Double> previousInputs, double step) {

        List<Neuron> results = new ArrayList<>();

        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            List<Double> newWeights = new ArrayList<>();
            for (int j = 0; j < neuron.getWeights().size(); j++) {
                Double weight = neuron.getWeights().get(j);
                newWeights.add(weight + step * errors.get(i) * previousInputs.get(j));
            }
            results.add(new Neuron(newWeights));
        }

        return results;
    }

    private static double calculateSigmoidalFrom(double value) {
        double e = Math.exp(value);
        return Math.pow(e / (1 + e), 2);
    }

}
