import lombok.Data;

import java.util.List;
import java.util.stream.Collectors;

@Data
public class TrainingPattern {

    protected List<Double> inputs;
    private List<Double> expectedOutputs;
    protected int inputCount;

    TrainingPattern(List<Double> inputs, List<Double> expectedOutputs) {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
        this.inputCount = inputs.size();
    }

    @Override
    public String toString() {
        return "Input pattern values:\n" +
                inputs.stream()
                        .map(Object::toString)
                        .map(i -> i.substring(0, Math.min(6, i.length())))
                        .collect(Collectors.joining(", ")) +
                "\nOutputs:\n" +
                expectedOutputs.stream()
                        .map(Object::toString)
                        .map(i -> i.substring(0, Math.min(6, i.length())))
                        .collect(Collectors.joining(", "));
    }
}
