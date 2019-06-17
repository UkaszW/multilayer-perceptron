import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ResultsPresenterUtil {

    static void printResults() {
        MultilayerPerceptron multilayer1 = new MultilayerPerceptron(2,4,-0.5,0.5, false);
        MultilayerPerceptron multilayer2 = new MultilayerPerceptron(2,4,-0.5,0.5, true);

        List<Double> inputs1 = new ArrayList<>(Arrays.asList(1.0,0.0,0.0,0.0));
        List<Double> outputs1 = new ArrayList<>(Arrays.asList(1.0,0.0,0.0,0.0));

        List<Double> inputs2 = new ArrayList<>(Arrays.asList(0.0,1.0,0.0,0.0));
        List<Double> outputs2 = new ArrayList<>(Arrays.asList(0.0,1.0,0.0,0.0));

        List<Double> inputs3 = new ArrayList<>(Arrays.asList(0.0,0.0,1.0,0.0));
        List<Double> outputs3 = new ArrayList<>(Arrays.asList(0.0,0.0,1.0,0.0));

        List<Double> inputs4 = new ArrayList<>(Arrays.asList(0.0,0.0,0.0,1.0));
        List<Double> outputs4 = new ArrayList<>( Arrays.asList(0.0,0.0,0.0,1.0));

        List<TrainingPattern> trainingPatterns = new ArrayList<>();
        TrainingPattern multilayerTrainingPattern1 = new TrainingPattern(inputs1,outputs1);
        TrainingPattern multilayerTrainingPattern2 = new TrainingPattern(inputs2,outputs2);
        TrainingPattern multilayerTrainingPattern3 = new TrainingPattern(inputs3,outputs3);
        TrainingPattern multilayerTrainingPattern4 = new TrainingPattern(inputs4,outputs4);
        trainingPatterns.add(multilayerTrainingPattern1);
        trainingPatterns.add(multilayerTrainingPattern2);
        trainingPatterns.add(multilayerTrainingPattern3);
        trainingPatterns.add(multilayerTrainingPattern4);

        multilayer1.teach(trainingPatterns, 50000, 0.005);
        multilayer2.teach(trainingPatterns, 30000, 0.03);

        System.out.println("BAIS off:");
        System.out.println("Hidden neurons:");
        System.out.println(multilayer1.getHiddenNeurons());
        System.out.println("===============");
        System.out.println("Output neurons:");
        System.out.println(multilayer1.getOutputNeurons());
        System.out.println("===============");

        System.out.println("Result of first input: ");
        System.out.println(multilayer1.calculateOutputs(inputs1));
        System.out.println("Result of second input: ");
        System.out.println(multilayer1.calculateOutputs(inputs2));
        System.out.println("Result of third input: ");
        System.out.println(multilayer1.calculateOutputs(inputs3));
        System.out.println("Result of fourth input: ");
        System.out.println(multilayer1.calculateOutputs(inputs4));

        System.out.println("\nBAIS active:");
        System.out.println("Hidden neurons:");
        System.out.println(multilayer2.getHiddenNeurons());
        System.out.println("===============");
        System.out.println("Output neurons:");
        System.out.println(multilayer2.getOutputNeurons());
        System.out.println("===============");

        System.out.println("Result of first input: ");
        System.out.println(multilayer2.calculateOutputs(inputs1));
        System.out.println("Result of second input: ");
        System.out.println(multilayer2.calculateOutputs(inputs2));
        System.out.println("Result of third input: ");
        System.out.println(multilayer2.calculateOutputs(inputs3));
        System.out.println("Result of fourth input: ");
        System.out.println(multilayer2.calculateOutputs(inputs4));

        // Showing hidden layer neurons after the training process for each of the training patterns
        MultilayerPerceptron multilayer3 = new MultilayerPerceptron(2,4,-0.5,0.5, true);

        multilayer3.teach(trainingPatterns, 30000, 0.03);

        System.out.println("=======================================");
        System.out.println("Showing hidden layers:");
        System.out.println("Pattern 1");
        multilayer3.calculateOutputs(inputs1);
        System.out.println(multilayer3.lastInputs);

        System.out.println("Pattern 2");
        multilayer3.calculateOutputs(inputs2);
        System.out.println(multilayer3.lastInputs);

        System.out.println("Pattern 3");
        multilayer3.calculateOutputs(inputs3);
        System.out.println(multilayer3.lastInputs);

        System.out.println("Pattern 4");
        multilayer3.calculateOutputs(inputs4);
        System.out.println(multilayer3.lastInputs);




        /**
         * THOUGHTS
         * BAIS off - two patterns can be trained
         * BAIS active - all 4 patterns trained
         */
    }

}
