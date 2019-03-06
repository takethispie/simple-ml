using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using simple_ml;

namespace simple_ml_tests
{
    [TestClass]
    public class PerceptronTests
    {
        [TestMethod]
        public void ShouldOutputExpectedResults()
        {
            var input = new[]{
                new[] {1.0, 0.0},
                new[] {0.0, 1},
                new[] {1.0, 1.0},
                new[] {2.0, 2.0}
            };

            var expectedOutput = new double[] {-1, 1, -1, 1};
            const double learningRate = 0.1;
            const int nbEpochs = 1000;
            
            double[] weights = Perceptron.CreateModel(2);
            weights = Perceptron.TrainLinearClassification(weights, input, expectedOutput, learningRate, nbEpochs);

            Console.WriteLine(Perceptron.ClassificationLinearInference(weights, input[0]));
            Console.WriteLine(Perceptron.ClassificationLinearInference(weights, input[1]));
            Console.WriteLine(Perceptron.ClassificationLinearInference(weights, input[2]));
            Console.WriteLine(Perceptron.ClassificationLinearInference(weights, input[3]));
            Console.WriteLine(weights[0]);
            Console.WriteLine(weights[1]);
            Console.WriteLine(weights[2]);
        }
    }
}
