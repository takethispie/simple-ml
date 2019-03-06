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
                new[] {1, 0.3, 1.5},
                new[] {1, 0.9, -2.5},
                new[] {1, 0.1, 0.3},
                new[] {1, -0.5, 1.2}
            };

            var expectedOutput = new double[] {1, -1, 1, -1};
            const double learningRate = 0.1;
            const int nbEpochs = 5000;
            
            double[] model = Perceptron.CreateModel(input.Length);
            model = Perceptron.TrainLinearClassification(model, input, expectedOutput, learningRate, nbEpochs);

            for (int i = 0; i < expectedOutput.Length; i++)
            {
                Assert.AreEqual(expectedOutput[i], model[i]);
            }
        }
    }
}
