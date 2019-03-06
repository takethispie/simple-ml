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
            //var input = new[]
            //{
            //    new[] { 1, 0.5, 0.1 },
            //    new[] { 1, 1.5, 0.9 },
            //    new[] { 1, 1.1, 1.2 }
            //};

            var input = new[]{
                new[] {1, 0.3, 1.5},
                new[] {1, 0.9, -2.5},
                new[] {1, 0.1, 0.3},
                new[] {1, -0.5, 1.2}
            };

            var expectedOutput = new double[] {1, -1, 1, -1};
            const double learningRate = 1;
            const int nbEpochs = 10000;
            
            double[] weights = Perceptron.CreateModel(input.Length);
            double[] output = Perceptron.TrainLinearClassification(weights, input, expectedOutput, learningRate, nbEpochs);

            for (int i = 0; i < expectedOutput.Length; i++)
            {
                Assert.AreEqual(expectedOutput[i], output[i]);
            }
        }
    }
}
