using Microsoft.VisualStudio.TestTools.UnitTesting;
using simple_ml;

namespace simple_ml_tests
{
    [TestClass]
    public class PerceptronTests
    {
        private Perceptron _perceptron;

        [TestMethod]
        public void WeightsShouldBeBetweenInclusiveNegativeOneAndInclusivePositiveOne()
        {
            const int minimumWeight = -1;
            const int maximumWeight = 1;
            var inputs = new[] { 1.0, 1.5, 0.6 };
            _perceptron = new Perceptron(inputs, minimumWeight, maximumWeight);

            foreach (var weight in _perceptron.Weights)
            {
                Assert.IsTrue(weight >= minimumWeight && weight <= maximumWeight);
            }
        }

        [TestMethod]
        public void GuessShouldBeEqualToNegativeOneOrPositiveOne()
        {
            const int minimumWeight = -1;
            const int maximumWeight = 1;
            var inputs = new[] { 1.0, 1.5, 0.6 };
            _perceptron = new Perceptron(inputs, minimumWeight, maximumWeight);

            int guess = _perceptron.Guess(inputs);

            Assert.IsTrue(guess == minimumWeight || guess == maximumWeight);
        }

        [TestMethod]
        public void OutputsShouldBeEmpty()
        {
            var inputs = new[] { 1.0, 1.5, 0.6 };
            _perceptron = new Perceptron(inputs);

            Assert.IsNull(_perceptron.Outputs);
        }
    }
}
