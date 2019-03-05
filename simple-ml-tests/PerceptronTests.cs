﻿using System;
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
            var inputs = new[] {1.0f, 1.5f, 0.6f};
            _perceptron = new Perceptron(minimumWeight, maximumWeight);

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
            var inputs = new[] { 1.0f, 1.5f, 0.6f };
            _perceptron = new Perceptron(minimumWeight, maximumWeight);

            int guess = _perceptron.Guess(inputs);

            Assert.IsTrue(guess == minimumWeight || guess == maximumWeight);
        }

        [TestMethod]
        public void OutputsShouldBeEmpty()
        {
            var inputs = new[] { 1.0f, 1.5f, 0.6f };
            _perceptron = new Perceptron();

            Assert.IsNull(_perceptron.Outputs);
        }
    }
}
