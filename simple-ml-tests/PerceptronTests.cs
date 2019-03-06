using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using simple_ml;

namespace simple_ml_tests
{
    [TestClass]
    public class PerceptronTests
    {
        [TestMethod]
        public void ShouldOutputExpectedClasses()
        {
            var input = new[]{
                new[] {1.0, 0.0},
                new[] {0.0, 1},
                new[] {1.0, 1.0},
                new[] {2.0, 2.0}
            };

            var expectedOutput = new double[] {-1, 1, -1, -1};
            const double learningRate = 0.1;
            const int nbEpochs = 1000;
            
            var model = Perceptron.TrainLinearClassifier(input, expectedOutput, learningRate, nbEpochs);
            var classes = Perceptron.GetInputClasses(model, input, expectedOutput);

            CollectionAssert.AreEqual(expectedOutput, classes);
        }

        [TestMethod]
        public void ShouldSignWithNegativeOne()
        {
            double result = Perceptron.Sign(-59.0);

            Assert.AreEqual(result, -1);
        }

        [TestMethod]
        public void ShouldSignZeroWithPositiveOne()
        {
            double result = Perceptron.Sign(0.0);

            Assert.AreEqual(result, 1);
        }

        [TestMethod]
        public void ShouldSignWithPositiveOne()
        {
            double result = Perceptron.Sign(16.0);

            Assert.AreEqual(result, 1);
        }

        [TestMethod]
        public void ShouldCreateModelWithInputSizePlusOne()
        {
            var input = new[]{
                new[] {1.0, 0.0},
                new[] {0.0, 1},
                new[] {1.0, 1.0},
                new[] {2.0, 2.0}
            };

            int nbInputColumns = input[0].Length;
            var model = Perceptron.CreateModel(nbInputColumns);

            Assert.IsTrue(model.Length - 1 == nbInputColumns);
        }

        [TestMethod]
        public void ShouldCreateModelWithRandomDoubleValues()
        {
            var input = new[]{
                new[] {1.0, 0.0},
                new[] {0.0, 1},
                new[] {1.0, 1.0},
                new[] {2.0, 2.0}
            };

            int nbInputColumns = input[0].Length;
            var model = Perceptron.CreateModel(nbInputColumns);

            CollectionAssert.AllItemsAreInstancesOfType(model, typeof(double));
        }

        [TestMethod]
        public void ShouldCreateModelWithRandomDoubleValuesBetweenNegativeOneAndPositiveOne()
        {
            var input = new[]{
                new[] {1.0, 0.0},
                new[] {0.0, 1},
                new[] {1.0, 1.0},
                new[] {2.0, 2.0}
            };

            int nbInputColumns = input[0].Length;
            var model = Perceptron.CreateModel(nbInputColumns);

            bool hasIncorrectValue = model.Any(x => x < -1.0 || x > 2.0);

            Assert.IsFalse(hasIncorrectValue);
        }

        [TestMethod]
        public void ShouldHaveIncorrectValue()
        {
            var input = new[]{
                new[] {1.0, 0.0},
                new[] {0.0, 1},
                new[] {1.0, 1.0},
                new[] {2.0, 2.0}
            };

            int nbInputColumns = input[0].Length;
            var model = Perceptron.CreateModel(nbInputColumns);
            model[0] = 2.1;

            bool hasIncorrectValue = model.Any(x => x < -1.0 || x > 2.0);

            Assert.IsTrue(hasIncorrectValue);
        }
    }
}
