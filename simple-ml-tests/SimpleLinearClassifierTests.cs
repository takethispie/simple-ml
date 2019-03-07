using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using simple_ml;

namespace simple_ml_tests
{
    [TestClass]
    public class SimpleLinearClassifierTests
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
            int expectedOutputLength = expectedOutput.Length;
            
            var model = SimpleLinearClassifier.TrainLinearClassifier(input, expectedOutput);
            var classes = SimpleLinearClassifier.GetInputClasses(model, input, expectedOutputLength);

            CollectionAssert.AreEqual(expectedOutput, classes);
        }

        [TestMethod]
        public void ShouldSignWithNegativeOne()
        {
            double result = SimpleLinearClassifier.Sign(-59.0);

            Assert.AreEqual(result, -1);
        }

        [TestMethod]
        public void ShouldSignZeroWithPositiveOne()
        {
            double result = SimpleLinearClassifier.Sign(0.0);

            Assert.AreEqual(result, 1);
        }

        [TestMethod]
        public void ShouldSignWithPositiveOne()
        {
            double result = SimpleLinearClassifier.Sign(16.0);

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
            var model = SimpleLinearClassifier.CreateModel(nbInputColumns);

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
            var model = SimpleLinearClassifier.CreateModel(nbInputColumns);

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
            var model = SimpleLinearClassifier.CreateModel(nbInputColumns);

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
            var model = SimpleLinearClassifier.CreateModel(nbInputColumns);
            model[0] = 2.1;

            bool hasIncorrectValue = model.Any(x => x < -1.0 || x > 2.0);

            Assert.IsTrue(hasIncorrectValue);
        }
    }
}
