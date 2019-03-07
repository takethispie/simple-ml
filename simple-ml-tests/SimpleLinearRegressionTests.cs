using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using simple_ml;

namespace simple_ml_tests
{
    [TestClass]
    public class SimpleLinearRegressionTests
    {
        [TestMethod]
        public void TestFunction() //TODO: Remove this.
        {
            var input = new[]{
                new[] {1.0, 0.0},
                new[] {0.0, 1},
                new[] {1.0, 1.0},
                new[] {2.0, 2.0}
            };

            var expectedOutput = new double[] { -1, 1, -1, -1 };

            var regression = new SimpleLinearRegression(input, expectedOutput);
            var result = regression.Compute();

            foreach (double d in result)
            {
                Console.WriteLine(d);
            }
        }
    }
}
