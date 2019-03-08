using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace simple_ml
{
    public class SimpleLinearRegression
    {
        public Matrix<double> X { get; set; }

        public Matrix<double> Y { get; set; }

        public Matrix<double> W { get; set; }

        public SimpleLinearRegression(double[][] inputs, double[] expectedResults)
        {
            if (inputs.Length > 0)
            {
                X = Matrix<double>.Build.Random(inputs.Length, inputs[0].Length + 1);
                for (int i = 0; i < inputs.Length; i++)
                {
                    X[i, 0] = 1;

                    for (int j = 0; j < inputs[i].Length; j++)
                    {
                        X[i, j + 1] = inputs[i][j];
                    }
                }

                Y = Matrix<double>.Build.Random(expectedResults.Length, 1);
                for (int k = 0; k < expectedResults.Length; k++)
                {
                    Y[k, 0] = expectedResults[k];
                }
            }
        }

        public double[] Compute()
        {
            W = ((X.Transpose() * X).Inverse() * X.Transpose()) * Y;
            return W.ToRowMajorArray();
        }

        /// <summary>
        /// Applying the regression linear inference formula
        /// </summary>
        /// <param name="model">Associated model</param>
        /// <param name="input">Associated input</param>
        /// <returns>A sum of ints</returns>
        public double LinearInference(double[] model, double[] input)
        {
            var total = model[0];

            for (int i = 1; i <= input.Length; i++)
            {
                total += model[i] * input[i - 1];
            }

            return total;
        }
    }
}
