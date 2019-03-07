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

        public void Compute()
        {
            W = ((X.Transpose() * X).Inverse() * X.Transpose()) * Y;
        }
    }
}
