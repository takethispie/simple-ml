using System;
using System.Linq;

namespace simple_ml
{
    public class Perceptron
    {
        public double[] Weights { get; private set; }
        public readonly double[] Inputs;
        public int[] Outputs;

        public Perceptron(double[] inputs, int minimumWeight = -1, int maximumWeight = 1)
        {
            Inputs = inputs;
            RandomizeWeights(Inputs.Length, minimumWeight, maximumWeight);
        }

        /// <summary>
        /// Computes the sum of each input with its respective weight
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public int Guess(double[] inputs)
        {
            if (inputs.Length == 0 || Weights.Length == 0 || Weights == null) return 0; //TODO: Y réfléchir

            double sum = 0f;

            for (int i = 0; i < Weights.Length; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            return Activate(sum);
        }

        /// <summary>
        /// Rosenblatt rule
        /// </summary>
        /// <param name="sum"></param>
        /// <returns></returns>
        private static int Activate(double sum)
        {
            if (sum >= 0) return 1;

            return -1;
        }

        private void RandomizeWeights(int size, int minimum, int maximum)
        {
            if (Weights == null) Weights = new double[size];

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = new Random().Next(minimum, maximum);
            }
        }

        public int Learn(int expectedResult, double[] inputs)
        {
            var guess = Guess(inputs);

            if (guess == expectedResult) return guess;

            double error = (expectedResult == 1 ? 1 : -1) - (guess == 1 ? 1 : -1);

            Weights = Weights.Zip(inputs, (weight, input) => weight + (LearningRate * error * input)).ToArray();

            return guess;
        }
    }
}
