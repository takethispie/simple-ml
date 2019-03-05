using System;

namespace simple_ml
{
    public class Perceptron
    {
        public float[] Weights { get; private set; }
        public int[] Outputs;

        public Perceptron(int minimumWeight = -1, int maximumWeight = 1)
        {
            RandomizeWeights(minimumWeight, maximumWeight);
        }

        /// <summary>
        /// Computes the sum of each input with its respective weight
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public int Guess(float[] inputs)
        {
            float sum = 0f;

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
        private static int Activate(float sum)
        {
            if (sum >= 0) return 1;

            return -1;
        }

        private void RandomizeWeights(int minimum, int maximum)
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = new Random().Next(minimum, maximum);
            }
        }
    }
}
