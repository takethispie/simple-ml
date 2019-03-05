using System;

namespace simple_ml
{
    public class Perceptron
    {
        public float[] Weights { get; private set; }
        public readonly float[] Inputs;
        public int[] Outputs;

        public Perceptron(float[] inputs, int minimumWeight = -1, int maximumWeight = 1)
        {
            Inputs = inputs;
            RandomizeWeights(Inputs.Length, minimumWeight, maximumWeight);
        }

        /// <summary>
        /// Computes the sum of each input with its respective weight
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public int Guess(float[] inputs)
        {
            if (inputs.Length == 0 || Weights.Length == 0 || Weights == null) return 0; //TODO: Y réfléchir

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

        private void RandomizeWeights(int size, int minimum, int maximum)
        {
            if (Weights == null) Weights = new float[size];

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = new Random().Next(minimum, maximum);
            }
        }
    }
}
