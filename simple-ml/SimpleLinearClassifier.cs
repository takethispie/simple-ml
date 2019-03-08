using System;
using System.Linq;

namespace simple_ml
{
    public static class SimpleLinearClassifier
    {
        /// <summary>
        /// Generates the model based on the number of columns in the input
        /// </summary>
        /// <param name="nbInputColumns">Number of columns in the input</param>
        /// <returns>The model with random double values</returns>
        public static double[] CreateModel(int nbInputColumns)
        {
            var model = new double[nbInputColumns + 1];
            var rdm = new Random();

            for (int i = 0; i < model.Length; i++)
            {
                model[i] = rdm.NextDouble() * 2.0 - 1.0;
            }

            return model;
        }

        /// <summary>
        /// Rosenblatt rule activation function
        /// </summary>
        /// <param name="sum"></param>
        /// <returns></returns>
        public static int Sign(double sum)
        {
            return sum >= 0 ? 1 : -1;
        }

        /// <summary>
        /// Applying the classifier linear inference formula
        /// </summary>
        /// <param name="model">Associated model</param>
        /// <param name="input">Associated input</param>
        /// <returns>1 or -1 based on the Rosenblatt rule</returns>
        public static int LinearInference(double[] model, double[] input)
        {
            var total = model[0];

            for (int i = 1; i <= input.Length; i++)
            {
                total += model[i] * input[i - 1];
            }

            return Sign(total);
        }

        /// <summary>
        /// Trains on a dataset input based on the linear classifier way
        /// </summary>
        /// <param name="inputs">One dimension per parameter like the age, the sex, the height, etc.</param>
        /// <param name="expectedOutput">Represents the expected class for each input</param>
        /// <param name="model">Model to use in the training. Can be null and created on the fly.</param>
        /// <param name="learningRate">Learning rate of the machine</param>
        /// <param name="nbEpochs">Nombre d'itérations</param>
        public static double[] Train(double[][] inputs, double[] expectedOutput, double[] model = null, double learningRate = 0.1, int nbEpochs = 1000)
        {
            if (model == null || !model.Any()) model = CreateModel(inputs[0].Length);

            for (int i = 0; i < nbEpochs; i++)
            {
                for (int k = 0; k < inputs.Length; k++)
                {
                    var gxk = LinearInference(model, inputs[k]);

                    var yk = expectedOutput[k];

                    for (int w = 0; w <= inputs[k].Length; w++)
                    {
                        model[w] += learningRate * (yk - gxk) * (w == 0 ? 1.0 : inputs[k][w - 1]);
                    }
                }
            }

            return model;
        }

        /// <summary>
        /// Retrieves the class for each input
        /// </summary>
        /// <param name="model">Associated model</param>
        /// <param name="input">Associated input</param>
        /// <param name="expectedOutputLength">Count of expected classes for each input</param>
        /// <returns>Input's classes</returns>
        public static double[] GetInputClasses(double[] model, double[][] input, int expectedOutputLength)
        {
            var classes = new double[expectedOutputLength];

            for (int i = 0; i < input.Length; i++)
            {
                classes[i] = LinearInference(model, input[i]);
            }

            return classes;
        }

        /// <summary>
        /// Generates a random array with -1, 0 or 1 corresponding to an expected output
        /// </summary>
        /// <param name="inputLength">Count of the number of inputs</param>
        /// <returns>The expectedOutput classes array</returns>
        public static double[] GenerateRandomExpectedOutput(int inputLength)
        {
            var output = new double[inputLength];

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = new Random().Next(-1, 1);
            }

            return output;
        }
    }
}
