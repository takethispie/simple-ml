using System;

namespace simple_ml
{
    public static class Perceptron
    {
        /// <summary>
        /// Generates the model based on the number of columns in the input
        /// </summary>
        /// <param name="nbInputColumns">Number of columns in the input</param>
        /// <returns>The model with random double values</returns>
        public static double[] CreateModel(int nbInputColumns)
        {
            var model = new double[nbInputColumns + 1];

            for (int i = 0; i < model.Length; i++)
            {
                model[i] = new Random().NextDouble() * 2.0 - 1.0;
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
            if (sum >= 0) return 1;

            return -1;
        }

        [Obsolete]
        public static double RegressiveLinearInference(double[] model, double[] input)
        {
            var total = model[0];

            for (int i = 1; i <= input.Length; i++)
            {
                total += model[i] * input[i - 1];
            }

            return total;
        }

        /// <summary>
        /// Applying the classifier linear inference formula
        /// </summary>
        /// <param name="model">Associated model</param>
        /// <param name="input">Associated input</param>
        /// <returns>1 or -1 based on the Rosenblatt rule</returns>
        public static int ClassifierLinearInference(double[] model, double[] input)
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
        /// <param name="learningRate">Learning rate of the machine</param>
        /// <param name="nbEpochs">Nombre d'itérations</param>
        public static double[] TrainLinearClassifier(double[][] inputs, double[] expectedOutput, double learningRate = 0.1, int nbEpochs = 1000)
        {
            var model = CreateModel(inputs[0].Length);

            for (int i = 0; i < nbEpochs; i++)
            {
                for (int k = 0; k < inputs.Length; k++)
                {
                    var gxk = ClassifierLinearInference(model, inputs[k]);

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
        /// <param name="expectedOutput">Expected classes for each input</param>
        /// <returns>Input's classes</returns>
        public static double[] GetInputClasses(double[] model, double[][] input, double[] expectedOutput)
        {
            var classes = new double[expectedOutput.Length];

            for (int i = 0; i < input.Length; i++)
            {
                classes[i] = ClassifierLinearInference(model, input[i]);
            }

            return classes;
        }
    }
}
