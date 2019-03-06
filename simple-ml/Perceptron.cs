using System;

namespace simple_ml
{
    public class Perceptron
    {
        public static double[] CreateModel(int nbInputColumns)
        {
            var weights = new double[nbInputColumns + 1];

            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = new Random().NextDouble() * 2.0 - 1.0;
            }

            return weights;
        }

        /// <summary>
        /// Rosenblatt rule
        /// </summary>
        /// <param name="sum"></param>
        /// <returns></returns>
        public static int Sign(double sum)
        {
            if (sum >= 0) return 1;

            return -1;
        }

        public static int ClassificationLinearInference(double[] model, double[] input)
        {
            var total = model[0];

            for (int i = 1; i <= input.Length; i++)
            {
                total += model[i] * input[i - 1];
            }

            return Sign(total);
        }

        private double RegressiveLinearInference(double[] model, double[] input)
        {
            var total = model[0];

            for (int i = 1; i <= input.Length; i++)
            {
                total += model[i] * input[i - 1];
            }

            return total;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="model">Tableau des poids</param>
        /// <param name="inputs">Une dimension par paramètre tel que l'âge, le sexe, la taille, etc.</param>
        /// <param name="expectedOutput"></param>
        /// <param name="learningRate"></param>
        /// <param name="nbEpochs">Nombre d'itérations</param>
        public static double[] TrainLinearClassification(double[] model, double[][] inputs, double[] expectedOutput, double learningRate, int nbEpochs)
        {
            for (int i = 0; i < nbEpochs; i++)
            {
                for (int k = 0; k < inputs.Length; k++)
                {
                    var gxk = ClassificationLinearInference(model, inputs[k]);

                    var yk = expectedOutput[k];

                    for (int w = 0; w <= inputs[k].Length; w++)
                    {
                        model[w] += learningRate * (yk - gxk) * (w == 0 ? 1.0 : inputs[k][w - 1]);
                    }
                }
            }

            return model;
        }
    }
}
