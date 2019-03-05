using System;
using System.Collections.Generic;
using System.Linq;

namespace simple_ml
{
    public class Perceptron
    {
        public double[] Weights;
        public double LearningRate, Threshold;
        private double[] inputs;

        public Perceptron(int inputCount, double learningRate = 0.1, double threshold = 0.5) {
            inputs = new double[inputCount];
            Weights = new double[inputCount];
            LearningRate = learningRate;
            Threshold = threshold;
        }

        public bool Process(double[] inputs) => inputs.Zip(Weights, (v, weight) => v*weight).Sum() > Threshold ? true : false;

        public bool Learn(bool expectedResult, double[] inputs) {
            //on prend les entrée et les multiplie par leur poid respectif pour mettre dans un nouveau array
            //si la sum de toute le weighted input > que threshold (0,5) alors true sinon false
            // => seuil de d'activation
            var res = Process(inputs);
            if(res == expectedResult) return res;

            double error = (expectedResult?1:0) - (res?1:0);
            for (int i = 0; i < Weights.Length; i++) {
                Weights[i] += LearningRate * error * inputs[i];
            }
            return res;
        }
    }
}
