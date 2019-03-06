using System;
using System.Collections.Generic;
using System.Linq;

namespace simple_ml
{
    public class Perceptron
    {
        public double[] Weights;
        public double LearningRate, Threshold;

        public Perceptron(int inputCount, double learningRate = 0.1, double threshold = 0.5) {
            Weights = new double[inputCount+1];
            LearningRate = learningRate;
            Threshold = threshold;
        }

        //on prend les entrée et les multiplie par leur poid respectif pour mettre dans un nouveau array
        //si la sum de toute le weighted input > que threshold (0,5) alors true sinon false
        // => seuil de d'activation
        public bool Process(double[] inputs) => inputs
                                                    .Zip(Weights, (v, weight) => v*weight)
                                                    .Sum() > Threshold;

        public bool Learn(bool expectedResult, double[] inputs) {
            var res = Process(inputs);
            if(res == expectedResult) return res;
            double error = (expectedResult?1:0) - (res?1:0);
            Weights = Weights.Zip(inputs, (weight, input) => weight + LearningRate * error * input).ToArray();
            return res;
        }
    }
}
