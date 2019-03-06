using System;
using System.Collections.Generic;
using System.Linq;

namespace simple_ml
{
    public class BinaryPerceptron
    {
        public double LearningRate { set; get; }
    
    
        public double[] Weights { set; get; }
    
    
        public const double Threshold = 0.5;
    
    
        public BinaryPerceptron(int inputCount, double learningRate = 0.1)
        {
            Weights = new double[inputCount];
            LearningRate = learningRate;
        }
    
        public bool Process(params double[] inputs)
        {
            if (inputs.Length != Weights.Length)
                throw new ArgumentException("nombre incorrecte d'inputs");
            
    
            // calculate the perceptron output and use Threshold to return a boolean
            return inputs.Zip(Weights, (value, weight) => value * weight).Sum() > Threshold;
        }
    
        
        public bool Learn(bool expectedResult, params double[] inputs)
        {
            bool result = Process(inputs);
            if (result == expectedResult) return result;
            double error = (expectedResult?1:0) - (result?1:0);
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] += LearningRate * error * inputs[i];
            }
    
            return result;
        }
    
    }
}
