using System;
using System.Collections.Generic;
using System.Linq;

namespace simple_ml
{
    public class Perceptron
    {
        public double LearningRate { set; get; }

        public List<NeuronConnection> Connections { get; set; }
    
    
        private double Input;
        private double outputSum;
        private double bias;
        public double Error { get; private set; }

        public Perceptron(double learningRate = 0.1)
        {
            var rnd = new Random();
            Connections = new List<NeuronConnection>();
            LearningRate = learningRate;
        }

        public void Backpropagation(double weight, double outputNeuronError, double outputNeuronDerivative)
        {
            Error = weight * outputNeuronDerivative * outputNeuronError;
        }
    
    
        public double Derivative(double output)
        {
            double activation = output;
            return activation * (1 - activation);
        }
        
        
        public double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        

        public double Output() => outputSum != double.MinValue ? outputSum : Sigmoid(Input + bias);
        
         
    
        public double Activate(params double[] inputs)
        {
            if (inputs.Length != Connections.Count) throw new ArgumentException("nombre incorrecte d'inputs");
            Input = inputs.Zip(Connections, (inp, con) => con.weight * inp).Sum();
            //inpuntSum = inpuntSum + bias * Weights[Weights.Length - 1];
            return Output();
        }
        

        public void AdjustWeights(double value)
        {
            Error = value;
            Connections.ForEach(con => con.weight += Error * Derivative(Output()) * LearningRate);
        }
    
    }
}
