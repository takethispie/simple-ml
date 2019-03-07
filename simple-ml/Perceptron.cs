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

        public double Output
        {
            get => Output != double.MinValue ? Output : Sigmoid(Input + bias);
            set => Output = value;
        }
        
        private double bias;
        public double Error { get; private set; }

        public Perceptron(double learningRate = 0.1)
        {
            var rnd = new Random();
            Connections = new List<NeuronConnection>();
            LearningRate = learningRate;
            bias = 0.0;
        }

        public double Backpropagation(Perceptron neuron)
        {
            var co = neuron.Connections.Find(x => x.From == this);
            return co.weight * neuron.Derivative() * neuron.Error;
        }
    
    
        public double Derivative()
        {
            double activation = Output;
            return activation * (1 - activation);
        }
        
        
        public double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
                
         
    
        public double Activate(params double[] inputs)
        {
            if (inputs.Length != Connections.Count) throw new ArgumentException("nombre incorrecte d'inputs");
            Input = inputs.Zip(Connections, (inp, con) => con.weight * inp).Sum();
            //inpuntSum = inpuntSum + bias * Weights[Weights.Length - 1];
            return Output;
        }
        

        public void AdjustWeights(double value)
        {
            Error = value;
            Connections.ForEach(con => con.weight += Error * Derivative() * LearningRate);
        }
    
    }
}
