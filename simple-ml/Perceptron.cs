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
        private double output;

        public double Output {
            get => output != double.MinValue ? output : 1 / (1 + Math.Exp(Input + bias));
            set => output = value;
        }
        
        private double bias;
        public double Error { get; private set; }

        public Perceptron(double learningRate = 0.1) {
            var rnd = new Random();
            Connections = new List<NeuronConnection>();
            LearningRate = learningRate;
            bias = 0.0;
            Input = 0.0;
        }

        public double Backpropagation(Perceptron neuron) {
            var co = neuron.Connections.Find(x => x.From.Equals(this));
            return co.weight * neuron.Derivative() * neuron.Error;
        }
    
    
        public double Derivative() {
            double activation = Output;
            return activation * (1 - activation);
        }
                
        public double Activate() {
            Input = Connections.Select(con => con.weight * con.From.Output).Sum();
            return Output;
        }
        

        public void AdjustWeights(double value) {
            Error = value;
            Connections.ForEach(con => con.weight += Error * Derivative() * LearningRate);
        }
    }
}
