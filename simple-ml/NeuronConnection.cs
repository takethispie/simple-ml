using System;

namespace simple_ml
{
    public class NeuronConnection
    {
        public double weight;
        public Perceptron From;

        public NeuronConnection(Perceptron from)
        {
            From = from;
            weight = new Random().NextDouble();
        }
    }
}