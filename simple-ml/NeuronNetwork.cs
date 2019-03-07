using System;
using System.Collections.Generic;

namespace simple_ml
{
    public class NeuronNetwork
    {
        public NeuronLayer[] Layers;
        
        public NeuronNetwork( params NeuronLayer[] layers)
        {
            Layers = layers;
        }

        public bool Train(double[] inputs, double[] expectedOutput)
        {
            return false;
        }

        public double[] Process(double[] inputs)
        {
            return new double[]{};
        }
        
    }
}