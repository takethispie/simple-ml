using System;
using System.Collections.Generic;

namespace simple_ml {
    public class NeuronNetwork {
        public NeuronLayer InputLayer, HiddenLayer;
        public Perceptron Output;
        private int nbEpoch = 5000;
        
        public NeuronNetwork() {
            ResetNetwork();
        }

        public void Train(double[] inputs, double expectedOutput) {
            double error;
            int iteration = 0;
            do {
                error = 0;
                var delta = expectedOutput - Activate(inputs);
                AdjustWeights(delta);
                error += Math.Pow(delta, 2);
                iteration++;
                if(iteration > nbEpoch) ResetNetwork();
            } while(error > 0.1);
        }

        public void ResetNetwork() {
            InputLayer = new NeuronLayer(2);
            HiddenLayer = new NeuronLayer(2);
            Output = new Perceptron();
            InputLayer.ConnectToNext(HiddenLayer);
            HiddenLayer.ForEach(x => Output.Connections.Add(new NeuronConnection(x)));
        }

        public double[] Process(double[] inputs) {
            return new double[]{};
        }

        public double Activate(double[] inputs) {
            for (int i = 0; i < inputs.Length; i++) InputLayer[i].Output = inputs[i];
            foreach (Perceptron neuron in HiddenLayer) neuron.Activate();
            Output.Activate();
            return Output.Output;
        }

        public void AdjustWeights(double delta) {
            Output.AdjustWeights(delta);
            foreach (var neuron in HiddenLayer) neuron.AdjustWeights(Output.Backpropagation(neuron));
        }
    }
}