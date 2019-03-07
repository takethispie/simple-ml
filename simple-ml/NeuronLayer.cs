using System.Collections.Generic;

namespace simple_ml
{
    public class NeuronLayer : List<Perceptron> {
        public NeuronLayer(int neuronCount) {
            for (int i = 0; i < neuronCount; i++) Add(new Perceptron());   
        }

        public void ConnectToNext(NeuronLayer layer) {
            layer.ForEach(neu => {
                ForEach(inp => neu.Connections.Add(new NeuronConnection(inp)));
            });
        }

        
        public void adjustWeight() {
            ForEach(perc => {
               perc.Connections.ForEach(child => child.From.AdjustWeights(child.From.Backpropagation(perc))); 
            });
        }
        
    }
}