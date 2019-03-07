using System;
using System.Linq;
using System.Runtime.InteropServices;
using NUnit.Framework;
using simple_ml;

namespace Tests
{
    public class Tests
    {
        private NeuronNetwork netw;
        
        [SetUp]
        public void Setup()
        {
            var input = new NeuronLayer(2);
            var hidden1 = new NeuronLayer(2);
            var output = new NeuronLayer(1);
            input.ConnectToNext(hidden1);
            hidden1.ConnectToNext(output);
            netw = new NeuronNetwork(input, hidden1, output);
        }


        [Test]
        public void AND()
        {
            Console.WriteLine("TestRun");
        }
    }
}