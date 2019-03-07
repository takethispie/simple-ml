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
            netw = new NeuronNetwork();
        }


        [Test]
        public void AND()
        {
            netw.Train(new double[]{1,1}, 1);
            netw.Train(new double[]{0,1}, 0);
            netw.Train(new double[]{1,0}, 1);
            netw.Train(new double[]{0,0}, 0);
            Assert.AreEqual(netw.Activate(new double[]{1,1}), 1);
        }
    }
}