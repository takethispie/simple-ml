using System.Linq;
using NUnit.Framework;
using simple_ml;

namespace Tests
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Test1()
        {
            TrainingItem[] trainingSet = 
            { 
                new TrainingItem(false, 1, 0), 
                new TrainingItem(false, 1, 0), 
                new TrainingItem(true, 1, 1)
            };
            

            var perc = new BinaryPerceptron(2);
            while (true)
            {
                var errorCount = 0;
                foreach (var item in trainingSet)
                {
                    var output = perc.Learn(item.Output, item.Inputs);
                    if (item.Output != output) errorCount++;
                }

                if (errorCount == 0) break;
            }
            
            Assert.IsTrue(perc.Process(1,0) == false);
            Assert.IsTrue(perc.Process(0,0) == false);
        }
    }
}