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

        private static BinaryPerceptron RunPerceptron(TrainingItem[] items, int inputCount)
        {
            var perc = new BinaryPerceptron(inputCount);
            var nbEpoch = 5000;
            var counter = 0;
            var errorCount = 0;
            while (counter < nbEpoch)
            {
                counter++;
                errorCount = 0;
                foreach (var item in items)
                {
                    if(perc.Learn(item.Output, item.Inputs) != item.Output) errorCount++;
                }
                if (errorCount == 0) break;
            }

            return perc;
        }

        [Test]
        public void AND()
        {
            TrainingItem[] trainingSet = 
            { 
                new TrainingItem(false, 1, 0), 
                new TrainingItem(false, 0, 1), 
                new TrainingItem(true, 1, 1)
            };


            var perc = RunPerceptron(trainingSet, 2);
            
            Assert.IsTrue(perc.Process(1,0) == false);
            Assert.IsTrue(perc.Process(0,0) == false);
            Assert.IsTrue(perc.Process(1,1));
        }


        [Test]
        public void OR()
        {
            TrainingItem[] trainingSet = 
            { 
                new TrainingItem(false, 1, 0, 0), 
                new TrainingItem(true, 1, 1, 0), 
                new TrainingItem(true, 1, 0, 1), 
                new TrainingItem(true, 1, 1, 1)
            };
            
            var perc = RunPerceptron(trainingSet, 3);
            
            Assert.IsTrue(perc.Process(1, 1,0));
            Assert.IsTrue(perc.Process(1, 0,0) == false);
        }

        [Test]
        public void NAND()
        {
            TrainingItem[] trainingSet = 
            { 
                new TrainingItem(true, 1, 1, 0), 
                new TrainingItem(true, 1, 0, 1), 
                new TrainingItem(false, 1, 1, 1)
            };

            var perc = RunPerceptron(trainingSet, 3);
            
            Assert.IsTrue(perc.Process(1, 1, 1) == false);
            Assert.IsTrue(perc.Process(1, 1, 0));

        }
        
        [Test]
        public void XORShouldFail()
        {
            TrainingItem[] trainingSet = 
            { 
                new TrainingItem(true, 1, 0, 0),
                new TrainingItem(false, 1, 1, 0), 
                new TrainingItem(false, 1, 0, 1), 
                new TrainingItem(true, 1, 1, 1)
            };

            var perc = RunPerceptron(trainingSet, 3);
            
            Assert.IsTrue(perc.Process(1, 1, 1), "this test should fail");
            Assert.IsTrue(perc.Process(1, 0, 0));
            Assert.IsTrue(perc.Process(1, 1, 0) == false);
            Assert.IsTrue(perc.Process(1, 0, 1) == false);

        }
    }
}