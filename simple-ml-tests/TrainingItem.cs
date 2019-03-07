namespace Tests
{
    public class TrainingItem
    {
        public double[] Inputs { get; private set; }

        public double[] Output { get; private set; }

        public TrainingItem(double[] expectedOutput, params double[] inputs)
        {
            Inputs = inputs;
            Output = expectedOutput;
        }
    }
}