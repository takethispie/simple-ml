namespace Tests
{
    public class TrainingItem
    {
        public double[] Inputs { get; private set; }

        public bool Output { get; private set; }

        public TrainingItem(bool expectedOutput, params double[] inputs)
        {
            Inputs = inputs;
            Output = expectedOutput;
        }
    }
}