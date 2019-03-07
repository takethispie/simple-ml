using UnityEngine;
using simple_ml;

namespace Assets.Scripts
{
    public class UnityPerceptron : MonoBehaviour
    {
        public int MethodToUse = 0;

        // Start is called before the first frame update
        void Start()
        {
            switch (MethodToUse)
            {
                case 0:
                    LinearClassifier();
                    break;
                case 1:
                    LinearRegression();
                    break;
            }
        }

        // Update is called once per frame
        void Update()
        {

        }

        private static void LinearClassifier()
        {
            var redSpheres = GameObject.Find("Red").transform;
            var input = new double[redSpheres.childCount][];
            var expectedOutput = new double[input.Length];
            int redSpheresCount = 0;

            foreach (Transform sphere in redSpheres)
            {
                var spherePosition = sphere.position;
                double x = spherePosition.x;
                double z = spherePosition.z;

                input[redSpheresCount] = new[] { x, z };

                if (sphere.position.y >= 0) expectedOutput[redSpheresCount] = 1;
                else expectedOutput[redSpheresCount] = -1;

                //Debug.Log($"X: {spherePosition.x} | Y: {spherePosition.y} | Z: {spherePosition.z}");
                redSpheresCount++;
            }
            
            var model = SimpleLinearClassifier.CreateModel(input[0].Length);
            var result = SimpleLinearClassifier.TrainLinearClassifier(input, expectedOutput, model);
            var whiteSpheres = GameObject.Find("White").transform;

            foreach (Transform sphere in whiteSpheres)
            {
                double[] coordinates = { sphere.position.x, sphere.position.z };
                float newCoordinates = SimpleLinearClassifier.ClassifierLinearInference(model, coordinates);
                sphere.position = new Vector3(sphere.position.x, newCoordinates, sphere.position.z);
            }
        }

        private static void LinearRegression()
        {
            var redSpheres = GameObject.Find("Red").transform;
            var input = new double[redSpheres.childCount][];
            var expectedOutput = new double[input.Length];
            int redSpheresCount = 0;

            foreach (Transform sphere in redSpheres)
            {
                var spherePosition = sphere.position;
                double x = spherePosition.x;
                double z = spherePosition.z;

                input[redSpheresCount] = new[] { x, z };

                if (sphere.position.y >= 0) expectedOutput[redSpheresCount] = 1;
                else expectedOutput[redSpheresCount] = -1;

                //Debug.Log($"X: {spherePosition.x} | Y: {spherePosition.y} | Z: {spherePosition.z}");
                redSpheresCount++;
            }

            var regression = new SimpleLinearRegression(input, expectedOutput);
            var result = regression.Compute();

            var whiteSpheres = GameObject.Find("White").transform;

            foreach (Transform sphere in whiteSpheres)
            {
                double[] coordinates = { sphere.position.x, sphere.position.z };
                //float newCoordinates = SimpleLinearClassifier.ClassifierLinearInference(model, coordinates);
                //sphere.position = new Vector3(sphere.position.x, newCoordinates, sphere.position.z);
            }
        }
    }
}
