using UnityEngine;
using simple_ml;

namespace Assets.Scripts
{
    public class UnityPerceptron : MonoBehaviour
    {
        public int MethodToUse;

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
                case 2:
                    NonLinearClassifier();
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

                redSpheresCount++;
            }
            
            var model = SimpleLinearClassifier.CreateModel(input[0].Length);
            var result = SimpleLinearClassifier.Train(input, expectedOutput, model);
            var whiteSpheres = GameObject.Find("White").transform;

            foreach (Transform sphere in whiteSpheres)
            {
                double[] coordinates = { sphere.position.x, sphere.position.z };
                float newCoordinates = SimpleLinearClassifier.LinearInference(result, coordinates);
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

                expectedOutput[redSpheresCount] = sphere.position.y;
                
                redSpheresCount++;
            }

            var regression = new SimpleLinearRegression(input, expectedOutput);
            var model = regression.Compute();

            var whiteSpheres = GameObject.Find("White").transform;

            foreach (Transform sphere in whiteSpheres)
            {
                double[] coordinates = { sphere.position.x, sphere.position.z };
                float newCoordinates = (float) regression.LinearInference(model, coordinates);
                sphere.position = new Vector3(sphere.position.x, newCoordinates, sphere.position.z);
            }
        }

        private static void NonLinearClassifier()
        {
            var redSpheres = GameObject.Find("Red").transform;
            var input = new double[redSpheres.childCount][];
            var expectedOutput = new double[input.Length];
            int redSpheresCount = 0;

            foreach (Transform sphere in redSpheres)
            {
                var spherePosition = sphere.position;
                double x = Mathf.Pow(spherePosition.x, 2);
                double z = Mathf.Pow(spherePosition.z, 2);

                input[redSpheresCount] = new[] { x, z };

                if (sphere.position.y >= 0) expectedOutput[redSpheresCount] = 1;
                else expectedOutput[redSpheresCount] = -1;

                Debug.Log($"X: {spherePosition.x} | Y: {spherePosition.y} | Z: {spherePosition.z}");
                redSpheresCount++;
            }

            var model = SimpleLinearClassifier.CreateModel(input[0].Length);
            var result = SimpleLinearClassifier.Train(input, expectedOutput, model);
            var whiteSpheres = GameObject.Find("White").transform;

            foreach (Transform sphere in whiteSpheres)
            {
                double[] coordinates = { sphere.position.x, sphere.position.z };
                float newCoordinates = SimpleLinearClassifier.LinearInference(model, coordinates);
                sphere.position = new Vector3(sphere.position.x, newCoordinates, sphere.position.z);
            }
        }
    }
}
