using System;
using System.Collections.Generic;
using UnityEngine;
using simple_ml;

namespace Assets.Scripts
{
    public class UnityInitPerceptron : MonoBehaviour
    {
        // Start is called before the first frame update
        void Start()
        {
            var redSpheres = GameObject.Find("Red").transform;
            var input = new double[redSpheres.childCount][];

            int i = 0;
            foreach (Transform sphere in redSpheres)
            {
                var spherePosition = sphere.position;
                double x = spherePosition.x;
                double z = spherePosition.z;
                input[i] = new[] { x, z };

                Debug.Log($"X: {spherePosition.x} | Y: {spherePosition.y} | Z: {spherePosition.z}");
                i++;
            }
            
            var expectedOutput = new double[input.Length];
            int sphereCount = 0;

            foreach (Transform sphere in redSpheres)
            {
                if (sphere.position.y >= 0) expectedOutput[sphereCount] = 1;
                else expectedOutput[sphereCount] = -1;

                sphereCount++;
            }

            var model = Perceptron.CreateModel(input[0].Length);
            var trainedResult = Perceptron.TrainLinearClassifier(input, expectedOutput, model);
            var whiteSpheres = GameObject.Find("White").transform;

            foreach (Transform sphere in whiteSpheres)
            {
                double[] coordinates = { sphere.position.x, sphere.position.z };
                float newCoordinates = Perceptron.ClassifierLinearInference(model, coordinates);
                sphere.position = new Vector3(sphere.position.x, newCoordinates, sphere.position.z);
            }
        }

        // Update is called once per frame
        void Update()
        {
        
        }
    }
}
