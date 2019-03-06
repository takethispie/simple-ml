using System;
using System.Collections.Generic;
using UnityEngine;
using simple_ml;

namespace Assets.Scripts
{
    public class Perceptron : MonoBehaviour
    {
        // Start is called before the first frame update
        void Start()
        {
            var spheres = GameObject.Find("Red");
            var positions = new List<Vector3>();

            foreach (Transform sphere in spheres.transform)
            {
                var pos = sphere.position;
                positions.Add(pos);
                Debug.Log($"X: {pos.x} | Y: {pos.y} | Z: {pos.z}");
            }

            var input = new double[][] { };
            for(int i = 0; i < positions.Count; i++)
            {
                double x = positions[i].x;
                double y = positions[i].y;
                var pos = new[] { x, y };
                input[i] = pos;
            }

            var model = simple_ml.Perceptron.CreateModel(2);
            var expectedOutput = new double[] {1, 1, -1};
            model = simple_ml.Perceptron.TrainLinearClassification(model, input, expectedOutput, 0.1, 1000);
        }

        // Update is called once per frame
        void Update()
        {
        
        }
    }
}
