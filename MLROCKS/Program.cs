using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLROCKS
{
    class Program
    {
        static void Main(string[] args)
        {

            int[] numbers = new int[] { 1, 2, 4, 3, 2, 4, 2, 1, 10, 2, 3, 2, 4, 5, 3, 2 };

            List<Withdrawl> withdrawls = numbers.Select(number => new Withdrawl { Amount = number }).ToList();

            var machineLearningsContext = new MLContext();

            var estimator = machineLearningsContext.Transforms.DetectIidSpike(
                outputColumnName: nameof(Prediction.Output),
                inputColumnName: nameof(Withdrawl.Amount),
                confidence: 99,
                pvalueHistoryLength: numbers.Length / 2);

            IDataView initialDataView = machineLearningsContext.Data.LoadFromEnumerable(withdrawls);
            IDataView transformedData = estimator.Fit(initialDataView).Transform(initialDataView);

            List<Prediction> predictions = machineLearningsContext.Data.CreateEnumerable<Prediction>(transformedData, false).ToList();

            foreach(Prediction prediction in predictions)
            {
                Console.WriteLine($"{prediction.Output[0]} \t {prediction.Output[1]} \t {prediction.Output[2]}");
            }

            Console.ReadKey();
        }

        class Prediction
        {
            [VectorType]
            public double[] Output { get; set; }
        }
    }
}
