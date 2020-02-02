using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLISAWESOME
{
    class Program
    {
        static void Main(string[] args)
        {
            // create some data
            int[] amounts = new int[] { 100, 150, 200, 300, 250, 3000, 100, 250, 300, 250 };

            List<Withdrawl> withdrawls =
                amounts.Select(amount => new Withdrawl { Amount = amount }).ToList();

            // instantiate machine learning context
            var machineLearningContext = new MLContext();

            // create your algorithm
            var estimator = machineLearningContext.Transforms.DetectIidSpike(
                outputColumnName: nameof(Predication.Output),
                inputColumnName: nameof(Withdrawl.Amount),
                confidence: 99,
                pvalueHistoryLength: amounts.Length / 2);

            // link data to algorithm
            IDataView amountsData = machineLearningContext.Data.LoadFromEnumerable(withdrawls);
            IDataView transformedAmountsData = estimator.Fit(amountsData).Transform(amountsData);

            // create output
            List<Predication> predictions =
                machineLearningContext.Data
                    .CreateEnumerable<Predication>(transformedAmountsData, reuseRowObject: false).ToList();

            foreach(Predication prediction in predictions)
            {
                double isAnomaly = prediction.Output[0];
                double originalValue = prediction.Output[1];
                double confidenceLevel = prediction.Output[2];

                Console.WriteLine($"{originalValue} \t {confidenceLevel} \t {isAnomaly}");
            }

            Console.ReadKey();
        }
    }

    class Withdrawl
    {
        public float Amount { get; set; }
    }

    class Predication
    {
        [VectorType]
        public double[] Output { get; set; }
    }
}
