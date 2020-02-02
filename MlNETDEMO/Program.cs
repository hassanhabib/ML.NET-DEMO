using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace MlNETDEMO
{
    class Program
    {
        static void Main(string[] args)
        {
            var withdrawals = new List<Withdrawl>
            {
                new Withdrawl
                {
                    Amount = 100
                },
                new Withdrawl
                {
                    Amount = 50
                },
                new Withdrawl
                {
                    Amount = 175
                },
                new Withdrawl{
                    Amount = 3000
                }
            };

            var context = new MLContext();

            var estimator = context.Data.LoadFromEnumerable(
                 nameof(Withdrawl.Amount));

            Console.WriteLine("Hello World!");
        }
    }
}
