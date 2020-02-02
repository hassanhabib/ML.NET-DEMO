using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace MlNETDEMO
{
    public class Prediction
    {
        [VectorType(3)]
        public double[] Estimation { get; set; }
    }
}
