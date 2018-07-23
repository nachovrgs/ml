using CsvHelper;
using System;
using System.IO;

namespace KNN
{
    class KNNProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Begin k-NN classification demo ");
            double[][] trainData = LoadData();
            int numFeatures = 2;
            int numClasses = 5;
            double[] unknown = new double[] { 5.25, 1.75 };
            Console.WriteLine("Predictor values: 5.25 1.75 ");
            int k = 1;
            Console.WriteLine("With k = 1");
            int predicted = Classify(unknown, trainData,
              numClasses, k);
            Console.WriteLine("Predicted class = " + predicted);
            k = 4;
            Console.WriteLine("With k = 4");
            predicted = Classify(unknown, trainData,
              numClasses, k);
            Console.WriteLine("Predicted class = " + predicted);
            Console.WriteLine("End k-NN demo ");
            Console.ReadLine();
        }
        
        
        public static double[][] LoadData()
        {
            System.IO.TextReader readFile = new StreamReader("C:\\data.csv");
            var csv = new CsvReader(readFile);
            var records = csv.GetRecords<Item>();
            double[][] data = new double[1000][];

            int i = 0;
            foreach (var row in records)
            {
                data[i] = new double[] { row.distance, row.age, row.result };
                i++;
            }
            return data;
        }

        static int Classify(double[] unknown, double[][] trainData, int numClasses, int k)
        {
            int n = trainData.Length;
            IndexAndDistance[] info = new IndexAndDistance[n];
            for (int i = 0; i < n; ++i)
            {
                IndexAndDistance curr = new IndexAndDistance();
                double dist = Distance(unknown, trainData[i]);
                curr.idx = i;
                curr.dist = dist;
                info[i] = curr;
            }
            int result = Vote(info, trainData, numClasses, k);
            return result;
        }

        static int Vote(IndexAndDistance[] info, double[][] trainData,
  int numClasses, int k)
        {
            int[] votes = new int[numClasses];  // One cell per class
            for (int i = 0; i < k; ++i)
            {       // Just first k
                int idx = info[i].idx;            // Which train item
                int c = (int)trainData[idx][2];   // Class in last cell
                ++votes[c];
            }
            int mostVotes = 0;
            int classWithMostVotes = 0;
            for (int j = 0; j < numClasses; ++j)
            {
                if (votes[j] > mostVotes)
                {
                    mostVotes = votes[j];
                    classWithMostVotes = j;
                }
            }
            return classWithMostVotes;
        }
        static double Distance(double[] unknown, double[] data)
        {
            double sum = 0.0;
            for (int i = 0; i < unknown.Length; ++i)
                sum += (unknown[i] - data[i]) * (unknown[i] - data[i]);
            return Math.Sqrt(sum);
        }

    } // Program class
    public class IndexAndDistance : IComparable<IndexAndDistance>
    {
        public int idx;  // Index of a training item
        public double dist;  // To unknown
                             // Need to sort these to find k closest
        public int CompareTo(IndexAndDistance other)
        {
            if (this.dist < other.dist) return -1;
            else if (this.dist > other.dist) return +1;
            else return 0;
        }
    }

    
}
