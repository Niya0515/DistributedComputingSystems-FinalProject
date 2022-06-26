package cn.edu.ecnu;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

public class SVM {

	public  static void main(String[] args) throws Exception {

		SparkConf conf = new SparkConf().setAppName("SVM").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);

		JavaRDD<String> csvData0  = sc.textFile(args[0],1);

		JavaRDD<LabeledPoint> training = csvData0.map(
			new Function<String,LabeledPoint> () {

				@Override
				public LabeledPoint call(String v1) throws Exception {

					int firstIndex = v1.indexOf(",");
					int secondIndex = v1.indexOf(",", firstIndex+1);


					double label = Double.parseDouble(v1.substring(0, firstIndex));
					String featureString[] = v1.substring(secondIndex + 1).trim().split(" ");
					double[] v = new double[featureString.length];
					int i = 0;
					for (String s : featureString) {
						if (s.trim().equals(""))
							continue;
						v[i++] = Double.parseDouble(s.trim());
					}
					return new LabeledPoint(label, Vectors.dense(v));
				}

			}
		);

		System.out.println("count ="  + training.count());

		JavaRDD<String> csvData1  = sc.textFile(args[1],1);
			
		JavaRDD<LabeledPoint> test = csvData1.map(

			new Function<String,LabeledPoint> () {

				@Override
				public LabeledPoint call(String v1) throws Exception {
					int firstIndex = v1.indexOf(",");
					int secondIndex = v1.indexOf(",", firstIndex+1);


					double label = Double.parseDouble(v1.substring(0, firstIndex));
					String featureString[] = v1.substring(secondIndex + 1).trim().split(" ");

					double[] v = new double[featureString.length];
					int i = 0;
					for (String s : featureString) {
						if (s.trim().equals(""))
							continue;
						v[i++] = Double.parseDouble(s.trim());
					}
					return new LabeledPoint(label, Vectors.dense(v));
				}

			}
		);

		System.out.println(test.count());

		final SVMModel svmModel = SVMWithSGD.train(training.rdd(), Integer.parseInt(args[2]));

		JavaPairRDD<Double, Double> predictionAndLabelSVM = test.mapToPair(

			new PairFunction<LabeledPoint, Double, Double>() {

				@Override
				public Tuple2<Double, Double> call(LabeledPoint p) {
					return new Tuple2<Double, Double>(svmModel.predict(p.features()), p.label());
				}
			}
		);

		double accuracySVM = 1.0* predictionAndLabelSVM.filter(
			new Function<Tuple2<Double, Double>, Boolean>() {
				@Override
				public Boolean call(Tuple2<Double, Double> pl) {
					
					return pl._1().intValue() == pl._2().intValue();
				}
			}
		).count() / (double)test.count();

		System.out.println("svm accuracy : " + accuracySVM);

	}
}