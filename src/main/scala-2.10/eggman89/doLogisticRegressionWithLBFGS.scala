package eggman89

import org.apache.spark.SparkContext
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.joda.time
import org.joda.time.DateTime


object doLogisticRegressionWithLBFGS {

  def trainAndTest(sc: SparkContext, dataset: DataFrame) {

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val rdd_total_data = dataset.map(x => LabeledPoint( x(11).toString.toInt - 3, Vectors.dense(x(0).toString.toDouble,
      x(1).toString.toDouble, x(2).toString.toDouble, x(3).toString.toDouble,
      x(4).toString.toDouble, x(5).toString.toDouble, x(6).toString.toDouble,
      x(7).toString.toDouble, x(8).toString.toDouble, x(9).toString.toDouble,x(10).toString.toDouble
      )))

    val splits = rdd_total_data.randomSplit(Array(0.7 , 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    println("Logistic regression")
    val startTime =  new DateTime()
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(7)
      .run(training)
    val endTime = new DateTime()
    val totalTime = new time.Interval(startTime,endTime)
    println("Time to test:" , totalTime.toDuration.getStandardSeconds, "seconds")

    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

      val metrics = new MulticlassMetrics(predictionAndLabels)
      val precision = metrics.precision
    println("Error Percentage = " + (1.0 - precision)* 100)

    predictionAndLabels.toDF("prediction", "label")
    val reg_evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = reg_evaluator.evaluate(predictionAndLabels.toDF("prediction", "label"))
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    println("confusion matrix")
    println(metrics.confusionMatrix)

    }


}
