package eggman89

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, RandomForestRegressor}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.joda.time
import org.joda.time.DateTime

object doRandomForest {
  def trainAndTest(sc: SparkContext, dataset: DataFrame)
  {

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._


    //merging features into ione
    val assembler = new VectorAssembler()
      .setInputCols(Array("fixed acidity", "volatile acidity", "citric acid","residual sugar",
        "chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"
      ))
      .setOutputCol("features")

    var df_total_dataset = assembler.transform(dataset).select("features", "quality")

    df_total_dataset = df_total_dataset.selectExpr("cast(quality as double) as quality", "features")

    val labelIndexer = new StringIndexer()
      .setInputCol("quality")
      .setOutputCol("indexedQuality")
      .fit(df_total_dataset)


    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(df_total_dataset)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedQuality")
      .setLabels(labelIndexer.labels)

    //splitting in the ratio of training/testing : 70/30
    val Array(df_trainingData, df_testData) = df_total_dataset.randomSplit(Array(0.7, 0.3))


    /*do Random Forest classifier*/
    val rfc = new RandomForestClassifier().setMaxBins(15).setMaxDepth(30).setMaxMemoryInMB(2048).setNumTrees(1000)
      .setLabelCol("indexedQuality")
      .setFeaturesCol("indexedFeatures")

    val pipeline = new Pipeline()
      .setStages(Array(  labelIndexer, featureIndexer, rfc, labelConverter))

    println("Random Forest : Classifier")
    var startTime =  new DateTime()
    var model = pipeline.fit(df_trainingData)
    var predictions = model.transform(df_testData)
    var endTime = new DateTime()
    var totalTime = new time.Interval(startTime,endTime)
    println("Time to test:" , totalTime.toDuration.getStandardSeconds, "seconds")
    // predictions.show(30)

    var evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedQuality")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    var accuracy = evaluator.evaluate(predictions)
    println("Error Percentage = " + (1.0 - accuracy)* 100)

    var  reg_evaluator = new RegressionEvaluator()
      .setLabelCol("indexedQuality")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    var rmse = reg_evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)
    //confsuion matrix
    println("confusion matrix")
    var predictionsAndLabels = predictions.map(p=>(p(0).toString.toDouble, p(7).toString.toDouble))
    var confusion_matrix = new  MulticlassMetrics(predictionsAndLabels)
    println(confusion_matrix.confusionMatrix)
    /*do Random Forest Regressor*/
    val rfr = new RandomForestRegressor().setMaxBins(15).setMaxDepth(30).setMaxMemoryInMB(4096).setNumTrees(2500)
      .setLabelCol("quality")
      .setFeaturesCol("indexedFeatures")

    val pipeline2 = new Pipeline()
      .setStages(Array(  featureIndexer, rfr))

    println("Random Forest : Regressor")
    startTime =  new DateTime()
    model = pipeline2.fit(df_trainingData)
    predictions = model.transform(df_testData)
    endTime = new DateTime()
    totalTime = new time.Interval(startTime,endTime)
    println("Time to test:" , totalTime.toDuration.getStandardSeconds, "seconds")

    //rounding the quality field
    val predictions2 = predictions.map(x =>  (  x(0).toString.toDouble ,x(1).toString,x(2).toString, math.round(x(3).toString.toDouble).toDouble)).toDF("quality","features","indexedFeatures", "prediction" )

    evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("quality")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    accuracy = evaluator.evaluate(predictions2)
    println("Error Percentage = " + (1.0 - accuracy)* 100)

    reg_evaluator = new RegressionEvaluator()
      .setLabelCol("quality")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    rmse = reg_evaluator.evaluate(predictions2)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    //confusion matrix
    println("confusion matrix")
    predictionsAndLabels = predictions2.map(p=>(p(0).toString.toDouble, p(3).toString.toDouble))
    confusion_matrix = new  MulticlassMetrics(predictionsAndLabels)
    println(confusion_matrix.confusionMatrix)

  }

}
