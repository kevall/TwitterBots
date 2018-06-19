import scala.util.{Try, Success, Failure}
import java.io.FileWriter
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.classification.{LogisticRegressionModel, LogisticRegression, 
                                           DecisionTreeClassifier, DecisionTreeClassificationModel,
                                           NaiveBayes, RandomForestClassifier, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object BotAnalysis {
  def limit(x: Double, min: Double, max: Double): Double =
    if (x < min) min else if (x > max) max else x
  
  def main(args: Array[String]) {
    val eval = new MulticlassClassificationEvaluator()
	val fw = new FileWriter("/s/bach/g/under/kevincb/BotOutput.txt", false)
    val spark = SparkSession.builder.appName("BotAnalysis").getOrCreate()
    val sc = SparkContext.getOrCreate()
    import spark.implicits._
    

    //Reads in tweets file, parses it, cleans it, converts it to doubles, merges on user_ids
    val botData = spark.read.textFile("/BotData").rdd.map(x => 
        x.split(",| |\t")).map(
        x => (x(0), x.drop(1).map(s => s.toDouble))).filter(
        x => x._2.count(_.isNaN) == 0 && x._2.sum != 0 && x._2.length == 23
    ).groupByKey()

    //Combines tweets by user_id, averages all relevant values
    //reply to status id left out s(15) 
    //Lang left out s(5)
    //Time zone left out s(6)
    //Reply_count left out (all zeores) s(18)
    val collatedData = botData.map{ case (key,value) =>
        val v = value.toList
        val length = v.length
        val numReplies = v.filter(x => x(0) != 0.0).length
        val s = v.reduce{ (a,b) =>
          Array[Double](a(0),a(1), a(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9), a(10), a(11), a(12), a(13), a(14), a(15),
           a(16), a(17)+b(17), a(18)+a(18), a(19)+b(19), a(20)+b(20), a(21)+b(21), a(22)+b(22))
        }
        if (s.count(_.hashCode.isNaN) == 0) {
            (s(13), Vectors.dense(Array[Double](s(0),s(1),s(2),s(3),s(4),s(7),s(8),s(9),s(10),s(11),s(12),s(14),
                    numReplies/length,s(17)/length,s(19)/length,s(20)/length,s(21)/length,s(22)/length)))
        } else { null }
    }.toDF("label", "features")
    
    //Standarization
    //val scaler = new StandardScaler().setWithStd(true).setWithMean(true).setInputCol("features").setOutputCol("scaledFeatures").fit(collatedData)
    //val input = scaler.transform(collatedData)
    
    /*
    val input = collatedData.keys.zip(dataValues).map{ case(key,value) =>
        new LabeledPoint(key, value)
    }
    */
    val splits = collatedData.randomSplit(Array(0.8, 0.2))
    val training = splits(0).cache()
    val test = splits(1).cache()
    
    fw.write("LOGISTIC REGRESSION\n")
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setFitIntercept(true).fit(training)
    var coef = lr.coefficientMatrix
    fw.write("Model coefficients: \n")
    fw.write(coef.toString(Int.MaxValue, Int.MaxValue) + "\n")
    var testPredictions = lr.transform(test).cache()
    var testZeroPrecision = testPredictions.filter(_(4) == 0.0)
    var testOnePrecision = testPredictions.filter(_(4) == 1.0)
    var testTwoPrecision = testPredictions.filter(_(4) == 2.0)
    
    var testZeroRecall = testPredictions.filter(r => r(0) == 0.0)
    var testOneRecall = testPredictions.filter(r => r(0) == 1.0)
    var testTwoRecall = testPredictions.filter(r => r(0) == 2.0)
    
    fw.write("Zero precision: " + eval.setMetricName("weightedPrecision").evaluate(testZeroPrecision) + "\n")
    fw.write("One precision: " + eval.setMetricName("weightedPrecision").evaluate(testOnePrecision) + "\n")
    fw.write("Two precision: " + eval.setMetricName("weightedPrecision").evaluate(testTwoPrecision) + "\n")
    
    fw.write("Zero recall: " + eval.setMetricName("weightedRecall").evaluate(testZeroRecall) + "\n")
    fw.write("One recall: " + eval.setMetricName("weightedRecall").evaluate(testOneRecall) + "\n")
    fw.write("Two recall: " + eval.setMetricName("weightedRecall").evaluate(testTwoRecall) + "\n")
    
    fw.write("Total accuracy: " + eval.setMetricName("accuracy").evaluate(testPredictions) + "\n")
    fw.write(s"\n\n\n")
    
    
    fw.write("DECISION TREE\n")
    val dtc = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").fit(training)
    fw.write(dtc.toDebugString + "\n")
    testPredictions = dtc.transform(test).cache()
    testZeroPrecision = testPredictions.filter(_(4) == 0.0)
    testOnePrecision = testPredictions.filter(_(4) == 1.0)
    testTwoPrecision = testPredictions.filter(_(4) == 2.0)
    
    testZeroRecall = testPredictions.filter(r => r(0) == 0.0)
    testOneRecall = testPredictions.filter(r => r(0) == 1.0)
    testTwoRecall = testPredictions.filter(r => r(0) == 2.0)
    
    fw.write("Zero precision: " + eval.setMetricName("weightedPrecision").evaluate(testZeroPrecision) + "\n")
    fw.write("One precision: " + eval.setMetricName("weightedPrecision").evaluate(testOnePrecision) + "\n")
    fw.write("Two precision: " + eval.setMetricName("weightedPrecision").evaluate(testTwoPrecision) + "\n")
    
    fw.write("Zero recall: " + eval.setMetricName("weightedRecall").evaluate(testZeroRecall) + "\n")
    fw.write("One recall: " + eval.setMetricName("weightedRecall").evaluate(testOneRecall) + "\n")
    fw.write("Two recall: " + eval.setMetricName("weightedRecall").evaluate(testTwoRecall) + "\n")
    
    fw.write("Total accuracy: " + eval.setMetricName("accuracy").evaluate(testPredictions) + "\n")
    fw.write(s"\n\n\n")
    
    
    fw.write("NAIVE BAYES\n")
    val nb = new NaiveBayes().setLabelCol("label").setFeaturesCol("features").fit(training)
    testPredictions = nb.transform(test).cache()
    testZeroPrecision = testPredictions.filter(_(4) == 0.0)
    testOnePrecision = testPredictions.filter(_(4) == 1.0)
    testTwoPrecision = testPredictions.filter(_(4) == 2.0)
    
    testZeroRecall = testPredictions.filter(r => r(0) == 0.0)
    testOneRecall = testPredictions.filter(r => r(0) == 1.0)
    testTwoRecall = testPredictions.filter(r => r(0) == 2.0)
    
    fw.write("Zero precision: " + eval.setMetricName("weightedPrecision").evaluate(testZeroPrecision) + "\n")
    fw.write("One precision: " + eval.setMetricName("weightedPrecision").evaluate(testOnePrecision) + "\n")
    fw.write("Two precision: " + eval.setMetricName("weightedPrecision").evaluate(testTwoPrecision) + "\n")
    
    fw.write("Zero recall: " + eval.setMetricName("weightedRecall").evaluate(testZeroRecall) + "\n")
    fw.write("One recall: " + eval.setMetricName("weightedRecall").evaluate(testOneRecall) + "\n")
    fw.write("Two recall: " + eval.setMetricName("weightedRecall").evaluate(testTwoRecall) + "\n")
    
    fw.write("Total accuracy: " + eval.setMetricName("accuracy").evaluate(testPredictions) + "\n")
    fw.write(s"\n\n\n")
    
    
    fw.write("RANDOM FOREST (30 trees)\n")
    val rfc = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(30).fit(training)
    testPredictions = rfc.transform(test).cache()
    testZeroPrecision = testPredictions.filter(_(4) == 0.0)
    testOnePrecision = testPredictions.filter(_(4) == 1.0)
    testTwoPrecision = testPredictions.filter(_(4) == 2.0)
    
    testZeroRecall = testPredictions.filter(r => r(0) == 0.0)
    testOneRecall = testPredictions.filter(r => r(0) == 1.0)
    testTwoRecall = testPredictions.filter(r => r(0) == 2.0)
    
    fw.write("Zero precision: " + eval.setMetricName("weightedPrecision").evaluate(testZeroPrecision) + "\n")
    fw.write("One precision: " + eval.setMetricName("weightedPrecision").evaluate(testOnePrecision) + "\n")
    fw.write("Two precision: " + eval.setMetricName("weightedPrecision").evaluate(testTwoPrecision) + "\n")
    
    fw.write("Zero recall: " + eval.setMetricName("weightedRecall").evaluate(testZeroRecall) + "\n")
    fw.write("One recall: " + eval.setMetricName("weightedRecall").evaluate(testOneRecall) + "\n")
    fw.write("Two recall: " + eval.setMetricName("weightedRecall").evaluate(testTwoRecall) + "\n")
    
    fw.write("Total accuracy: " + eval.setMetricName("accuracy").evaluate(testPredictions) + "\n")
    fw.write(s"\n\n\n")
    

    fw.write("MULTI-LAYER PERCEPTRON [18,30,30,3] \n")
    val mlp = new MultilayerPerceptronClassifier().setLayers(Array[Int](18,30,30,3)).setLabelCol("label").setFeaturesCol("features").fit(training)
    testPredictions = mlp.transform(test).cache()
    testZeroPrecision = testPredictions.filter(_(4) == 0.0)
    testOnePrecision = testPredictions.filter(_(4) == 1.0)
    testTwoPrecision = testPredictions.filter(_(4) == 2.0)

    testZeroRecall = testPredictions.filter(r => r(0) == 0.0)
    testOneRecall = testPredictions.filter(r => r(0) == 1.0)
    testTwoRecall = testPredictions.filter(r => r(0) == 2.0)

    fw.write("Zero precision: " + eval.setMetricName("weightedPrecision").evaluate(testZeroPrecision) + "\n")
    fw.write("One precision: " + eval.setMetricName("weightedPrecision").evaluate(testOnePrecision) + "\n")
    fw.write("Two precision: " + eval.setMetricName("weightedPrecision").evaluate(testTwoPrecision) + "\n")

    fw.write("Zero recall: " + eval.setMetricName("weightedRecall").evaluate(testZeroRecall) + "\n")
    fw.write("One recall: " + eval.setMetricName("weightedRecall").evaluate(testOneRecall) + "\n")
    fw.write("Two recall: " + eval.setMetricName("weightedRecall").evaluate(testTwoRecall) + "\n")

    fw.write("Total accuracy: " + eval.setMetricName("accuracy").evaluate(testPredictions) + "\n")
    fw.write(s"\n\n\n")


    fw.write("FULL ENSEMBLE\n")
    val outputs = rfc.transform(test).select("prediction").rdd.zip(nb.transform(test).select("prediction").rdd).zip(
                  dtc.transform(test).select("prediction").rdd).zip(lr.transform(test).select("prediction").rdd).zip(mlp.transform(test).select("prediction").rdd).map{ 
        case (((((a, b), c), d), e)) => Array[String](a(0).toString,b(0).toString,
                                               c(0).toString,d(0).toString,e(0).toString).groupBy(identity).maxBy(_._2.size)._1
    }.map(_.toDouble).toDF("prediction")
    testPredictions = testPredictions.drop("prediction").withColumn("id", monotonically_increasing_id()).join(outputs.withColumn("id", 
                      monotonically_increasing_id()), Seq("id")).drop("id").cache()
    testZeroPrecision = testPredictions.filter(_(4) == 0.0)
    testOnePrecision = testPredictions.filter(_(4) == 1.0)
    testTwoPrecision = testPredictions.filter(_(4) == 2.0)
    
    testZeroRecall = testPredictions.filter(r => r(0) == 0.0)
    testOneRecall = testPredictions.filter(r => r(0) == 1.0)
    testTwoRecall = testPredictions.filter(r => r(0) == 2.0)
    
    fw.write("Zero precision: " + eval.setMetricName("weightedPrecision").evaluate(testZeroPrecision) + "\n")
    fw.write("One precision: " + eval.setMetricName("weightedPrecision").evaluate(testOnePrecision) + "\n")
    fw.write("Two precision: " + eval.setMetricName("weightedPrecision").evaluate(testTwoPrecision) + "\n")
    
    fw.write("Zero recall: " + eval.setMetricName("weightedRecall").evaluate(testZeroRecall) + "\n")
    fw.write("One recall: " + eval.setMetricName("weightedRecall").evaluate(testOneRecall) + "\n")
    fw.write("Two recall: " + eval.setMetricName("weightedRecall").evaluate(testTwoRecall) + "\n")
    
    fw.write("Total accuracy: " + eval.setMetricName("accuracy").evaluate(testPredictions) + "\n")
    fw.write(s"\n\n\n")
    fw.close()
  }
}
