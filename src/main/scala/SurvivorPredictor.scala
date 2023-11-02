import com.sun.jmx.mbeanserver.Util.cast
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions.{avg, udf}
import org.apache.spark.sql.types.BooleanType

object SurvivorPredictor extends App {

  val sparkConf = new SparkConf()
  sparkConf.setMaster("local[*]")

  val spark = SparkSession.builder.config(sparkConf).getOrCreate()

  // Load the training and testing datasets as Spark DataFrames
  val test = spark.read.option("inferSchema", value = true).option("header", value = true).csv("C:\\Users\\mehul\\Documents\\College_Grad\\Sem_3\\INFO_7200\\Assingment\\Spark 2\\test.csv")
  val train  = spark.read.option("inferSchema", value = true).option("header", value = true).csv("C:\\Users\\mehul\\Documents\\College_Grad\\Sem_3\\INFO_7200\\Assingment\\Spark 2\\train.csv")

  train.printSchema()
  train.show()

  import spark.implicits._

  //Follow up on the previous spark assignment 1 and explained a few statistics.
  train.describe().show()
  train.groupBy("Pclass").count().show()
  train.groupBy("Embarked").count().show()
  train.groupBy("sex", "pclass")
    .agg(avg("Age").as("Average Age")).show()

  //Create new attributes that may be derived from the existing attributes
  val isAlone = udf((sibsp: Int, parch: Int, age: Int) => sibsp == 0 && parch == 0 && age > 17)

  train.filter($"Age" < 18).show()


  val trainIsAlone = train.withColumn("isAlone",
    isAlone($"sibsp", $"parch", $"Age"))
  val testIsAlone = test.withColumn("isAlone",
    isAlone($"sibsp", $"parch", $"Age"))

  trainIsAlone.show()

  val companion = udf((sibsp: Int, parch: Int, age: Int) =>
    if ((sibsp + parch == 0) && age < 18) 1 else sibsp + parch)

  val trainCompanion = train.withColumn("Companions",
    companion($"sibsp", $"parch", $"Age"))
  val testCompanion = test.withColumn("Companions",
    companion($"sibsp", $"parch", $"Age"))

  trainCompanion.show()

  // Cleaning the data for prediction
  val trainCleaned = trainCompanion.na.drop()
  val testCleaned = testCompanion.na.drop()

  val avgAge = trainCleaned.agg(avg("Age")).first()(0).asInstanceOf[Double]
  val avgFare = trainCleaned.agg(avg("Fare")).first()(0).asInstanceOf[Double]
  val trainFilled = trainCleaned.na.fill(Map("Age" -> avgAge, "Fare" -> avgFare, "Embarked" -> "S"))
  val testFilled = testCleaned.na.fill(Map("Age" -> avgAge, "Fare" -> avgFare, "Embarked" -> "S"))

  val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")
  val sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
  val assembler = new VectorAssembler()
    .setInputCols(Array("Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "EmbarkedIndex", "Companions"))
    .setOutputCol("features")

  val rf = new RandomForestClassifier().setLabelCol("Survived").setFeaturesCol("features").setNumTrees(100)
  val pipeline = new Pipeline().setStages(Array(embarkedIndexer, sexIndexer, assembler, rf))
  val model = pipeline.fit(trainFilled)

  val predictions = model.transform(testFilled)

  val predictionValue: DataFrame = predictions.select("PassengerId", "prediction")
  predictionValue.show()

  //predictionValue.write.mode(SaveMode.Overwrite).csv("path/to/predictions_output")

  val rddPrediction = predictionValue.rdd
  rddPrediction.saveAsTextFile("C:/Users/mehul/Documents/College_Grad/Sem_3/INFO_7200/Assingment/Spark 2/result.csv")
  /*predictionValue.write
    .format("csv")
    .option("header", "true")
    .option("delimiter", ",")
    .mode("overwrite")
    .save("C:/Users/mehul/Documents/College_Grad/Sem_3/INFO_7200/Assingment/Spark 2/result.csv")

   */
}
