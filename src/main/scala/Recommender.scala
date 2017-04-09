/**
  * Created by weiyang on 4/6/17.
  */

//import java.nio.file.Paths
package com.recommender


import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.{ALS, ALSModel}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object Recommender {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Recommender").config("spark.master","local").getOrCreate()
    import spark.implicits._
    val rawUserArtistData = spark.read.textFile(fsPath("/data/user_artist_data.txt"))
    val rawArtistData = spark.read.textFile(fsPath("/data/artist_data.txt"))
    val rawArtistAlias = spark.read.textFile(fsPath("/data/artist_alias.txt"))

    // convert the DataSet[String] to DataFrame with field names "id" and "name" for further reference.
    val artistByID = rawArtistData.flatMap(line => {
      val (id,name) = line.span(_ != '\t')
      if (name.isEmpty){
        None
      } else {
        try {
          Some((id.toInt,name.trim))
        } catch {
          case e: NumberFormatException => None
        }
      }
    }).toDF("id","name")
    artistByID.select("id","name").limit(5).show()

    // convert DataSet[String] rawArtistAlias to Map[Int,Int]
    // then make the result into broadcast variable that all worker can share.
    val artistAlias = rawArtistAlias.flatMap{line =>
      val tokens = line.split("\t")
      if (tokens(0).isEmpty){
        None
      } else {
        Some((tokens(0).toInt,tokens(1).toInt))
      }
    }.collect().toMap
    val bArtistAlias = spark.sparkContext.broadcast(artistAlias)


    // convert DataSet[String] rawUserArtistData to DataFrame with field names : "user","artist","count"
    // to make up a data set: allData
    val allData = rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      (userID, finalArtistID, count)
    }.toDF("user", "artist", "count").persist()

    artistAlias.take(5).foreach(println)

    //initialize a recommender
    // for method prepare : find basic information about the allData
    // for method model : train a ALS, and give out the recommendation in resprect to the ALS model.
    // for method evaluate : to test different configuaration on ALS and find out the best config for ALS based on AUC.
    val recommender = new Recommender(spark)
    recommender.prepare(rawUserArtistData,artistByID,artistAlias)
    recommender.model(allData,artistByID)
    recommender.evaluate(allData)
  }
  /** @return The filesystem path of the given resource */
  def fsPath(resource: String): String =
    getClass.getResource(resource).getPath

}

class Recommender(private val spark:SparkSession) {
  import spark.implicits._

  def prepare(
             rawUserArtistData: Dataset[String],
             artistByID : DataFrame,
             aritstAlias: Map[Int,Int]
             ):Unit = {
    val userArtistDF = rawUserArtistData.map(line => {
      val Array(user,artist,_) = line.split(' ')
      (user.toInt,artist.toInt)
    }
    ).toDF("user","artist")

    userArtistDF.agg(max($"user"),min($"user"),max($"artist"),max($"artist")).show()

    val (badID,goodID) = aritstAlias.head
    artistByID.filter($"id" isin(badID,goodID)).show()
    println(badID + " " + goodID)

  }

  def model(trainData: DataFrame,
           artistByID: DataFrame) : Unit = {

    val model = new ALS()
    .setSeed(Random.nextLong())
    .setImplicitPrefs(true)
    .setRank(10)
    .setRegParam(0.01)
    .setAlpha(1.0)
    .setMaxIter(10)
    .setUserCol("user")
    .setItemCol("artist")
    .setRatingCol("count")
    .setPredictionCol("prediction")
    .fit(trainData)

    model.userFactors.select("features").show(truncate = false)

    val userID = 2093760

    val existingArtistID = trainData.filter($"user" === userID)
                                    .select($"artist").as[Int].collect()

    artistByID.filter($"id" isin(existingArtistID:_*)).show()

    val toRecommend = model.itemFactors.select($"id".as("artist")).withColumn("user",lit(userID))
    toRecommend.show(5)
    model.transform(toRecommend).show(5)
    val topRecommends = model.transform(toRecommend).select($"artist",$"prediction").orderBy($"prediction".desc).limit(10)
    val recommendIDs = topRecommends.select($"artist").as[Int].collect()

    artistByID.filter($"id" isin(recommendIDs:_*)).show()
  }

  def evaluate(allData: DataFrame): Unit = {

    val Array(trainData,cvData) = allData.randomSplit(Array(0.9,0.1))
    trainData.persist()
    cvData.persist()

    val allArtistID = allData.select($"artist").as[Int].distinct().collect()
    val bAllArtistID =  spark.sparkContext.broadcast(allArtistID)
    val evaluations =
      for (rank <- Seq(5,30);
           regParam <- Seq(1.0,0.001);
           alpha <- Seq(1.0,40.0)
          )
      yield{
        val model = new ALS().setSeed(Random.nextLong())
                            .setImplicitPrefs(true)
                            .setMaxIter(10)
                            .setAlpha(alpha)
                            .setRegParam(regParam)
                            .setUserCol("user")
                            .setItemCol("artist")
                            .setRank(rank)
                            .setRatingCol("count")
                            .setPredictionCol("prediction")
                            .fit(trainData)

        val auc = areaUnderCurve(cvData, bAllArtistID, model.transform)

        model.userFactors.unpersist()
        model.itemFactors.unpersist()

        (auc, (rank, regParam, alpha))
      }

    evaluations.sorted.reverse.foreach(println)

    trainData.unpersist()
    cvData.unpersist()
  }
  def areaUnderCurve(positiveData: DataFrame,
                     bAllArtistIDs: Broadcast[Array[Int]],
                     predictFunction: (DataFrame => DataFrame)): Double = {

    // What this actually computes is AUC, per user. The result is actually something
    // that might be called "mean AUC".

    // Take held-out data as the "positive".
    // Make predictions for each of them, including a numeric score
    val positivePredictions = predictFunction(positiveData.select("user", "artist")).
      withColumnRenamed("prediction", "positivePrediction")

    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other artists, excluding those that are "positive" for the user.
    val negativeData = positiveData.select("user", "artist").as[(Int,Int)].
      groupByKey { case (user, _) => user }.
      flatMapGroups { case (userID, userIDAndPosArtistIDs) =>
        val random = new Random()
        val posItemIDSet = userIDAndPosArtistIDs.map { case (_, artist) => artist }.toSet
        val negative = new ArrayBuffer[Int]()
        val allArtistIDs = bAllArtistIDs.value
        var i = 0
        // Make at most one pass over all artists to avoid an infinite loop.
        // Also stop when number of negative equals positive set size
        while (i < allArtistIDs.length && negative.size < posItemIDSet.size) {
          val artistID = allArtistIDs(random.nextInt(allArtistIDs.length))
          // Only add new distinct IDs
          if (!posItemIDSet.contains(artistID)) {
            negative += artistID
          }
          i += 1
        }
        // Return the set with user ID added back
        negative.map(artistID => (userID, artistID))
      }.toDF("user", "artist")

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeData).
      withColumnRenamed("prediction", "negativePrediction")

    // Join positive predictions to negative predictions by user, only.
    // This will result in a row for every possible pairing of positive and negative
    // predictions within each user.
    val joinedPredictions = positivePredictions.join(negativePredictions, "user").
      select("user", "positivePrediction", "negativePrediction").cache()

    // Count the number of pairs per user
    val allCounts = joinedPredictions.
      groupBy("user").agg(count(lit("1")).as("total")).
      select("user", "total")
    // Count the number of correctly ordered pairs per user
    val correctCounts = joinedPredictions.
      filter($"positivePrediction" > $"negativePrediction").
      groupBy("user").agg(count("user").as("correct")).
      select("user", "correct")

    // Combine these, compute their ratio, and average over all users
    val meanAUC = allCounts.join(correctCounts, "user").
      select($"user", ($"correct" / $"total").as("auc")).
      agg(mean("auc")).
      as[Double].first()

    joinedPredictions.unpersist()

    meanAUC
  }

}
