name := "Recommender"

version := "1.0"

scalaVersion := "2.11.9"

val v = "2.1.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % v,
  "org.apache.spark" %% "spark-mllib" % v
)
        