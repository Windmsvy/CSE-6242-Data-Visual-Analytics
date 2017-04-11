package edu.gatech.cse6242

// Xu Zhang
// CSE 6242 Homework 3 Task 3


import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Task2 {
  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("Task2"))
    // read the file
    val file = sc.textFile("hdfs://localhost:8020" + args(0))

    val l = file.map(_.split("\t").map(_.toInt))
    val mapper_filter = l.filter(a => a(2) != 1)
    val sce = mapper_filter.map(a=>(a(0),a(2)*(-1.0)))
    val tar = mapper_filter.map(a=>(a(1),a(2)*(1.0)))  
    val t = sce.union(tar)
    val mapreduce = t.reduceByKey(_+_)
    val output = mapreduce.map( a=>(a._1+"\t"+a._2) )

    output.saveAsTextFile("hdfs://localhost:8020" + args(1))
  }
}
