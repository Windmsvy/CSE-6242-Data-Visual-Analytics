package edu.gatech.cse6242;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.StringTokenizer;
import java.util.HashSet;

public class Task4 {


  public static class Map_1 extends
        Mapper<Object, Text, Text, IntWritable> {
    private Text word = new Text();
    private IntWritable one = new IntWritable(1);

    public void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {
        String line = value.toString();
        StringTokenizer itr = new StringTokenizer(line,"\t");
        if(itr.hasMoreElements()){
            word.set(itr.nextToken().toLowerCase());
            context.write(word, one);
        }
        if(itr.hasMoreElements()){
            word.set(itr.nextToken().toLowerCase());
            context.write(word, one);
        }
      }	  		  
	  }
  
  public static class First_Red
       extends Reducer<Text,IntWritable,Text,IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context
                       ) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable p : values) {
        sum += p.get();
      }
      result.set(sum);
      context.write(key, result);
    }
  }


  public static class Map_2 extends
        Mapper<Object, Text, Text, IntWritable> {
    
    private Text times = new Text();
    private IntWritable one = new IntWritable(1);

    public void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {
        String line = value.toString();
        StringTokenizer itr = new StringTokenizer(line,"\t");
        String[] row = new String[itr.countTokens()];
        for(int i=0;i<2;i++) {
           row[i]=itr.nextToken();  
        } 
        times.set(row[1]) ;  
        context.write(times,one);  
      }         
    }
  
  public static class Second_Red
       extends Reducer<Text,IntWritable,Text,IntWritable> {
    private IntWritable result = new IntWritable();
    public void reduce(Text key, Iterable<IntWritable> values, Context context
                       ) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable p:values) {
        sum += p.get();
      }
      result.set(sum);
      context.write(key,result);
    }
  }

  // First Reduce Process
  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job first_job = Job.getInstance(conf, "Task4");
	first_job.setJarByClass(Task4.class);
    first_job.setMapperClass(Map_1.class);
    first_job.setCombinerClass(First_Red.class);
    first_job.setReducerClass(First_Red.class);
    first_job.setOutputKeyClass(Text.class);
    first_job.setOutputValueClass(IntWritable.class);

    FileInputFormat.addInputPath(first_job, new Path(args[0]));
    FileOutputFormat.setOutputPath(first_job, new Path("wasbs://cse6242@windblowing.blob.core.windows.net/temfile_large"));
    first_job.waitForCompletion(true);

    // second Reduce process
    Configuration conf2 = new Configuration();
    Job second_job = Job.getInstance(conf2, "Task4");
    second_job.setJarByClass(Task4.class);
    second_job.setMapperClass(Map_2.class);
    second_job.setCombinerClass(Second_Red.class);
    second_job.setReducerClass(Second_Red.class);
    second_job.setOutputKeyClass(Text.class);
    second_job.setOutputValueClass(IntWritable.class);

    FileInputFormat.addInputPath(second_job, new Path("wasbs://cse6242@windblowing.blob.core.windows.net/temfile_large"));
    FileOutputFormat.setOutputPath(second_job, new Path(args[1]));
    System.exit(second_job.waitForCompletion(true) ? 0 : 1);

  }
}
