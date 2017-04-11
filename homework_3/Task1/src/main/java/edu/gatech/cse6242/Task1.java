package edu.gatech.cse6242;

// Xu Zhang
// Homework 3 Task 1

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapred.*;
import java.io.IOException;
import java.util.*;

public class Task1 {
	public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, IntWritable> {
		private Text word = new Text();
		public void map(LongWritable key, Text value, OutputCollector<Text, IntWritable> output, Reporter reporter) throws IOException {
			StringTokenizer tokenizer = new StringTokenizer(value.toString());
			String[] items = value.toString().split("\\t");
			word.set(items[1]);
			output.collect(word,new IntWritable(Integer.parseInt(items[2])));
		}
	}
	public static class Red_process extends MapReduceBase implements Reducer<Text, IntWritable, Text, IntWritable> {
		public void reduce(Text key, Iterator<IntWritable> values, OutputCollector<Text, IntWritable> output, Reporter reporter) throws IOException {
			int wgt = 0;
			while (values.hasNext()) {
				wgt = Math.max(wgt,values.next().get());
			}
			output.collect(key, new IntWritable(wgt));
		}
	}
	public static void main(String[] args) throws Exception {
		JobConf result = new JobConf(Task1.class);
		result.setJobName("Task1");
		result.setOutputKeyClass(Text.class);
		result.setOutputValueClass(IntWritable.class);
		result.setMapperClass(Map.class);
		result.setCombinerClass(Red_process.class);
		result.setReducerClass(Red_process.class);
		result.setInputFormat(TextInputFormat.class);
		result.setOutputFormat(TextOutputFormat.class);

		FileInputFormat.setInputPaths(result, new Path(args[0]));
		FileOutputFormat.setOutputPath(result, new Path(args[1]));
		JobClient.runJob(result);
	}
}