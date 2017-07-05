# -*- coding: utf-8 -*-
import os
import sys
import datetime
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn import metrics
# this is a example test;

# hello = tf.constant("hello word!")
# sess = tf.Session()
# print sess.run(hello)
# a= tf.constant(100)
# b= tf.constant(200)
# print  sess.run(a+b)
# sess.close()

#this is my example 60000 trainingData, 10000 testData
#导入input_data用于自动下载和安装MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
import time

from dataHelper import LoadData
import numpy as np

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))


class LR(object):
	"""docstring for LR"""
	def __init__(self,filename):
		super(LR, self).__init__()
		if(len(sys.argv)>=2):
			filename=sys.argv[1]
		# self.myData=LoadData(filename)
		self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		self.vecSize=784

	def myStart(self):

		#加载数据

		#放置占位符，用于在计算时接收输入值
		x = tf.placeholder("float", [None, self.vecSize])
		#使用Tensorflow提供的回归模型softmax，y代表输出,二分类问题，n_class=1
		n_classes = 10 # MNIST total classes (0-9 digits)
		y = tf.placeholder("float", [None,n_classes])

		# Parameters
		learning_rate = 0.02
		training_epochs = 10
		batch_size = 100
		display_step = 1

		# Network Parameters
		n_hidden_1 = 512 # 1st layer number of features
		n_hidden_2 =128 # 2nd layer number of features
		n_hidden_3 = 64 # 3nd layer number of features
		# Store layers weight & bias
		weights = {
			'h1': tf.Variable(tf.random_normal([self.vecSize, n_hidden_1],stddev=0.1)),
			'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=0.1)),
			'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],stddev=0.1)),
			'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes],stddev=0.1))
		}
		biases = {
			'b1': tf.Variable(tf.random_normal([n_hidden_1],stddev=0.1)),#random_normal
			'b2': tf.Variable(tf.random_normal([n_hidden_2],stddev=0.1)),
			'b3': tf.Variable(tf.random_normal([n_hidden_3],stddev=0.1)),
			'out':tf.Variable(tf.random_normal([n_classes],stddev=0.1))
		}

		# Construct model
		pred = self.multilayer_perceptron(x, weights, biases)

		# Define loss and optimizer
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

		# Initializing the variables
		init = tf.initialize_all_variables()
		# Launch the graph
		with tf.Session() as sess:
			sess.run(init)
			# Training cycle
			start_time = time.time()

			for epoch in range(training_epochs):
				avg_cost = 0.
				total_batch =1000
				# Loop over all batches
				for i in range(total_batch):

					batch=self.mnist.train.next_batch(100)
					batch_xs, batch_ys = batch[0],batch[1] #self.myData.next_batch(100)

					# Run optimization op (backprop) and cost op (to get loss value)
					_, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,y: batch_ys})
					# Compute average loss
					# avg_cost += c / total_batch
				# Display logs per epoch step
				# if epoch % display_step == 0:
					print "Epoch:", '%04d' % (i), "cost=", "{:.9f}".format(c)
			print "Optimization Finished!"

			# Test model
			x_test,y_test =self.mnist.test.images, self.mnist.test.labels,#self.myData.test_data()
			print "get data type",type(x_test),type(y_test),x_test.shape ,y_test.shape 
			# batch_xs, batch_ys = self.myData.next_batch(1000)		
			batch=self.mnist.train.next_batch(400)
			batch_xs, batch_ys = batch[0],batch[1]

			correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
			# Calculate accuracy
			accuracy =tf.reduce_mean(tf.cast(correct_prediction, "float"))
			rsult=accuracy.eval({x: batch_xs, y: batch_ys})
			print "################ train every accuracy:",rsult

			pidx=tf.argmax(pred,1)

			pd=sess.run(pidx,feed_dict={x: batch_xs})
			print "result data type",type(pd),pd.shape

			print "model Accuracy:", self.getScore_1( batch_ys, pd)

			rsult=accuracy.eval({x: x_test, y: y_test})
			print "################ test every accuracy",rsult
			pd=sess.run(pidx,feed_dict={x: x_test, y: y_test})
			print type(pd),pd.shape
			print "test  Accuracy:", self.getScore_1(y_test, pd)

			end_time = time.time()
			print  "total time is ", end_time-start_time
			# saver = tf.train.Saver()
			# saver_path = saver.save(sess, "mingSave/my-model-ML.ckpt")
			# print "Model saved in file: ", saver_path
	# Create model
	def multilayer_perceptron(self,x, weights, biases):
		# Hidden layer with RELU activation   wx+b
		layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer_1 = tf.nn.relu(layer_1)
		# Hidden layer with RELU activation
		layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
		layer_2 = tf.nn.relu(layer_2)  #relu log_softmax 
		layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
		layer_3 = tf.nn.relu(layer_3)
		# Output layer with linear activation
		out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
		return out_layer

	 #    #load model
		# feed_dict={x: x_test, y_: y_test}
		# self.myRestore(accuracy,feed_dict)
	def getScore(self,y_true,y_pred):
		y_true =y_true.reshape(-1).tolist()
		y_pred =y_pred.reshape(-1).tolist()
		print (classification_report(y_true,y_pred))
		print (metrics.confusion_matrix(y_true,y_pred))
	def getScore_1(self,y_true,p):
		y=[]
		p=p.tolist()
		for item in y_true:
			mlabel=0
			# print "#!!!!#:",item.ndim
			for i in range(y_true.ndim):
				if item[i]==1:
					mlabel=i
					break
			y.append(mlabel)
  		# print y[0:10],y[len(y)-10:len(y)]
  		# print p[0:10],p[len(p)-10:len(p)]
		print (classification_report(y,p))
		print (metrics.confusion_matrix(y,p))


	def myRestore(self,acc,feed_dict):
		saver = tf.train.Saver()
		with tf.Session() as sess:
			start_time = time.time()
			saver.restore(sess,"mingSave/my-model-LR.ckpt")
			print sess.run(acc, feed_dict)
			end_time = time.time()
			print "predict time is ", end_time - start_time

class test(object):
	"""docstring for test"""
	def __init__(self, arg):
		super(test, self).__init__()
		self.arg = arg
	def pf(self):
		print self.arg
		print sys.argv[0]
		print "ming"

if __name__ == '__main__':

	# print "limingming"
	# lr = test(sys.argv[0])
	# lr.pf()
	filename='ch_waimai2_corpus.txt'
	lr = LR(filename)
	lr.myStart()


