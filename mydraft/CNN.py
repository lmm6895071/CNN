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
		self.myData=LoadData(filename)
		self.vecSize=400

	def myStart(self):

		#加载数据
		self.myData.train_data()


		#放置占位符，用于在计算时接收输入值
		x = tf.placeholder("float", [None, self.vecSize])
		#使用Tensorflow提供的回归模型softmax，y代表输出,二分类问题，n_class=1
		n_classes = 2 # MNIST total classes (0-9 digits)
		y = tf.placeholder("float", [None,n_classes])

		# Parameters
		learning_rate = 0.003
		training_epochs = 50
		batch_size = 100
		display_step = 1

		# Network Parameters
		n_hidden_1 = 512 # 1st layer number of features
		n_hidden_2 = 256 # 2nd layer number of features
		n_hidden_3 = 128 # 3nd layer number of features
		# Store layers weight & bias
		weights = {
			'h1': tf.Variable(tf.random_normal([self.vecSize, n_hidden_1])),#stddev=0.01
			'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
			'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
			'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
		}
		biases = {
			'b1': tf.Variable(tf.random_normal([n_hidden_1])),
			'b2': tf.Variable(tf.random_normal([n_hidden_2])),
			'b3': tf.Variable(tf.random_normal([n_hidden_3])),
			'out':tf.Variable(tf.random_normal([n_classes]))
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
					batch_xs, batch_ys = self.myData.next_batch(100)
					# Run optimization op (backprop) and cost op (to get loss value)
					_, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,y: batch_ys})
					# Compute average loss
					# avg_cost += c / total_batch
				# Display logs per epoch step
				# if epoch % display_step == 0:
					print "Epoch:", '%04d' % (i), "cost=", "{:.9f}".format(c)
			print "Optimization Finished!"

			# Test model
			x_test,y_test = self.myData.test_data()
			print "get data type",type(x_test),type(y_test),x_test.shape ,y_test.shape 
			batch_xs, batch_ys = self.myData.next_batch(1000)		

			correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
			# Calculate accuracy
			accuracy =tf.reduce_mean(tf.cast(correct_prediction, "float"))
			rsult=accuracy.eval({x: batch_xs, y: batch_ys})
			print "every accuracy:",rsult

			pidx=tf.argmax(pred,1)

			pd=sess.run(pidx,feed_dict={x: batch_xs})
			print "result data type",type(pd),pd.shape
			print "model Accuracy:", self.getScore_1( batch_ys, pd)

			rsult=accuracy.eval({x: x_test, y: y_test})
			print "every accuracy",rsult
			pd=sess.run(pidx,feed_dict={x: x_test, y: y_test})
			print type(pd),pd.shape
			print "test  Accuracy:", self.getScore_1(y_test, pd)

			end_time = time.time()
			print  "total time is ", end_time-start_time
			saver = tf.train.Saver()
			saver_path = saver.save(sess, "mingSave/my-model-ML.ckpt")
			print "Model saved in file: ", saver_path
	# Create model
	def multilayer_perceptron(self,x, weights, biases):
		# Hidden layer with RELU activation   wx+b
		layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer_1 = tf.nn.relu(layer_1)
		# Hidden layer with RELU activation
		layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
		layer_2 = tf.nn.relu(layer_2)
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
		print y_true.shape,y_pred.shape
		for item in y_true:
			mlabel=0
			# print "#!!!!#:",item.ndim
			for i in range(y_true.ndim):
				if item[i]==1:
					mlabel=i
					break
			y.append(mlabel)
  		print y[0:10],y[len(y)-10:len(y)]
  		print p[0:10],p[len(p)-10:len(p)]
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



class CNN(object):
	"""docstring for CNN"""
	def __init__(self,filename):
		super(CNN, self).__init__()
		if(len(sys.argv)>=2):
			filename=sys.argv[1]
		self.myData=LoadData(filename)
		self.vecSize=400#784,10
		self.n_classes=2

	#权重初始化函数
	def weight_variable(self,shape):
		#输出服从截尾正态分布的随机值
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	#偏置初始化函数
	def bias_variable(self,shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	#创建卷积op
	#x 是一个4维张量，shape为[batch,height,width,channels]
	#卷积核移动步长为1。填充类型为SAME,可以不丢弃任何像素点
	def conv2d(self,x, W):
		return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

	#创建池化op
	#采用最大池化，也就是取窗口中的最大值作为结果
	#x 是一个4维张量，shape为[batch,h eight,width,channels]
	#ksize表示pool窗口大小为2x2,也就是高2，宽2
	#strides，表示在height和width维度上的步长都为2
	def max_pool_2x2(self,x):
	    return tf.nn.max_pool(x, ksize=[1,2,2,1],
	                          strides=[1,2,2,1], padding="SAME")

	def init(self):
		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

		#创建一个交互式Session
		sess = tf.InteractiveSession()
		


		#创建两个占位符，x为输入网络的图像，y_为输入网络的图像类别
		x = tf.placeholder("float", shape=[None, self.vecSize])
		y_ = tf.placeholder("float", shape=[None, self.n_classes])

		#第1层，卷积层
		#初始化W为[5,5,1,32]的张量，表示卷积核大小为5*5，第一层网络的输入和输出神经元个数分别为1和32 32个特征map
		#CNN  窗口大小为3,4,5；map 大小为100；  卷积核大小为h=[3,4,5],h*100;
		W_conv1 = self.weight_variable([5,5,1,32]) 
		#初始化b为[32],即输出大小
		b_conv1 = self.bias_variable([32])

		#把输入x(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1]
		#-1表示自动推测这个维度的size
		x_image = tf.reshape(x, [-1,28,28,1])

		#把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling
		#h_pool1的输出即为第一层网络输出，shape为[batch,14,14,1]
		h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		#第2层，卷积层
		#卷积核大小依然是5*5，这层的输入和输出神经元个数为32和64
		W_conv2 = self.weight_variable([5,5,32,64])
		b_conv2 = self.weight_variable([64])

		#h_pool2即为第二层网络输出，shape为[batch,7,7,1]
		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2)

		#第3层, 全连接层
		#这层是拥有1024个神经元的全连接层
		#W的第1维size为7*7*64，7*7是h_pool2输出的size，64是第2层输出神经元个数
		W_fc1 = self.weight_variable([7*7*64, 1024])
		b_fc1 = self.bias_variable([1024])

		#计算前需要把第2层的输出reshape成[batch, 7*7*64]的张量
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		#Dropout层
		#为了减少过拟合，在输出层前加入dropout
		keep_prob = tf.placeholder("float")
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		#输出层
		#最后，添加一个softmax层
		#可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率
		W_fc2 = self.weight_variable([1024, 10])
		b_fc2 = self.bias_variable([10])

		y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

		#预测值和真实值之间的交叉墒
		cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

		#train op, 使用ADAM优化器来做梯度下降。学习率为0.0001
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

		#评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。
		#因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
		correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

		#计算正确预测项的比例，因为tf.equal返回的是布尔值，
		#使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值
		accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))

		#初始化变量
		sess.run(tf.initialize_all_variables())

		#开始训练模型，循环20000次，每次随机从训练集中抓取50幅图像
		for i in range(200):
		    batch = mnist.train.next_batch(50)
		    if i%10 == 0:
		        #每100次输出一次日志
		        train_accuracy = accuracy.eval(feed_dict={
		            x:batch[0], y_:batch[1], keep_prob:1.0})
		        print "step %d, training accuracy %g" % (i, train_accuracy)

		    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

		print "test accuracy %g" % accuracy.eval(feed_dict={
		x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
			
		pidx=tf.argmax(y_conv,1)
		batch = mnist.train.next_batch(500)
		pd=sess.run(pidx,feed_dict={x: batch[0], y_:batch[1], keep_prob:1.0})
		print "result data type",type(pd),pd.shape
		print "model Accuracy:", self.getScore_1( batch[1], pd)

		pd=sess.run(pidx,feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
		print "result data type",type(pd),pd.shape
		print "model Accuracy:", self.getScore_1(mnist.test.labels, pd)


	def getScore_1(self,y_true,p):
		y=[]
		print y_true.shape,p.shape
		for item in y_true:
			mlabel=0
			# print "#!!!!#:",item.ndim
			for i in range(y_true.ndim):
				if item[i]==1:
					mlabel=i
					break
			y.append(mlabel)
	  	print y[0:10],y[len(y)-10:len(y)]
	  	print p[0:10],p[len(p)-10:len(p)]
		print (classification_report(y,p))
		print (metrics.confusion_matrix(y,p))


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
	# lr = LR(filename)
	# lr.myStart()

	cnn = CNN(filename)
	cnn.init()


