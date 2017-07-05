import numpy as np
import tensorflow as tf
import logging
import os
import sys

logging.getLogger().setLevel(logging.INFO)

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))

class TextCNN(object):
	
	def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,data_type='random'):
		# Placeholders for input, output and dropout
		logging.info("the sequence_length is:%d",sequence_length)
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Embedding layer
		with tf.name_scope('embedding'):#tf.device('/cpu:0'), 
			if data_type=='random':
				WS = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='WS')
				self.embedded_chars = tf.nn.embedding_lookup(WS, self.input_x)  #find vertor for word
			elif data_type=='word2vector':
				self.input_x = tf.placeholder(tf.float32,[None,sequence_length,embedding_size],name='input_x')
				# WS=self.get_wordVec(sequence_length,embedding_size)
				self.embedded_chars = self.input_x#tf.Variable(self.input_x, name='WS')#this vocab_size is the vecotr of data
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
		
		logging.info("input X shape is {}".format(self.input_x))

		# Create a convolution + maxpool layer for each filter size
		logging.info(self.embedded_chars)
		logging.info("this embedded_chars_expanded size:{}".format(self.embedded_chars_expanded))
		logging.info((self.embedded_chars_expanded))

		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope('conv-maxpool-%s' % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters] # kenrel 3*50,input=1,output=32|100
				logging.info(filter_shape)
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
				logging.info("++++W++++:%s",str(tf.shape(W)))
				logging.info(W)
				logging.info("++++b++++:%s",str(tf.shape(b)))

				conv = tf.nn.conv2d(
					tf.reshape(self.embedded_chars_expanded,[-1,sequence_length,embedding_size,1]),
					# self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='conv')

				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
				logging.info("pooling input shape:{}".format(conv))	
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='pool')
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		logging.info("Combine all the pooled features")
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(pooled_outputs,3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		# Add dropout
		with tf.name_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		# Final (unnormalized) scores and predictions
		logging.info("-------------output layer------")
		with tf.name_scope('output'):
			W = tf.get_variable(
				'W',
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
			self.predictions = tf.argmax(self.scores, 1, name='predictions')

		# Calculate mean cross-entropy loss
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.scores) #  only named arguments accepted            
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		# Accuracy
		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
		with tf.name_scope('num_correct'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
