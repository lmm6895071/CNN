import os
import sys
import json
import time
import logging
import dataHelper
import numpy as np
import tensorflow as tf
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from dataHelper import LoadData
from sklearn.metrics import classification_report
from sklearn import metrics
logging.getLogger().setLevel(logging.INFO)

def getScore(y_true,y_pred):
	y_true =y_true.reshape(-1).tolist()
	y_pred =y_pred.reshape(-1).tolist()
	logging.info((classification_report(y_true,y_pred)))
	logging.info(metrics.confusion_matrix(y_true,y_pred))

def train_cnn(filename,data_type):
	"""Step 0: load sentences, labels, and training parameters"""
	
	n_classes=2
	wordvec_size=256
	myData=LoadData(filename,data_type)

	parameter_file ="./parameters.json"
	params = json.loads(open(parameter_file).read())

	"""Step 1: split the original dataset into train and test sets"""
	x_train, y_train =myData.train_data()
	


	x_test,y_test = myData.test_data()

	x_dev,y_dev = myData.next_batch(300)


	logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
	logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

	"""Step 3: build a graph and cnn object"""
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
		sess = tf.Session(config=session_conf)
		sess=tf.Session()
		with sess.as_default():
			vocab_size = myData.vocab_size
			if data_type=='random':
				vocab_processor = myData.vocab_processor
				cnn = TextCNN(
					sequence_length=x_train.shape[1],#maxlength
					num_classes=y_train.shape[1],#n_classes
					vocab_size=vocab_size,
					embedding_size=params['embedding_dim'],
					filter_sizes=list(map(int, params['filter_sizes'].split(","))),#feature windows 3,4,5
					num_filters=params['num_filters'],#map 
					l2_reg_lambda=params['l2_reg_lambda'],
					data_type=data_type)
			elif data_type=='word2vector':
				cnn = TextCNN(
					sequence_length=myData.max_document_length,#maxlength
					num_classes=y_train.shape[1],#n _classes
					vocab_size=vocab_size,# 
					embedding_size=wordvec_size,#word2vector dim
					filter_sizes=list(map(int, params['filter_sizes'].split(","))),#feature windows 3,4,5
					num_filters=params['num_filters'],#map 
					l2_reg_lambda=params['l2_reg_lambda'],
					data_type=data_type)

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			# my_predict = tf.argmax(cnn.predictions,1)
			my_lable = tf.argmax(cnn.input_y,1)


			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.curdir+"/out_train_model/")
			logging.info("save path is {}".format(out_dir))

			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.all_variables())


			# One training step: train the model with one batch
			def train_step(x_batch, y_batch):
				feed_dict = {
					cnn.input_x: x_batch,
					cnn.input_y: y_batch,
					cnn.dropout_keep_prob: params['dropout_keep_prob']}
				_, step, loss, acc,p,l= sess.run([train_op, global_step, cnn.loss, cnn.accuracy,cnn.predictions,my_lable], feed_dict)
				logging.info("the step={} of training ,loss={},acc={}".format(step,loss,acc))
				return p,l

			# One evaluation step: evaluate the model with one batch
			def dev_step(x_batch, y_batch):
				feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
				step, loss, acc, num_correct,p,l = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct,cnn.predictions,my_lable], feed_dict)
				logging.info("the step={} of dev_step ,loss={},acc={},num_correct={}".format(step,loss,acc,num_correct))
				return num_correct,p,l

			def test_step(x_batch, y_batch):
				feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
				step, loss, acc, num_correct,p,l = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct,cnn.predictions,my_lable], feed_dict)
				logging.info("the step={} of dev_step ,loss={},acc={},num_correct={}".format(step,loss,acc,num_correct))

				return num_correct,p,l
			if data_type=='random':
				# Save the word_to_id map since predict.py needs it
				vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
			sess.run(tf.initialize_all_variables())

			# Training starts here
			train_batches = myData.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
			logging.info("train_batches type is {}".format(type(train_batches)))
			best_accuracy, best_at_step = 0, 0
			
			"""Step 6: train the cnn model with x_train and y_train (batch by batch)"""

			logging.info("start CNN train: batch_size={},num_epochs={}".format(params['batch_size'],params['num_epochs']))
			PP_train=np.array([])
			LL_train=np.array([])
			for train_batch in train_batches:
				logging.info("+++++++++++++++train_batches type is {},{}".format(type(train_batch),train_batch.shape))

				try:
					x_train_batch, y_train_batch = zip(*train_batch)
				except Exception as err:
					logging.info("this zip is error;{}".format(err))
				p,l=train_step(x_train_batch, y_train_batch)
				PP_train=np.concatenate((PP_train,p))
				LL_train=np.concatenate((LL_train,l))
				current_step = tf.train.global_step(sess, global_step)

				"""Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
				if current_step % params['evaluate_every'] == 0:
					dev_batches = myData.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 10)
					total_dev_correct = 0
					PP_DEV=np.array([])
					LL_DEV=np.array([])
					for dev_batch in dev_batches:
						try:
							x_dev_batch, y_dev_batch = zip(*dev_batch)
						except Exception as err:
							logging.info("this dev zip is error;{}".format(err))
						num_dev_correct,p,l = dev_step(x_dev_batch, y_dev_batch)
						PP_DEV=np.concatenate((PP_DEV,p))
						LL_DEV=np.concatenate((LL_DEV,l))
						total_dev_correct += num_dev_correct
					dev_accuracy = float(total_dev_correct) / len(y_dev)
					logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))
					getScore(LL_DEV,PP_DEV)

					"""Step 6.2: save the model if it is the best based on accuracy of the dev set"""
					if dev_accuracy >= best_accuracy:
						best_accuracy, best_at_step = dev_accuracy, current_step
						outpath = saver.save(sess, checkpoint_prefix, global_step=current_step)
						logging.critical('Saved model {} at step {}'.format(outpath, best_at_step))
						logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

			logging.info("train result is:++++++===+++++++")
			getScore(LL_train,PP_train)
			"""Step 7: predict x_test (batch by batch)"""
			test_batches = myData.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
			total_test_correct = 0

			PP=np.array([])
			LL=np.array([])
			for test_batch in test_batches:
				try:
					x_test_batch, y_test_batch = zip(*test_batch)
				except Exception as err:
					logging.info("this test zip is error;{}".format(err))
				num_test_correct,p,l= test_step(x_test_batch, y_test_batch)
				PP=np.concatenate((PP,p))
				LL=np.concatenate((LL,l))
				total_test_correct += num_test_correct
			test_accuracy = float(total_test_correct) / len(y_test)
			logging.critical('Accuracy on test set is {} based on the best model'.format(test_accuracy))
			getScore(LL,PP)
			logging.critical('The processes is completed')



if __name__ == '__main__':
	# python3 train.py ./data/consumer_complaints.csv.zip ./parameters.json
	filename='ch_waimai2_corpus.txt'
	if(len(sys.argv)>=2):
		if sys.argv[1] =='hotel':
			filename='ch_hotel_corpus.txt'
	data_type='random'
	data_type='word2vector'
	train_cnn(filename,data_type)
