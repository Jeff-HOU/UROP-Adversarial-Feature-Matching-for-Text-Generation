import logging
import tensorflow as tf
import pickle
import logging
import os
from utils import set_logger
# import numpy as np

# reproducibility
tf.set_random_seed(117)

class Discriminator(object):
	def __init__(self, window, vocab_size, paramSavePath, logPath, input_dim, keep_prob, reuse, generator, timestr, debug):
		# params = {'lambda_r': 0.001, 'lambda_m': 0.001, 'word_dim': 300}
		self.name = 'd'
		self.window = window
		self.vocab_size = vocab_size
		self.input_dim = input_dim
		self.paramSavePath = paramSavePath
		self.logPath = logPath
		self.timestr = timestr
		#self.cnn_out = tf.get_variable(name=self.name + '_f',
		#							shape=[],
		#							initializer=tf.zeros_initializer())
		self.keep_prob = keep_prob
		self.logger = set_logger(self.logPath, self.timestr, os.path.basename(__file__))
		if reuse:
			self.Wemb = generator.Wemb
		else:
			self.Wemb = tf.get_variable(name=self.name + '_Wemb', shape=[self.vocab_size, self.input_dim],
										dtype=tf.float32, initializer=tf.random_uniform_initializer())
		with tf.variable_scope('d'):
			for i, n in enumerate(self.window):
				W = tf.get_variable(name=self.name + '_W' + str(i),
									shape=[n, 1, 1, 1],
									initializer=tf.contrib.layers.xavier_initializer())
				b = tf.get_variable(name=self.name + '_b' + str(i),
									shape=[1],
									initializer=tf.zeros_initializer())
				#c = tf.get_variable(name=self.name + '_c' + str(i), # c is each cnn_out
				#					shape=[-1, self.input_dim],
				#					initializer=tf.zeros_initializer())
	def cp(self, input_sents, i=0):
		# conv and pool
		# https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-11-2-mnist_deep_cnn.py
		with tf.variable_scope('d', reuse=True):
			c = tf.nn.conv2d(input_sents, tf.get_variable(self.name + '_W' + str(i)),
										  strides=[1, 1, 1, 1], padding='VALID')
			c = tf.nn.tanh(c)
			tmp = input_sents.get_shape().as_list()[1] - self.window[i] + 1
			c = tf.nn.max_pool(c, ksize=[1, tmp, 1, 1],
								  strides=[1, 1, 1, 1],
								  padding='VALID')
			c = tf.nn.dropout(c, keep_prob=self.keep_prob)
			return tf.reshape(c, [-1, self.input_dim])
	def out(self):
		# I don't know how to implement next!!!😭😭😭
		self.fc = tf.contrib.layers.fully_connected(inputs=self.cnn_out,
												num_outputs=1,
												activation_fn=tf.sigmoid,
												biases_initializer=tf.ones_initializer())
		self.fc = tf.tanh(self.fc) # [batch_size, 1]
		return self.cnn_out, self.fc # need to output cnn_out for KDE calculation

	def discriminate(self, input_sents):
		input_sents = input_sents # dim: None * self.input_dim
		for i in range(len(self.window)):
			c = self.cp(input_sents=input_sents, i=i)
			if i != 0:
				self.cnn_out = tf.concat([self.cnn_out, c], axis=1)
			else:
				self.cnn_out = c
		# self.cnn_out: [batch_size, input_dim * len(window)]
		return self.out()
	
	# Following code copied here:
	# https://stackoverflow.com/questions/17219481/save-to-file-and-load-an-instance-of-a-python-class-with-its-attributes
	def save(self, fileName):
		assert fileName is not None

		with open(self.paramSavePath + fileName, 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
	
	@staticmethod
	def load(self, fileName):
		assert fileName is not None

		with open(self.paramSavePath + fileName, 'rb') as f:
			return pickle.load(f)

