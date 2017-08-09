import logging
import tensorflow as tf
import pickle
import os
# from tensorflow.contrib import rnn
from utils import dot_product as dot
from utils import set_logger
# Use license_1 in LICENSE (BSD 3-clause)
# lstm implementation details: https://apaszke.github.io/lstm-explained.html
# and here, http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# and here, though written in theano: https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/lstm.py
# And it seems that most of author's code is from here. But he does not attach the BSD 3-clause License.
# Is that usual????

tf.set_random_seed(117)

class Generator(object):
	def __init__(self, timestep, window, batch_size, vocab_size, paramSavePath, logPath, input_dim, hidden_size, keep_prob, L, timestr, debug):
		self.name = 'g'
		self.timestep = timestep
		self.hidden_size = hidden_size
		self.input_dim = input_dim
		self.window = window
		self.keep_prob = keep_prob
		self.L = L # options['L'] in author's code, for numerical stability. But why? Author doesn't explain...
		self.paramSavePath = paramSavePath
		self.logPath = logPath
		self.timestr = timestr
		# first input
		self.batch_size = batch_size if not debug else 10
		self.vocab_size = vocab_size
		# self.bhid = params['bhid']
		# self.Vhid = dot(params['Vhid'], self.Wemb) # (500, vocab_size)
		self.logger = set_logger(self.logPath, self.timestr, os.path.basename(__file__))
		self.init_param()

		# lstm = rnn.BasicLSTMCell(num_units=self.hidden_size, state_is_tuple=True)
		# lstm = rnn.DropoutWrapper(cell=lstm, output_keep_prob=keep_prob)
		# outputs, _states = rnn.static_rnn(lstm, z, dtype=tf.float32)

	def init_param(self):
		idm = self.input_dim
		hs = self.hidden_size
		ws = len(self.window)
		nf = idm * ws
		# author's special initlaization strategy.
		self.Wemb = tf.get_variable(name=self.name + '_Wemb', shape=[self.vocab_size, idm], dtype=tf.float32, initializer=tf.random_uniform_initializer())
		self.bhid = tf.get_variable(name=self.name + '_bhid', shape=[self.vocab_size], dtype=tf.float32, initializer=tf.zeros_initializer())
		self.Vhid = tf.get_variable(name=self.name + '_Vhid', shape=[hs, idm], dtype=tf.float32, initializer=tf.random_uniform_initializer())
		self.Vhid = dot(self.Vhid, self.Wemb) # [hidden_size, vocab_size]
		self.i2h_W = tf.get_variable(name=self.name + '_i2h_W', shape=[idm, hs * 4], dtype=tf.float32, initializer=tf.random_uniform_initializer())
		self.h2h_W = tf.get_variable(name=self.name + '_h2h_W', shape=[hs, hs * 4], dtype=tf.float32, initializer=tf.orthogonal_initializer())
		self.z2h_W = tf.get_variable(name=self.name + '_z2h_W', shape=[nf, hs * 4], dtype=tf.float32, initializer=tf.random_uniform_initializer())
		b_init_1 = tf.zeros((hs,))
		b_init_2 = tf.ones((hs,)) * 3
		b_init_3 = tf.zeros((hs,))
		b_init_4 = tf.zeros((hs,))
		b_init = tf.concat([b_init_1, b_init_2, b_init_3, b_init_4], axis=0)
		# b_init = tf.constant(b_init)
		# self.b = tf.get_variable(name=self.name + '_b', shape=[hs * 4], dtype=tf.float32, initializer=b_init)
		self.b = tf.get_variable(name=self.name + '_b', dtype=tf.float32, initializer=b_init) # ValueError: If initializer is a constant, do not specify shape.
		self.C0 = tf.get_variable(name=self.name + '_C0', shape=[nf, hs], dtype=tf.float32, initializer=tf.random_uniform_initializer())
		self.b0 = tf.get_variable(name=self.name + '_b0', shape=[hs], dtype=tf.float32, initializer=tf.zeros_initializer())
	
	def lstm(self, prev_y, prev_h, prev_c, z):
		hs = self.hidden_size

		preact = tf.einsum('ijk,ka->ija', prev_h, self.h2h_W) + \
				 tf.einsum('ijk,ka->ija', prev_y, self.i2h_W) + \
				 tf.matmul(z, self.z2h_W) + \
				 self.b # preactivation
		# [1, batch_size, hidden_size * 4]
		i = tf.sigmoid(preact[:, :, 0*hs: 1*hs])
		f = tf.sigmoid(preact[:, :, 1*hs: 2*hs])
		o = tf.sigmoid(preact[:, :, 2*hs: 3*hs])
		c = tf.tanh(preact[:, :, 3*hs: 4*hs])
		c = f * prev_c + i * c # [1, batch_size, hidden_size] (element-wise multiply)
		h = o * tf.tanh(c) # [1, batch_size, hidden_size]
		y = tf.einsum('ijk,ka->ija', h, self.Vhid) + self.bhid # [1, batch_size, vocab_size]

		# Author doesn't mention this part in his paper, but it appers in his code
		# So I assume this is part of his soft-max approx. strategy ---|
		max_y = tf.reduce_max(y, axis=1, keep_dims=True) # [1, 1, vocab_size]
		e = tf.exp((y - max_y) * self.L)  # [1, batch_size, vocab_size]
		w = e / tf.reduce_sum(e, axis=1, keep_dims=True) # [1, batch_size, vocab_size]
		# Assumption ends here ----------------------------------------|

		y = tf.einsum('ijk,ka->ija', w, self.Wemb) # [1, batch_size, input_dim]
		
		return y, h, c

	def generate(self, z):
		# forgive me. I also don't know what this is. The author doesn't mention
		# it in the paper and gives no comments for it in his code.
		h = tf.tanh(tf.matmul(z, self.C0) + self.b0) # [batch_size, hidden_size]
		# The expanded dimension is timestep. Following codes are used to prepare the first h
		h = tf.expand_dims(h, axis=0) # [1, batch_size, hidden_size]
		y = tf.einsum('ijk,ka->ija', h, self.Vhid) # [1, batch_size, vocab_size]
		c = tf.zeros(tf.shape(h)) # [1, batch_size, hidden_size]
		y_max = tf.reduce_max(y, axis=1, keep_dims=True) # [1, 1, vocab_size]
		e = tf.exp((y - y_max) * self.L) # [1, batch_size, vocab_size]
		w0 = e / tf.reduce_sum(e, axis=1, keep_dims=True) # [1, batch_size, vocab_size]
		y = tf.einsum('ijk,ka->ija', w0, self.Wemb) # [1, batch_size, input_dim]
		h_all = tf.identity(h) # [1, batch_size, hidden_size]
		for i in range(self.timestep - 1):
			y, h, c = self.lstm(y, h, c, z)
			h_all = tf.concat((h_all, h), axis=0)
		# h_all: [timestep, batch_size, hidden_size]
		shape_w = tf.shape(h_all) # [timestep, batch_size, hidden_size]
		h_all = tf.reshape(h_all, [shape_w[0] * shape_w[1], shape_w[2]]) # [timestep * batch_size, hidden_size]
		pred_w = tf.matmul(h_all, self.Vhid) + self.bhid # [timestep * batch_size, vocab_size]

		max_w = tf.reduce_max(pred_w, axis=1, keep_dims=True) # [timestep * batch_size, 1]
		e0 = tf.exp((pred_w - max_w) * self.L) # [timestep * batch_size, vocab_size]
		pred_w = e0 / tf.reduce_sum(e0, axis=1, keep_dims=True) # [timestep * batch_size, vocab_size]
		
		self.max_print = tf.reduce_max(pred_w, axis=1) # [timestep * batch_size, 1]
		self.max_print = tf.transpose(tf.reshape(self.max_print, [self.timestep, self.batch_size]), [1, 0]) # [batch_size, timestep]

		# Add PAD before beginning and after ending
		# NOTICE that timestep also modified after this part!
		pred_w = tf.reshape(pred_w, [self.timestep, self.batch_size, self.vocab_size]) # [timestep, batch_size, vocab_size]
		pred_w = tf.transpose(pred_w, [1, 0, 2]) # [batch_size, timestep, vocab_size]
		pad = max(self.window) - 1
		end_mat = tf.concat([tf.ones([self.batch_size, pad, 1]), tf.zeros([self.batch_size, pad, self.vocab_size - 1])], axis=2) # [batch_size, pad, vocab_size]
		pred_w = tf.concat([end_mat, pred_w, end_mat],axis=1) # [batch_size, timestep, vocab_size]
		self.timestep = self.timestep + 2 * pad
		pred_w = tf.reshape(pred_w, [self.timestep * self.batch_size, self.vocab_size]) # [timestep * batch_size, vocab_size]

		generatee = tf.matmul(pred_w, self.Wemb) # [timestep * batch_size, input_dim]
		generatee = tf.nn.dropout(generatee, keep_prob=self.keep_prob)
		generatee = tf.reshape(generatee, [self.batch_size, self.timestep, self.input_dim, 1]) # [batch_size, timestep, input_dim, 1]
		return pred_w, generatee

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


