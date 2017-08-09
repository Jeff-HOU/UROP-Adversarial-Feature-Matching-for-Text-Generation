# TODO: 1. makes PAD and split the big string into sentences.
#       2. make batch first and then split (or maybe not, given that there is no need for validation set and test set to have batches)

from nltk.tokenize import word_tokenize
import collections
import itertools
import logging
import re
import os
import numpy as np
import time
import pickle
from utils import set_logger

class Data(object):
	def __init__(self, dataPath, savePath, paramSavePath, logPath, debug, split_percent, batch_size, timestr, timestep, window):
		'''
			* dataPath is way to find the data. We have two data files.
				One is the real size as described in the paper.
				Another is a much smaller dataset with 100 sentences
				from both arXiv and book dataset used for early code test.
			* debug is the indicator whether we are testing our code or real training.
				default: debug = True, testing code mode.
			# split_percent: training set : validation set : testing set
		'''
		self.debug = debug
		self.savePath = savePath
		self.dataPath = dataPath if not self.debug else '../data/data_pre.txt'
		self.paramSavePath = paramSavePath
		self.logger = set_logger(logPath, timestr, os.path.basename(__file__))
		self.split_percent = split_percent
		self.timestep = timestep
		self.window = window
		self.load_data()
	#   self.data is the list containing all the contents in data file
	#   self.sentSize: how many sentences.
		self.clean_str()
		self.word2num()
	#   self.dataArr: an np.ndarray version of self.data
	#   self.mapToNum is the word - index map. A word's index can be visited by self.mapToNum['word'].
	#   self.dataNum maps words in self.dataStr into number. (np.ndarray)
	#   self.vocabSize is vocabulary size
		self.split_tvt()
	#   self.train training set
	#   self.validation validation set
	#   self.test testing set
	#   self.shift() Shift first 10% of self.dataNum and split tvt sets again.
		self.batch_size = batch_size if not self.debug else 10
	def load_data(self):
		'''	Load data from self.dataPath into one string.
		'''
		try:
			with open(self.dataPath) as f:
				self.data = f.read().splitlines()
			self.sentSize = len(self.data)
			if self.debug:
				self.logger.info('load_data finished')
		except:
			msg = 'File does not exist.\n' + \
				'Sorry the dataset used here is protected under copyright.\n\n' + \
				'If you would like to use the dateset,' + \
				'please kindly read the README.md under data folder.'
			self.logger.info(msg)
			raise Exception(msg)
	
	def clean_str(self):
		"""	Tokenization/string cleaning for all datasets except for SST.
			https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
			It seems that they usually use this function to do some cleaning,
			though I really don't know why. Hope I can figure it out later.
		"""
		for i in range(self.sentSize):
			string = self.data[i]
			string = re.sub(r"[^A-Za-z0-9(),!?\'\`_]", " ", string)
			string = re.sub(r"\'s", " \'s", string)
			string = re.sub(r"\'ve", " \'ve", string)
			string = re.sub(r"n\'t", " n\'t", string)
			string = re.sub(r"\'re", " \'re", string)
			string = re.sub(r"\'d", " \'d", string)
			string = re.sub(r"\'ll", " \'ll", string)
			string = re.sub(r",", " , ", string)
			string = re.sub(r"!", " ! ", string)
			string = re.sub(r"\(", " \( ", string)
			string = re.sub(r"\)", " \) ", string)
			string = re.sub(r"\?", " \? ", string)
			string = re.sub(r"\s{2,}", " ", string)
			string = re.sub("__LaTex__", "", string) # consider how to deal with latex later.
			# prevent some special words from converting to lowercase.
			specialwords = ['EOS', 'GOO', '__LaTex__']
			toLower = lambda x: " ".join( a if a in specialwords else a.lower() \
				for a in x.split() )
			self.data[i] = toLower(string.strip())
		if self.debug:
			self.logger.info('clean_str finished')
	
	def word2num(self):
		''' Index each word to a unique index.
			Here we use its frequency rank to index, though it does
			not really matter how to index.
		'''
		# remove sentences with more than self.timestep words.
		# or else, there would be too many PADs in the end.
		sentenceSplit = [word_tokenize(x) for x in self.data]
		#sentenceLength = np.asarray([len(x) for x in sentenceSplit])
		# add 'PAD' to the end if a sentence is shorter than self.timestep words.
		# Then add 'PAD' before and after the whole sentence.
		sentenceSplitPAD = []
		for i in range(len(sentenceSplit)):
			if self.timestep - len(sentenceSplit[i]) >= 0:
				sentenceSplitPAD.append(['PAD'] * (max(self.window) - 1) + sentenceSplit[i] + \
										['PAD'] * (self.timestep - len(sentenceSplit[i]) + max(self.window) - 1))
		#self.dataArr = np.asarray(sentenceSplitPAD)
		# There won't be significant difference if I just remove some sentences.
		# So, when counting how many words in the whole dataset
		# I just use the original one.
		words = list(itertools.chain(*sentenceSplitPAD))
		a = collections.Counter(words)
		# add some symbols. see url below for details
		# https://github.com/nicolas-ivanov/tf_seq2seq_chatbot/issues/15
		a['UNK'] = 1
		# a['PAD'] = 1
		self.vocabSize = len(a.keys())
		b = a.most_common(self.vocabSize)
		self.mapToNum = collections.defaultdict(list)
		self.mapToWord = collections.defaultdict(list)
		i = 0
		for k, _ in b:
			self.mapToNum[k].append(i)
			self.mapToWord[i].append(k)
			i += 1
		self.dataNum = []
		for sentence in sentenceSplitPAD:
			sentenceNum = []
			for word in sentence:
				if word in self.mapToNum:
					sentenceNum.extend(self.mapToNum[word])
				else:
					sentenceNum.extend(self.mapToNum['UNK'])
			self.dataNum.append(sentenceNum)
		self.dataNum = np.asarray(self.dataNum)
		self.sentSize = len(self.dataNum)
		# save dataNum, in case we may use it later.
		fileName = 'dataNum_' + time.strftime("%Y%m%d_%H%M%S")
		np.save(self.savePath + fileName, self.dataNum)
		self.logger.info("'dataNum' save to " + self.savePath + fileName)
	
	def split_tvt(self, shift=False):
		'''	split data into Training set, Validation Set, Testing set.
			* shift: if True, shift first 10% of self.dataNum and split tvt sets again.
		'''
		if shift:
			self.dataNum = np.concatenate((self.dataNum[int(self.sentSize * 0.1):],
											self.dataNum[:int(self.sentSize * 0.1)]))
		self.train = self.dataNum[:int(self.sentSize * self.split_percent[0])]
		self.validation = self.dataNum[int(self.sentSize * self.split_percent[0]):
				int(self.sentSize * (self.split_percent[0] + self.split_percent[1]))]
		self.test = self.dataNum[int(self.sentSize * self.split_percent[1]):]
		if self.debug:
			self.logger.info('split_tvt finished')

	def shift(self):
		'''	A dump version for split_tvt, with argument shift=True
			Shift first 10% of self.dataNum and split tvt sets again.
		'''
		self.split_tvt(shift=True)

	def get_first_batch(self, whichSet='train'):
		'''	Used for get the first batch of t/v/t set.
			whichSet: choose from 'train', 'validation', 'test'
		'''
		if whichSet == 'train':
			self.trainBatchCnt = 0
			self.trainMaxBatch = int(len(self.train) / self.batch_size)
			return self.train[:self.batch_size]
		elif whichSet == 'validation':
			self.validationBatchCnt = 0
			self.validationMaxBatch = int(len(self.validation) / self.batch_size)
			return self.validation[:self.batch_size]
		elif whichSet == 'test':
			self.testBatchCnt = 0
			self.testMaxBatch = int(len(self.test) / self.batch_size)
			return self.test[:self.batch_size]
		else:
			msg = 'Wrong set name!\n'+ \
				  'Should be train / validation / test.'
			raise Exception(msg)

	def next_batch(self, whichSet='train'):
		if whichSet == 'train':
			self.trainBatchCnt += 1
			assert self.trainBatchCnt < self.trainMaxBatch
			return self.train[self.trainBatchCnt * self.batch_size: (self.trainBatchCnt + 1) * self.batch_size]
		elif whichSet == 'validation':
			self.validationBatchCnt += 1
			assert self.validationBatchCnt < self.validationMaxBatch
			return self.validation[self.validationBatchCnt * self.batch_size: (self.validationBatchCnt + 1) * self.batch_size]
		elif whichSet == 'test':
			self.testBatchCnt += 1
			assert self.testBatchCnt < self.testMaxBatch
			return self.test[self.testBatchCnt * self.batch_size: (self.testBatchCnt + 1) * self.batch_size]
		else:
			msg = 'Wrong set name!\n'+ \
				  'Should be train / validation / test.'
			raise Exception(msg)
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







