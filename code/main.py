import logging
import time
import os
import tensorflow as tf
import numpy as np
from utils import cal_BLEU
from utils import calculate_mmd as cal_mmd
from utils import dot_product as dot
from utils import set_logger
from data import Data
from discriminator import Discriminator
from generator import Generator
from sklearn.neighbors.kde import KernelDensity
import pickle

model_build_start_time = time.time()

# Here begins some basic setup -----------------------------------------------------------------------------|
# miscellanous but most important
seed = 117
debug = True            # Debug or not. Temp variable when building. Can be remove later.

# params about data / file handling
dataPath = '../data/data.txt' if not debug else '../data/data_pre.txt'# check data.py for details
savePath = '../save/'   # path for save some log / parameter.
logPath = '../log/'
paramSavePath='../param/'
timestr = time.strftime("%Y%m%d-%H%M%S")
split_percent = [0.6, 0.2, 0.2] # train : validation : test

# params about model
batch_size = 10
hidden_size = 500
input_dim = 300
keep_prob = 0.8         # Drop out keep prob
L = 1000                # Check Class Generator for details
label_smoothing = 0.01  # Relating to a technique author uses in his code, not mentioned in paper.
						# Search label_smoothing in this file, you will find a detailed explanation below.
timestep = 60           # max length of a sentenced, but will be modified in later part
						# since 'PAD' needs to be added in the front of and after a sentence.
window = [3, 4, 5]      # CNN window size
reuse = True            # resue Word embedding matrix (word is represented in number in Class Data, but we
						# need to convert it into a 300D vector), enable this to share this embedding matrix.

# params about cost (MMD, KDE, BLEU, COST_G, COST_D...)
diag = 0.1
kde_bandwidth = 0.2
lambda_m = 0.001        # parameter in (2) in paper
lambda_q = 0            # should be a parameter in (2) in paper. Author included it in his code.
lambda_r = 0.001        # parameter in (2) in paper
ngram = [2, 3, 4]       # Used for BLEU, n-gram.
seen_size = 0
mmd_param = {'logistic': 0.2, 'logistic_s': 5, # Weight in multi-kernel MMD
			 'gaussian': 0.8, 'gaussian_sigma': 20}

# params about training
training_epochs = 50
learning_rate_g = 0.00005
learning_rate_d = 0.0001

# params about print
dispFreq = 10                   # display frequency. How many batches before display some results / log on screen.
dg_ratio = 1                    # Train how many times Generator before train once Discriminator
validFreq = int(batch_size * 2) # validation frequency. How many batches before validate once.
saveFreq = int(batch_size * 2)  # save frequency. How many batches before save some paramters / results / log into a file.

# check existence of paths.
if not os.path.exists(dataPath):
	os.mkdir(dataPath)
if not os.path.exists(savePath):
	os.mkdir(savePath)
if not os.path.exists(logPath):
	os.mkdir(logPath)
if not os.path.exists(paramSavePath):
	os.mkdir(paramSavePath)

# init config for tools
tf.set_random_seed(seed)
logger = set_logger(logPath, timestr, os.path.basename(__file__))
# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# Basic setup finishes here --------------------------------------------------------------------------------|
# Graph building begins here -------------------------------------------------------------------------------|
data = Data(dataPath=dataPath, savePath=savePath, paramSavePath=paramSavePath, logPath=logPath,
			debug=debug, split_percent=split_percent, batch_size=batch_size, timestr=timestr,
			timestep=timestep, window=window)
vocab_size = data.vocabSize

z = tf.placeholder(tf.float32, [batch_size, input_dim * len(window)])
x = tf.placeholder(tf.int32, [batch_size, timestep + 2 * (max(window) - 1)])
gen = Generator(timestep=timestep, window=window, batch_size=batch_size,
				vocab_size=vocab_size, paramSavePath=paramSavePath, logPath=logPath,
				input_dim=input_dim, hidden_size=hidden_size, keep_prob=keep_prob,
				L=L, timestr=timestr, debug=debug)
print('Generator building finished!')
dis = Discriminator(window=window, vocab_size=vocab_size, paramSavePath=paramSavePath, logPath=logPath,
					input_dim=input_dim, keep_prob=keep_prob, reuse=reuse, generator=gen, timestr=timestr, debug=debug)
print('Discriminator building finished!')
pred_w, gee = gen.generate(z) # gee stands for generatee (I made this word. hhh)
# pred_w: [timestep * batch_size, vocab_size]
# gee   : [batch_size, timestep, input_dim, 1]
timestep = gen.timestep # The addition of PAD makes timestep bigger. So adjustion needs here.
gee_cnn_out, gee_dised = dis.discriminate(gee) # gee_cnn_out is the cnn output before the FC layer.
											   # gee_dised stands for generatee that has been discriminated.
# gee_cnn_out: [batch_size, input_dim * len(window)]
# gee_dised  : [batch_size, 1]
result3 = tf.reshape(tf.argmax(pred_w, axis=1), [batch_size, timestep]) # [batch_size, timestep]
print('result3 prepared')
result5 = gen.max_print[0] # [1, timestep]
print('result5 prepared')
# This is also not mentioned in the paper.
# but seems to be the reverse part for exp in generator.
# Not mentioned starts here ---------------------------|
# gee_recon = (gee_dised + 1)/2
# gee_recon = tf.log(gee_recon)
# z_code = tensor.cast(z[:, 0], dtype='int32')
# z_index = tensor.arange(n_batch)
# fake_logent = gee_recon[z_index ,z_code]
# l_I = tf.reduce_sum(fake_logent)
# not mentioned ends here -----------------------------|

# But let still keep the sigmoid activition first -----|
gee_dised = tf.sigmoid(gee_dised)
# And this will be replaced then ----------------------|
result2 = tf.reduce_mean(gee_dised) # There will be result(i), just for display train process.
									# Check below. There is a big print function.
									# [1]
print('result2 prepared')
real = tf.gather_nd(dis.Wemb, tf.reshape(x, [-1, 1]))
real = tf.reshape(real, [batch_size, timestep, input_dim, 1])
real_cnn_out, real_dised = dis.discriminate(real)
# real_cnn_out: [batch_size, input_dim * len(window)]
# real_dised  : [batch_size, 1]
# This is also not mentioned in the paper.
# but seems to be the reverse part for exp in generator ----------------------------|
real_dised = tf.sigmoid(real_dised * (1 - label_smoothing) + 0.5 * label_smoothing)
# not mentioned ends here ----------------------------------------------------------|
# This turns out to be an implementation of a overfitting-avoidance technique.
# proposed by https://arxiv.org/pdf/1512.00567.pdf
# also used here: https://arxiv.org/pdf/1606.03498.pdf
# nice illusion here: http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
# But these three seem to have confict with each other
# I will add this technique after I work it out
# I worked it out, slightly changed the weight. Check 584-586 lines in URL below:
# https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/losses/losses_impl.py
# In case it is change, it says:
# multi_class_labels = (multi_class_labels * (1 - label_smoothing) + 0.5 * label_smoothing)
# And then losses = nn.sigmoid_cross_entropy_with_logits(labels=multi_class_labels, logits=logits)
# Now this would be pretty clear.
# But let still keep the sigmoid activition first -----|
# real_dised = tf.sigmoid(real_dised)
# And this will be replaced then ----------------------|

result1 = tf.reduce_mean(real_dised) # [1]
print('result1 prepared')

# KDE_INPUT
#gee_kde = tf.reshape(gee, [batch_size, timestep * input_dim])
#real_kde = tf.reshape(real, [batch_size, timestep * input_dim])
#kernel = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(real_kde)
#kde_input = tf.reduce_mean(kernel.score_samples(gee_kde))
#print('KDE_INPUT prepared')

# KDE
#kernel = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(real_cnn_out)
#kde = tf.reduce_mean(kernel.score_samples(gee_cnn_out))
#print('KDE prepared')

# MMD
mmd = cal_mmd(gee_dised, real_dised, mmd_param, batch_size)
print('MMD prepared')

# COST
gan_cost = - tf.reduce_sum(tf.log(1 - gee_dised + 1e-6)) / (input_dim * len(window)) - \
			tf.reduce_sum(tf.log(real_dised + 1e-6)) / batch_size 
print('gan_cost prepared')
# no idea why, author implemented like this in his code but not in his paper ---|
r_t  = gee_dised / 2.0 + .5
z_t = z / 2.0 + .5
recon_cost = tf.reduce_sum(- z_t * tf.log(r_t + 0.0001) - (1. - z_t) * tf.log(1.0001 - r_t)) / \
			(batch_size * input_dim * len(window))
print('recon_cost prepared')
# no idea ends here ------------------------------------------------------------|

# The COST in his code is different from the one described in his paper!!!! ----|
# Look above, and you would find that the code to generate this ----------------|
# d_cost = gan_cost + lambda_m * mmd - lambda_r * recon_cost - lambda_q * l_I / (input_dim * len(window))
# g_cost = mmd + lambda_q * l_I / (input_dim * len(window))
# not mentioned ends here ------------------------------------------------------|
# So, use code below instead.
d_cost = gan_cost - lambda_m * mmd + lambda_r * recon_cost
g_cost = mmd

print('d_cost, g_cost prepared')
# Optimizer
d_optimizer = tf.train.AdamOptimizer(learning_rate_d).minimize(d_cost)
g_optimizer = tf.train.AdamOptimizer(learning_rate_g).minimize(g_cost)
print('d_optimizer, g_optimizer prepared')
# sufficient statistics for log display and record
cur_size = seen_size * 1.0
identity = tf.eye(input_dim * len(window)) * diag
fake_mean = tf.reduce_mean(gee_cnn_out, axis=0)
real_mean = tf.reduce_mean(real_cnn_out, axis=0)
fake_xx = tf.matmul(gee_dised, gee_dised, True)
real_xx = tf.matmul(real_dised, real_dised, True)
acc_fake_xx = (tf.eye(input_dim * len(window)) * cur_size + fake_xx) / batch_size
acc_real_xx = (tf.eye(input_dim * len(window)) * cur_size + real_xx) / batch_size
acc_fake_mean = (tf.zeros([input_dim * len(window)]) * cur_size + fake_mean * batch_size) / batch_size
acc_real_mean = (tf.zeros([input_dim * len(window)]) * cur_size + real_mean * batch_size) / batch_size

cov_fake = acc_fake_xx - dot(tf.expand_dims(acc_fake_mean, 1), tf.expand_dims(acc_fake_mean, 1)) + identity
cov_real = acc_real_xx - dot(tf.expand_dims(acc_real_mean, 1), tf.expand_dims(acc_real_mean, 1)) + identity
  
cov_fake_inv = tf.matrix_inverse(cov_fake)
cov_real_inv = tf.matrix_inverse(cov_real)

result4 = tf.trace(dot(cov_fake_inv, cov_real) + dot(cov_real_inv, cov_fake))
print('result4 prepared')
result6 = tf.reduce_sum((fake_mean - real_mean)**2)
print('result6 prepared')
print('mdoel building finished. time left {}'.format(time.time() - model_build_start_time))
logger.info('mdoel building finished. time left {}'.format(time.time() - model_build_start_time))
# graph building ends here ---------------------------------------------------------------------------------|
# Training starts
print('training starts')
train = data.get_first_batch('train')
# validation = data.get_first_batch('validation')
# Suprisingly the author didn't use a validation set!
test = data.test
total_batch = 0
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	if debug:
		logger.info('gen.Wemb before run')
		logger.info(sess.run(tf.Print(gen.Wemb, [gen.Wemb])))
	total_start_time = time.time()
	for epoch in range(training_epochs):
		epoch_start_time = time.time()
		for batch in range(data.trainMaxBatch):
			if debug:
				print('batch ' + str(batch) + ' begins')
			total_batch += 1
			train_x = train
			train_z = np.random.uniform(-1, 1, [batch_size, input_dim * len(window)]).astype('float32')
			train_z[:, 0] = np.random.randint(2, size=batch_size).astype('float32')
			if debug:
					logger.info('g_cost， mmd before train: ')
					logger.info(sess.run([g_cost, mmd], feed_dict={x: train_x, z: train_z}))
			_, gc = sess.run([g_optimizer, g_cost], feed_dict={x: train_x, z: train_z})
			if debug:
					logger.info('gen.Wemb after run')
					logger.info(sess.run(tf.Print(gen.Wemb, [gen.Wemb])))
					logger.info('g_cost, mmd before train: ')
					logger.info(sess.run([g_cost, mmd], feed_dict={x: train_x, z: train_z}))
			if np.mod(total_batch, dispFreq) == 0:
				temp_out = sess.run([result1, result2, result3, result4, result5, result6], feed_dict={x: train_x, z: train_z})
				dc = sess.run([d_cost], feed_dict={x: train_x, z: train_z})
				dc = dc[0]
				print('real '             + str(round(temp_out[0], 2)) + \
					  ' fake '            + str(round(temp_out[1], 2)) + \
					  ' Covariance loss ' + str(round(temp_out[3], 2)) + \
					  ' mean loss '       + str(round(temp_out[5], 2)) + \
					  ' cost_g '          + str(gc) + \
					  ' cost_d '          + str(dc))
				print("Generated:" + " ".join([data.mapToWord[x][0] for x in temp_out[2][0] if x != data.mapToNum['PAD'][0]]))
				
				logger.info('Epoch {} Update {} Cost G {} Real {} Fake {} loss_cov {}  meanMSE {}'.format(epoch, total_batch, gc, round(temp_out[0], 2), round(temp_out[1], 2), temp_out[3], temp_out[5])) 
				logger.info('Generated: {}'.format(" ".join([data.mapToWord[x][0] for x in temp_out[2][0] if x != data.mapToNum['PAD'][0]]))) 
			if np.mod(epoch, dg_ratio) == 0:
				if debug:
					logger.info('d_cost before train: ')
					logger.info(sess.run([d_cost], feed_dict={x: train_x, z: train_z}))
				_, dc = sess.run([d_optimizer, d_cost], feed_dict={x: train_x, z: train_z})
				if debug:
					logger.info('d_cost after train: ')
					logger.info(sess.run([d_cost], feed_dict={x: train_x, z: train_z}))
				if np.mod(epoch, dispFreq) == 0:
					logger.info('Cost D {}'.format(dc))
			
			if np.mod(epoch, saveFreq) == 0:
				logger.info('Saving ...')

				save_path = saver.save(sess, logPath + timestr + ".ckpt")
				logger.info('Model saved in file: %s' %save_path)

				#data.save('data-' +         timestr)
				#dis.save('discriminator-' + timestr)
				#gen.save('generator-' +     timestr)

				logger.info('Done ...')

			if np.mod(epoch, validFreq) == 0:
				val_z = np.random.uniform(-1, 1, [batch_size, input_dim * len(window)])
				#val_z = tf.random_uniform([batch_size, input_dim * len(window)], minval=-1, maxval=1)
				predset = sess.run([result3], feed_dict={x: train_x, z: val_z})
				predset = predset[0]
				[bleu2s, bleu3s, bleu4s] = cal_BLEU(predset, test, data, ngram, debug) # Check def of this func why need to pass <data> object
				
				logger.info('Valid BLEU2 = {}, BLEU3 = {}, BLEU4 = {}'.format(bleu2s, bleu3s, bleu4s))
				print ('Valid BLEU (2, 3, 4): ' + ' '.join([str(round(x, 3)) for x in (bleu2s, bleu3s, bleu4s)]))


				#print ('Valid KDE_INPUT = {} and KDE = {}'.format(kde_input, kde))
		print('epoch {} finished, total time left {}, this epoch {}'.format(epoch, time.time() - total_start_time, time.time() - epoch_start_time))
