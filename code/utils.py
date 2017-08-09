import tensorflow as tf
import logging
from nltk.translate.bleu_score import modified_precision

def dot_product(a, b, transpose_a=False, transpose_b=True):
	return tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)

def log_mean_exp(A,b,sigma):
    a = - 0.5 * tf.reduce_sum(((A - tf.tile(b,[A.get_shape().as_list()[0],1]))**2), axis=1) / (sigma**2)
    max_ = a.max()
    return max_ + tf.log(tf.reduce_mean(tf.exp(a - tf.tile(max_, a.get_shape().as_list()[0]))))

def cal_KDE(x, mu, sigma):
	s1, updates = tf.scan(fn=lambda i,s: s + log_mean_exp(mu,X[i, :],sigma))
def cal_nkde(X,mu,sigma):
    s1,updates=theano.scan(lambda i,s: s+log_mean_exp(mu,X[i,:],sigma), sequences=[theano.tensor.arange(X.shape[0])],outputs_info=[np.asarray(0.,dtype="float32")])
    E=s1[-1]
    Z=mu.shape[0]*theano.tensor.log(sigma*np.sqrt(np.pi*2))
    return (Z-E)/mu.shape[0]
'''
def calculate_gaussian_mmd(tempxx, tempxy, tempyy, sigma_range):
	kxx, kxy, kyy = [0., 0., 0.]
	for sigma in sigma_range:
		kxx += tf.reduce_mean(tf.exp(-tempxx/2/(sigma**2)))
		kxy += tf.reduce_mean(tf.exp(-tempxy/2/(sigma**2)))
		kyy += tf.reduce_mean(tf.exp(-tempyy/2/(sigma**2)))
	return tf.sqrt(kxx + kyy - 2 * kxy) # Author uses square in his paper but root in his code!
def calculate_linear_mmd(tempxx, tempxy, tempyy):
	kxx = tf.reduce_mean(tempxx)
	kxy = tf.reduce_mean(tempxy)
	kyy = tf.reduce_mean(tempyy)
	return tf.sqrt(kxx + kyy - 2 * kxy)
def calculate_mmd(x, y, mmd_w, dividend=1.):
	dist_x, dist_y = x / dividend, y / dividend
	x_sq = tf.reduce_sum(tf.square(dist_x), axis=1, keep_dims=True) # [batch_size, 1]
	y_sq = tf.reduce_sum(tf.square(dist_y), axis=1, keep_dims=True) # [batch_size, 1]

	tempxx = -2 * dot_product(dist_x, dist_x) + x_sq + x_sq  # (x -x')^2
	tempxy = -2 * dot_product(dist_x, dist_y) + x_sq + y_sq  # (x -y)^2
	tempyy = -2 * dot_product(dist_y, dist_y) + y_sq + y_sq  # (y -y')^2
	gaussian_mmd = calculate_gaussian_mmd(tempxx, tempxy, tempyy, sigma_range)
	linear_mmd = calculate_linear_mmd(tempxx, tempxy, tempyy, sigma_range)
	mmd = mmd_w['linear'] * linear_mmd + mmd_w['gaussian'] * gaussian_mmd
	return mmd
'''
def logistic_kernel(x, y, param):
	# useful for calculate_logistic_mmd, same symbol as https://en.wikipedia.org/wiki/Logistic_distribution
	s = param['logistic_s']
	numerator = tf.exp(-(x - y) / s)
	denominator = s * tf.square(1 + tf.exp(-(x - y) / s))
	return numerator / denominator
def calculate_logistic_mmd(x0, y0, x1, y1, param, batch_size):
	kxx = tf.reduce_sum(logistic_kernel(x0, x1, param)) / batch_size**2
	kxy = tf.reduce_sum(logistic_kernel(x0, y1, param)) / batch_size**2
	kyy = tf.reduce_sum(logistic_kernel(y0, y1, param)) / batch_size**2
	return kxx - 2*kxy + kyy
def calculate_gaussian_mmd(x0, y0, x1, y1, param, batch_size):
	kxx = tf.reduce_sum(tf.exp(-tf.square(x0 - x1) / 2 / param['gaussian_sigma'])) / batch_size**2
	kxy = tf.reduce_sum(tf.exp(-tf.square(x0 - y1) / 2 / param['gaussian_sigma'])) / batch_size**2
	kyy = tf.reduce_sum(tf.exp(-tf.square(y0 - y1) / 2 / param['gaussian_sigma'])) / batch_size**2
	return kxx - 2*kxy + kyy
def calculate_mmd(x, y, param, batch_size):
	xt = tf.transpose(x)
	yt = tf.transpose(y)
	x0 = tf.identity(x)
	y0 = tf.identity(y)
	x1 = tf.identity(xt)
	y1 = tf.identity(yt)
	for i in range(batch_size - 1):
		x0 = tf.concat([x0, x], axis=1)
		y0 = tf.concat([y0, y], axis=1)
		x1 = tf.concat([x1, xt], axis=0)
		y1 = tf.concat([y1, yt], axis=0)
	gaussian_mmd = calculate_gaussian_mmd(x0, y0, x1, y1, param, batch_size)
	logistic_mmd = calculate_logistic_mmd(x0, y0, x1, y1, param, batch_size)
	mmd = param['logistic'] * logistic_mmd + param['gaussian'] * gaussian_mmd
	return mmd
def prepare_for_bleu(sentence, data, ngram):
	# Need to calculate bleu 2-4, so the length of a sentence need to at least 4.
	# If not, add PAD at the end.
	sent = [data.mapToWord[x][0] for x in sentence if x != data.mapToNum['PAD'][0]]
	while len(sent) < max(ngram):
		sent.extend(['PAD'])
	return sent

def cal_BLEU(hypp, reff, data, ngram, debug):
	hyp = []
	for s in hypp:
		hyp.append(prepare_for_bleu(s, data, ngram))
	ref = []
	for s in reff:
		ref.append(prepare_for_bleu(s, data, ngram))
	# Check URL below for this powerful package (modified_precision):
	# http://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.modified_precision
	bleu = [0.] * len(ngram)
	if debug:
		print(hyp)
	for s in hyp:
		#print(s)
		for i in range(len(ngram)):
			bleu[i] += round(modified_precision(ref, s, n=ngram[i]), 5)
	return [x / len(hyp) for x in bleu]

def set_logger(logPath, timestr, name):
	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	fh = logging.FileHandler(logPath + timestr + '.log')
	fh.setLevel(logging.INFO)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	logger.addHandler(fh)
	return logger