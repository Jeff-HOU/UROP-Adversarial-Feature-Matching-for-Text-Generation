''' save to pickle file. meaningless! maybe remove later.
import os
import pandas as pd
from nltk.tokenize import sent_tokenize
import pickle

read_dir = 'raw_data/'
save_dir = 'abstract/'

if not os.path.exists(save_dir):
	os.mkdir(save_dir)

for filename in os.listdir(read_dir):
	a = pd.read_csv(read_dir + filename)
	b = a['abstract'].replace(to_replace='\n', value=' ', regex=True).replace(to_replace='\$[^$]*\$|\S+_{[^$\s]+}|[a-zA-Z]+\([^$\s]+\)', value='__LaTex__', regex=True)
	sentences = []
	for _, row in b.iteritems():
		try:
			sentences.extend(sent_tokenize(row))
		except:
			continue
	with open(save_dir + filename.replace('.csv', '') + '.pkl', 'wb') as f:
		pickle.dump(sentences, f)
'''

# TODO: Fix this!
# Use list to temp store all str to reduce I/O usage.
import os
import re
import pandas as pd
from nltk.tokenize import sent_tokenize

read_dir = 'raw_data/'
save_dir = 'abstracts/'

if not os.path.exists(save_dir):
	os.mkdir(save_dir)

for filename in os.listdir(read_dir):
	a = pd.read_csv(read_dir + filename)
	b = a['abstract'].replace(to_replace='\n', value=' ', regex=True).replace(to_replace='\$[^$]*\$|\S+_{[^$\s]+}|[a-zA-Z]+\([^$\s]+\)', value='__LaTex__', regex=True)
	with open(save_dir + filename.replace('.csv', '') + '.txt', 'w') as f:
		for _, row in b.iteritems():
			try:
				write_string = sent_tokenize(row)[0].strip("'")
				write_string = re.sub(r'(.)([.!?])$', r"\1 \2", write_string)
				f.write("%s\n" % write_string)
			except:
				continue