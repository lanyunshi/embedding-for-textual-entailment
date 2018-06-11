import nltk 
from collections import defaultdict
import numpy as np
import argparse
import os

#np.random.seed(134)

def FrequentSNLI(datapath, WP):
	string = dict()
	output = defaultdict(list)
	for i in ['train', 'dev', 'test']:# 
		idx = 0
		data = open(datapath + i + ".txt", "r").readlines()
		for line in data:
			idx += 1
			line = line.replace("\n", "")
			string[str(i) + str(idx)] = line
			text, hyper, label = line.split("\t")
			for t in text.split(" "):
				for h in hyper.split(" "):
					if (t, h) in WP and t!=h:
						output[(t, h)] += [str(i) + str(idx)]
			if idx % 10000 == 0:
				print("process %s ... %s" %(i, idx*1.0/len(data)))
	return output, string

def ReadWordPair(datapath):
	f = open(datapath + 'strict.txt', 'r')
	data = f.readlines()
	f.close()
	WP = []
	for line in data:
		line = line.replace("\n", "")
		WP += [(line.split("\t")[0], line.split("\t")[2])]
	return set(WP)

def launch_main(datapath, sourcepath, wordpairpath):
	# get all word pairs of word pair data
	WP = ReadWordPair(wordpairpath)
	# get all word pairs of source textual entailment sentences
	WPStatistics, string = FrequentSNLI(datapath + sourcepath, WP)

	top = 10
	idx = 0
	train = []
	dev = []
	test = []
	# write a file to describe statistics of word pairs
	f = open('%sWPStatistics4.txt' %datapath, 'w')
	# assign sentences of different word pairs into either train/valid or test
	for i in sorted(WPStatistics, key=lambda i: len(WPStatistics[i]), reverse=True):
		idx_list = []
		if (np.random.random() < 0.9 and len(WPStatistics[i]) < 100) or \
			(np.random.random() < 0.5 and len(WPStatistics[i]) >= 100):
			for l in WPStatistics[i]:
				try:
					test += [string[l]]
					idx_list += ['test%s' %len(test)]
					del string[l]
				except:
					pass
		else:
			for l in WPStatistics[i]:
				try:
					if np.random.random() < 0.9:
						train += [string[l]]
						idx_list += ['train%s' %len(train)]
					else:
						dev += [string[l]]
						idx_list += ['valid%s' %len(valid)]
					del string[l]
				except:
					pass
		line = i[0] + ' ' + i[1] + ' ' + str(len(WPStatistics[i])) + '\t' + str(idx_list) + '\n'
		f.write(line)
		idx += 1			
	f.close()
	idx = 0 
	for l in string:
		idx += 1
		num = np.random.random()
		if num < 0.9:
			train += [string[l]]
		else:
			dev += [string[l]]
		if idx % 1000 ==0:
			print("... %s" %(idx*1.0/len(string)))

	# write data with non overlap split 
	if not os.path.exists('%ssequenceNS' %datapath):
		os.makedirs('%ssequenceNS' %datapath)
	for i in ['train', 'dev', 'test']:
		f = open('%ssequenceNS/%s.txt' %(datapath, i), 'w')
		f.write("\n".join(eval(i)))
		f.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('datapath', help='path contains textual entailment data')
	parser.add_argument('sourcepath', help='path contains original textual entailment data')
	parser.add_argument('wordpairpath', help='path of word pairs data')
	args = parser.parse_args()

	launch_main(args.datapath, args.sourcepath, args.wordpairpath)