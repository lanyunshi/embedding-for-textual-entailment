import numpy as np
import pickle
from sklearn.decomposition import PCA
import re
import argparse
import cPickle

def w2v_load(file):
	f = open(file, "r")
	w2v = {}
	idx = 0
	while True:
		line = f.readline()
		if not line:
			break
		else:
			line = line.replace("\n", "")
			line = re.sub('\s$', '', line)
			v = []
			for i in line.split(" ")[1:]:
				v += [float(i)]
			w2v[line.split(" ")[0]] = np.asarray(v)
	return w2v

def obtain_vocab(path):
	entity2idx = {}
	with open(path) as f:
		for line_idx, line in enumerate(f):
			line = line.replace('\n', '')

			h, _, t = line.split('\t')
			if h not in entity2idx:
				entity2idx[h] = len(entity2idx)
			if t not in entity2idx:
				entity2idx[t] = len(entity2idx)
	return entity2idx

def Con_con(w2v, entity2idx, Word):
	concrete = set()
	with open('LogicWordNetSICK/data/abstract_dic.txt') as f:
		for line_idx, line in enumerate(f):
			concrete.add(line.replace('\n', ''))
	dim = Word.shape[1]
	print("dim num: " + str(dim))
	dic = {}
	idx = 0
	for w in w2v:
		if w in entity2idx.keys() and w in concrete:
			try:
				#a = entity2idx[w]
				dic[w] = np.concatenate([w2v[w], Word[entity2idx[w]]])
				idx += 1
				# print(w)
			except:
				pass
		else:
			dic[w] = np.concatenate([w2v[w], np.random.uniform(-0.1, 0.1, dim)])
	print("enrich word number: " + str(idx))
	#print("dim num: " + str(len(dic[w])))
	return dic

def Sin_con(w2v, entity2idx, Word):
	dim = 50
	print("dim num: " + str(dim))
	dic = {}
	idx = 0
	for w in w2v:
		if w in entity2idx.keys():
			try:
				dic[w] = Word[entity2idx[w]]
				idx += 1
			except:
				pass
		else:
			dic[w] = np.random.uniform(-0.1, 0.1, dim)
	print("enrich word number: " + str(idx))
	print(len(w2v))
	return dic

def sig(Word):
	word = 1/(1+np.exp(-Word))
	return word

def main(wordpairpath, datapath, evpath, mode):
	w2v = w2v_load(('%sglove/w2v.txt' %datapath).replace('/glove', 'glove'))
	_, _, _, _, Word, _ = pickle.load(open('%s/%s.pkl' %(wordpairpath, evpath), "r"))

	entity2idx = cPickle.load(open('%s/entity2idx.pkl' %wordpairpath))
	print(len(entity2idx))

	if mode == 'our':
		dic = Con_con(w2v, entity2idx, Word)
	elif mode == 'nn':
		dic = Con_con(w2v, entity2idx, Word)

	output = []
	for d in dic:
		o = []
		for v in dic[d]:
			o += [str(v)]
		output += [d + " " + " ".join(o)]
	print("vab num: " + str(len(output)))
	f = open(('%sglove/w2v%s_abstract.txt' %(datapath, mode)).replace('/glove', 'glove'), "w")
	f.write("\n".join(output))
	f.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('wordpairpath', help='path of word pairs data')
	parser.add_argument('datapath', help='path contains textual entailment data')
	parser.add_argument('evpath', help='path of entailment vectors')
	parser.add_argument('mode', help='mode of entailment vectors: My or NN')
	args = parser.parse_args()

	main(args.wordpairpath, args.datapath, args.evpath, args.mode)