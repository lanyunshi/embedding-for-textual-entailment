import numpy as np
import pickle
from sklearn.decomposition import PCA
import re
from nltk.corpus import wordnet as wn

datapath = "/home/yunshi/BinaryCode/ERP/"

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

#w2v = pickle.load(open(datapath + "SNLI/" + "w2v.pkl", "r"))
w2v = w2v_load(datapath + "TextualEntailment/multiffn-nlitmp/data/SNLIglove/w2v.txt")
#w2v_retrofit = w2v_load("/home/yunshi/Retrofitting/retrofitting/out_vec.txt")
#entity2idx = pickle.load(open(datapath + "LogicWordNetSNLI/data/" + "entity2idx.pkl", "rb"))
W1, b1, W, b, Word, Label = pickle.load(open(datapath + "LogicWordNetSNLI/data/My/" + "2017_01_11_22_16_46.pkl", "r"))

def Cover(w2v, w2v_retrofit):
	for w in w2v:
		if w in w2v_retrofit:
			w2v[w] = w2v_retrofit[w]
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
	dim = Word.shape[1]
	print("dim num: " + str(dim))
	dic = {}
	idx = 0
	nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
	for w in w2v:
		if w in entity2idx.keys() and w in nouns:
			try:
				a = entity2idx[w]
				#dic[w] = np.concatenate([w2v[w], Word[entity2idx[w]]])
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

def Pca(dic, component):
	matrix = []
	for d in dic:
#		print(dic[d].shape)
		matrix += [dic[d]]
	matrix = np.asmatrix(matrix)
	matrix = PCA(component).fit_transform(matrix)
	index = 0
	for d in dic:
		dic[d] = matrix[index, :]
		index += 1
	return dic

entity2idx = obtain_vocab(datapath + "LogicWordNetSNLI/label.txt")
print(len(entity2idx))

#Word = sig(Word)
dic = Con_con(w2v, entity2idx, Word)
#dic = Sin_con(w2v, entity2idx, Word)
#dic = Pca(dic, 300)
#dic = Cover(w2v, w2v_retrofit)
output = []
for d in dic:
	o = []
	for v in dic[d]:
		o += [str(v)]
	output += [d + " " + " ".join(o)]
print("vab num: " + str(len(output)))
# f = open(datapath + "TextualEntailment/multiffn-nlitmp/data/SICKglove/" + "w2vour.txt", "w")
# f.write("\n".join(output))
# f.close()