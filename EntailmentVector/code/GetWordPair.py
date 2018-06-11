import nltk
import pickle
import word2vec
from gensim import models
import numpy as np
from collections import defaultdict
import operator
import re
import inflect
import itertools
from treelib import Node, Tree

p = inflect.engine()

def comma(string):
	string = string.replace(",", " ,")
	string = string.replace(".", " .")
	return string

def load_SICK(datapath):
	f = open(datapath + 'SICKtv.txt', 'r')
	idx = 0
	index = 0
	entity2idx = {}
	idx2entity = {}
	entity2tag = {}
	idxtext = []
	while True:
		line = f.readline()
		if idx!= 0:
			if not line:
				break
			else:
				t1 = comma(line.split("\t")[1]).split(" ")
				t2 = comma(line.split("\t")[2]).split(" ")
#				t1 = nltk.word_tokenize(line.split("\t")[1])
				t1 = nltk.pos_tag(t1)
#				t2 = nltk.word_tokenize(line.split("\t")[2])
				t2 = nltk.pos_tag(t2)
				s1 = []; s2 = []
				for t in t1:
					if re.search("NN\w*", t[1]):
						t = t[0].lower()
						if t not in entity2idx.keys():
							entity2idx[t] = index
							idx2entity[index] = t
							index += 1
						s1 += [t]
				for t in t2:
					if re.search("NN\w*", t[1]):
						t = t[0].lower()
						if t not in entity2idx.keys():
							entity2idx[t] = index
							idx2entity[index] = t
							index += 1
						s2 += [t]
				idxtext += [(s1, s2)]
		idx += 1
		if idx%500 == 0:
			print("POS tag SICK, finish ... " + str(1.0*idx/10000))
	return entity2idx, idx2entity, idxtext

def load_bin_vec(w2v_file, vocab):
    model = models.Word2Vec.load_word2vec_format(w2v_file, binary=True)
    word_vecs = {}
    idx = 0
    for word in model.vocab:
#        print(word)
        if  word in vocab.keys():
            word_vecs[word] = model[word]
        if idx % 10000 ==0:
            print("already process ... %s" %((idx + 0.1)/len(model.vocab)))
        idx += 1
    return word_vecs

def add_unknown_words(word_vecs, vocab, k=300):
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

datapath = "/home/yunshi/ERP/SNLI/"

# output = []
# for w in w2v:
# 	vector = []
# 	for v in w2v[w]:
# 		vector += [str(v)]
# 	output += [w + " " +  " ".join(vector)]
# f = open(datapath + "w2v.txt", "w")
# f.write("\n".join(output))
# f.close()

def compare(idxtext, w2v):
	pair2sid = defaultdict(list)
	pair2distance = {}
	for i in range(len(idxtext)):
		t = idxtext[i]
		p2d = {}
		for i1 in t[0]:
			for i2 in t[1]:
				if i1>i2:
					p2d[(i1, i2)] = np.linalg.norm(w2v[i1] - w2v[i2])
				elif i1<i2:
					p2d[(i2, i1)] = np.linalg.norm(w2v[i1] - w2v[i2])
		for x in p2d.keys():
			pair2sid[x] += [(i + 1)]
			pair2distance[x] = p2d[x]
	return pair2sid, pair2distance

def rank_all(datapath):
	p2sid = pickle.load(open(datapath + "pair2sid.pkl", "r"))
	p2d = pickle.load(open(datapath + "pair2distance.pkl", "r"))
	d = []
	s = []
	d = np.asarray([p2d[p] for p in p2sid])
	s = np.asarray([len(p2sid[p]) for p in p2sid])
	normalize = (s/max(s) + 1 - d/max(d))/2
	rank = {}
	i = 0
	for p in p2sid:
		rank[p] = normalize[i]
		i += 1
	sorted_r = sorted(rank.items(), key=operator.itemgetter(1), reverse= True)
	output = []
	for i in range(int(0.1*len(sorted_r))):
		output += [sorted_r[i][0][0] + "\t" + sorted_r[i][0][1]]
	f = open(datapath + "label.txt", "w")
	f.write("\n".join(output))
	f.close()

# entity2idx, idx2entity, idxtext = load_SICK(datapath)
# print("num words in SICK: " + str(len(entity2idx)))
# f = open(datapath + "entity2idx.pkl", "w")
# pickle.dump(entity2idx, f)
# f.close()
# f = open(datapath + "idx2entity.pkl", "w")
# pickle.dump(idx2entity, f)
# f.close()
# f = open(datapath + "idxtext.pkl", "w")
# pickle.dump(idxtext, f)
# f.close()
# w2v_file = datapath + 'GoogleNews-vectors-negative300.bin'
# print("loading word2vec")
# w2v = load_bin_vec(w2v_file, entity2idx)
# print ("word2vec loaded!")
# print ("num words already in word2vec: " + str(len(w2v)))
# add_unknown_words(w2v, entity2idx, k = 300)
# f = open(datapath + "w2v.pkl", "w")
# pickle.dump(w2v, f)
# f.close()

# idxtext = pickle.load(open(datapath + "idxtext.pkl", "r"))
# w2v = pickle.load(open(datapath + "w2v.pkl", "r"))
# pair2sid, pair2distance = compare(idxtext, w2v)
# f = open(datapath + "pair2sid.pkl", "w")
# pickle.dump(pair2sid, f)
# f.close()
# f = open(datapath + "pair2distance.pkl", "w")
# pickle.dump(pair2distance, f)
# f.close()

# rank_all(datapath)

def single(output, data):
	un_change = []
	for d in data:
		if p.singular_noun(d[0]) == d[1] or p.singular_noun(d[1]) == d[0]:
			output += [d[0] + "\t" + "=" + "\t" + d[1]]
		else:
			un_change += [d]
	print("before single: " + str(len(data)) + " after: " + str(len(un_change)))
	output += ["\n"]
	return output, un_change

def use_syn(output, data):
	un_change = []
	word2syn = defaultdict(list)
	syn2word = defaultdict(list)
	f = open(datapath + "wn_s.pl", "r")
	while True:
		line = f.readline()
		if not line:
			break
		else:
			word2syn[line.split(",")[2][1 : -1]] += [line.split(",")[0][2 : ]]
			syn2word[line.split(",")[0][2 : ]] += [line.split(",")[2][1 : -1]]
	f.close()
	ex_syn = set()
	for s in syn2word:
		if len(syn2word[s])>1:
			for i in syn2word[s]:
				for j in syn2word[s]:
					ex_syn.add((i, j))
	for d in data:
		d0 = d[0]; d1 = d[1]
		if p.singular_noun(d[0])!= False:
			d0 = p.singular_noun(d[0])
		if p.singular_noun(d[1])!= False:
			d1 = p.singular_noun(d[1])
		if (d0, d1) in ex_syn:
			output += [d[0] + "\t" + "=" + "\t" + d[1]]
		else:
			un_change += [d]
	print("before syn: " + str(len(data)) + " after: " + str(len(un_change)))
	output += ["\n"]
	return output, un_change, word2syn

def use_hyp(word2syn, output, data):
	un_change = []
	dic = Tree()
	dic.create_node("100001740", "100001740")
	add = -1
	while add != 0:
		add = 0
		f = open(datapath + "wn_hyp.pl", "r")
		while True:
			line = f.readline()
			if not line:
				break
			else:
				l, r = re.findall('\d+', line)
				try:
					dic.create_node(l, l, parent=r)
					add += 1
				except:
					pass
		print(dic.size())
	entail = defaultdict(list)
	for n in dic.all_nodes():
		for m in dic.subtree(n.tag).all_nodes():
			if m.tag!=n.tag:
				entail[n.tag].append(m.tag)
	label = set()
	for d in data:
		d0 = d[0]; d1 = d[1]
		if p.singular_noun(d[0])!= False:
			d0 = p.singular_noun(d[0])
		if p.singular_noun(d[1])!= False:
			d1 = p.singular_noun(d[1])
		for i in word2syn[d0]:
			for j in word2syn[d1]:
				if j in entail[i]:
					if d[0] + "\t" + ">" + "\t" + d[1] not in output:
						output += [d[0] + "\t" + ">" + "\t" + d[1]]
						label.add(d)
				elif i in entail[j]:
					if d[0] + "\t" + "<" + "\t" + d[1] not in output:
						output += [d[0] + "\t" + "<" + "\t" + d[1]]
						label.add(d)
		if d not in un_change and d not in label:
			un_change += [d]
	print("before single: " + str(len(data)) + " after: " + str(len(un_change)))
	output += ["\n"]
	del entail
	data = un_change
	del un_change 
	un_change = []
	alter = defaultdict(list)
	for n in dic.all_nodes():
		for m in dic.siblings(n.tag):
			if m.tag!=n.tag and n.bpointer!=m.tag:
				alter[n.tag].append(m.tag)
	label = set()
	for d in data:
		d0 = d[0]; d1 = d[1]
		if p.singular_noun(d[0])!= False:
			d0 = p.singular_noun(d[0])
		if p.singular_noun(d[1])!= False:
			d1 = p.singular_noun(d[1])
		for i in word2syn[d0]:
			for j in word2syn[d1]:
				if j in alter[i]:
					if d[0] + "\t" + "|" + "\t" + d[1] not in output:
						output += [d[0] + "\t" + "|" + "\t" + d[1]]
						label.add(d)
				elif i in alter[j]:
					if d[0] + "\t" + "|" + "\t" + d[1] not in output:
						output += [d[0] + "\t" + "|" + "\t" + d[1]]
						label.add(d)
		if d not in un_change and d not in label:
			un_change += [d]
	del alter
	print("before single: " + str(len(data)) + " after: " + str(len(un_change)))
	output += ["\n"]
	return output, un_change

# label dataset
# f = open(datapath + "label.txt", "r")
# raw_data = f.readlines()
# f.close()
# data = []
# for line in raw_data:
# 	line = re.sub("\n", "", line)
# 	data += [(line.split("\t")[0], line.split("\t")[1])]

# output = []
# output, data = single(output, data)
# output, data, word2syn = use_syn(output, data)
# output, data = use_hyp(word2syn, output, data)
# print("automaticaly label: " + str(1.0*len(output)/len(raw_data)))

# unlabel = []
# for d in data:
# 	unlabel += [d[0] + "\t" + d[1]]
# f = open(datapath + "unlabel.txt", "w")
# f.write("\n".join(unlabel))
# f.close()
# f = open(datapath + "autolabel.txt", "w")
# f.write("\n".join(output))
# f.close()

def clear_unccess(datapath):
	f = open("/home/yunshi/ERP/Logic/label.txt", "r")
	data = f.readlines()
	f.close()
	dic = {}
	for line in data:
		h, r, t = line.replace("\n", "").split("\t")
		dic[(h, t)] = r
	f = open(datapath + "label.txt", "r")
	data = f.readlines()
	f.close()
	output = []
	unlabel = []
	for line in data:
		line = line.replace("\n", "")
		h, t = line.split("\t")
		if (h, t) in dic:
			output +=[h + "\t" + dic[(h, t)] + "\t" + t]
		else:
			unlabel += [line]
	return output, unlabel

output, unlabel = clear_unccess(datapath)
f = open(datapath + "autolabel.txt", "w")
f.write("\n".join(output))
f.close()
f = open(datapath + "unlabel.txt", "w")
f.write("\n".join(unlabel))
f.close()