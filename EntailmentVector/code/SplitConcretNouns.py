

def LoadConcretWP(datapath):
	ConcretDic = {}
	with open('%s/LogicWordNetSICK/Concretness.txt' %datapath) as f:
		for line_idx, line in enumerate(f):
			if line_idx > 0:
				w, _, score, _, _, _, _, _, _ = line.split('\t')
				if float(score) < 3.5:
					ConcretDic[w] = float(score)
	print(len(ConcretDic))
	print(sorted(ConcretDic.items(), key=lambda x: x[1])[::-1][:20])
	return ConcretDic

def SplitTestFromWP(datapath, ConcretDic):
	g2 = open('%s/LogicWordNetSICK/data/abstract_dic.txt' %datapath, 'w')
	for d in ConcretDic:
		g2.write(d + '\n')
	g2.close()
	g = open('%s/LogicWordNetSICK/data/abstract_test.txt' %datapath, 'w')
	with open('%s/LogicWordNetSICK/data/strict.txt' %datapath) as f:
		for line_idx, line in enumerate(f):
			line = line.replace('\n', '')
			h, r, t = line.split('\t')
			if h in ConcretDic and t in ConcretDic:
				g.write(line + '\n')
	g.close()

datapath = '/home/yunshi/BinaryCode/ERP'
ConcretDic = LoadConcretWP(datapath)
SplitTestFromWP(datapath, ConcretDic)