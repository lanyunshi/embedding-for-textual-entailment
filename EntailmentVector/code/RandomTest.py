import numpy as np

g = open('data/test.txt', 'w')
with open('data/wholedata.txt') as f:
	for line_idx, line in enumerate(f):
		line = line.replace('\n', '')
		h, r, t = line.split('\t')
		if r == '=' and np.random.random() < 0.5:
			g.write(line + '\n')
		elif np.random.random() < 0.2:
			g.write(line + '\n')
g.close()
