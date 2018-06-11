def check(f1, f2):
	f1_scores = []
	with open(f1) as f:
		for line_idx, line in enumerate(f):
			#print(line)
			line = line.replace('\n', '')
			idx, golden, pred = line.split('\t')
			golden = int(golden)
			pred = int(float(pred))
			f1_scores += [(golden, pred)]

	f2_scores = []
	with open(f2) as f:
		for line_idx, line in enumerate(f):
			line = line.replace('\n', '')
			idx, golden, pred = line.split('\t')
			golden = int(golden)
			pred = int(float(pred))
			f2_scores += [(golden, pred)]

	for idx in range(len(f1_scores)): # (f1_scores[idx] == (2, 1) and f2_scores[idx] == (2, 2))
		if (f1_scores[idx] == (0, 1) and f2_scores[idx] == (0, 1)):
			print(idx+1)

check('saved-model/sick_w2v/predict.txt', 'saved-model/sick_w2vour/predict.txt')