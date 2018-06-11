import scipy.sparse as sp
import pickle
import os
import sys
import argparse

def load_data(file):
    with open(file, 'r') as f:
        row_data = f.readlines()
    l = []
    r = []
    t = []
    output = []
    for line in row_data:
        line = line.replace('\n','').replace('\r','')
        line = line.split("\t")

        output += ['\t'.join([line[0], line[1], line[2]])]
        l += [line[0]]
        r += [line[1]]
        t += [line[2]]
    return l, r, t, output

def launch_main(datapath):
    # creat dictionary
    l = []
    r= []
    t = []
    output = []
    for f in ['wholedata']:
        subl, subr, subt, suboutput = load_data(datapath + "/%s.txt" %f)
        l += subl
        r += subr
        t += subt
        output += suboutput
    overlap_set = set(l) & set(t)
    left_set = set(l) - overlap_set
    right_set = set(t) - overlap_set
    rel_set = set(r)
    # make dictionary
    idx = 0
    rel_idx = 0
    entity2idx = {}
    idx2entity = {}
    rel2idx = {}
    idx2rel = {}
    for entity in left_set:
        idx2entity[idx] = entity
        entity2idx[entity] = idx
        idx +=1
    for entity in overlap_set:
        idx2entity[idx] = entity
        entity2idx[entity] = idx
        idx +=1
    for entity in right_set:
        idx2entity[idx] = entity
        entity2idx[entity] = idx
        idx +=1
    for entity in rel_set:
        idx2entity[idx] = entity
        entity2idx[entity] = idx
        rel2idx[entity] = rel_idx
        idx2rel[rel_idx]= entity
        rel_idx += 1
        idx +=1
    f = open(datapath + '/entity2idx2.pkl', 'wb')
    pickle.dump(entity2idx, f, -1)
    f.close()
    f = open(datapath + '/idx2entity2.pkl', 'wb')
    pickle.dump([idx2entity, rel2idx, idx2rel], f, -1)
    f.close()

    print(r'entity >>> %s  left entity >>> %s overlap entity >>> %s  right entity >>> %s  relation entity >>> %s' %(
            idx, len(left_set), len(overlap_set), len(right_set), len(rel_set)))
    f = open(datapath + '/info2.txt', 'w')
    f.write(r'entity >>> %s  left entity >>> %s overlap entity >>> %s  right entity >>> %s  relation entity >>> %s' %(
            idx, len(left_set), len(overlap_set), len(right_set), len(rel_set)))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datapath', help='the path containing extracted word pairs from WordNet')
    args = parser.parse_args()

    launch_main(args.datapath)