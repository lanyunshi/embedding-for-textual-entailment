import numpy as np
import os
import re
import pickle
import sys
import datetime

import Model
import theano.tensor as T
from collections import defaultdict, OrderedDict
import theano
import time
from sklearn import svm
import zipfile
import argparse

import sys
#import MyModel as model

np.random.seed(124)

def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

def LogicPredict(U, lr_decay, batch_size, n_epochs, datasets, n_in, n_out, model):
    rng = np.random
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")

    # transform input index into embeddings
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape(
        (x.shape[0],x.shape[1]*Words.shape[1]))

    # squeeze input data whithin [0, 1]
    if model in ["My"]:
        layer0_input = Model.Sig(layer0_input)

    # define neural network graph
    classifier = eval('Model.%sRegression' %re.sub('[0-9]', '', model))(rng, 
        layer0_input , n_in, n_out) 
    params = classifier.params 
    if model not in ['NN2']:
        params += [Words]
    cost, show = classifier.loss_function(y)
    grad_updates = sgd(params, cost, lr_decay)
    
    # prepare train, validation, test data
    n_val_batches = int(0.5*(np.round(datasets[1].shape[0]/batch_size)))
    val_set = datasets[0][:n_val_batches*batch_size,:]
    datasets[0] = datasets[0][n_val_batches*batch_size:,:]
    test_set_x = datasets[1][:,:-1] 
    test_set_y = np.asarray(datasets[1][:,-1],"int32")

    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = rng.permutation(datasets[0])
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    train_set = rng.permutation(new_data)
    n_train_batches = new_data.shape[0]/batch_size

    train_set_x, train_set_y = shared_dataset((train_set[:,:-1],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:-1],val_set[:,-1]))

    print("  After adjusting, Train num: %s Vali num: %s Test num: %s" 
        %(datasets[0].shape[0], val_set.shape[0], test_set_x.shape[0]))

    # define train/valid/test models
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
             y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)

    valtrain_model = theano.function([index], classifier.errors(y),
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)  
            
    train_model = theano.function([index], [cost, classifier.show], updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)     

    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape(
        (x.shape[0],x.shape[1]*Words.shape[1]))
    if model in ["My"]:
        test_layer0_input = Model.Sig(test_layer0_input)
    test_y_pred, test_y_pred_given_x = classifier.predict(test_layer0_input)
    test_perf = T.mean(T.eq(test_y_pred, y))
    test_model_all = theano.function([x,y], [test_perf, test_y_pred, classifier.W1, 
        classifier.b1, classifier.W, classifier.b, Words, classifier.W2], 
        allow_input_downcast = True) 

    # start training 
    print( '... training')
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    new_W = []
    while (epoch < n_epochs):
        average_cost = []
        start_time = time.time()
        epoch = epoch + 1
        it = 0
        for minibatch_index in rng.permutation(range(n_train_batches)):
            cost_epoch = train_model(minibatch_index)
            #if it%1000 == 0:
                #print(cost_epoch[0])
                #print(" epoch %s, process... %s" %(epoch, 100.00*it/n_train_batches))
                # sys.stdout.flush()
            average_cost += [cost_epoch[0]]
            it += 1
        valtrain_perf = np.mean([valtrain_model(i) for i in xrange(n_train_batches)])
        val_perf = np.mean([val_model(i) for i in xrange(n_val_batches)])
        if epoch%2 == 0:         
        	print(('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%, average cost: %.4f' 
                % (epoch, time.time()-start_time, valtrain_perf*100., val_perf*100., 
                np.mean(average_cost))))
        if val_perf >= best_val_perf:
            test_perfs, test_y_preds, W1, b1, W, b, w, ls = test_model_all(test_set_x,
                test_set_y) 

            p = [W1, b1, W, b, w, ls]
            best_val_perf = val_perf
        sys.stdout.flush()
    return test_perfs, test_y_preds, test_set_y, p

def LogicPredict_svm(U, lr_decay, batch_size, n_epochs, datasets, n_in, n_out, model):
    Words = U

    print( '... W2v emebdding + SVM')
    n_val_batches = 0 
    train_set = datasets[0]
    test_set = datasets[1]
    datasets = [train_set, test_set]
    train_set, test_set = datasets
    train_set_x, train_set_y = train_set[:, :-1], train_set[:, -1]
    test_set_x, test_set_y = test_set[:, :-1], test_set[:, -1]
    print("  After adjusting, Train num: %s Test num: %s" %(datasets[0].shape[0], datasets[1].shape[0]))

    if model in ["ConW2v"]:
        X = np.concatenate([Words[train_set_x[:, 0]], Words[train_set_x[:, -1]]], axis=1)
        x = np.concatenate([Words[test_set_x[:, 0]], Words[test_set_x[:, -1]]], axis=1)
    elif model in ["DiffW2v"]:
        X = - Words[train_set_x[:, 0]] + Words[train_set_x[:, -1]]
        x = - Words[test_set_x[:, 0]] + Words[test_set_x[:, -1]]
    
    clf = svm.SVC()
    clf.fit(X, train_set_y)
    print(clf)
    test_y_pred = clf.predict(x)
    test_perf = np.mean(test_y_pred == test_set_y)
    p = [None, None, None, clf, Words, None]
    return test_perf, test_y_pred, test_set_y, p

def sgd(params, cost, lr):
    updates = OrderedDict({})
    for param in params:
        d_param = T.grad(cost, param)
        updates[param] = param - (1 - lr) * d_param
    return updates

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def generateDataset(relationLogic, entity2idx, target = None):
    if target == None:
        target = {"<": 0, ">":1, "=": 2, "|":3, "^":4, "v":5, "#":6}
    datasets = []
    for d in relationLogic:
        dataset = []
        for line in d:
            line = line.replace("\n", "").replace("\r", "")
            h, label, t = line.split("\t")
            X = [entity2idx[h], entity2idx[t], target[label]]
            dataset += [X]
        datasets += [np.asarray(dataset)]
    return datasets

def generateU(D, entitynum):
    return np.random.uniform(low=0, high=1, size=(D, entitynum)).T

def getU(fname, entity2idx, D):
    U = np.random.uniform(low=0, high=1, size=(D, len(entity2idx)))
    index = 0
    seen = []
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in entity2idx:
                U[:, entity2idx[word]] = np.fromstring(f.read(binary_len), dtype='float32')  
                seen += [word]
                index += 1
            else:
                f.read(binary_len)
    print("num words already in word2vec: %s, randomly generate %s" %(index, len(entity2idx)-index))
    U = U/np.sqrt(np.sum(U**2, axis = 0))
    return U.T, seen

def cross_valid5(data, split_num):
    split = {}
    num = len(data)
    interval = 1.0/split_num
    split_dic = {}
    start = 0
    for i in range(split_num):
        if i == (split_num - 1):
            split_dic[(start, start + interval + 0.001)] = i
        else:
            split_dic[(start, start + interval)] = i
        start += (interval)
    for i in range(num):
        random_num = np.random.random()
        for (s, e) in split_dic:
            if s <= random_num < e:
                split[i] = split_dic[(s, e)]
    return split

def split_data(data, split, e):
    idx = 0
    train = []
    test = []
    data = np.random.permutation(data)
    for line in data:
        if split[idx] == e:
            test += [line]
        else:
            train += [line]
        idx += 1
    return train, test

def split_data_below(data, split, e):
    idx = 0
    train = []
    test = []
    data = np.random.permutation(data)
    for line in data:
        if split[idx] == 5:
            test += [line]
        elif split[idx] <= e:
            train += [line]
        idx += 1
    return train, test

def load_text(file):
    output = []
    data = open(file, "r").readlines()
    for line in data:
        line = line.replace("\n", "").replace("\r", "")
        output += [line.split("\t")]
    return output

def exp(datapath, folder, mode, tp=None, pm = None):
    # load entity2idx files
    entity2idx = pickle.load(open(datapath + "entity2idx.pkl", "rb"))
    try:
        idx2entity, rel2idx, idx2rel = pickle.load(open(datapath + "idx2entity.pkl", "rb"))
        rel_num = len(rel2idx)
    except:
        rel2idx = {"<": 0, ">":1, "=": 2, "|":3}
        rel_num = 4
    
    if mode == 'train-test':
        # create new trained model path
        timestamp = (datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S'))
        timestamp = folder + '/' + timestamp
        # save print out into a file
        #sys.stdout = open(datapath + '%s.txt'% timestamp, "w")

        # set some parameters 
        entitynum = len(entity2idx) - len(rel2idx)
        print(entitynum)
        model = folder.replace('/', '')
        D = 50 # word2vec embedding dimensionality
        epoch = 1 # epoch number
        split_num = 3 # validation split number
        batch_size = 5 # batch_size
        lr = 0.9 # learning rate

        print('load data ...')
        Data = open(datapath + "wholedata.txt", "r").readlines()
        split = cross_valid5(Data, split_num)
        print('data loaded!')
        print(rel2idx)

        total_accuracy = []
        total_f1 = []
        print("Label... entitynum " + str(entitynum) + " dimension:" + str(D))
        print("model %s epoch %s split num %s" %(str(model), str(epoch), str(split_num)))

        sys.stdout.flush()
        for e in range(split_num):
            Train, Test = split_data(Data, split, e)
            datasets = generateDataset([Train, Test], entity2idx, target = rel2idx)
            print("\n >>>  Cross: " + str(e) + " Train num: " + str(len(datasets[0])) + " Test num: " + str(len(datasets[1])))
        
            if model in ["ConW2v", "DiffW2v"]:
                U, _ = getU('/home/yunshi/Word2vec/GoogleNews-vectors-negative300.bin', 
                    entity2idx, D)
                test_perf, predy, test_y_pred, p = LogicPredict_svm(U, lr, batch_size, epoch, 
                    datasets, 2*D, rel_num, model)
            else:
                U = generateU(D, entitynum)
                #U, _ = getU('/home/yunshi/Word2vec/GoogleNews-vectors-negative300.bin', entity2idx, D)
                test_perf, predy, test_y_pred, p = LogicPredict(U, lr, batch_size, epoch, 
                    datasets, 2*D, rel_num, model)
            print("so far .... accuracy: " + str(test_perf))

            # evaluation test based on labels
            average_f1 = []
            for i in range(rel_num):
                recall, precision = recallPrecision(predy, test_y_pred, i)
                print("label: %s recall: %s precision: %s" %(i, recall, precision))
                if recall!="nan":
                    alpha = 0; beta = 0
                    if recall == 0:
                        alpha = 0.00001
                    if precision ==0:
                        beta = 0.00001
                    average_f1 += [2.0/(1.0/(recall+alpha) + 1.0/(precision+beta))]
            average_f1 = np.mean(average_f1)
        
            total_accuracy += [test_perf]; total_f1 += [average_f1]
            print("so far ... f1 score: " + str(average_f1))
            # save models
            pickle.dump(p, open('saved-model/%s.pkl' %timestamp, "w"))
            sys.stdout.flush()
        sd_accuracy = np.std(total_accuracy); total_accuracy = np.mean(total_accuracy); total_f1 = np.mean(total_f1)

        print("\nfinal accuracy ### " + str(total_accuracy) + " final f1 ### " + str(total_f1) + " std: " + str(sd_accuracy))
    
    if mode == 'only-test':
        entitynum = 0
        # assert testpath and pre-trained model
        assert (tp is not None or pm is not None)

        # load data and pre-trained model
        Test = load_text("data/%s.txt" %tp)
        W1, b1, W, b, U, Label = pickle.load(open("saved-model/%s/%s.pkl" %(folder, pm), "r"))
        model = folder.replace("/", "")
        Test = check(entity2idx, Test, entitynum, target = rel2idx)
        check_set_x = Test[:, :-1]
        check_set_y = Test[:, -1]
        Words = theano.shared(value = U, name = "Words")
        Labels = theano.shared(value = Label, name = "Labels")
        index = T.lscalar()
        rng = np.random
        x = T.matrix('x')
        test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],x.shape[1]*Words.shape[1]))

        # start test 
        classifier = eval('Model.' + model + 'Regression')(rng, test_layer0_input, 600, 
            rel_num, Labels, W1, b1, W, b)
        test_y_pred, _ = classifier.predict(test_layer0_input)
        test_model_all = theano.function([x], [test_y_pred], allow_input_downcast = True)
        check_predy = test_model_all(check_set_x)[0]

        # print out results
        average_f1 = []
        recalls, precisions = [], []
        for i in range(rel_num):
            recall, precision = recallPrecision(check_predy,check_set_y , i)
            if recall is not 'nan' and precision is not 'nan': 
                f1 = 2.*(recall*precision) /(recall + precision)
            else:
                f1 = 0.
            if recall is not 'nan':
	            recalls += [recall]
	            precisions += [precision]
	            print("label: %s\t recall: %s\t precision: %s\t f1 score: %s" 
	                %(str(i), str(recall), str(precision), str(f1)))

        print("\nfinal f1 ### %s" %(np.mean(f1)))

def check(entity2idx, checkdata, entitynum, target = None):
    if target == None:
        target = {"<": 0, ">":1, "=": 2, "|":3, "^":4, "v":5, "#":6}
    else:
        print(target)
    checkdata_copy = []
    for line in checkdata:
        h, label, t = line
        X = [(entity2idx[h] - entitynum), (entity2idx[t] - entitynum)]
        X = X + [target[label]]
        checkdata_copy += [X]
    checkdata_copy = np.asarray(checkdata_copy)
    return checkdata_copy

def recallPrecision(predy, test_set_y, num):
    if np.sum(test_set_y == num)==0:
        recall = "nan"
    else:
        recall = 100. * np.sum(predy[test_set_y == num] == num)/(np.sum(test_set_y == num))
    if np.sum(predy == num) == 0:
        precision = 'nan'
    else:
        precision = 100. * np.sum(test_set_y[predy == num] == num)/(np.sum(predy == num))
    return recall, precision

def launch_main(datapath, modelpath, mode, tp = None, pm = None):
    exp(datapath, modelpath, mode = mode, tp = tp, pm = pm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datapath', help='path to data files')
    parser.add_argument('modelpath', help='name of model: My/ or ConW2v/')
    parser.add_argument('mode', help='whether train-test or just test: train-test ot only-test')
    parser.add_argument('-tp',dest = 'testpath', help='path to test file')
    parser.add_argument('-pm',dest = 'pretrainmodel', help='path to pre-trained model')
    args = parser.parse_args()

    launch_main(args.datapath, args.modelpath, args.mode, 
        tp = args.testpath, pm = args.pretrainmodel)