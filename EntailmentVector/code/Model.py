import numpy
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import math

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Sig(x):
    y = T.nnet.sigmoid(x)
    return y
def Sig_fast(lam, x):
    y = T.nnet.ultra_fast_sigmoid(lam*x)
    return y
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)
def Lr(x):
    y = T.maximum(0.0, x) + 0.01*T.minimum(0.0, x)
    return(y)
def dis(x, y):
    return T.sum(T.abs_(x-y), axis = 1)

class NNRegression(object):
    """NN layer to do logic deduction
    """

    def __init__(self, rng, input, n_in, n_out, W2 = None, W1=None, b1 = None, W = None,  b=None):
        """ Initialize the parameters 

        rng: random seed
        input: input matrix N*2n
        n_in: input dimension 2n
        n_out: output dimension k=7
        W2: useless parameter
        W1, b1: softmax parameter k*n, k
        W, b: nn layer, matrix, n*2n, n
    """
        f = n_in/2
        # initialize parameters
        if W is None:
            self.W = theano.shared(
                    value=rng.uniform(low=-0.05, high=0.05, size=(n_in, f)),
                    name='W')
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(
                    value=rng.uniform(low=-0.05, high=0.05, size=(f,)),
                    name='b')
        else:
            self.b = b

        if W1 is None:
            self.W1 = theano.shared(
                    value=rng.uniform(low=-0.05, high=0.05, size=(f, n_out)),
                    name='W1')
        else:
            self.W1 = W1

        if b1 is None:
            self.b1 = theano.shared(
                    value=rng.uniform(low=-0.05, high=0.05, size=(n_out,)),
                    name='b1')
        else:
            self.b1 = b1
        self.W2 = 0*self.W

        # nn layer computation Tanh
        self.tmp = T.dot(input, self.W) + self.b
        self.show = self.tmp
        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(Tanh(self.tmp), self.W1) + self.b1)

        # compute prediction as class whose probability is maximal in
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # update parameters of the model
        self.params = [self.W, self.b, self.W1, self.b1]

    def loss_function(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

    .. math::
        
    y: corresponds to a vector that gives for each example the
    correct label
    
    Note: we use the mean instead of the sum so that
    the learning rate is less dependent on the batch size
    """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]), self.p_y_given_x

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
    zero one loss over the size of the minibatch
    
    y: corresponds to a vector that gives for each example the
    correct label
    """

        # check if y has same dimension of y_pred
        return T.mean(T.eq(self.y_pred, y))

    def predict(self, new_data):
        tmp = T.dot(new_data, self.W) + self.b
        p_y_given_x = T.nnet.softmax(T.dot(Tanh(tmp), self.W1) + self.b1)
        y_pred = T.argmax(p_y_given_x, axis=1)
        return y_pred, p_y_given_x


class MyRegression(object):
    """Symbolic representation to do logic deduction
    """

    def __init__(self, rng, input, n_in, n_out, W2=None, W1=None, b1 = None, W = None,  b=None):
        """ Initialize the parameters 

        rng: random seed
        input: input matrix N*2n
        n_in: input dimension 2n
        n_out: output dimension k=7
        W2: useless parameters
        W, b: softmax parameter k*n, k
        W1, b1: useless parameters
    """
        self.d = 4 # number of tensors

        # initialize parameters

        if W is None:
            self.W = theano.shared(
                    value=rng.uniform(low=-0.05, high=0.05, size=(self.d, n_out)),
                    name='W')
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(
                        value=rng.uniform(low=-0.05, high=0.05, size=(n_out, )), 
                        name='b')
        else:
            self.b = b

        self.W1 = 0*self.W
        self.b1 = 0*self.W
        self.W2 = 0*self.W

        num = input.shape[1]/2
        lefto = input[:, :num]
        righto = input[:, num:]

        # convert N*n matrix to predicate calculas p = [c; 1-c]
        leftc = 1 - lefto
        rightc = 1 - righto
        oo = T.mean(lefto*righto, axis = 1)
        co = T.mean(leftc*righto, axis = 1)
        oc = T.mean(lefto*rightc, axis = 1)
        cc = T.mean(leftc*rightc, axis = 1)
        self.tmp2 = T.stack([oo, co, oc, cc], axis=1)

        # softmax layer
        self.p_y_given_x = T.nnet.softmax(T.dot(self.tmp2, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.show = self.tmp2
        # updaate parameters of the model
        self.params = [self.W, self.b]

    def loss_function(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

    .. math::
        
    y: corresponds to a vector that gives for each example the
    correct label
    
    Note: we use the mean instead of the sum so that
    the learning rate is less dependent on the batch size
    """
        loss = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return loss, self.p_y_given_x

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
    zero one loss over the size of the minibatch
    
    y: corresponds to a vector that gives for each example the
    correct label
    """
        return T.mean(T.eq(self.y_pred, y))

    def predict(self, new_data):
        num = new_data.shape[1]/2
        lefto = new_data[:, :num]
        righto = new_data[:, num:]

        leftc = 1 - lefto
        rightc = 1 - righto
        oo = T.mean(lefto*righto, axis = 1)
        co = T.mean(leftc*righto, axis = 1)
        oc = T.mean(lefto*rightc, axis = 1)
        cc = T.mean(leftc*rightc, axis = 1)
        tmp2 = T.stack([oo, co, oc, cc], axis=1)

        p_y_given_x = T.nnet.softmax(T.dot(tmp2, self.W) + self.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        return y_pred, p_y_given_x
