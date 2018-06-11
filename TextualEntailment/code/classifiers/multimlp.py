# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import json
import os
import logging
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

import utils

tf.set_random_seed(123)

def variable_summaries(var, name):
    """
    Create tensorflow variable summaries
    """
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def attention_softmax3d(values):
    """
    Performs a softmax over the attention values.

    :param values: 3d tensor with raw values
    :return: 3d tensor, same shape as input
    """
    original_shape = tf.shape(values)
    num_units = original_shape[2]
    reshaped_values = tf.reshape(values, tf.stack([-1, num_units]))
    softmaxes = tf.nn.softmax(reshaped_values)
    return tf.reshape(softmaxes, original_shape)


def clip_sentence(sentence, sizes):
    """
    Clip the input sentence placeholders to the length of the longest one in the
    batch. This saves processing time.

    :param sentence: tensor with shape (batch, time_steps)
    :param sizes: tensor with shape (batch)
    :return: tensor with shape (batch, time_steps)
    """
    max_batch_size = tf.reduce_max(sizes)
    clipped_sent = tf.slice(sentence, [0, 0],
                            tf.stack([-1, max_batch_size]))
    return clipped_sent


def mask_values_after_sentence_end(values, sentence_sizes, value):
    """
    Given a batch of matrices, each with shape m x n, mask the values in each row
    after the positions indicated in sentence_sizes.

    :param values: tensor with shape (batch_size, m, n)
    :param sentence_sizes: tensor with shape (batch_size) containing the
        sentence sizes that should be limited
    :param value: scalar value to assign to items after sentence size
    :return: a tensor with the same shape
    """
    time_steps = tf.shape(values)[2]

    ones = tf.ones_like(values, dtype=tf.int32)
    mask = value * tf.cast(ones, tf.float32)

    # This piece of code is pretty ugly. We create a tensor with the same shape
    # as the values with each index from 0 to max_size and compare it against
    # another tensor with the same shape which holds the length of each batch.
    # We use tf.select, and then set values past sentence size to -inf.
    # If/when tensorflow had better indexing capabilities, we could simplify it.
    range_ = tf.range(time_steps)
    positions = ones * tf.reshape(range_, [1, 1, -1])
    sizes = ones * tf.reshape(sentence_sizes, [-1, 1, 1])
    cond = tf.less(positions, sizes)

    return tf.select(cond, values, mask)


def get_weights_and_biases():
    """
    Return all weight and bias variables
    :return:
    """
    var_list = []
    for var in tf.global_variables():
        if 'embeddings' not in var.name:
            var_list += [var]
            #print(var.name)
    return var_list


class MultiFeedForwardClassifier(object):
    """
    Implementation of the multi feed forward network model described in
    the paper "A Decomposable Attention Model for Natural Language
    Inference" by Parikh et al., 2016.

    It applies feedforward MLPs to combinations of parts of the two sentences,
    without any recurrent structure.
    """
    def __init__(self, num_units, num_classes,
                 vocab_size, embedding_size, max_len, mode, use_intra_attention=False,
                 training=True, learning_rate=0.001, clip_value=None,
                 l2_constant=0.0, project_input=False, distance_biases=10):

        self.num_units = num_units
        self.num_classes = num_classes
        self.use_intra = use_intra_attention
        self.project_input = project_input
        self.distance_biases = distance_biases
        self.mode = mode
        self.max_len = max_len

        # we have to supply the vocab size to allow validate_shape on the
        # embeddings variable, which is necessary down in the graph to determine
        # the shape of inputs at graph construction time
        self.embeddings_ph = tf.placeholder(tf.float32, (vocab_size, embedding_size),
                                            'embeddings')
        # sentence plaholders have shape (batch, time_steps)
        self.sentence1 = tf.placeholder(tf.int32, (None, None), 'sentence1')
        self.sentence2 = tf.placeholder(tf.int32, (None, None), 'sentence2')
        self.sentence1_size = tf.placeholder(tf.int32, [None], 'sent1_size')
        self.sentence2_size = tf.placeholder(tf.int32, [None], 'sent2_size')
        self.label = tf.placeholder(tf.int32, [None], 'label')
        self.l2_constant = l2_constant
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        self.dropout_keep = tf.placeholder(tf.float32, None, 'dropout')
        self.embedding_size = embedding_size
        self._extra_init()
        self.w2v_size = 300
        self.extra_size = self.embedding_size - self.w2v_size

        # we initialize the embeddings from a placeholder to circumvent
        # tensorflow's limitation of 2 GB nodes in the graph
        self.embeddings = tf.Variable(self.embeddings_ph, trainable=False,
                                      validate_shape=True)

        # clip the sentences to the length of the longest one in the batch
        # this saves processing time
        clipped_sent1 = clip_sentence(self.sentence1, self.sentence1_size)
        clipped_sent2 = clip_sentence(self.sentence2, self.sentence2_size)
        embedded1 = tf.nn.embedding_lookup(self.embeddings, clipped_sent1)
        embedded2 = tf.nn.embedding_lookup(self.embeddings, clipped_sent2)

        if project_input:
            projected1 = self.project_embeddings(embedded1)
            projected2 = self.project_embeddings(embedded2, True)
            self.representation_size = self.num_units
        else:
            projected1 = embedded1
            projected2 = embedded2
            self.representation_size = self.embedding_size

        if use_intra_attention:
            # here, repr's have shape (batch , time_steps, 2*num_units)
            repr1 = self.compute_intra_attention(projected1)
            repr2 = self.compute_intra_attention(projected2, True)
            self.representation_size *= 2
        else:
            # in this case, repr's have shape (batch, time_steps, num_units)
            repr1 = projected1
            repr2 = projected2

        # the architecture has 3 main steps: soft align, compare and aggregate
        # alpha and beta have shape (batch, time_steps, embeddings)
        self.alpha, self.beta = self.attend(repr1, repr2)

        if self.mode == 'w2vw2v':
            self.v1 = self.compare(repr1, self.beta, self.sentence1_size)
            self.v2 = self.compare(repr2, self.alpha, self.sentence2_size, True)
        elif self.mode == 'w2vw2vour':
            self.v1 = self.compare(repr1[:, :, :self.w2v_size], self.beta[:, :, :self.w2v_size], self.sentence1_size)
            self.v2 = self.compare(repr2[:, :, :self.w2v_size], self.alpha[:, :, :self.w2v_size], self.sentence2_size, True)
            self.v1 = tf.concat([self.v1, self.our(repr1[:, :, self.w2v_size:], self.beta[:, :, self.w2v_size:], self.sentence1_size)], 2)
            self.v2 = tf.concat([self.v2, self.our(repr2[:, :, self.w2v_size:], self.alpha[:, :, self.w2v_size:], self.sentence2_size, True)], 2)
        elif self.mode == 'w2vw2vnn':
            self.v1 = self.compare(repr1[:, :, :self.w2v_size], self.beta[:, :, :self.w2v_size], self.sentence1_size)
            self.v2 = self.compare(repr2[:, :, :self.w2v_size], self.alpha[:, :, :self.w2v_size], self.sentence2_size, True)
            self.v1 = tf.concat([self.v1, self.comparenn(repr1[:, :, self.w2v_size:], self.beta[:, :, self.w2v_size:], self.sentence1_size)], 2)
            self.v2 = tf.concat([self.v2, self.comparenn(repr2[:, :, self.w2v_size:], self.alpha[:, :, self.w2v_size:], self.sentence2_size, True)], 2)
        
        self.display = [self.inter_att1]
        self.logits = self.aggregate(self.v1, self.v2, self.sentence1_size, self.sentence2_size)
        self.answer = tf.argmax(self.logits, 1, 'answer')

        if training:
            self._create_training_tensors()
            self.merged_summaries = tf.summary.merge_all()


    def _extra_init(self):
        """
        Entry point for subclasses initialize more stuff
        """
        pass

    def project_embeddings(self, embeddings, reuse_weights=False):
        """
        Project word embeddings into another dimensionality

        :param embeddings: embedded sentence, shape (batch, time_steps, embedding_size)
        :param reuse_weights: reuse weights in internal layers
        :return: projected embeddings with shape (batch, time_steps, num_units)
        """
        time_steps = tf.shape(embeddings)[1]
        embeddings_2d = tf.reshape(embeddings, [-1, self.embedding_size])

        with tf.variable_scope('projection', reuse=reuse_weights):
            initializer = tf.random_normal_initializer(0.0, 0.1)
            weights = tf.get_variable('weights', [self.embedding_size, self.num_units],
                                      initializer=initializer)
            if not reuse_weights:
                variable_summaries(weights, 'projection/weights')

            projected = tf.matmul(embeddings_2d, weights)

        projected_3d = tf.reshape(projected, tf.stack([-1, time_steps, self.num_units]))
        return projected_3d

    def _relu_layer(self, inputs, weights, bias):
        """
        Apply dropout to the inputs, followed by the weights and bias,
        and finally the relu activation
        :param inputs: 2d tensor
        :param weights: 2d tensor
        :param bias: 1d tensor
        :return: 2d tensor
        """
        after_dropout = tf.nn.dropout(inputs, self.dropout_keep)
        raw_values = tf.nn.xw_plus_b(after_dropout, weights, bias)
        return tf.nn.relu(raw_values)

    def _tanh_layer(self, inputs, weights, bias):
        """
        Apply dropout to the inputs, followed by the weights and bias,
        and finally the relu activation
        :param inputs: 2d tensor
        :param weights: 2d tensor
        :param bias: 1d tensor
        :return: 2d tensor
        """
        after_dropout = tf.nn.dropout(inputs, self.dropout_keep)
        raw_values = tf.nn.xw_plus_b(after_dropout, weights, bias)
        return tf.nn.tanh(raw_values)

    def _apply_feedforward(self, inputs, num_input_units, scope,
                           reuse_weights=False):
        """
        Apply two feed forward layers with self.num_units on the inputs.
        :param inputs: tensor in shape (batch, time_steps, num_input_units)
            or (batch, num_units)
        :param num_input_units: a python int
        :return: a tensor with shape (batch, time_steps, num_units)
        """
        rank = len(inputs.get_shape())
        if rank == 3:
            time_steps = tf.shape(inputs)[1]

            # combine batch and time steps in the first dimension
            inputs2d = tf.reshape(inputs, tf.stack([-1, num_input_units]))
        else:
            inputs2d = inputs

        scope = scope or 'feedforward'
        with tf.variable_scope(scope, reuse=reuse_weights):
            initializer = tf.random_normal_initializer(0.0, 0.1)

            with tf.variable_scope('layer1'):
                shape = [num_input_units, self.num_units]
                weights1 = tf.get_variable('weights', shape, initializer=initializer)
                zero_init = tf.zeros_initializer()
                bias1 = tf.get_variable('bias', shape =[self.num_units], dtype=tf.float32,
                                        initializer=zero_init)

            with tf.variable_scope('layer2'):
                shape = [self.num_units, self.num_units]
                weights2 = tf.get_variable('weights', shape, initializer=initializer)
                bias2 = tf.get_variable('bias', shape =[self.num_units], dtype=tf.float32,
                                        initializer=zero_init)

            num_output_units = self.num_units
            # relus are (time_steps * batch, num_units)
            relus2 = self._relu_layer(inputs2d, weights1, bias1)
            #relus2 = self._relu_layer(relus1, weights2, bias2)

        if rank == 3:
            output_shape = tf.stack([-1, time_steps, num_output_units])
            return tf.reshape(relus2, output_shape)

        return relus2

    def _get_distance_biases(self, time_steps, reuse_weights=False):
        """
        Return a 2-d tensor with the values of the distance biases to be applied
        on the intra-attention matrix of size sentence_size
        :param time_steps: tensor scalar
        :return: 2-d tensor (time_steps, time_steps)
        """
        with tf.variable_scope('distance-bias', reuse=reuse_weights):
            # this is d_{i-j}
            distance_bias = tf.get_variable('dist_bias', [self.distance_biases],
                                            initializer=tf.zeros_initializer())

            # messy tensor manipulation for indexing the biases
            r = tf.range(0, time_steps)
            r_matrix = tf.tile(tf.reshape(r, [1, -1]), tf.stack([time_steps, 1]))
            raw_inds = r_matrix - tf.reshape(r, [-1, 1])
            clipped_inds = tf.clip_by_value(raw_inds, 0, self.distance_biases - 1)
            values = tf.nn.embedding_lookup(distance_bias, clipped_inds)

        return values

    def compute_intra_attention(self, sentence, reuse_weights=False):
        """
        Compute the intra attention of a sentence. It returns a concatenation
        of the original sentence with its attended output.

        :param sentence: tensor in shape (batch, time_steps, num_units)
        :return: a tensor in shape (batch, time_steps, 2*num_units)
        """
        time_steps = tf.shape(sentence)[1]
        with tf.variable_scope('intra-attention') as scope:
            # this is F_intra in the paper
            # f_intra1 is (batch, time_steps, num_units) and
            # f_intra1_t is (batch, num_units, time_steps)
            f_intra = self._apply_feedforward(sentence, self.num_units,
                                              scope,
                                              reuse_weights=reuse_weights)
            f_intra_t = tf.transpose(f_intra, [0, 2, 1])

            # these are f_ij
            # raw_attentions is (batch, time_steps, time_steps)
            raw_attentions = tf.batch_matmul(f_intra, f_intra_t)

            # bias has shape (time_steps, time_steps)
            bias = self._get_distance_biases(time_steps, reuse_weights=reuse_weights)

            # bias is broadcast along batches
            raw_attentions += bias
            attentions = attention_softmax3d(raw_attentions)

            attended = tf.batch_matmul(attentions, sentence)

        return tf.concat([sentence, attended], 2)

    def attend(self, sent1, sent2):
        """
        Compute inter-sentence attention. This is step 1 (attend) in the paper

        :param sent1: tensor in shape (batch, time_steps, num_units),
            the projected sentence 1
        :param sent2: tensor in shape (batch, time_steps, num_units)
        :return: a tuple of 3-d tensors, alfa and beta.
        """
        with tf.variable_scope('inter-attention') as self.attend_scope:
            # this is F in the paper
            num_units = self.w2v_size

            # repr1 has shape (batch, time_steps, num_units)
            # repr2 has shape (batch, num_units, time_steps)
            repr1 = self._transformation_attend(sent1[:, :, :self.w2v_size], num_units,
                                                self.sentence1_size)
            repr2 = self._transformation_attend(sent2[:, :, :self.w2v_size], num_units,
                                                self.sentence2_size, True)
            repr2 = tf.transpose(repr2, [0, 2, 1])

            # compute the unnormalized attention for all word pairs
            # raw_attentions has shape (batch, time_steps1, time_steps2)
            raw_attentions = tf.matmul(repr1, repr2)

            # now get the attention softmaxes
            att_sent1 = attention_softmax3d(raw_attentions)

            att_transposed = tf.transpose(raw_attentions, [0, 2, 1])
            att_sent2 = attention_softmax3d(att_transposed)

            self.inter_att1 = att_sent1
            self.inter_att2 = att_sent2
            alpha = tf.matmul(att_sent2, sent1, name='alpha')
            beta = tf.matmul(att_sent1, sent2, name='beta')

        return alpha, beta

    def _transformation_attend(self, sentence, num_units, length,
                               reuse_weights=False):
        """
        Apply the transformation on each sentence before attending over each
        other. In the original model, it is a two layer feed forward network.
        :param sentence: a tensor with shape (batch, time_steps, num_units)
        :param num_units: a python int indicating the third dimension of sentence
        :param length: real length of the sentence. Not used in this class.
        :param reuse_weights: whether to reuse weights inside this scope
        :return: a tensor with shape (batch, time_steps, num_units)
        """
        return self._apply_feedforward(sentence, num_units, self.attend_scope,
                                       reuse_weights)

    def compare(self, sentence, soft_alignment, sentence_length,
                reuse_weights=False):
        """
        Apply a feed forward network to compare one sentence to its
        soft alignment with the other.

        :param sentence: embedded and projected sentence,
            shape (batch, time_steps, num_units)
        :param soft_alignment: tensor with shape (batch, time_steps, num_units)
        :param reuse_weights: whether to reuse weights in the internal layers
        :return: a tensor (batch, time_steps, num_units)
        """
        with tf.variable_scope('comparison', reuse=reuse_weights) \
                as self.compare_scope:

            num_units = 2 * self.w2v_size

            # sent_and_alignment has shape (batch, time_steps, num_units)
            sent_and_alignment = tf.concat([sentence, soft_alignment], 2)

            output = self._transformation_compare(sent_and_alignment, num_units,
                                                  sentence_length, reuse_weights)

        return output

    def comparenn(self, sentence, soft_alignment, sentence_length,
                reuse_weights=False):
        """
        Apply a feed forward network to compare one sentence to its
        soft alignment with the other.

        :param sentence: embedded and projected sentence,
            shape (batch, time_steps, num_units)
        :param soft_alignment: tensor with shape (batch, time_steps, num_units)
        :param reuse_weights: whether to reuse weights in the internal layers
        :return: a tensor (batch, time_steps, num_units)
        """
        with tf.variable_scope('nn', reuse=reuse_weights) \
                as self.compare_scope:

            num_units = 2 * self.extra_size

            # sent_and_alignment has shape (batch, time_steps, num_units)
            sent_and_alignment = tf.concat([sentence, soft_alignment], 2)

            output = self._transformation_compare(sent_and_alignment, num_units,
                                                  sentence_length, reuse_weights)

        return output

    def our(self, sentence1, sentence2, sentence_length,
                reuse_weights=False):
        """
        Apply a feed forward network to compare one sentence to its
        soft alignment with the other.

        :param sentence: embedded and projected sentence,
            shape (batch, time_steps, num_units)
        :param soft_alignment: tensor with shape (batch, time_steps, num_units)
        :param reuse_weights: whether to reuse weights in the internal layers
        :return: a tensor (batch, time_steps, num_units)
        """
        with tf.variable_scope('our', reuse=reuse_weights) \
                as self.compare_scope:

            num_units = self.extra_size
            time_steps = tf.shape(sentence1)[1]
            # sent_and_alignment has shape (batch, time_steps, num_units)
            sentence1 = tf.reshape(sentence1, tf.stack([-1, num_units]))
            sentence2 = tf.reshape(sentence2, tf.stack([-1, num_units]))

            # sent_and_alignment has shape (batch, time_steps, num_units)
            xx = tf.multiply(sentence1, sentence2)
            #self.idx = tf.reshape(tf.reduce_mean(xx, 1)/(tf.reduce_max(tf.reduce_mean(xx, 1))), [-1, 1])
            idx = tf.expand_dims(tf.reduce_sum(tf.concat([sentence1, sentence2], -1), 1), 1)
            self.idx = tf.where(tf.greater(idx, 10), tf.ones_like(idx), tf.zeros_like(idx))
            xxw = tf.multiply(sentence1, 1 - sentence2) * self.idx
            xwx = tf.multiply(1 - sentence1, sentence2) * self.idx
            xwxw = tf.multiply( 1- sentence1, 1 - sentence2) * self.idx

            sent_and_alignment = tf.concat([xx, xwx, xxw, xwxw], axis = 1)
            output_shape = tf.stack([-1, time_steps, 4*num_units])
            self.sent_and_alignment = tf.reshape(sent_and_alignment, output_shape)

            output = self._transformation_compare(self.sent_and_alignment, 4*num_units,
                                                  sentence_length, reuse_weights)

        return output

    def aggregate(self, v1, v2, v1_length, v2_length):
        """
        Aggregate the representations induced from both sentences and their
        representations
        :param v1: tensor with shape (batch, time_steps, num_units)
        :param v2: tensor with shape (batch, time_steps, num_units)
        :return: logits over classes, shape (batch, num_classes)
        """
        # sum over time steps; resulting shape is (batch, num_units)
        # v1_sum = tf.reduce_sum(v1, [1])
        # v2_sum = tf.reduce_sum(v2, [1])
        num_units = self._num_units_on_aggregate()
        v1 = self.RNN(v1, num_units/2, v1_length)
        v2 = self.RNN(v2, num_units/2, v2_length, reuse_weights = True)
        v1_sum = tf.reduce_max(v1, 1)
        v2_sum = tf.reduce_max(v2, 1)
        concat_v = tf.concat([v1_sum,  v2_sum], 1)
        #concat_v = v1_sum - v2_sum

        with tf.variable_scope('aggregation') as self.aggregate_scope:
            initializer = tf.random_normal_initializer(0.0, 0.1)
            with tf.variable_scope('linear'):
                shape = [self.num_units, self.num_classes]
                weights_linear = tf.get_variable('weights', shape,
                                                 initializer=initializer)
                bias_linear = tf.get_variable('bias', [self.num_classes],
                            initializer=tf.zeros_initializer())

            pre_logits = self._apply_feedforward(concat_v, self.num_units,
                                                 self.aggregate_scope)
            logits = tf.nn.xw_plus_b(pre_logits, weights_linear, bias_linear)

        return logits

    def RNN(self, x, num_units, x_length, reuse_weights = False):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        with tf.variable_scope('LSTM', reuse=reuse_weights) as lstm_scope:
            x.set_shape([None, None, num_units])
            #x = tf.transpose(x, [1, 0, 2])
            #x = tf.unpack(tf.transpose(x, [1, 0, 2]), self.max_len+1)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_units/2), 
                                            output_keep_prob = self.dropout_keep)

            outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, sequence_length = x_length, 
                                                dtype=tf.float32)

        return outputs

    def _num_units_on_aggregate(self):
        """
        Return the number of units used by the network when computing
        the aggregated representation of the two sentences.
        """ 
        if self.mode in ['w2vw2vour', 'w2vw2vnn']:
            return 4 * self.num_units
        if self.mode in ['w2vour', 'w2vnn', 'w2vw2v']:
            return 2 * self.num_units

    def _transformation_compare(self, sentence, num_units, length,
                                reuse_weights=False):
        """
        Apply the transformation on each attended token before comparing.
        In the original model, it is a two layer feed forward network.

        :param sentence: a tensor with shape (batch, time_steps, num_units)
        :param num_units: a python int indicating the third dimension of sentence
        :param length: real length of the sentence. Not used in this class.
        :param reuse_weights: whether to reuse weights inside this scope
        :return: a tensor with shape (batch, time_steps, num_units)
        """
        return self._apply_feedforward(sentence, num_units, self.compare_scope,
                                       reuse_weights)

    def _create_training_tensors(self):
        """
        Create the tensors used for training
        """
        hits = tf.equal(tf.cast(self.answer, tf.int32), self.label)
        self.accuracy = tf.reduce_mean(tf.cast(hits, tf.float32),
                                       name='accuracy')
        with tf.name_scope('training'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits,
                                                                           labels = self.label)
            labeled_loss = tf.reduce_mean(cross_entropy)
            weights = [v for v in tf.global_variables() if 'weight' in v.name]
            if self.l2_constant > 0:
                l2_partial_sum = sum([tf.nn.l2_loss(weight) for weight in weights])
                l2_loss = tf.mul(self.l2_constant, l2_partial_sum, 'l2_loss')
                self.loss = tf.add(labeled_loss, l2_loss, 'loss')
            else:
                self.loss = labeled_loss

            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            if self.clip_value is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
            self.train_op = optimizer.apply_gradients(zip(gradients, v))

    def initialize_embeddings(self, session, embeddings):
        """
        Initialize word embeddings
        :param session: tensorflow session
        :param embeddings: the contents of the word embeddings
        :return:
        """
        init_op = tf.initialize_variables([self.embeddings])
        session.run(init_op, {self.embeddings_ph: embeddings})

    def initialize(self, session, embeddings):
        """
        Initialize all tensorflow variables.
        :param session: tensorflow session
        :param embeddings: the contents of the word embeddings
        """
        init_op = tf.global_variables_initializer()
        session.run(init_op, {self.embeddings_ph: embeddings})

    def initialize_pretrain(self, session, dirname):
        tensorflow_file = os.path.join(dirname, 'model')
        saver = tf.train.Saver(get_weights_and_biases())
        # print(saver)
        saver.restore(session, tensorflow_file)
        print('successfully re-load pre-trained model ...')

    @classmethod
    def load(cls, dirname, session):
        """
        Load a previously saved file.
        :param dirname: directory with model files
        :param session: tensorflow session
        :return: an instance of MultiFeedForward
        """
        params = utils.load_parameters(dirname)

        model = cls(params['num_units'], params['num_classes'],
                    params['vocab_size'], params['embedding_size'], 
                    params['max_len'], params['mode'], training=False)

        tensorflow_file = os.path.join(dirname, 'model')
        saver = tf.train.Saver(get_weights_and_biases())
        saver.restore(session, tensorflow_file)

        return model

    def _get_params_to_save(self):
        """
        Return a dictionary with data for reconstructing a persisted object
        """
        vocab_size = self.embeddings.get_shape()[0].value
        data = {'num_units': self.num_units,
                'num_classes': self.num_classes,
                'vocab_size': vocab_size,
                'mode': self.mode,
                'max_len' : self.max_len,
                'embedding_size': self.embedding_size}

        return data

    def save(self, dirname, session, saver):
        """
        Persist a model's information
        """
        params = self._get_params_to_save()
        tensorflow_file = os.path.join(dirname, 'model')
        params_file = os.path.join(dirname, 'model-params.json')

        with open(params_file, 'wb') as f:
            json.dump(params, f)

        saver.save(session, tensorflow_file)

    def train(self, session, train_dataset, valid_dataset, test_dataset,
              num_epochs, batch_size, dropout_keep, save_dir,
              report_interval=100):
        """
        Train the model with the specified parameters
        :param session: tensorflow session
        :param train_dataset: an RTEDataset object with training data
        :param valid_dataset: an RTEDataset object with validation data
        :param num_epochs: number of epochs to run the model. During each epoch,
            all data points are seen exactly once
        :param batch_size: how many items in each minibatch.
        :param dropout_keep: dropout keep probability (applied at LSTM input and output)
        :param save_dir: path to directory to save the model
        :param report_interval: how many minibatches between each performance report
        :return:
        """
        logger = utils.get_logger('rte_network')
        handler = logging.FileHandler(save_dir + '/result.log')
        logger.addHandler(handler)

        # this tracks the accumulated loss in a minibatch (to take the average later)
        accumulated_loss = 0

        best_acc = 0

        # batch counter doesn't reset after each epoch
        batch_counter = 0

        # get all weights and biases, but not the embeddings
        # (embeddings are huge and saved separately)
        vars_to_save = get_weights_and_biases()

        saver = tf.train.Saver(vars_to_save, max_to_keep=1)
        # summ_writer = tf.train.SummaryWriter(log_dir, session.graph)
        # summ_writer.add_graph(session.graph)

        for i in range(num_epochs):
            train_dataset.shuffle_data()
            batch_index = 0

            while batch_index < train_dataset.num_items:
                batch_index2 = batch_index + batch_size

                feeds = {self.sentence1: train_dataset.sentences1[batch_index:batch_index2],
                         self.sentence2: train_dataset.sentences2[batch_index:batch_index2],
                         self.sentence1_size: train_dataset.sizes1[batch_index:batch_index2],
                         self.sentence2_size: train_dataset.sizes2[batch_index:batch_index2],
                         self.label: train_dataset.labels[batch_index:batch_index2],
                         self.dropout_keep: dropout_keep
                         }

                ops = [self.train_op, self.loss]
                _, loss = session.run(ops, feed_dict=feeds)
                accumulated_loss += loss

                batch_index = batch_index2
                batch_counter += 1
                if batch_counter % report_interval == 0:
                    # summ_writer.add_summary(summaries, batch_counter)
                    avg_loss = accumulated_loss / report_interval
                    accumulated_loss = 0

                    batch_index3 = 0
                    valid_losses = 0
                    idx = 0
                    acces = 0
                    while batch_index3 < valid_dataset.num_items:
                        batch_index4 = min([batch_index3 + batch_size, valid_dataset.num_items])

                        feeds = {self.sentence1: valid_dataset.sentences1[batch_index3:batch_index4],
                                 self.sentence2: valid_dataset.sentences2[batch_index3:batch_index4],
                                 self.sentence1_size: valid_dataset.sizes1[batch_index3:batch_index4],
                                 self.sentence2_size: valid_dataset.sizes2[batch_index3:batch_index4],
                                 self.label: valid_dataset.labels[batch_index3:batch_index4],
                                 self.dropout_keep: 1.0
                                 }

                        valid_loss, acc = session.run([self.loss, self.accuracy],
                                                      feed_dict=feeds)
                        valid_losses += valid_loss
                        acces += acc*(batch_index4 - batch_index3)
                        idx += (batch_index4 - batch_index3)
                        batch_index3 = batch_index4

                    valid_loss = valid_losses/idx
                    acc = acces/idx

                    msg = '%d epochs, %d batches' % (i, batch_counter)
                    msg += '\tTraining loss: %f' % avg_loss
                    msg += '\tValid loss: %f' % valid_loss
                    msg += '\tValid accuracy: %f' % acc

                    if acc > 0.72:#best_acc:
                        batch_index3 = 0
                        test_losses = 0
                        idx = 0
                        acces = 0
                        answers = np.zeros((test_dataset.num_items))
                        while batch_index3 < test_dataset.num_items:
                            batch_index4 = min([batch_index3 + 5000, test_dataset.num_items])

                            feeds = {self.sentence1: test_dataset.sentences1[batch_index3:batch_index4],
                                     self.sentence2: test_dataset.sentences2[batch_index3:batch_index4],
                                     self.sentence1_size: test_dataset.sizes1[batch_index3:batch_index4],
                                     self.sentence2_size: test_dataset.sizes2[batch_index3:batch_index4],
                                     self.label: test_dataset.labels[batch_index3:batch_index4],
                                     self.dropout_keep: 1.0
                                     }

                            _, acc_test, answer, display = session.run([self.loss, self.accuracy, self.answer, self.display],
                                                      feed_dict=feeds)
                            acces += acc_test*(batch_index4 - batch_index3)
                            idx += (batch_index4 - batch_index3)
                            answers[batch_index3: batch_index4] = answer
                            batch_index3 = batch_index4

                        acc_test = acces/idx
                        msg += '\tTest accuracy: %f' %acc_test
                        if acc_test and acc and acc_test > best_acc:
                            self.save(save_dir, session, saver)
                            self.print_acc(answers, test_dataset.labels, save_dir, 
                                display)
                            best_acc = acc_test
                            msg += '\t(saved model)'

                    logger.info(msg)

    def test(self, session, valid_dataset, test_dataset, batch_size, save_dir):

        saver = tf.train.Saver(get_weights_and_biases(), max_to_keep=1)
        logger = utils.get_logger('rte_network')
        handler = logging.FileHandler(save_dir + '/result.log')
        logger.addHandler(handler)

        best_acc = 0
        batch_index3 = 0
        valid_losses = 0
        idx = 0
        acces = 0
        while batch_index3 < valid_dataset.num_items:
            batch_index4 = min([batch_index3 + batch_size, valid_dataset.num_items])

            feeds = {self.sentence1: valid_dataset.sentences1[batch_index3:batch_index4],
                     self.sentence2: valid_dataset.sentences2[batch_index3:batch_index4],
                     self.sentence1_size: valid_dataset.sizes1[batch_index3:batch_index4],
                     self.sentence2_size: valid_dataset.sizes2[batch_index3:batch_index4],
                     self.label: valid_dataset.labels[batch_index3:batch_index4],
                     self.dropout_keep: 1.0
                     }

            valid_loss, acc = session.run([self.loss, self.accuracy],
                                          feed_dict=feeds)
            valid_losses += valid_loss
            acces += acc*(batch_index4 - batch_index3)
            idx += (batch_index4 - batch_index3)
            batch_index3 = batch_index4

        valid_loss = valid_losses/idx
        acc = acces/idx

        msg = 'Valid loss: %f' % valid_loss
        msg += '\tValid accuracy: %f' % acc

        if acc > 0.72:#best_acc:
            batch_index3 = 0
            test_losses = 0
            idx = 0
            acces = 0
            answers = np.zeros((test_dataset.num_items))
            while batch_index3 < test_dataset.num_items:
                batch_index4 = min([batch_index3 + 5000, test_dataset.num_items])

                feeds = {self.sentence1: test_dataset.sentences1[batch_index3:batch_index4],
                         self.sentence2: test_dataset.sentences2[batch_index3:batch_index4],
                         self.sentence1_size: test_dataset.sizes1[batch_index3:batch_index4],
                         self.sentence2_size: test_dataset.sizes2[batch_index3:batch_index4],
                         self.label: test_dataset.labels[batch_index3:batch_index4],
                         self.dropout_keep: 1.0
                         }

                _, acc_test, answer, display = session.run([self.loss, self.accuracy, self.answer, self.display],
                                          feed_dict=feeds)
                acces += acc_test*(batch_index4 - batch_index3)
                idx += (batch_index4 - batch_index3)
                answers[batch_index3: batch_index4] = answer
                batch_index3 = batch_index4
            self.print_acc(answers, test_dataset.labels, save_dir, display)

            acc_test = acces/idx
            msg += '\tTest accuracy: %f' %acc_test
            if acc_test > best_acc:
                best_acc = acc_test
                msg += '\t(saved model)'

        logger.info(msg)

    def print_acc(self, y_hat, y, save_dir, display):
        f = open(save_dir + '/predict.txt', "w")
        for idx in range(len(y)):
            f.write(str(idx+1) + "\t" + str(y[idx]) + "\t" + str(y_hat[idx]) + "\n")
        # for i in range(display[0][9, :, :].shape[0]):
        #     f.write(str(display[0][9, i, :]) + "\n")
        f.close()