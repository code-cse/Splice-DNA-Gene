import tensorflow as tf
import numpy as np
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import CrossValDataPreparation as cvp
import os

class BaseModel:
    
    def __init__(self, n_units=90, n_layers=3, n_classes=3,
                n_seq=3, seq_len=20, word_size=64):
        '''
        n_units: number of hidden recurrent units in a single layer
        n_layers: number of layers in a single stack of the model
        n_classes: number of classifiation categories
        n_seq: number of shift sequences 0-shift, 1-shift and 2-shift, i.e., 3 in our case
        seq_len: length of the encoded sequences, in terms of states
        word_size: size of vocabulary
        '''
        self.n_units = n_units
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_seq = n_seq
        self.seq_len = seq_len
        self.word_size = word_size
        
    def get_a_cell(self, cell_size, keep_prob=1):
        # tf.nn.rnn_cell.LSTMCell
        cell = tf.nn.rnn_cell.LSTMCell(cell_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return drop
    
    def rnn_base(self):
        tf.reset_default_graph()
        
        self.input_data = tf.placeholder(tf.float32, [None, self.n_seq, self.seq_len, self.word_size])
        self.target = tf.placeholder(tf.float32, [None, self.n_classes])
        
        with tf.name_scope('RNN_Base'):
            cell0 = tf.nn.rnn_cell.MultiRNNCell(
             [self.get_a_cell(self.n_units, 1) for _ in range(self.n_layers)]
             )
            
            cell1 = tf.nn.rnn_cell.MultiRNNCell(
             [self.get_a_cell(self.n_units, 1) for _ in range(self.n_layers)]
             )

            cell2 = tf.nn.rnn_cell.MultiRNNCell(
             [self.get_a_cell(self.n_units, 1) for _ in range(self.n_layers)]
             )
        with tf.variable_scope("RNNOutput", reuse = tf.AUTO_REUSE):
            outputs0, self.states0 = tf.nn.dynamic_rnn(cell0, self.input_data[:, 0, :, :], dtype=tf.float32)
            outputs1, self.states1 = tf.nn.dynamic_rnn(cell1, self.input_data[:, 1, :, :], dtype=tf.float32)
            outputs2, self.states2 = tf.nn.dynamic_rnn(cell2, self.input_data[:, 2, :, :], dtype=tf.float32)
        
        weights0 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_units, self.n_classes], mean =0, stddev=0.01))}
        biases0 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_classes], mean =0, stddev=0.01))}

        weights1 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_units, self.n_classes], mean =0, stddev=0.01))}
        biases1 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_classes], mean =0, stddev=0.01))}

        weights2 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_units, self.n_classes], mean =0, stddev=0.01))}
        biases2 = {"linear_layer":tf.Variable(tf.truncated_normal([self.n_classes], mean =0, stddev=0.01))}
        
        self.final_output0 = tf.matmul(outputs0[:,-1,:], weights0["linear_layer"]) + biases0["linear_layer"]
        self.final_output1 = tf.matmul(outputs1[:,-1,:], weights1["linear_layer"]) + biases1["linear_layer"]
        self.final_output2 = tf.matmul(outputs2[:,-1,:], weights2["linear_layer"]) + biases2["linear_layer"]
        
    def model_optimizer_define(self, lrate=0.001):
        '''
        lrate: learning rate
        '''
        self.lrate = lrate
        softmax0 = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.final_output0, labels = self.target)
        cross_entropy0 = tf.reduce_mean(softmax0)

        softmax1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.final_output1, labels = self.target)
        cross_entropy1 = tf.reduce_mean(softmax1)

        softmax2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.final_output2, labels = self.target)
        cross_entropy2 = tf.reduce_mean(softmax2)
        
        fo = self.final_output = tf.reduce_mean([self.final_output0, self.final_output1, self.final_output2],0)
        ce = self.cross_entropy = tf.reduce_mean([cross_entropy0, cross_entropy1, cross_entropy2],0)
        
        ts = self.train_step = tf.train.RMSPropOptimizer(self.lrate).minimize(self.cross_entropy)
        cp = self.correct_prediction = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.final_output,1))
        ac = self.accuracy = (tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32)))
        in_d = self.input_data
        tar = self.target
        return fo, ce, ts, cp, ac, in_d, tar
        

    def base_model():
        bs = BaseModel()
        bs.rnn_base()
        print("In BaseModel")
        final_output, cross_entropy, train_step, \
            correct_prediction, accuracy, input_data, target = bs.model_optimizer_define()
        return final_output, cross_entropy, train_step, correct_prediction, accuracy, input_data, target
