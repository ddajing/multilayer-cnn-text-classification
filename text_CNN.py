import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    CNN 2-chanels model for text classification.
    """
    def __init__(self, sentence_len, vocab_size, embedding_size, num_classes, 
             static_embedding_filter, filter_sizes, num_filters, l2_reg_lambda = 0.0):
        
        # Initialize placeholder
        self.inputs = tf.placeholder(tf.int32,  [None, sentence_len], name = "inputs")
        self.labels = tf.placeholder(tf.float32, [None, num_classes], name = "labels")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prop")
        
        # Embedding words in to vectors 1x300 with both static and nonsatic filters
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # Initialize nonstatic filters
            nonstatic_embedding_filter = tf.get_variable("nonstatic_filter", initializer = static_embedding_filter)
            self.nonstatic_embedding = tf.nn.embedding_lookup(nonstatic_embedding_filter, self.inputs)
            self.static_embedding = tf.nn.embedding_lookup(static_embedding_filter, self.inputs)
            self.embedded_sentences = tf.concat([self.nonstatic_embedding, self.static_embedding], -1)
            self.embedded_layer = tf.reshape(self.embedded_sentences, [-1, sentence_len, embedding_size, 2])

        # Convolutional and maxpooling layers        
        pooled_layer = []
        for filter_size in filter_sizes:
            filter_shape = [filter_size, embedding_size, 2, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "weights")
            b = tf.Variable(tf.constant(0.1, shape = [num_filters]), name = "bias")
            # Convolutional layer
            conv = tf.nn.conv2d(
                    self.embedded_layer,
                    W,
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "conv")
            
            # Add bias and apply rectifier linear unit function
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
            
            # Maxpooling layer
            pooled = tf.nn.max_pool(
                    h, 
                    ksize = [1, sentence_len - filter_size + 1, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "pooled")

            pooled_layer.append(pooled)
        
        total_num_filters = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_layer, 3, name = "h_pool")
        self.h_flat = tf.reshape(self.h_pool, [-1   , total_num_filters], name = "h_flat")
        
        # Add dropout
        self.h_drop = tf.nn.dropout(self.h_flat, self.dropout_keep_prob, name = "h_drop")
        
        l2_loss = 0.0
        # Outputs layer
        with tf.name_scope("outputs"):
            W = tf.get_variable("weights", shape = [total_num_filters, num_classes],
                                initializer = tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape = [num_classes]), name = "bias")
            self.scores = tf.nn.xw_plus_b(self.h_flat, W, b, name = "scores")
            self.predictions = tf.argmax(self.scores, 1)
            l2_loss += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        
        # Compute loss 
        losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.labels, logits = self.scores)
        self.loss = tf.add(tf.reduce_mean(losses), l2_loss * l2_reg_lambda, name = "loss")
        
        # Compute accuracy
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")