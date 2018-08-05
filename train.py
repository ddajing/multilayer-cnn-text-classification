import tensorflow as tf
import numpy as np
import os
import time
import pickle
import datetime
from absl import flags
from data_helpers import split_data
from text_CNN import TextCNN

# Parameter
# ========================================

for name in list(flags.FLAGS):
  delattr(flags.FLAGS, name)
  
# Data split params
tf.flags.DEFINE_float("devset_percentage", 0.1, "Devset split percentage")

# Model hyperparameters
tf.flags.DEFINE_string("filter_sizes", "3, 4, 5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter_sizes")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch size")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on devset after this many step")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many step")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_integer("patience_threshold", 10, "Patience parameter for early stopping")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def train(x, y, pretrained_embedding_filter):
    
    # Split data into developement set and training set
    x_train, y_train, x_dev, y_dev = split_data(x, y, FLAGS.devset_percentage)
    
    # Training
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
                allow_soft_placement = FLAGS.allow_soft_placement,
                log_device_placement = FLAGS.log_device_placement)
        sess = tf.Session(config = session_conf)
        with sess.as_default():
            cnn = TextCNN(
                    sentence_len = x_train.shape[1],
                    vocab_size = pretrained_embedding_filter.shape[0],
                    embedding_size = pretrained_embedding_filter.shape[1],
                    static_embedding_filter = pretrained_embedding_filter,
                    filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters = FLAGS.num_filters,
                    num_classes = y_train.shape[1],
                    l2_reg_lambda = FLAGS.l2_reg_lambda)
            # Define training precedure
            global_step = tf.Variable(tf.constant(0), name = "global_step", trainable = False)
            optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
            train_op = optimizer.minimize(cnn.loss, global_step = global_step)
            
            # Output directory for model
            timestamp = str(time.time())
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            
            #Checkpoint_directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep = FLAGS.num_checkpoints)
            
            # Initializer all variables
            sess.run(tf.global_variables_initializer())
            
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                        cnn.inputs: x_batch,
                        cnn.labels: y_batch,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
                _, step, loss, accuracy = sess.run(
                        [train_op, global_step, cnn.loss, cnn.accuracy],
                        feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            
            def dev_step(x_batch, y_batch):
                """
                A single developement step
                """
                feed_dict = {
                        cnn.inputs: x_batch,
                        cnn.labels: y_batch,
                        cnn.dropout_keep_prob: 1.0}
                step, loss, accuracy = sess.run(
                        [global_step, cnn.loss, cnn.accuracy],
                        feed_dict)
                
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                return loss, accuracy
            
            # Batch Gradient iter
            best_loss = 0.0
            final_accuracy = 0.0
            patience = 0
            should_stop = False
            num_batches = int((len(y_train) - 1) / FLAGS.batch_size + 1) 
            start_training_time = datetime.datetime.now().isoformat()
            for epoch in range(FLAGS.num_epochs):
                if should_stop:
                    break
                
                shuffled_indices = np.random.permutation(np.arange(len(y_train)))
                x_shuffled = x_train[shuffled_indices]
                y_shuffled = y_train[shuffled_indices]
                
                for batch in range(num_batches):
                    start_index = batch * FLAGS.batch_size
                    end_index = min(start_index + FLAGS.batch_size, len(y_train))
                    train_step(x_shuffled[start_index:end_index], y_shuffled[start_index:end_index])
                    current_step = tf.train.global_step(sess, global_step)
                    
                    if current_step % FLAGS.evaluate_every == 0:
                        print("Evaluation...")
                        loss_value, accuracy_value = dev_step(x_dev, y_dev)
                        print("")
                        
                        if current_step == FLAGS.evaluate_every or loss_value < best_loss:
                            patience -= patience
                            print(best_loss, loss_value)
                            best_loss = loss_value
                            final_accuracy = accuracy_value
                            path = saver.save(sess, checkpoint_prefix, global_step = current_step)
                            print("Saved model checkpoint to {}\n".format(path))
                        else:
                            patience += 1
                            if patience > FLAGS.patience_threshold:
                                should_stop = True
                                print ("Early stopping after {} step".format(current_step))
                                break

            print("Accuracy: {}, Loss: {}".format(final_accuracy, best_loss))            
            print("Training Completed!")
            end_training_time = datetime.datetime.now().isoformat()
            print("Started training: {}\nCompleted Training: {}".format(start_training_time, end_training_time))

if __name__ == "__main__":
    data = pickle.load(open("data.p", "rb"))
    x, y, pretrained_embedding_filter = data[0], data[1], data[2]
    train(x, y, pretrained_embedding_filter)