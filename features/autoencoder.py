# Authors: LI AO <aoli@hit.edu.cn>
#
# License: 

from .. import logger

import math
import numpy as np
import tensorflow as tf
from sklearn.utils.validation import check_array

class AutoEncoder(object):
    """Three-Level Auto Encoder for feature extraction.


    Warning: This class should be used based on Tensorflow.
    """

    def __init__(self, 
                 n_hidden_1=256, 
                 n_hidden_2=128, 
                 n_hidden_3=64,
                 training_epochs=20, 
                 learning_rate=0.01, 
                 batch_size=256, 
                 display_step=1,
                 use_gpu=True,
                 gpu_id=0):
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_hidden_3 = n_hidden_3
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.display_step = display_step
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.fitted = False


    def __variables(self):
        self.__X = tf.placeholder(tf.float32, [None, self.n_input])
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'encoder_h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
            'decoder_h3': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input]))
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([self.n_hidden_3])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b3': tf.Variable(tf.random_normal([self.n_input]))
        }

    
    def __next_batch(self, X):
        total_batch = int(math.ceil(float(X.shape[0])/self.batch_size))
        for i in range(total_batch):
            start = i * self.batch_size
            end = min((i+1)*self.batch_size, X.shape[0])
            yield X[start:end]


    def __network(self):
        # network
        encoder_op = self.__encoder(self.__X)
        decoder_op = self.__decoder(encoder_op)
        # loss function
        cost = tf.losses.mean_squared_error(self.__X, decoder_op)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        
        return encoder_op, cost, optimizer


    def __encoder(self, x):
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                      self.biases['encoder_b1']))
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, self.weights['encoder_h2']),
                                      self.biases['encoder_b2']))
        layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, self.weights['encoder_h3']),
                                      self.biases['encoder_b3']))
        return layer3
    
    
    def __decoder(self, x):
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                      self.biases['decoder_b1']))
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, self.weights['decoder_h2']),
                                      self.biases['decoder_b2']))
        layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, self.weights['decoder_h3']),
                                      self.biases['decoder_b3']))
        return layer3


    def fit(self, X):
        """Build an autoencoder from the fit data X.

        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        
        Returns
        -------
        self : object
            Returns the instance itself.
        """  
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)
        if np.isnan(np.min(X)): raise ValueError("It contains NaN.")

        self.n_samples, self.n_features = X.shape
        self.n_input = self.n_features     

        self.__variables()
        # construct network
        _, cost_op, optimizer_op = self.__network()

        init = tf.initialize_all_variables()
        # create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.__sess = tf.Session(config=config)
        if self.use_gpu: self.device = '/gpu:' + str(self.gpu_id)
        else: self.device = '/cpu'
        logger.info('Used device: ' + self.device)
        with tf.Graph().as_default():
            with tf.device(self.device):
                self.__sess.run(init)
                for epoch in range(self.training_epochs):
                    for batch in self.__next_batch(X):
                        cost, _ = self.__sess.run([cost_op, optimizer_op], \
                                           feed_dict={self.__X: batch})
                    if epoch % self.display_step == 0:
                        logger.info("Epoch: %04d, cost = %.6f"%(epoch+1,cost))
                logger.info("Optimization Finished!")
        self.fitted = True
        
        return self


    def transform(self, X):
        """Apply autoencoder on X to extract features.


        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_hidden_3)

        """
        assert self.fitted, "This instance is not fitted yet. Call 'fit'\
                with appropriate arguments before using this method."
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)
        if np.isnan(np.min(X)): raise ValueError("It contains NaN.")
        n_s, n_f = X.shape
        assert n_f == self.n_input,"number of cols(%d) != n_input(%d)"\
                        %(n_f,self.n_input)

        feats_op, _, _ = self.__network()
        X_new = np.empty([0, self.n_hidden_3])
        with tf.Graph().as_default():
            with tf.device(self.device):
                for batch in self.__next_batch(X):
                    feats = self.__sess.run(feats_op,feed_dict={self.__X: batch})
                    X_new = np.row_stack((X_new, feats))

        return X_new
