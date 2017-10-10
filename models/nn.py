# Authors: LI AO <aoli@hit.edu.cn>
#
# License: 

from .. import logger

import math
import numpy as np
import tensorflow as tf
from sklearn.utils.validation import check_array

class ANN(object):
    """Artificial Neural Network for Machine Learning Problems.


    Warning: This class should be used based on Tensorflow.
    """

    def __init__(self, 
                 n_hiddens,
                 n_classes=2,
                 training_epochs=20, 
                 learning_rate=0.01, 
                 batch_size=256, 
                 objective='softmax',
                 display_step=1,
                 use_gpu=True,
                 gpu_id=0):
        self.n_hiddens = check_array(n_hiddens,ensure_2d=False,dtype=int)
        assert self.n_hiddens.ndim == 1, "n_hiddens must be 1-d array(list of n.nodes)" 
        self.n_classes = n_classes
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.display_step = display_step
        self.obj = objective
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.fitted = False


    def __variables(self):
        self.__X = tf.placeholder(tf.float32, [None, self.n_input])
        self.__y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.weights = dict(); self.biases = dict(); h_lens = len(self.n_hiddens)
        self.weights['w1'] = tf.Variable(tf.random_normal([self.n_input, self.n_hiddens[0]]))
        self.biases['b1'] = tf.Variable(tf.random_normal([self.n_hiddens[0]]))
        for i in range(h_lens-1):
            self.weights['w'+str(i+2)] = tf.Variable(
                             tf.random_normal([self.n_hiddens[i], self.n_hiddens[i+1]]))
            self.biases['b'+str(i+2)] = tf.Variable(tf.random_normal([self.n_hiddens[i+1]]))
        self.weights['w'+str(h_lens+1)] = tf.Variable(
                             tf.random_normal([self.n_hiddens[h_lens-1], self.n_classes]))
        self.biases['b'+str(h_lens+1)] = tf.Variable(tf.random_normal([self.n_classes])),

    
    def __next_batch(self, X, y=None):
        total_batch = int(math.ceil(float(X.shape[0])/self.batch_size))
        for i in range(total_batch):
            start = i * self.batch_size
            end = min((i+1)*self.batch_size, X.shape[0])
            if y is None: yield X[start:end]
            else: yield X[start:end], y[start:end]


    def __network(self):
        # network
        logits = self.__forward(self.__X)
        # loss function
        if self.obj == 'softmax': cost = tf.losses.softmax_cross_entropy(self.__y, logits)
        elif self.obj == 'logloss': cost = tf.losses.log_loss(self.__y, logits)
        else: cost = tf.losses.mean_squared_error(self.__y, logits)  # mse
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        
        return logits, cost, optimizer


    def __forward(self, x):
        layer = x   # for convenience
        for i in range(1, len(self.weights)+1):
            layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, self.weights['w'+str(i)]),
                                         self.biases['b'+str(i)]))
        return layer
    
    
    def fit(self, X, y):
        """Build a neural network classifier from the training set (X, y).

        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_classes] (one-hot), 
            The target values (class labels) as integers.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """  
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)
        y = check_array(y, ensure_2d=False)
        if np.isnan(np.min(X)): raise ValueError("It contains NaN.")
        y = np.atleast_1d(y)

        if y.ndim == 1:     # convert to one hot code
            from sklearn.preprocessing import OneHotEncoder as OHE
            y = np.reshape(y, (-1, 1))
            model = OHE(sparse=False)
            y = np.array(model.fit_transform(y))

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
                    for batch_X, batch_y in self.__next_batch(X, y):
                        cost, _ = self.__sess.run([cost_op, optimizer_op], \
                                feed_dict={self.__X: batch_X, self.__y: batch_y})
                    if epoch % self.display_step == 0:
                        logger.info("Epoch: %04d, cost = %.6f"%(epoch+1,cost))
                logger.info("Optimization Finished!")
        self.fitted = True
        
        return self


    def predict_proba(self, X, check_input=True):
        """Predict class probabilities on the input samples X.


        check_input : boolean, (default=True)
            Allow to bypass serveral input checking.
            Don't use this parameter unless you know what you do.


        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        
        check_input : bool
            Run check_array on X.


        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_classes
            such arrays if n_classes > 1.
            The class probabilities of the input samples.
        """
        assert self.fitted, "This instance is not fitted yet. Call 'fit'\
                with appropriate arguments before using this method."
        if check_input: 
            X = check_array(X, ensure_min_samples=2, ensure_min_features=2)
            if np.isnan(np.min(X)): raise ValueError("It contains NaN.")
        n_s, n_f = X.shape
        assert n_f == self.n_input,"number of cols(%d) != n_input(%d)"\
                        %(n_f,self.n_input)

        feats_op, _, _ = self.__network()
        p = np.empty([0, self.n_classes])
        with tf.Graph().as_default():
            with tf.device(self.device):
                for batch in self.__next_batch(X):
                    feats = self.__sess.run(feats_op,feed_dict={self.__X: batch})
                    p = np.row_stack((p, feats))

        return p


    def predict(self, X):
        """Predict class value for X.

        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.


        Returns
        -------
        y : array of shape = [n_samples]  The predicted classes.  
        """
        p = self.predict_proba(X)
        y = np.argmax(p, axis=1)
        return y

    
    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.


        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_classes)
            True labels for X.


        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """ 
        y = check_array(y, ensure_2d=False)
        y = np.atleast_1d(y)

        if y.ndim == 2:     # from one-hot code
            y = np.argmax(p, axis=1)
 
        pred_y = self.predict(X)
        score = sum(pred_y == y) / float(len(y)) 
        return score
