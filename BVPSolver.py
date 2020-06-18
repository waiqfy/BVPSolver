# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:56:37 2020

@author: hshcc
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
tf.random.set_seed(666)

class BVPSolver:
    # Initialize the class
    def __init__(self, X, EBC_pos,EBC,NBC_pos,NBC,layers):

        #lower/upper bound of X
        self.lb = X.min()
        self.ub = X.max()

        # Collocation point
        self.x = tf.reshape(tf.convert_to_tensor(X,dtype=tf.float32),[100,1])

        #boundary conditions
        self.ebc_pos=tf.reshape(tf.convert_to_tensor(EBC_pos,dtype=tf.float32),[1,1])
        self.nbc_pos=tf.reshape(tf.convert_to_tensor(NBC_pos,dtype=tf.float32),[1,1])
        self.ebc=tf.convert_to_tensor(EBC, dtype=tf.float32)
        self.nbc=tf.convert_to_tensor(NBC, dtype=tf.float32)

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        self.layers = layers

        self.optimizer_Adam = tf.keras.optimizers.Adam()

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,
                                                      dtype=tf.float32), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights)
        H=X
#        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-1):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(H@W+b)
        W = weights[-1]
        b = biases[-1]
        Y = H@W+b
        return Y

    def net_u(self, x):
        u = self.neural_net(x, self.weights, self.biases)
        return u

    def net_BCs(self):
        u = self.net_u(self.ebc_pos)
        with tf.GradientTape() as g:
            g.watch(self.nbc_pos)
            tempu= self.net_u(self.nbc_pos)
        u_x = g.gradient(tempu, self.nbc_pos)
        return u,u_x

    def net_f(self, x):

        with tf.GradientTape() as uxx:
            with tf.GradientTape() as ux:
                ux.watch(x)
                uxx.watch(x)
                u = self.net_u(x)
            u_x = ux.gradient(u, x)
        u_xx = uxx.gradient(u_x, x)
        f = -u_xx -u + tf.square(x)
        return f

    def MSE_loss(self):
        ebc_pred, nbc_pred = self.net_BCs()
        f_pred = self.net_f(self.x)

        loss=tf.reduce_mean(tf.square(self.ebc - ebc_pred)) + \
                    tf.reduce_mean(tf.square(self.nbc - nbc_pred)) + \
                    tf.reduce_mean(tf.square(f_pred))
        return loss

    def train(self, tol,nIter):
        MSE=[]
        plt.figure('Training loss')
        niter=nIter
        for it in range(nIter):

            if self.MSE_loss().numpy() < tol:
                niter=it+1
                print('Solved, iteration='+str(niter))
                break
            var_list=[self.weights,self.biases]
            self.optimizer_Adam.minimize(self.MSE_loss, var_list)
            MSE.append(self.MSE_loss().numpy())
        plt.plot(range(1,niter+1),MSE)
        plt.title('Training loss='+str(np.min(MSE)))
        plt.xlabel('iteration')
        plt.ylabel('MSE')
        plt.show()
        return MSE

    def predict(self, X_star=0):
        if X_star==0:
            X_star=self.x
        u_star = self.net_u(X_star)
        f_star = self.net_f(X_star)
        return u_star.numpy(), f_star.numpy()


if __name__ == "__main__":

    iteration=2000

    x=np.linspace(0,1,100)
    true_u=(2*np.cos(1-x)-np.sin(x))/np.cos(1)+np.square(x)-2

    EBC_pos=np.array([0.])
    EBC=np.array([0.])
    NBC_pos=np.array([1.])
    NBC=np.array([1.])

    layers = [1, 20, 20,20,20, 1]

    model = BVPSolver(x, EBC_pos,EBC,NBC_pos,NBC,layers)
    result=model.train(1e-5,iteration) # MSE

    u_pred, f_pred = model.predict()
#%%
u_FEM=-x/2153*(-2776+301*x+7*np.square(x))

MSE_NN=np.mean(np.square(true_u-u_pred[:,0]))
MSE_FEM=np.mean(np.square(true_u-u_FEM))

plt.figure()
plt.plot(x,u_pred,'b+')
#plt.plot(x,u_FEM,'b')
plt.plot(x,true_u,'g+')
plt.xlabel('x')
plt.ylabel('U')
plt.title('MSE='+str(MSE_NN)+' \n iteration='+str(iteration)+' \n layer='+str(layers))
plt.legend(['NN','analytical'])

plt.show()
