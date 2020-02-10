import tensorflow as tf
import numpy as np
import random
import gym
import tensorflow.contrib.eager as tfe
from carsm_util_2 import *
import multiprocessing
import sys
import baselines.common.tf_util as U
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from itertools import permutations, repeat
import itertools
import copy
from gym import utils

class ARSM_TRPO_AGENT_mujoco(tf.keras.Model):
    def __init__(self, nA, network, tau, Grad_Clip_Norm, K, C, KL_penalty, d_target):
        super(ARSM_TRPO_AGENT_mujoco, self).__init__()
        """ Define here the layers used during the forward-pass 
            of the neural network.
        """
        self.network = network
        self.tau = tau
        self.K = K
        self.C = C
        self.Grad_Clip_Norm = Grad_Clip_Norm
        self.KL_penalty = KL_penalty
        self.d_target = d_target
        if self.network == 'Linear':        
            # Hidden layer.
            self.dense_layer1 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal())
            self.dense_layer2 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            self.dense_layer3 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            self.dense_layer4 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            # Output layer. No activation.
            self.policy_layer = tf.layers.Dense(nA, activation=None, kernel_initializer=tf.keras.initializers.he_normal())
            self.q_layer = tf.layers.Dense(1, activation=None,kernel_initializer=tf.keras.initializers.he_normal())

            
            ### target network
            self.dense_layer_c1 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal())
            self.dense_layer_c2 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            self.dense_layer_c3 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            self.dense_layer_c4 = tf.layers.Dense(64, activation=tf.nn.tanh,kernel_initializer=tf.keras.initializers.he_normal()) 
            # Output layer. No activation.
            self.policy_layer_c = tf.layers.Dense(nA, activation=None, kernel_initializer=tf.keras.initializers.he_normal())
            self.q_layer_c = tf.layers.Dense(1, activation=None,kernel_initializer=tf.keras.initializers.he_normal())            
            
        elif self.network == 'Convolution':
            self.cnn1 = tf.layers.Conv2D(32, 8, 4, kernel_initializer=ortho_init(np.sqrt(2)),\
                                  activation = tf.nn.tanh, padding = 'valid')
            self.cnn2 = tf.layers.Conv2D(64, 4, 2,kernel_initializer=ortho_init(np.sqrt(2)),\
                                         activation = tf.nn.tanh, padding = 'valid')
            self.cnn3 = tf.layers.Conv2D(64, 3, 1,\
                                  kernel_initializer=ortho_init(np.sqrt(2)), activation = tf.nn.tanh,\
                                  padding = 'valid')
            self.dense_layer = tf.layers.Dense(512, activation=tf.nn.tanh, kernel_initializer=ortho_init(np.sqrt(2))) 
            # Output layer. No activation.
            self.policy_layer = tf.layers.Dense(nA, activation=None,kernel_initializer=ortho_init(np.sqrt(1)))
            self.q_layer = tf.layers.Dense(nA, activation=None,kernel_initializer=ortho_init(np.sqrt(1)))
            
            ### target network            
            self.cnn_c1 = tf.layers.Conv2D(32, 8, 4, kernel_initializer=ortho_init(np.sqrt(2)),\
                                  activation = tf.nn.tanh, padding = 'valid')
            self.cnn_c2 = tf.layers.Conv2D(64, 4, 2,kernel_initializer=ortho_init(np.sqrt(2)),\
                                         activation = tf.nn.tanh, padding = 'valid')
            self.cnn_c3 = tf.layers.Conv2D(64, 3, 1,\
                                  kernel_initializer=ortho_init(np.sqrt(2)), activation = tf.nn.tanh,\
                                  padding = 'valid')
            self.dense_layer_c = tf.layers.Dense(512, activation=tf.nn.tanh, kernel_initializer=ortho_init(np.sqrt(2))) 
            # Output layer. No activation.
            self.policy_layer_c = tf.layers.Dense(nA, activation=None,kernel_initializer=ortho_init(np.sqrt(1)))
            self.q_layer_c = tf.layers.Dense(nA, activation=None,kernel_initializer=ortho_init(np.sqrt(1)))            
    def duplicate(self):
        if self.network == 'Linear':        
            # copy the parameters
            [v_t.assign(v) for v_t, v in zip(self.dense_layer_c1.variables, self.dense_layer1.variables)]                
            [v_t.assign(v) for v_t, v in zip(self.dense_layer_c2.variables, self.dense_layer2.variables)]                
            [v_t.assign(v) for v_t, v in zip(self.dense_layer_c3.variables, self.dense_layer3.variables)]                
            [v_t.assign(v) for v_t, v in zip(self.dense_layer_c4.variables, self.dense_layer4.variables)]                
            [v_t.assign(v) for v_t, v in zip(self.policy_layer_c.variables, self.policy_layer.variables)]                
            [v_t.assign(v) for v_t, v in zip(self.q_layer_c.variables, self.q_layer.variables)]                            
        elif self.network == 'Convolution':
            [v_t.assign(v) for v_t, v in zip(self.cnn_c1.variables, self.cnn1.variables)]
            [v_t.assign(v) for v_t, v in zip(self.cnn_c2.variables, self.cnn2.variables)]
            [v_t.assign(v) for v_t, v in zip(self.cnn_c3.variables, self.cnn3.variables)]
            [v_t.assign(v) for v_t, v in zip(self.dense_layer_c.variables, self.dense_layer.variables)]
            [v_t.assign(v) for v_t, v in zip(self.policy_layer_c.variables, self.policy_layer.variables)]
            [v_t.assign(v) for v_t, v in zip(self.q_layer_c.variables, self.q_layer.variables)]
        
        
    def  dup_q_estimation(self,state):
        if self.network == 'Linear':
                
            h1 = self.dense_layer_c3(state)
            h2 = self.dense_layer_c4(h1)
            q_values = self.q_layer_c(h2)        

            return q_values
        elif self.network == 'Convolution':
            h1 = self.cnn_c1(tf.cast(state, tf.float32)/255.)
            h2 = self.cnn_c2(h1)
            h3 = self.cnn_c3(h2)            
            h3 = conv_to_fc(h3)
            h4 = self.dense_layer_c(h3)
            q_values = self.q_layer_c(h4)
        
            return q_values        
        
    def dup_policy(self, state):
        if self.network == 'Linear':
                
            h1 = self.dense_layer_c1(state)
            h2 = self.dense_layer_c2(h1)
            logits = self.policy_layer_c(h2)

            return logits
        elif self.network == 'Convolution':
            h1 = self.cnn_c1(tf.cast(state, tf.float32)/255.)
            h2 = self.cnn_c2(h1)
            h3 = self.cnn_c3(h2)            
            h3 = conv_to_fc(h3)
            h4 = self.dense_layer_c(h3)
            logits = self.policy_layer_c(h4)
        
            return logits
    
    def update_target(self):
        if self.network == 'Linear':
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.dense_layer_c1.variables, self.dense_layer1.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.dense_layer_c2.variables, self.dense_layer2.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.dense_layer_c3.variables, self.dense_layer3.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.dense_layer_c4.variables, self.dense_layer4.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.policy_layer_c.variables, self.policy_layer.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.q_layer_c.variables, self.q_layer.variables)]      
        elif self.network == 'Convolution':                      
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.cnn_c1.variables, self.cnn1.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.cnn_c2.variables, self.cnn2.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.cnn_c3.variables, self.cnn3.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.dense_layer_c.variables, self.dense_layer.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.policy_layer_c.variables, self.policy_layer.variables)]                
            [v_t.assign((1-self.tau)*v+self.tau*v_t) for v_t, v in zip(self.q_layer_c.variables, self.q_layer.variables)]   

    
    def policy(self, state):
        if self.network == 'Linear':
                
            h1 = self.dense_layer1(state)
            h2 = self.dense_layer2(h1)
            logits = self.policy_layer(h2)

            return logits
        elif self.network == 'Convolution':
            h1 = self.cnn1(tf.cast(state, tf.float32)/255.)
            h2 = self.cnn2(h1)
            h3 = self.cnn3(h2)            
            h3 = conv_to_fc(h3)
            h4 = self.dense_layer(h3)
            logits = self.policy_layer(h4)
        
            return logits
    
    def q_estimation(self, state):
        if self.network == 'Linear':
                
            h1 = self.dense_layer3(state)
            h2 = self.dense_layer4(h1)
            q_values = self.q_layer(h2)        

            return q_values
        elif self.network == 'Convolution':
            h1 = self.cnn1(tf.cast(state, tf.float32)/255.)
            h2 = self.cnn2(h1)
            h3 = self.cnn3(h2)            
            h3 = conv_to_fc(h3)
            h4 = self.dense_layer(h3)
            q_values = self.q_layer(h4)
        
            return q_values        

    
    def loss_fn_q(self, states, q_target, actions, coef_q):
        """ Defines the loss function used during 
            training.         
        """
        q_input = tf.concat([states,actions], axis =1)
        q_values= self.q_estimation(q_input)
        q_values = tf.reshape(q_values, [-1])
        loss = coef_q * tf.reduce_mean(tf.square(q_values - q_target))
        return loss
    
    def grads_fn_q(self, states, q_target, actions,coef_q):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn_q(states, q_target, actions, coef_q)
        return tape.gradient(loss, self.variables)
    
    def entropy(self, states):
        logits = self.policy(states)
        K = self.K
        C = self.C
        logits = tf.reshape(logits, (-1,K,C))
        ent = np.float32(0)
        for kk in range(K):
            logits_tmp = logits[:,kk,:]
            a0 = logits_tmp - tf.reduce_max(logits_tmp, 1, keep_dims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
            p0 = ea0 / z0
            ent += tf.reduce_sum(p0 * (tf.log(z0) - a0))
        return ent
    
    def KL_div(self, states, logits_old):  # can only use 10% of data to accelerate the computation
        logits_new = self.policy(states)
        K = self.K
        C = self.C
        logits_new = tf.reshape(logits_new, (-1,K,C))
        logits_old = tf.stop_gradient(tf.reshape(logits_old, (-1,K,C)))

        logits_new_rescale = logits_new - tf.reduce_max(logits_new, 2, keep_dims = True)
        logits_old_rescale = logits_old - tf.reduce_max(logits_old, 2, keep_dims = True)
        a1 = tf.reduce_sum((logits_old_rescale - logits_new_rescale) * tf.exp(logits_old_rescale),axis = 2)
        a2 = tf.reduce_sum(tf.exp(logits_old_rescale),axis = 2)
        a4 = tf.log(tf.reduce_sum(tf.exp(logits_new_rescale), axis = 2)+1e-8)
        a3 = tf.log(a2+1e-8)
        kl = tf.reduce_sum(a1 / a2 - a3 + a4)
        return kl    
    
    def get_prob(self, states, actions_onehot, compress):
        K = self.K
        C = self.C
        logit_new = self.policy(states)
        logit_new = tf.reshape(logit_new, (-1, K, C))
        prob_new = tf.nn.softmax(logit_new, axis = 2)
        prob_new = tf.clip_by_value(prob_new, clip_value_max=2,clip_value_min=1e-10)#avoid NA when taking log
        if compress:
            prob_new = tf.reshape(tf.reduce_prod(tf.reduce_sum(prob_new * actions_onehot, axis = 2), axis = 1), (-1,1))                
        return prob_new
        
    
    def loss_fn_policy(self, states, f_delta, ent_par, K, C):
        logit = self.policy(states)
        logit = tf.reshape(logit, (-1, K, C))
        ent = self.entropy(states) #/ (K*len(states))
        return tf.reduce_mean(tf.multiply(logit, f_delta)) - (ent_par/(K*len(states)))*ent
    
    def grads_fn_policy(self, states, f_delta, ent_par, K, C):

        with tfe.GradientTape() as tape:
            loss = self.loss_fn_policy(states, f_delta, ent_par, K, C)
        return tape.gradient(loss, self.variables)
        
    def update_policy(self, states, optimizer, f_delta, ent_par, K, C):
        
        grad_policy = self.grads_fn_policy(states, f_delta, ent_par, K, C)
        optimizer.apply_gradients(zip(grad_policy, self.variables))
        return grad_policy
        
    def update_q(self, states, optimizer, q_target, Grad_Clip_NN, actions, coef_q):
        
        grad_q = self.grads_fn_q(states, q_target, actions, coef_q)
        if Grad_Clip_NN:
            grad_q = tf.clip_by_global_norm(grad_q, self.Grad_Clip_Norm)[0]
        optimizer.apply_gradients(zip(grad_q, self.variables))
        return grad_q
    
    def loss_q_one_hot(self, states, q_target, actions):
        actions = actions.astype(int)
        actions = action_one_hot(actions, self.K, self.C)
        q_input = tf.concat([states,actions], axis =1)
        q_values= self.q_estimation(q_input)
        q_values = tf.reshape(q_values, [-1])
        loss = tf.reduce_mean(tf.square(q_values - q_target))
        return loss        
    
    def grads_fn_q_one_hot(self, states, q_target, actions):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_q_one_hot(states, q_target, actions)
        return tape.gradient(loss, self.variables)    

    def update_q_one_hot(self, states, optimizer, q_target, Grad_Clip_NN, actions):
        grad_q = self.grads_fn_q_one_hot(states, q_target, actions)
        if Grad_Clip_NN:
            grad_q = tf.clip_by_global_norm(grad_q, self.Grad_Clip_Norm)[0]
        optimizer.apply_gradients(zip(grad_q, self.variables))
            
    def loss_a2c(self,states, A, actions_onehot):
        prob = self.get_prob(states, actions_onehot, True)

        A -= tf.reduce_mean(A)
        mean,var = tf.nn.moments(A,0)
        A /= tf.sqrt(var)
        obj = - A * tf.math.log(prob)        
        return tf.reduce_mean(obj)
    
    def grads_fn_policy_a2c(self, states, A, actions_onehot):

        with tfe.GradientTape() as tape:
            loss = self.loss_a2c(states, A, actions_onehot)
        return tape.gradient(loss, self.variables)
    
    def test(self, states, logits_old, v):
        K = self.K
        C = self.C        
        logits_new = self.policy(states)     
        logits_new = tf.reshape(logits_new, (-1,K,C))
        logits_old = tf.stop_gradient(tf.reshape(logits_old, (-1,K,C)))
        prob_old = tf.nn.softmax(logits_old)
        return -tf.reduce_sum(tf.reduce_mean(tf.stop_gradient(prob_old) * tf.log(tf.nn.softmax(logits_new)+1e-8),axis=0))

    
def fisher_vector_product(v,model,states,logits_old):
    with tfe.GradientTape() as t1:
        with tfe.GradientTape() as t2:
            t1.watch(model.variables)
            t2.watch(model.variables)
            loss = model.test(states,logits_old,v)
            grads = t2.gradient(loss, model.variables)
        grads_flat = tf.concat([tf.reshape(grad_,[-1])for grad_ in grads if grad_ is not None],axis = 0)
        grads_v = tf.reduce_sum(grads_flat * v)
    
        grads_grads_v = t1.gradient(grads_v, model.variables)
    
    return np.array(tf.concat([tf.reshape(grad_,[-1])for grad_ in grads_grads_v if grad_ is not None],axis = 0))


def set_from_flat(grad, grad_prev):
    assigns = []
    start = 0
    for i in range(len(grad_prev)):
        current_layer = grad_prev[i]
        if current_layer is not None:
            shape = current_layer.shape
            size = int(np.prod(shape))
            assigns.append(tf.reshape(grad[start:start + size], shape))
            start += size
        else:
            assigns.append(None)
    return assigns
            
import numpy as np
def cg_arsm(f_Ax, b, model, states, logits_old, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p,model,states,logits_old)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print(fmtstr % (i+1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x
            
def action_one_hot(actions_idx, K, C):
    actions_idx = actions_idx.astype(int)
    N = len(actions_idx)
    idx_tmp1 = np.reshape(np.tile(range(K),N),(N,-1)) * C
    seq_2_rep = np.linspace(start = 0, stop = (N-1)*K*C, num = N).astype(int)
    idx_tmp2 = np.reshape(np.repeat(seq_2_rep, K),(N,-1))
    idx_vec = np.hstack(idx_tmp1 + idx_tmp2 + actions_idx)
    action_one_hot_return = np.zeros((N*K*C))
    action_one_hot_return[idx_vec] = 1
    action_one_hot_return = np.reshape(action_one_hot_return,(N,K*C))
    return action_one_hot_return
    