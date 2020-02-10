
import numpy as np
import tensorflow as tf
import copy
import scipy.stats
import gym
import baselines.common.tf_util as U
import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def pseudo_action_swap_matrix_multi(pi,phi): # for multi-variable, only return vector instead of matrix
    C=len(pi)
    RaceAllSwap = np.log(pi[:,np.newaxis])-phi[np.newaxis,:]
    Race = np.diag(RaceAllSwap)
    action_true = np.argmin(Race)

    #tic()
    if C<=6: # True: #True:
        #Slow version for large C
        pseudo_actions=np.full((C, C), action_true)
        for m in range(C):
            for jj in  range(m):
                RaceSwap = Race.copy()
                RaceSwap[m], RaceSwap[jj]=RaceAllSwap[jj,m],RaceAllSwap[m,jj]
                s_action = np.argmin(RaceSwap)
                pseudo_actions[m,jj], pseudo_actions[jj,m] = s_action, s_action

    else:
        Race_min = Race[action_true]
        #Fast version for large C
        pseudo_actions=np.full((C, C), action_true)
        SwapSuccess = RaceAllSwap<=Race_min
        SwapSuccess[action_true,:]=True
        np.fill_diagonal(SwapSuccess,0)
        m_idx,j_idx = np.where(SwapSuccess)
        for i in range(len(m_idx)):
            m,jj = m_idx[i],j_idx[i]
            RaceSwap = Race.copy()
            RaceSwap[m], RaceSwap[jj]=RaceAllSwap[jj,m],RaceAllSwap[m,jj]
            if m==action_true or jj == action_true:
                s_action = np.argmin(RaceSwap)
                pseudo_actions[m,jj], pseudo_actions[jj,m] = s_action, s_action
            else:
                if RaceSwap[m]<RaceSwap[jj]:
                    pseudo_actions[m,jj], pseudo_actions[jj,m] = m, m
                else:
                    pseudo_actions[m,jj], pseudo_actions[jj,m] = jj, jj
    return np.reshape(pseudo_actions, (-1))


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

def discount_reward(rewards, gamma): # no normalization
    dr = np.sum(np.power(gamma,np.arange(len(rewards)))*rewards)
    return dr

def discount_rewards(rewards, gamma): # no normalization
    drs = np.sum(np.power(gamma,np.arange(len(rewards)))*rewards)[None]
    return drs

def swap(array, a,b):
    array[a], array[b] = array[b], array[a]
    return array

def discount_reward_multi(rewards, gamma): # no normalization
    discounted = []
    r = 0
    for reward in rewards[::-1]:
        r = reward + gamma*r # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

    
def scheduler(lr, n_total, n_current, factor):
    return lr * (1 - factor * n_current/n_total)


def polynomial_decay(lr_start, global_step, decay_steps, end_lr, power):
    global_step = min(global_step, decay_steps)
    decayed_learning_rate = (lr_start - end_lr) \
    * (1 - global_step / decay_steps) ** (power) \
    + end_lr
    return decayed_learning_rate

def find_location(Arr, elem, K):
    return (np.where(np.sum(Arr==elem, axis=1)==K)[0])
    